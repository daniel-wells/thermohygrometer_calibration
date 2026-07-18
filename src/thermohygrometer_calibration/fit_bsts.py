from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


REQUIRED_COLUMNS = {"timestamp", "device_id", "line", "position", "epoch", "temp_c", "humidity_rh"}


def _write_trace_atomic(idata: az.InferenceData, trace_path: Path, retries: int = 6, delay_s: float = 0.8) -> None:
    """Write NetCDF via temp file + atomic replace to avoid clobber on lock failures."""
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None

    for attempt in range(retries):
        temp_path = trace_path.with_name(f"{trace_path.stem}.tmp-{os.getpid()}-{attempt}{trace_path.suffix}")
        try:
            if temp_path.exists():
                temp_path.unlink()

            idata.to_netcdf(temp_path, engine="h5netcdf")
            os.replace(temp_path, trace_path)
            return
        except (BlockingIOError, OSError, PermissionError) as exc:
            last_err = exc
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
            if attempt < retries - 1:
                time.sleep(delay_s)
            continue

    raise RuntimeError(
        f"Failed to write trace after {retries} attempts due to file locking: {trace_path}"
    ) from last_err


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a Bayesian structural time-series (BSTS) model with a shared "
            "local-level drift and effects for line, position, epoch, and device."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["actual", "simulated"],
        default="actual",
        help="Dataset label used for default input/output paths.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Processed long-format CSV from thermohygrometer_calibration.analyze.",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["temp_c", "humidity_rh"],
        default="temp_c",
        help="Response variable to model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where model outputs are written.",
    )
    parser.add_argument("--draws", type=int, default=1000, help="Posterior draws per chain.")
    parser.add_argument("--tune", type=int, default=1000, help="Tuning steps per chain.")
    parser.add_argument("--chains", type=int, default=4, help="Number of MCMC chains.")
    parser.add_argument(
        "--target-accept",
        type=float,
        default=0.9,
        help="NUTS target acceptance probability.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--position-effects",
        action="store_true",
        default=False,
        help=(
            "Include placement effects (both line and position) in the model. "
            "Off by default: line/position are confounded with device when each device "
            "occupies one fixed location (single-epoch data). Enable once crossover epoch "
            "data is available so placement effects are separately identified."
        ),
    )
    return parser.parse_args()


def _prepare_design(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp", target_col, "device_id", "line", "position", "epoch"])
    work = work.sort_values(["timestamp", "device_id"]).reset_index(drop=True)

    if work.empty:
        raise ValueError("No valid rows remain after filtering missing values.")

    devices = np.sort(work["device_id"].astype(str).unique())
    lines = np.sort(work["line"].astype(int).unique())
    positions = np.sort(work["position"].astype(int).unique())
    epochs = np.sort(work["epoch"].astype(str).unique())
    time_points = np.sort(work["timestamp"].unique())

    device_map = {name: idx for idx, name in enumerate(devices)}
    line_map = {name: idx for idx, name in enumerate(lines)}
    position_map = {name: idx for idx, name in enumerate(positions)}
    epoch_map = {name: idx for idx, name in enumerate(epochs)}
    time_map = {name: idx for idx, name in enumerate(time_points)}

    index_data = {
        "device_idx": work["device_id"].astype(str).map(device_map).to_numpy(dtype="int64"),
        "line_idx": work["line"].astype(int).map(line_map).to_numpy(dtype="int64"),
        "position_idx": work["position"].astype(int).map(position_map).to_numpy(dtype="int64"),
        "epoch_idx": work["epoch"].astype(str).map(epoch_map).to_numpy(dtype="int64"),
        "time_idx": work["timestamp"].map(time_map).to_numpy(dtype="int64"),
        "y": work[target_col].to_numpy(dtype="float64"),
    }

    coords = {
        "obs": np.arange(len(work), dtype="int64"),
        "device": devices,
        "line": lines,
        "position": positions,
        "epoch": epochs,
        "time": np.arange(len(time_points), dtype="int64"),
    }

    return work, index_data, coords


def _build_bsts_model(
    index_data: dict[str, np.ndarray],
    coords: dict[str, np.ndarray],
    *,
    has_multiple_epochs: bool,
    position_effects: bool = False,
    observed: bool = True,
) -> pm.Model:
    with pm.Model(coords=coords) as model:
        observed_y = pm.Data("y_obs", index_data["y"], dims="obs") if observed else None
        device_idx = pm.Data("device_idx", index_data["device_idx"], dims="obs")
        line_idx = pm.Data("line_idx", index_data["line_idx"], dims="obs")
        position_idx = pm.Data("position_idx", index_data["position_idx"], dims="obs")
        epoch_idx = pm.Data("epoch_idx", index_data["epoch_idx"], dims="obs")
        time_idx = pm.Data("time_idx", index_data["time_idx"], dims="obs")

        intercept = pm.Normal("intercept", mu=float(np.mean(index_data["y"])), sigma=5.0)

        sigma_device = pm.HalfNormal("sigma_device", sigma=0.5)
        # Centered hierarchical parameterization improves mixing for sigma_device
        # when each device has many observations.
        device_offset = pm.Normal("device_offset", mu=0.0, sigma=sigma_device, dims="device")
        # Enforce sum-to-zero so intercept represents grand mean and device effects
        # are identified as relative offsets rather than sharing a free common mode.
        device_effect = pm.Deterministic("device_effect", device_offset - pm.math.mean(device_offset), dims="device")

        # Placement effects (line + position) are only identified when devices
        # move across locations (requires crossover epoch data).
        if position_effects:
            # Line effect: only 2 levels, so sum-to-zero reduces to a single scalar.
            # line_delta > 0 means line 0 (sorted first) is warmer than the grand mean.
            n_lines = len(coords["line"])
            if n_lines == 2:
                line_delta = pm.Normal("line_delta", mu=0.0, sigma=1.0)
                line_signs = pm.Data("line_signs", np.array([1.0, -1.0]), dims="line")
                line_effect = pm.Deterministic("line_effect", line_delta * line_signs, dims="line")
            else:
                line_raw = pm.Normal("line_raw", mu=0.0, sigma=1.0, dims="line")
                line_effect = pm.Deterministic("line_effect", line_raw - pm.math.mean(line_raw), dims="line")

            position_raw = pm.Normal("position_raw", mu=0.0, sigma=1.0, dims="position")
            position_effect = pm.Deterministic(
                "position_effect", position_raw - pm.math.mean(position_raw), dims="position"
            )

        if has_multiple_epochs:
            epoch_raw = pm.Normal("epoch_raw", mu=0.0, sigma=1.0, dims="epoch")
            epoch_effect = pm.Deterministic(
                "epoch_effect", epoch_raw - pm.math.mean(epoch_raw), dims="epoch"
            )

        # sigma_level controls per-minute latent drift.
        # SD after T steps is sqrt(T) * sigma_level.
        # HalfNormal(0.05) → ~1.9°C daily SD; HalfNormal(0.1) → ~3.8°C daily SD.
        sigma_level = pm.HalfNormal("sigma_level", sigma=0.05)
        # GaussianRandomWalk is equivalent to cumsum of Normal(0, sigma_level) steps
        # but expressed as a single multivariate distribution that PyMC handles more
        # efficiently. init_dist anchors the first state near 0.
        _level_raw = pm.GaussianRandomWalk(
            "local_level_raw",
            sigma=sigma_level,
            dims="time",
            init_dist=pm.Normal.dist(mu=0.0, sigma=5.0),
        )
        # Mean-center so the intercept is not confounded with the average
        # elevation of the random walk over the observation window.
        local_level = pm.Deterministic("local_level", _level_raw - pm.math.mean(_level_raw), dims="time")

        # sigma_obs: SwitchBot sensors have ~0.4°C stated accuracy. Tightened to HalfNormal(0.2)
        # to reduce trading off with sigma_level.
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.2)

        mu = intercept + device_effect[device_idx]
        if position_effects:
            mu = mu + line_effect[line_idx]
            mu = mu + position_effect[position_idx]
        if has_multiple_epochs:
            mu = mu + epoch_effect[epoch_idx]
        mu = mu + local_level[time_idx]

        pm.Normal("likelihood", mu=mu, sigma=sigma_obs, observed=observed_y, dims="obs")

    return model


def fit_bsts(df: pd.DataFrame, target_col: str, draws: int, tune: int, chains: int, target_accept: float, seed: int, position_effects: bool = False) -> az.InferenceData:
    work, index_data, coords = _prepare_design(df, target_col)
    has_multiple_epochs = work["epoch"].nunique() > 1

    model = _build_bsts_model(
        index_data,
        coords,
        has_multiple_epochs=has_multiple_epochs,
        position_effects=position_effects,
        observed=True,
    )

    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
        )

    return idata


def run(input_path: Path, target_col: str, output_dir: Path, draws: int, tune: int, chains: int, target_accept: float, seed: int, position_effects: bool = False) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Processed input file not found: {input_path}")

    df = pd.read_csv(input_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Processed input missing columns: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    idata = fit_bsts(
        df=df,
        target_col=target_col,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        seed=seed,
        position_effects=position_effects,
    )

    summary_var_names = [
        "intercept",
        "sigma_obs",
        "sigma_device",
        "sigma_level",
        "device_effect",
        "local_level",
    ]
    if position_effects and "line_effect" in idata.posterior:
        summary_var_names.append("line_effect")
    if position_effects and "position_effect" in idata.posterior:
        summary_var_names.append("position_effect")
    if "epoch_effect" in idata.posterior:
        summary_var_names.append("epoch_effect")

    model_summary = az.summary(idata, var_names=summary_var_names)

    summary_path = output_dir / f"{target_col}_summary.csv"
    trace_path = output_dir / f"{target_col}_trace.nc"

    model_summary.to_csv(summary_path)
    _write_trace_atomic(idata, trace_path)


def main() -> None:
    args = parse_args()

    input_path = args.input
    if input_path is None:
        input_path = Path("data/processed") / args.dataset / "measurements_long.csv"

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("results") / args.dataset / "model"

    run(
        input_path=input_path,
        target_col=args.target,
        output_dir=output_dir,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        position_effects=args.position_effects,
        target_accept=args.target_accept,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
