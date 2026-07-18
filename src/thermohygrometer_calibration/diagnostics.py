from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import matplotlib
import numpy as np
import pandas as pd
import pymc as pm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermohygrometer_calibration.fit_bsts import _build_bsts_model, _prepare_design, REQUIRED_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostic plots: pairs plot from trace + prior predictive trajectories."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["actual", "simulated"],
        default="actual",
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--target", type=str, choices=["temp_c", "humidity_rh"], default="temp_c")
    parser.add_argument("--trace", type=Path, default=None, help="Path to existing .nc trace file.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prior-samples", type=int, default=50, help="Number of prior predictive draws.")
    parser.add_argument(
        "--trace-include-time",
        action="store_true",
        default=False,
        help=(
            "Include time-indexed latent variables (e.g. level_step/local_level) in trace plots. "
            "Disabled by default because these can create very large figures."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _expand_vars(posterior, var_names: list[str]) -> list[tuple[str, np.ndarray]]:
    """Return (label, 1-D samples) for scalar and vector parameters.

    For vector parameters (device_effect, line_effect, etc.) each element is
    expanded to a separate entry using its coordinate label when available.
    """
    entries: list[tuple[str, np.ndarray]] = []
    stacked = posterior.stack(sample=("chain", "draw"))
    for var in var_names:
        if var not in stacked:
            continue
        da = stacked[var]
        extra_dims = [d for d in da.dims if d != "sample"]
        if not extra_dims:
            entries.append((var, da.values.ravel()))
        else:
            coord_dim = extra_dims[0]
            n_levels = da.sizes[coord_dim]
            coord_vals = da.coords[coord_dim].values if coord_dim in da.coords else np.arange(n_levels)
            for k, label in enumerate(coord_vals):
                # isel is dimension-order-independent, ravel collapses to 1-D
                entries.append((f"{var}[{label}]", da.isel({coord_dim: k}).values.ravel()))
    return entries


def _pairs_plot(idata: az.InferenceData, output_path: Path) -> None:
    # Prefer free (sampled) scalars over deterministic transforms.
    # line_delta / epoch_delta are the single-scalar reparameterisations;
    # only fall back to the full line_effect / epoch_effect vectors when the
    # scalar form is absent (i.e. >2 levels, sum-to-zero version used instead).
    posterior = idata.posterior.to_dataset() if hasattr(idata.posterior, "to_dataset") else idata.posterior

    all_vars = [
        "intercept", "sigma_obs", "sigma_device", "sigma_level",
        # free scalars preferred; vectors only when scalar absent
        "line_delta",
        "line_effect" if "line_delta" not in posterior else None,
        "epoch_delta",
        "epoch_effect" if "epoch_delta" not in posterior else None,
        "device_effect", "position_effect",
    ]
    all_vars = [v for v in all_vars if v is not None]
    present = [v for v in all_vars if v in posterior]
    entries = _expand_vars(posterior, present)
    labels = [e[0] for e in entries]
    samples = [e[1] for e in entries]

    n = len(entries)
    cell = 1.6
    fig, axes = plt.subplots(n, n, figsize=(cell * n, cell * n))
    if n == 1:
        axes = np.array([[axes]])

    # Collect divergences — must match the stacked sample dimension of the posterior
    div_flat: np.ndarray | None = None
    n_samples = len(samples[0]) if samples else 0
    if hasattr(idata, "sample_stats"):
        try:
            ss = idata.sample_stats.to_dataset() if hasattr(idata.sample_stats, "to_dataset") else idata.sample_stats
            if "diverging" in ss:
                raw = ss["diverging"].values.ravel().astype(bool)
                if len(raw) == n_samples:
                    div_flat = raw
        except Exception:
            pass

    for i in range(n):
        xi = samples[i]
        for j in range(n):
            ax = axes[i, j]
            yj = samples[j]
            if i == j:
                ax.hist(xi, bins=25, color="steelblue", edgecolor="none", density=True)
            else:
                ax.scatter(yj, xi, alpha=0.08, s=1, color="steelblue", rasterized=True)
                if div_flat is not None and div_flat.any():
                    ax.scatter(yj[div_flat], xi[div_flat], alpha=0.9, s=6, color="red", zorder=5)

            # Axis labels only on edges
            if i == n - 1:
                ax.set_xlabel(labels[j], fontsize=6, rotation=45, ha="right")
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=6)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=5)

    fig.suptitle("Posterior pairs — all effects (red = divergences)", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _trace_plot(idata: az.InferenceData, output_path: Path, include_time: bool = False) -> None:
    posterior = idata.posterior.to_dataset() if hasattr(idata.posterior, "to_dataset") else idata.posterior

    # Expand all posterior variables into scalar series entries, optionally
    # excluding high-dimensional time-indexed latent states.
    selected_vars = []
    for var in posterior.data_vars:
        dims = set(posterior[var].dims)
        if not include_time and "time" in dims:
            continue
        selected_vars.append(var)

    entries = _expand_vars(posterior, selected_vars)
    if not entries:
        raise ValueError("No posterior parameters selected for trace plot.")

    # Keep figures readable by paginating when there are many parameters.
    per_page = 18
    n_pages = int(np.ceil(len(entries) / per_page))

    # Overlay divergences by chain/draw when available.
    div = None
    if hasattr(idata, "sample_stats"):
        try:
            ss = idata.sample_stats.to_dataset() if hasattr(idata.sample_stats, "to_dataset") else idata.sample_stats
            if "diverging" in ss:
                div = ss["diverging"].values.astype(bool)
        except Exception:
            div = None

    for page in range(n_pages):
        chunk = entries[page * per_page : (page + 1) * per_page]
        n_rows = len(chunk)
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 2.2 * n_rows), squeeze=False)

        for row, (var_name, values_flat) in enumerate(chunk):
            trace_ax = axes[row, 0]
            dist_ax = axes[row, 1]

            # Recover chain/draw shape if available; fallback to single-chain layout.
            n_chains = int(getattr(idata.posterior.sizes, "get", lambda *_: 1)("chain", 1))
            n_draws = int(getattr(idata.posterior.sizes, "get", lambda *_: len(values_flat))("draw", len(values_flat)))
            if n_chains * n_draws == len(values_flat):
                values = values_flat.reshape(n_chains, n_draws)
            else:
                values = values_flat.reshape(1, -1)
                n_chains, n_draws = values.shape

            draw_idx = np.arange(n_draws)
            for c in range(n_chains):
                chain_vals = values[c]
                trace_ax.plot(draw_idx, chain_vals, linewidth=0.8, alpha=0.8, label=f"chain {c}")
                dist_ax.hist(chain_vals, bins=30, density=True, alpha=0.35, edgecolor="none", label=f"chain {c}")

                if div is not None and div.shape == values.shape:
                    chain_div = div[c]
                    if chain_div.any():
                        trace_ax.scatter(
                            draw_idx[chain_div],
                            chain_vals[chain_div],
                            color="red",
                            s=8,
                            alpha=0.7,
                            zorder=5,
                        )

            trace_ax.set_ylabel(var_name)
            trace_ax.set_xlabel("Draw")
            trace_ax.set_title(f"{var_name} trace")
            dist_ax.set_xlabel(var_name)
            dist_ax.set_ylabel("Density")
            dist_ax.set_title(f"{var_name} posterior")
            if row == 0:
                trace_ax.legend(fontsize=8)
                dist_ax.legend(fontsize=8)

        title_scope = "all parameters" if include_time else "all non-time parameters"
        fig.suptitle(f"Posterior trace — {title_scope} (page {page + 1}/{n_pages})", fontsize=11)
        fig.tight_layout()

        if n_pages == 1:
            page_output = output_path
        else:
            page_output = output_path.with_name(f"{output_path.stem}_p{page + 1:02d}{output_path.suffix}")
        fig.savefig(str(page_output), dpi=150, bbox_inches="tight")
        plt.close(fig)


def _device_effect_trace_plot(idata: az.InferenceData, output_path: Path) -> None:
    posterior = idata.posterior.to_dataset() if hasattr(idata.posterior, "to_dataset") else idata.posterior
    if "device_effect" not in posterior:
        raise ValueError("device_effect not found in posterior; cannot create device trace plot.")

    values = posterior["device_effect"].values
    if values.ndim != 3:
        raise ValueError("Unexpected shape for device_effect posterior.")

    n_chains, n_draws, n_devices = values.shape
    draw_idx = np.arange(n_draws)

    device_coord = posterior["device_effect"].coords.get("device")
    if device_coord is not None:
        device_labels = [str(v) for v in np.asarray(device_coord.values)]
    else:
        device_labels = [f"device_{i}" for i in range(n_devices)]

    n_cols = 3
    n_rows = int(np.ceil(n_devices / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 2.4 * n_rows), squeeze=False)

    for i in range(n_devices):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        for chain in range(n_chains):
            ax.plot(draw_idx, values[chain, :, i], linewidth=0.7, alpha=0.85, label=f"chain {chain}")
        ax.set_title(device_labels[i], fontsize=9)
        ax.set_xlabel("Draw")
        ax.set_ylabel("device_effect")
        if i == 0:
            ax.legend(fontsize=7)

    for i in range(n_devices, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r, c].axis("off")

    fig.suptitle("Posterior trace — device effects", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _device_effect_forest_plot(idata: az.InferenceData, output_path: Path) -> None:
    posterior = idata.posterior.to_dataset() if hasattr(idata.posterior, "to_dataset") else idata.posterior
    if "device_effect" not in posterior:
        raise ValueError("device_effect not found in posterior; cannot create device forest plot.")

    values = posterior["device_effect"].values
    if values.ndim != 3:
        raise ValueError("Unexpected shape for device_effect posterior.")

    flat = values.reshape(-1, values.shape[-1])  # (samples, devices)
    device_coord = posterior["device_effect"].coords.get("device")
    if device_coord is not None:
        device_labels = [str(v) for v in np.asarray(device_coord.values)]
    else:
        device_labels = [f"device_{i}" for i in range(values.shape[-1])]

    means = flat.mean(axis=0)
    lo = np.quantile(flat, 0.01, axis=0)
    hi = np.quantile(flat, 0.99, axis=0)
    order = np.argsort(means)

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")

    for rank, idx in enumerate(order):
        color = cmap(idx % 10)
        ax.hlines(rank, lo[idx], hi[idx], color=color, linewidth=4.0, alpha=0.85)
        ax.plot(means[idx], rank, marker="o", color=color, markersize=6)
        ax.text(hi[idx] + 0.002, rank, f"{means[idx]:+.3f}", va="center", ha="left", fontsize=8, color=color)

    ax.axvline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([device_labels[i] for i in order])
    ax.set_xlabel("device_effect")
    ax.set_ylabel("Device")
    ax.set_title("Posterior forest plot for device effects (98% interval + posterior mean)")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _prior_predictive_plot(
    df: pd.DataFrame,
    target_col: str,
    n_samples: int,
    seed: int,
    output_path: Path,
    position_effects: bool,
) -> None:
    work, index_data, coords = _prepare_design(df, target_col)
    has_multiple_epochs = work["epoch"].nunique() > 1

    timestamps = work.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")["timestamp"]
    t_numeric = (timestamps - timestamps.min()).dt.total_seconds() / 3600
    t_days = t_numeric / 24.0

    model = _build_bsts_model(
        index_data,
        coords,
        has_multiple_epochs=has_multiple_epochs,
        position_effects=position_effects,
        observed=False,
    )

    with model:
        prior = pm.sample_prior_predictive(draws=n_samples, random_seed=seed)

    # In this PyMC/ArviZ version observed vars land in prior group when prior_predictive is empty
    prior_ds = prior.prior.ds

    fig, ax = plt.subplots(figsize=(8, 5))

    # Aggregate across devices at each timestamp for the observed reference line.
    time_idx = index_data["time_idx"]
    n_time = len(t_numeric)
    counts = np.bincount(time_idx, minlength=n_time).astype(float)

    observed_by_time = np.bincount(time_idx, weights=index_data["y"], minlength=n_time) / counts

    # --- Local level prior trajectories over time ---
    if "local_level" in prior_ds:
        prior_level = prior_ds["local_level"].values.reshape(-1, n_time)
    else:
        raw = prior_ds["local_level_raw"].values.reshape(-1, n_time)
        prior_level = raw - raw.mean(axis=1, keepdims=True)
    prior_level = prior_level - prior_level[:, [0]]
    for i in range(prior_level.shape[0]):
        ax.plot(t_days, prior_level[i], color="darkorange", alpha=0.2, linewidth=0.5)
    # Overlay observed mean anchored to its start so the series begins at 0.
    observed_centered = observed_by_time - observed_by_time[0]
    ax.plot(
        t_days,
        observed_centered,
        color="black",
        linewidth=1.0,
        alpha=0.9,
        label="Observed mean (start=0)",
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Days since start")
    ax.set_ylabel("Latent level")
    ax.set_title("Prior local-level trajectories")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(
    input_path: Path,
    target_col: str,
    trace_path: Path | None,
    output_dir: Path,
    prior_samples: int,
    seed: int,
    trace_include_time: bool,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Processed input file not found: {input_path}")

    df = pd.read_csv(input_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Processed input missing columns: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    position_effects = False

    # 1 — Trace + pairs plots from trace
    if trace_path is not None and trace_path.exists():
        try:
            idata = az.from_netcdf(str(trace_path))
        except Exception:
            idata = az.from_netcdf(str(trace_path), engine="h5netcdf")

        posterior = idata.posterior.to_dataset() if hasattr(idata.posterior, "to_dataset") else idata.posterior
        position_effects = any(name in posterior for name in ("line_effect", "line_delta", "position_effect"))

        trace_plot_path = output_dir / f"{target_col}_traceplot.png"
        _trace_plot(idata, trace_plot_path, include_time=trace_include_time)
        print(f"Trace plot saved: {trace_plot_path}")

        device_trace_plot_path = output_dir / f"{target_col}_device_effect_traceplot.png"
        _device_effect_trace_plot(idata, device_trace_plot_path)
        print(f"Device-effect trace plot saved: {device_trace_plot_path}")

        device_forest_plot_path = output_dir / f"{target_col}_device_effect_forest.png"
        _device_effect_forest_plot(idata, device_forest_plot_path)
        print(f"Device-effect forest plot saved: {device_forest_plot_path}")

        pairs_path = output_dir / f"{target_col}_pairs.png"
        _pairs_plot(idata, pairs_path)
        print(f"Pairs plot saved: {pairs_path}")
    else:
        print("No trace file provided or found; skipping trace/pairs plots.")

    # 2 — Prior predictive trajectories
    prior_path = output_dir / f"{target_col}_prior_predictive.png"
    _prior_predictive_plot(df, target_col, prior_samples, seed, prior_path, position_effects=position_effects)
    print(f"Prior predictive plot saved: {prior_path}")


def main() -> None:
    args = parse_args()

    input_path = args.input
    if input_path is None:
        input_path = Path("data/processed") / args.dataset / "measurements_long.csv"

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("results") / args.dataset / "model"

    trace_path = args.trace
    if trace_path is None:
        default_trace = output_dir / f"{args.target}_trace.nc"
        if default_trace.exists():
            trace_path = default_trace

    run(
        input_path=input_path,
        target_col=args.target,
        trace_path=trace_path,
        output_dir=output_dir,
        prior_samples=args.prior_samples,
        seed=args.seed,
        trace_include_time=args.trace_include_time,
    )


if __name__ == "__main__":
    main()
