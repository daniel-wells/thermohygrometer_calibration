from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_LAYOUT_COLUMNS = {"device_id", "device_type", "line", "position", "valid_from"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate thermo-hygrometer data with hierarchical device effects "
            "and structured shared time-series dynamics."
        )
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=Path("data/layout.csv"),
        help="CSV layout file with device_id, device_type, line, position.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/simulated"),
        help="Directory for simulated outputs.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2026-07-01 00:00:00",
        help="Start timestamp for simulation.",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=7 * 24 * 4,
        help="Number of time points (default is 7 days at 15-minute sampling).",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="15min",
        help="Sampling frequency compatible with pandas date_range.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_layout(layout_path: Path, at_time: pd.Timestamp) -> pd.DataFrame:
    if not layout_path.exists():
        raise FileNotFoundError(f"Layout file not found: {layout_path}")
    layout = pd.read_csv(layout_path)

    missing = REQUIRED_LAYOUT_COLUMNS - set(layout.columns)
    if missing:
        raise ValueError(f"Layout file missing columns: {sorted(missing)}")

    layout = layout.copy()
    layout["device_type"] = layout["device_type"].str.lower().str.strip()
    valid_types = {"ip65", "standard"}
    bad_types = sorted(set(layout["device_type"]) - valid_types)
    if bad_types:
        raise ValueError(f"Invalid device_type values: {bad_types}")

    layout["line"] = pd.to_numeric(layout["line"], errors="raise").astype(int)
    layout["position"] = pd.to_numeric(layout["position"], errors="raise")
    layout["valid_from"] = pd.to_datetime(layout["valid_from"], errors="coerce")
    layout["valid_to"] = pd.to_datetime(layout.get("valid_to"), errors="coerce")

    active = layout["valid_from"] <= at_time
    not_expired = layout["valid_to"].isna() | (layout["valid_to"] >= at_time)
    layout = layout[active & not_expired].drop(columns=["valid_from", "valid_to"])

    if layout.empty:
        raise ValueError(f"No layout entries are active at {at_time}")

    if layout["device_id"].duplicated().any():
        dupes = layout.loc[layout["device_id"].duplicated(), "device_id"].tolist()
        raise ValueError(f"Multiple active layout entries for device(s): {dupes}")

    return layout.sort_values(["line", "position", "device_id"]).reset_index(drop=True)


def simulate_latent_environment(n_times: int, freq: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    step_minutes = pd.to_timedelta(freq).total_seconds() / 60.0
    points_per_day = max(1.0, (24.0 * 60.0) / step_minutes)
    t = np.arange(n_times)

    temp_level = np.empty(n_times)
    temp_level[0] = 21.0
    for i in range(1, n_times):
        temp_level[i] = temp_level[i - 1] + rng.normal(0.0, 0.03)

    temp_daily = 1.4 * np.sin(2 * np.pi * t / points_per_day - 0.8) + 0.35 * np.cos(
        2 * np.pi * t / points_per_day
    )
    true_temp = temp_level + temp_daily

    humidity_level = np.empty(n_times)
    humidity_level[0] = 52.0
    for i in range(1, n_times):
        humidity_level[i] = humidity_level[i - 1] + rng.normal(0.0, 0.10)

    humidity_daily = 2.8 * np.sin(2 * np.pi * t / points_per_day + 0.6)
    true_humidity = humidity_level + humidity_daily - 0.85 * (true_temp - np.mean(true_temp))
    true_humidity = np.clip(true_humidity, 25.0, 90.0)

    return true_temp, true_humidity


def _line_membership_matrix(lines: np.ndarray) -> np.ndarray:
    return (lines[:, None] == lines[None, :]).astype(float)


def _distance_matrix(layout: pd.DataFrame, line_gap: float = 1.5) -> np.ndarray:
    xy = np.column_stack([layout["position"].to_numpy(dtype=float), layout["line"].to_numpy(dtype=float) * line_gap])
    diffs = xy[:, None, :] - xy[None, :, :]
    return np.sqrt(np.sum(diffs**2, axis=2))


def _correlation_kernel(layout: pd.DataFrame, rho: float, line_strength: float, nugget: float) -> np.ndarray:
    dist = _distance_matrix(layout)
    lines = layout["line"].to_numpy(dtype=float)
    base = np.exp(-dist / rho)
    same_line = _line_membership_matrix(lines)
    k = 0.55 * base + line_strength * same_line + nugget * np.eye(len(layout))

    # Normalize to a proper correlation matrix.
    d = np.sqrt(np.diag(k))
    corr = k / np.outer(d, d)
    return corr


def _simulate_ar1(n_times: int, n_devices: int, phi: float, innovation_sd: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros((n_times, n_devices), dtype=float)
    for t in range(1, n_times):
        x[t] = phi * x[t - 1] + rng.normal(0.0, innovation_sd, size=n_devices)
    return x


def _simulate_bias_drift(n_times: int, n_devices: int, drift_sd: float, rng: np.random.Generator) -> np.ndarray:
    drift = np.zeros((n_times, n_devices), dtype=float)
    innovations = rng.normal(0.0, drift_sd, size=(n_times - 1, n_devices))
    drift[1:] = np.cumsum(innovations, axis=0)
    return drift


def simulate_measurements(
    layout: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_times = len(timestamps)
    n_devices = len(layout)

    true_temp, true_humidity = simulate_latent_environment(n_times, timestamps.freqstr, rng)

    device_type = layout["device_type"].to_numpy()
    type_temp_mu = np.where(device_type == "ip65", 0.12, -0.03)
    type_humidity_mu = np.where(device_type == "ip65", -1.10, 0.35)

    line_ids = layout["line"].astype(int).to_numpy()
    line_temp_effects = {line: rng.normal(0.0, 0.08) for line in np.unique(line_ids)}
    line_humidity_effects = {line: rng.normal(0.0, 0.45) for line in np.unique(line_ids)}

    temp_bias0 = type_temp_mu + np.array([line_temp_effects[line] for line in line_ids]) + rng.normal(0.0, 0.10, size=n_devices)
    humidity_bias0 = (
        type_humidity_mu
        + np.array([line_humidity_effects[line] for line in line_ids])
        + rng.normal(0.0, 0.70, size=n_devices)
    )

    temp_bias_drift = _simulate_bias_drift(n_times, n_devices, drift_sd=0.002, rng=rng)
    humidity_bias_drift = _simulate_bias_drift(n_times, n_devices, drift_sd=0.010, rng=rng)

    temp_corr = _correlation_kernel(layout, rho=1.8, line_strength=0.25, nugget=0.20)
    humidity_corr = _correlation_kernel(layout, rho=1.3, line_strength=0.30, nugget=0.25)

    temp_chol = np.linalg.cholesky(temp_corr)
    humidity_chol = np.linalg.cholesky(humidity_corr)

    temp_shared_noise = rng.normal(0.0, 0.06, size=(n_times, n_devices)) @ temp_chol.T
    humidity_shared_noise = rng.normal(0.0, 0.35, size=(n_times, n_devices)) @ humidity_chol.T

    temp_ar = _simulate_ar1(n_times, n_devices, phi=0.45, innovation_sd=0.025, rng=rng)
    humidity_ar = _simulate_ar1(n_times, n_devices, phi=0.50, innovation_sd=0.09, rng=rng)

    observed_temp = (
        true_temp[:, None]
        + temp_bias0[None, :]
        + temp_bias_drift
        + temp_shared_noise
        + temp_ar
    )
    observed_humidity = (
        true_humidity[:, None]
        + humidity_bias0[None, :]
        + humidity_bias_drift
        + humidity_shared_noise
        + humidity_ar
    )
    observed_humidity = np.clip(observed_humidity, 0.0, 100.0)

    records: list[dict[str, object]] = []
    for j, row in layout.reset_index(drop=True).iterrows():
        for t, ts in enumerate(timestamps):
            records.append(
                {
                    "timestamp": ts,
                    "device_id": row["device_id"],
                    "device_type": row["device_type"],
                    "line": int(row["line"]),
                    "position": float(row["position"]),
                    "true_temp_c": float(true_temp[t]),
                    "true_humidity_rh": float(true_humidity[t]),
                    "temp_c": float(observed_temp[t, j]),
                    "humidity_rh": float(observed_humidity[t, j]),
                }
            )

    out = pd.DataFrame(records)
    return out.sort_values(["timestamp", "line", "position", "device_id"]).reset_index(drop=True)


def _saturation_vapor_pressure_kpa(temp_c: np.ndarray) -> np.ndarray:
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def _dew_point_c(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    rh_safe = np.clip(rh, 1e-4, 100.0)
    a = 17.27
    b = 237.7
    gamma = (a * temp_c / (b + temp_c)) + np.log(rh_safe / 100.0)
    return (b * gamma) / (a - gamma)


def _absolute_humidity_g_m3(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    rh_safe = np.clip(rh, 0.0, 100.0)
    return 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5)) * rh_safe * 2.1674 / (273.15 + temp_c)


def to_switchbot_export(df: pd.DataFrame) -> pd.DataFrame:
    temp = df["temp_c"].to_numpy()
    rh = df["humidity_rh"].to_numpy()

    svp = _saturation_vapor_pressure_kpa(temp)
    vpd = svp * (1.0 - (rh / 100.0))
    dpt = _dew_point_c(temp, rh)
    abs_h = _absolute_humidity_g_m3(temp, rh)

    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(df["timestamp"]).dt.strftime("%b %d, %Y %H:%M"),
            "Temperature_Celsius(℃)": np.round(temp, 1),
            "Relative_Humidity(%)": np.round(rh).astype(int),
            "DPT(℃)": np.round(dpt, 1),
            "VPD(kPa)": np.round(vpd, 2),
            "Abs Humidity(g/m³)": np.round(abs_h, 2),
        }
    )
    return out


def write_outputs(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep this directory strictly SwitchBot-style exports from the latest run.
    for old_csv in output_dir.glob("*.csv"):
        old_csv.unlink()

    for device_id, g in df.groupby("device_id"):
        export = to_switchbot_export(g.sort_values("timestamp"))
        safe_name = str(device_id).replace("/", "_")
        export.to_csv(output_dir / f"{safe_name}_data.csv", index=False)


def main() -> None:
    args = parse_args()

    start = pd.Timestamp(args.start)
    layout = load_layout(args.layout, at_time=start)
    timestamps = pd.date_range(start=start, periods=args.periods, freq=args.freq)

    df = simulate_measurements(layout=layout, timestamps=timestamps, seed=args.seed)
    write_outputs(df, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
