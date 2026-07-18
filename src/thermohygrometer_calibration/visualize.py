from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    coord_equal,
    facet_wrap,
    geom_line,
    geom_point,
    geom_pointrange,
    geom_segment,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_color_manual,
    scale_color_gradient,
    scale_fill_gradient,
    scale_fill_gradient2,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_bw,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create simple per-device time-series plots using plotnine."
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
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write generated figures.",
    )
    return parser.parse_args()


def _build_device_color_map(df: pd.DataFrame) -> dict[str, str]:
    indoor_palette = ["#A9D6FF", "#0B4FA8"]
    outdoor_palette = ["#FFC3C3", "#B22222"]

    type_series = df.get("device_type")
    if type_series is None:
        device_ids = sorted(df["device_id"].dropna().unique().tolist())
        fallback = indoor_palette + outdoor_palette
        return {device_id: fallback[idx % len(fallback)] for idx, device_id in enumerate(device_ids)}

    first_type = (
        df[["device_id", "device_type"]]
        .dropna(subset=["device_id"])
        .drop_duplicates(subset=["device_id"], keep="first")
    )

    outdoor_types = {"ip65", "outdoor"}
    indoor_types = {"standard", "std", "indoor"}

    color_map: dict[str, str] = {}
    indoor_devices: list[str] = []
    outdoor_devices: list[str] = []
    unknown_devices: list[str] = []

    for row in first_type.itertuples(index=False):
        device_id = str(row.device_id)
        device_type = str(row.device_type).strip().lower() if pd.notna(row.device_type) else ""
        if device_type in outdoor_types:
            outdoor_devices.append(device_id)
        elif device_type in indoor_types:
            indoor_devices.append(device_id)
        else:
            unknown_devices.append(device_id)

    for idx, device_id in enumerate(sorted(indoor_devices)):
        color_map[device_id] = _make_shades(indoor_palette[0], indoor_palette[1], len(indoor_devices))[idx]
    for idx, device_id in enumerate(sorted(outdoor_devices)):
        color_map[device_id] = _make_shades(outdoor_palette[0], outdoor_palette[1], len(outdoor_devices))[idx]
    for idx, device_id in enumerate(sorted(unknown_devices)):
        color_map[device_id] = _make_shades(indoor_palette[0], indoor_palette[1], len(unknown_devices))[idx]

    return color_map


def _interpolate_hex(start_hex: str, end_hex: str, t: float) -> str:
    start_hex = start_hex.lstrip("#")
    end_hex = end_hex.lstrip("#")
    sr, sg, sb = int(start_hex[0:2], 16), int(start_hex[2:4], 16), int(start_hex[4:6], 16)
    er, eg, eb = int(end_hex[0:2], 16), int(end_hex[2:4], 16), int(end_hex[4:6], 16)
    r = round(sr + (er - sr) * t)
    g = round(sg + (eg - sg) * t)
    b = round(sb + (eb - sb) * t)
    return f"#{r:02X}{g:02X}{b:02X}"


def _make_shades(start_hex: str, end_hex: str, n: int) -> list[str]:
    if n <= 1:
        return [_interpolate_hex(start_hex, end_hex, 0.5)]
    return [_interpolate_hex(start_hex, end_hex, i / (n - 1)) for i in range(n)]


def _build_group_split_colors(
    df: pd.DataFrame, split_col: str, label_prefix: str
) -> tuple[pd.DataFrame, dict[str, str], str]:
    out = df.copy()
    color_col = f"{split_col}_group"
    na_label = f"{label_prefix} NA"

    levels = sorted(out[split_col].dropna().unique().tolist())
    if not levels:
        out[color_col] = na_label
        return out, {na_label: "#808080"}, color_col

    split_idx = max(1, len(levels) // 2)
    blue_levels = levels[:split_idx]
    red_levels = levels[split_idx:]

    blue_labels = [f"{label_prefix} {int(level)}" for level in blue_levels]
    red_labels = [f"{label_prefix} {int(level)}" for level in red_levels]
    blue_shades = _make_shades("#A9D6FF", "#0B4FA8", max(1, len(blue_labels)))
    red_shades = _make_shades("#FFC3C3", "#B22222", max(1, len(red_labels)))

    color_map: dict[str, str] = {na_label: "#808080"}
    for label, color in zip(blue_labels, blue_shades):
        color_map[label] = color
    for label, color in zip(red_labels, red_shades):
        color_map[label] = color

    out[color_col] = out[split_col].apply(
        lambda value: f"{label_prefix} {int(value)}" if pd.notna(value) else na_label
    )

    return out, color_map, color_col


def _save_plot(
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    output_path: Path,
    color_map: dict[str, str],
    color_col: str = "device_id",
    color_label: str = "Device",
    group_col: str = "device_id",
) -> None:
    p = (
        ggplot(df, aes(x="timestamp", y=y_col, color=color_col, group=group_col))
        + geom_line(size=0.6, alpha=0.9)
        + scale_color_manual(values=color_map)
        + theme_bw()
        + labs(title=title, x="Time", y=y_label, color=color_label)
    )
    p.save(filename=str(output_path), width=12, height=7, dpi=180, verbose=False)


def _save_heatmap(
    df: pd.DataFrame,
    value_col: str,
    label: str,
    title: str,
    output_path: Path,
    facet_col: str | None = None,
) -> None:
    group_cols = ["device_id", "line", "position"]
    if facet_col is not None:
        group_cols.append(facet_col)

    means = (
        df.groupby(group_cols, as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "mean_value"})
    )

    if facet_col is not None and facet_col in means.columns:
        epoch_avg = (
            df.groupby(facet_col, as_index=False)[value_col]
            .mean()
            .rename(columns={value_col: "epoch_avg"})
        )
        means = means.merge(epoch_avg, on=facet_col, how="left")
        means["offset_value"] = means["mean_value"] - means["epoch_avg"]
    else:
        overall_avg = float(df[value_col].mean())
        means["offset_value"] = means["mean_value"] - overall_avg

    means["label"] = means["device_id"] + "\n" + means["offset_value"].round(2).astype(str)
    means["line"] = means["line"].astype(int)
    means["position"] = means["position"].astype(int)
    line_breaks = sorted(means["line"].unique().tolist())
    position_breaks = sorted(means["position"].unique().tolist())
    max_abs = float(np.nanmax(np.abs(means["offset_value"]))) if len(means) else 1.0
    if max_abs == 0:
        max_abs = 1.0

    p = (
        ggplot(means, aes(x="position", y="line", fill="offset_value"))
        + geom_tile(color="white", size=1)
        + geom_text(aes(label="label"), size=8, color="black")
        + scale_fill_gradient2(
            low="#2B6CB0",
            mid="#F7FAFC",
            high="#C53030",
            midpoint=0.0,
            limits=(-max_abs, max_abs),
            name=f"{label} offset",
        )
        + scale_x_continuous(breaks=position_breaks, minor_breaks=[])
        + scale_y_continuous(breaks=line_breaks, minor_breaks=[])
        + coord_equal()
        + theme_bw()
        + theme(figure_size=(6, 4))
        + labs(title=title, x="Position", y="Line")
    )
    if facet_col is not None and facet_col in means.columns:
        p = p + facet_wrap(f"~{facet_col}") + theme(figure_size=(11, 4.5))
    p.save(filename=str(output_path), dpi=180, verbose=False)


def _save_temp_humidity_scatter_by_sensor(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    contour_df = _build_dewpoint_contour_df(df)

    p = (
        ggplot(df, aes(x="temp_c", y="humidity_rh", color="days_since_start"))
        + geom_line(
            contour_df,
            aes(x="temp_c", y="humidity_rh", group="dewpoint_c"),
            inherit_aes=False,
            color="#444444",
            alpha=0.45,
            size=0.4,
        )
        + geom_point(size=0.9, alpha=0.55)
        + facet_wrap("~device_id")
        + scale_color_gradient(low="#d0e8f5", high="#08306b")
        + scale_y_continuous(limits=(20, 65))
        + theme_bw()
        + theme(figure_size=(12, 8))
        + labs(
            title="Temperature vs Humidity by Sensor",
            x="Temperature (deg C)",
            y="Relative Humidity (%RH)",
            color="Days since start",
        )
    )
    p.save(filename=str(output_path), dpi=180, verbose=False)


def _saturation_vapor_pressure_hpa(temp_c: np.ndarray) -> np.ndarray:
    # Tetens approximation to Clausius-Clapeyron over typical ambient temperatures.
    return 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))


def _build_dewpoint_contour_df(df: pd.DataFrame) -> pd.DataFrame:
    t_min = float(df["temp_c"].min())
    t_max = float(df["temp_c"].max())
    t_grid = np.linspace(t_min, t_max, 240)
    e_sat_t = _saturation_vapor_pressure_hpa(t_grid)

    # Select practical dew-point contours for indoor/outdoor calibration range.
    td_candidates = np.arange(-5.0, 31.0, 5.0)
    td_levels = [td for td in td_candidates if td <= t_max + 1.0]
    if not td_levels:
        td_levels = [t_min]

    frames: list[pd.DataFrame] = []
    for td in td_levels:
        e_td = _saturation_vapor_pressure_hpa(np.array([td]))[0]
        rh = 100.0 * (e_td / e_sat_t)
        mask = (rh >= 0.0) & (rh <= 100.0)
        if not np.any(mask):
            continue

        frames.append(
            pd.DataFrame(
                {
                    "temp_c": t_grid[mask],
                    "humidity_rh": rh[mask],
                    "dewpoint_c": f"Td={td:.0f}C",
                }
            )
        )

    if not frames:
        return pd.DataFrame({"temp_c": [], "humidity_rh": [], "dewpoint_c": []})

    return pd.concat(frames, ignore_index=True)


def _save_device_summary_scatter_90ci(summary_path: Path, output_path: Path) -> None:
    if not summary_path.exists():
        raise FileNotFoundError(f"Device summary file not found: {summary_path}")

    summary = pd.read_csv(summary_path)
    required = {
        "device_id",
        "n_obs",
        "temp_mean_c",
        "temp_sd_c",
        "humidity_mean_rh",
        "humidity_sd_rh",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Device summary missing columns: {sorted(missing)}")

    z_90 = 1.6448536269514722
    n = summary["n_obs"].clip(lower=1).astype(float)
    temp_se = summary["temp_sd_c"].astype(float) / np.sqrt(n)
    humidity_se = summary["humidity_sd_rh"].astype(float) / np.sqrt(n)

    summary = summary.copy()
    summary["temp_lo"] = summary["temp_mean_c"].astype(float) - z_90 * temp_se
    summary["temp_hi"] = summary["temp_mean_c"].astype(float) + z_90 * temp_se
    summary["humidity_lo"] = summary["humidity_mean_rh"].astype(float) - z_90 * humidity_se
    summary["humidity_hi"] = summary["humidity_mean_rh"].astype(float) + z_90 * humidity_se

    p = (
        ggplot(summary, aes(x="temp_mean_c", y="humidity_mean_rh"))
        + geom_segment(
            aes(x="temp_lo", xend="temp_hi", y="humidity_mean_rh", yend="humidity_mean_rh"),
            color="#666666",
            size=0.6,
            alpha=0.9,
        )
        + geom_segment(
            aes(x="temp_mean_c", xend="temp_mean_c", y="humidity_lo", yend="humidity_hi"),
            color="#666666",
            size=0.6,
            alpha=0.9,
        )
        + geom_point(aes(color="device_id"), size=2.3, alpha=0.95)
        + geom_text(aes(label="device_id"), nudge_y=0.15, size=7)
        + theme_bw()
        + labs(
            title="Device Mean Temperature vs Humidity (90% CI)",
            x="Mean temperature (deg C)",
            y="Mean relative humidity (%RH)",
            color="Device",
        )
    )
    p.save(filename=str(output_path), dpi=180, verbose=False)


def _wilson_interval(k: np.ndarray, n: np.ndarray, z: float = 1.959963984540054) -> tuple[np.ndarray, np.ndarray]:
    # Wilson score interval for binomial proportions.
    k = k.astype(float)
    n = n.astype(float)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    lo = np.clip(center - half, 0.0, 1.0)
    hi = np.clip(center + half, 0.0, 1.0)
    return lo, hi


def _build_adjacent_step_frequency_df(df: pd.DataFrame) -> pd.DataFrame:
    work = df[["timestamp", "device_id", "device_type", "temp_c", "humidity_rh"]].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp", "device_id", "device_type", "temp_c", "humidity_rh"])

    rows: list[dict[str, object]] = []
    for (device_type, device_id), group in work.groupby(["device_type", "device_id"], dropna=False):
        g = group.sort_values("timestamp").reset_index(drop=True)
        for variable, col in (("Temperature", "temp_c"), ("Humidity", "humidity_rh")):
            step = (g[col] - g[col].shift(1)).abs().dropna().round(6)
            if step.empty:
                continue

            n_total = int(len(step))
            vc = step.value_counts().sort_index()
            for step_size, count in vc.items():
                rows.append(
                    {
                        "device_type": str(device_type),
                        "device_id": str(device_id),
                        "variable": variable,
                        "step_size": float(step_size),
                        "count": int(count),
                        "n_total": n_total,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["proportion"] = out["count"] / out["n_total"]
    lo, hi = _wilson_interval(out["count"].to_numpy(), out["n_total"].to_numpy())
    out["ci_low"] = lo
    out["ci_high"] = hi
    return out


def _save_adjacent_step_frequency_plot(df: pd.DataFrame, output_path: Path) -> None:
    freq_df = _build_adjacent_step_frequency_df(df)
    if freq_df.empty:
        raise ValueError("No adjacent steps available to plot.")

    # Zero-step frequency is the complement of all non-zero steps; drop it to focus the plot.
    freq_df = freq_df[freq_df["step_size"] > 0].copy()
    if freq_df.empty:
        raise ValueError("No non-zero adjacent steps available to plot.")

    p = (
        ggplot(freq_df, aes(x="step_size", y="proportion", color="device_type"))
        + geom_pointrange(aes(ymin="ci_low", ymax="ci_high"), size=0.4, alpha=0.9)
        + facet_wrap("~variable", scales="free_x")
        + theme_bw()
        + theme(figure_size=(12, 5))
        + labs(
            title="Adjacent Step Frequencies by Device Type",
            subtitle="Points are per-device proportions; error bars are 95% Wilson intervals",
            x="Adjacent step size",
            y="Frequency (proportion)",
            color="Device type",
        )
    )
    p.save(filename=str(output_path), dpi=180, verbose=False)


def run(input_path: Path, output_dir: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Processed input file not found: {input_path}")

    df = pd.read_csv(input_path)
    required = {"timestamp", "device_id", "temp_c", "humidity_rh", "line", "position", "epoch"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Processed input missing columns: {sorted(missing)}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "temp_c", "humidity_rh"])
    df["days_since_start"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() / 86400.0
    color_map = _build_device_color_map(df)
    line_split_df, line_split_map, line_split_col = _build_group_split_colors(df, "line", "Line")
    position_split_df, position_split_map, position_split_col = _build_group_split_colors(
        df, "position", "Position"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    _save_plot(
        df,
        y_col="temp_c",
        y_label="Temperature (deg C)",
        title="Temperature Over Time by Device",
        output_path=output_dir / "temperature_by_device.png",
        color_map=color_map,
    )

    _save_plot(
        df,
        y_col="humidity_rh",
        y_label="Relative Humidity (%RH)",
        title="Humidity Over Time by Device",
        output_path=output_dir / "humidity_by_device.png",
        color_map=color_map,
    )

    _save_plot(
        line_split_df,
        y_col="temp_c",
        y_label="Temperature (deg C)",
        title="Temperature Over Time by Device (Blue/Red Split by Line)",
        output_path=output_dir / "temperature_by_line_split.png",
        color_map=line_split_map,
        color_col=line_split_col,
        color_label="Line",
    )

    _save_plot(
        line_split_df,
        y_col="humidity_rh",
        y_label="Relative Humidity (%RH)",
        title="Humidity Over Time by Device (Blue/Red Split by Line)",
        output_path=output_dir / "humidity_by_line_split.png",
        color_map=line_split_map,
        color_col=line_split_col,
        color_label="Line",
    )

    _save_plot(
        position_split_df,
        y_col="temp_c",
        y_label="Temperature (deg C)",
        title="Temperature Over Time by Device (Blue/Red Split by Position)",
        output_path=output_dir / "temperature_by_position_split.png",
        color_map=position_split_map,
        color_col=position_split_col,
        color_label="Position",
    )

    _save_plot(
        position_split_df,
        y_col="humidity_rh",
        y_label="Relative Humidity (%RH)",
        title="Humidity Over Time by Device (Blue/Red Split by Position)",
        output_path=output_dir / "humidity_by_position_split.png",
        color_map=position_split_map,
        color_col=position_split_col,
        color_label="Position",
    )

    _save_heatmap(
        df,
        value_col="temp_c",
        label="Mean temp (deg C)",
        title="Mean Temperature by Layout Position and Epoch",
        output_path=output_dir / "temperature_heatmap.png",
        facet_col="epoch",
    )

    _save_heatmap(
        df,
        value_col="humidity_rh",
        label="Mean humidity (%RH)",
        title="Mean Humidity by Layout Position and Epoch",
        output_path=output_dir / "humidity_heatmap.png",
        facet_col="epoch",
    )

    _save_temp_humidity_scatter_by_sensor(
        df,
        output_path=output_dir / "temp_vs_humidity_by_sensor.png",
    )

    _save_device_summary_scatter_90ci(
        input_path.with_name("device_summary.csv"),
        output_path=output_dir / "device_summary_temp_humidity_90ci.png",
    )

    _save_adjacent_step_frequency_plot(
        df,
        output_path=output_dir / "adjacent_step_frequency_wilson95.png",
    )


def main() -> None:
    args = parse_args()
    input_path = args.input
    if input_path is None:
        input_path = Path("data/processed") / args.dataset / "measurements_long.csv"

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("results") / args.dataset / "figures"

    run(input_path, output_dir)


if __name__ == "__main__":
    main()
