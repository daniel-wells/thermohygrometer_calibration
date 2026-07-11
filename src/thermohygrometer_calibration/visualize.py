from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    coord_equal,
    geom_line,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_gradient,
    scale_y_reverse,
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


def _build_blue_red_split_map(df: pd.DataFrame, split_col: str) -> dict[str, str]:
    first_pos = (
        df[["device_id", split_col]]
        .dropna(subset=["device_id", split_col])
        .drop_duplicates(subset=["device_id"], keep="first")
    )
    if first_pos.empty:
        return _build_device_color_map(df)

    sorted_levels = sorted(first_pos[split_col].unique().tolist())
    split_idx = max(1, len(sorted_levels) // 2)
    blue_levels = set(sorted_levels[:split_idx])

    blue_devices = sorted(
        first_pos[first_pos[split_col].isin(blue_levels)]["device_id"].astype(str).tolist()
    )
    red_devices = sorted(
        first_pos[~first_pos[split_col].isin(blue_levels)]["device_id"].astype(str).tolist()
    )

    blue_shades = _make_shades("#A9D6FF", "#0B4FA8", len(blue_devices))
    red_shades = _make_shades("#FFC3C3", "#B22222", len(red_devices))

    color_map: dict[str, str] = {}
    for device_id, color in zip(blue_devices, blue_shades):
        color_map[device_id] = color
    for device_id, color in zip(red_devices, red_shades):
        color_map[device_id] = color

    # Keep all devices colored even if split column is missing for a subset.
    all_devices = sorted(df["device_id"].dropna().astype(str).unique().tolist())
    fallback_cycle = _make_shades("#A9D6FF", "#0B4FA8", max(1, len(all_devices)))
    for idx, device_id in enumerate(all_devices):
        if device_id not in color_map:
            color_map[device_id] = fallback_cycle[idx % len(fallback_cycle)]

    return color_map


def _save_plot(
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    output_path: Path,
    color_map: dict[str, str],
) -> None:
    p = (
        ggplot(df, aes(x="timestamp", y=y_col, color="device_id"))
        + geom_line(size=0.6, alpha=0.9)
        + scale_color_manual(values=color_map)
        + theme_bw()
        + labs(title=title, x="Time", y=y_label, color="Device")
    )
    p.save(filename=str(output_path), width=12, height=7, dpi=180, verbose=False)


def _save_heatmap(
    df: pd.DataFrame,
    value_col: str,
    label: str,
    title: str,
    output_path: Path,
) -> None:
    means = (
        df.groupby(["device_id", "line", "position"], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "mean_value"})
    )
    means["label"] = means["device_id"] + "\n" + means["mean_value"].round(2).astype(str)
    means["line"] = means["line"].astype(int)
    means["position"] = means["position"].astype(int)

    p = (
        ggplot(means, aes(x="position", y="line", fill="mean_value"))
        + geom_tile(color="white", size=1)
        + geom_text(aes(label="label"), size=8, color="black")
        + scale_fill_gradient(low="#d0e8f5", high="#08306b", name=label)
        + coord_equal()
        + theme_bw()
        + theme(figure_size=(6, 4))
        + labs(title=title, x="Position", y="Line")
    )
    p.save(filename=str(output_path), dpi=180, verbose=False)


def run(input_path: Path, output_dir: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Processed input file not found: {input_path}")

    df = pd.read_csv(input_path)
    required = {"timestamp", "device_id", "temp_c", "humidity_rh", "line", "position"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Processed input missing columns: {sorted(missing)}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "temp_c", "humidity_rh"])
    color_map = _build_device_color_map(df)
    line_split_map = _build_blue_red_split_map(df, "line")
    position_split_map = _build_blue_red_split_map(df, "position")

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
        df,
        y_col="temp_c",
        y_label="Temperature (deg C)",
        title="Temperature Over Time by Device (Blue/Red Split by Line)",
        output_path=output_dir / "temperature_by_line_split.png",
        color_map=line_split_map,
    )

    _save_plot(
        df,
        y_col="humidity_rh",
        y_label="Relative Humidity (%RH)",
        title="Humidity Over Time by Device (Blue/Red Split by Line)",
        output_path=output_dir / "humidity_by_line_split.png",
        color_map=line_split_map,
    )

    _save_plot(
        df,
        y_col="temp_c",
        y_label="Temperature (deg C)",
        title="Temperature Over Time by Device (Blue/Red Split by Position)",
        output_path=output_dir / "temperature_by_position_split.png",
        color_map=position_split_map,
    )

    _save_plot(
        df,
        y_col="humidity_rh",
        y_label="Relative Humidity (%RH)",
        title="Humidity Over Time by Device (Blue/Red Split by Position)",
        output_path=output_dir / "humidity_by_position_split.png",
        color_map=position_split_map,
    )

    _save_heatmap(
        df,
        value_col="temp_c",
        label="Mean temp (deg C)",
        title="Mean Temperature by Layout Position",
        output_path=output_dir / "temperature_heatmap.png",
    )

    _save_heatmap(
        df,
        value_col="humidity_rh",
        label="Mean humidity (%RH)",
        title="Mean Humidity by Layout Position",
        output_path=output_dir / "humidity_heatmap.png",
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
