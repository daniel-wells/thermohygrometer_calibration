from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    coord_equal,
    facet_wrap,
    geom_line,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_gradient,
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
    means["label"] = means["device_id"] + "\n" + means["mean_value"].round(2).astype(str)
    means["line"] = means["line"].astype(int)
    means["position"] = means["position"].astype(int)
    line_breaks = sorted(means["line"].unique().tolist())
    position_breaks = sorted(means["position"].unique().tolist())

    p = (
        ggplot(means, aes(x="position", y="line", fill="mean_value"))
        + geom_tile(color="white", size=1)
        + geom_text(aes(label="label"), size=8, color="black")
        + scale_fill_gradient(low="#d0e8f5", high="#08306b", name=label)
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
