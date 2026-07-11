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


def _save_plot(df: pd.DataFrame, y_col: str, y_label: str, title: str, output_path: Path) -> None:
    p = (
        ggplot(df, aes(x="timestamp", y=y_col, color="device_id"))
        + geom_line(size=0.6, alpha=0.9)
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

    output_dir.mkdir(parents=True, exist_ok=True)

    _save_plot(
        df,
        y_col="temp_c",
        y_label="Temperature (deg C)",
        title="Temperature Over Time by Device",
        output_path=output_dir / "temperature_by_device.png",
    )

    _save_plot(
        df,
        y_col="humidity_rh",
        y_label="Relative Humidity (%RH)",
        title="Humidity Over Time by Device",
        output_path=output_dir / "humidity_by_device.png",
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
