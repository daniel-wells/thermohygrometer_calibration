from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SWITCHBOT_REQUIRED_COLUMNS = {"Date", "Temperature_Celsius(℃)", "Relative_Humidity(%)"}
LAYOUT_REQUIRED_COLUMNS = {"device_id", "device_type", "line", "position", "valid_from"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read SwitchBot-style CSV exports and create processed tidy data "
            "for downstream modeling and visualization."
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
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing per-device SwitchBot export CSV files.",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=Path("data/layout.csv"),
        help="Layout config CSV with device_id, device_type, line, position, valid_from, valid_to.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where processed outputs are written.",
    )
    return parser.parse_args()


def _device_id_from_filename(path: Path) -> str:
    stem = path.stem.strip()
    if stem.lower().endswith("_data"):
        return stem[:-5]
    return stem if stem else "unknown_device"


def _read_switchbot_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = SWITCHBOT_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"File {path.name} missing SwitchBot columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df["Date"], errors="coerce"),
            "device_id": _device_id_from_filename(path),
            "temp_c": pd.to_numeric(df["Temperature_Celsius(℃)"], errors="coerce"),
            "humidity_rh": pd.to_numeric(df["Relative_Humidity(%)"], errors="coerce"),
            "source_file": path.name,
        }
    )
    out = out.dropna(subset=["timestamp", "temp_c", "humidity_rh"]).copy()
    return out


def load_layout_config(layout_path: Path) -> pd.DataFrame:
    if not layout_path.exists():
        raise FileNotFoundError(f"Layout config not found: {layout_path}")
    layout = pd.read_csv(layout_path)
    missing = LAYOUT_REQUIRED_COLUMNS - set(layout.columns)
    if missing:
        raise ValueError(f"Layout config missing columns: {sorted(missing)}")
    layout = layout.copy()
    layout["valid_from"] = pd.to_datetime(layout["valid_from"], errors="coerce")
    layout["valid_to"] = pd.to_datetime(layout.get("valid_to"), errors="coerce")
    return layout


def join_layout(df: pd.DataFrame, layout: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(layout, on="device_id", how="left")
    in_range = merged["timestamp"] >= merged["valid_from"]
    not_expired = merged["valid_to"].isna() | (merged["timestamp"] <= merged["valid_to"])
    matched = merged[in_range & not_expired].copy()
    matched = matched.drop(columns=["valid_from", "valid_to"])
    matched_ids = set(matched["device_id"].unique())
    unmatched = sorted(set(df["device_id"].unique()) - matched_ids)
    if unmatched:
        raise ValueError(f"No layout entry found for device(s): {unmatched}")
    return matched.sort_values(["timestamp", "device_id"]).reset_index(drop=True)


def load_switchbot_exports(input_dir: Path) -> pd.DataFrame:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(p for p in input_dir.glob("*_data.csv") if p.is_file())
    if not files:
        raise FileNotFoundError(
            f"No SwitchBot export files matching '*_data.csv' found in: {input_dir}"
        )

    frames = [_read_switchbot_file(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise ValueError("No valid rows found after parsing input files.")

    return df.sort_values(["timestamp", "device_id"]).reset_index(drop=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("device_id", as_index=False)
        .agg(
            n_obs=("timestamp", "size"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
            temp_mean_c=("temp_c", "mean"),
            temp_sd_c=("temp_c", "std"),
            humidity_mean_rh=("humidity_rh", "mean"),
            humidity_sd_rh=("humidity_rh", "std"),
        )
        .sort_values("device_id")
        .reset_index(drop=True)
    )
    return summary


def run(input_dir: Path, layout_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_switchbot_exports(input_dir)
    layout = load_layout_config(layout_path)
    df = join_layout(df, layout)
    summary = summarize(df)

    df.to_csv(output_dir / "measurements_long.csv", index=False)
    summary.to_csv(output_dir / "device_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    if input_dir is None:
        input_dir = Path("data/raw") if args.dataset == "actual" else Path("data/simulated")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("data/processed") / args.dataset

    run(input_dir, args.layout, output_dir)


if __name__ == "__main__":
    main()
