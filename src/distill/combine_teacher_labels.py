from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.distill.data import canonicalize_image_key
from src.distill.generate_teacher_labels import OUTPUT_FIELDNAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine per-split teacher-label CSVs into one training manifest.")
    parser.add_argument("--input_csv", action="append", required=True, help="Teacher-label CSV. Repeat this flag.")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def load_teacher_label_csv(path_value: str | Path) -> pd.DataFrame:
    path = Path(path_value).expanduser().resolve(strict=False)
    frame = pd.read_csv(path)
    missing_columns = [column for column in OUTPUT_FIELDNAMES if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Teacher-label CSV is missing required columns: {path} :: {', '.join(missing_columns)}")
    frame = frame.copy()
    frame["__image_key"] = frame["image_path"].apply(canonicalize_image_key)
    frame["__source_csv"] = str(path)
    frame = frame.dropna(subset=["__image_key"]).reset_index(drop=True)
    return frame


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_csv = Path(args.output_csv).expanduser().resolve(strict=False)
    if output_csv.exists() and not args.overwrite:
        raise ValueError(f"Output already exists. Use --overwrite to replace it: {output_csv}")

    frames = [load_teacher_label_csv(path_value) for path_value in args.input_csv]
    combined = pd.concat(frames, ignore_index=True)
    duplicate_mask = combined.duplicated(subset=["__image_key", "split"], keep="first")
    duplicate_rows = combined.loc[duplicate_mask, ["image_path", "split", "__source_csv"]].copy()
    combined = combined.drop_duplicates(subset=["__image_key", "split"], keep="first").reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined[OUTPUT_FIELDNAMES].to_csv(output_csv, index=False)

    summary = {
        "output_csv": str(output_csv),
        "input_csvs": [str(Path(path_value).expanduser().resolve(strict=False)) for path_value in args.input_csv],
        "rows_written": int(len(combined)),
        "split_counts": {str(key): int(value) for key, value in combined["split"].value_counts().to_dict().items()},
        "duplicate_rows_dropped": int(len(duplicate_rows)),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
