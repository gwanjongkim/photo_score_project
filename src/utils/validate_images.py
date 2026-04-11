
"""
A command-line utility to validate images in a CSV file.

This script reads a CSV file with an 'image_path' column, checks each image for validity,
and reports statistics. It can also produce a cleaned CSV file containing only the
valid images and a text file listing the invalid image paths.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils.image_utils import is_image_valid


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate images in a CSV file.")
    parser.add_argument("csv_path", type=Path, help="Path to the input CSV file.")
    parser.add_argument(
        "--cleaned_csv_path",
        type=Path,
        help="Optional path to save the cleaned CSV file.",
    )
    parser.add_argument(
        "--invalid_log_path",
        type=Path,
        help="Optional path to save a log of invalid image paths.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    if "image_path" not in df.columns:
        raise ValueError("CSV file must have an 'image_path' column.")

    valid_indices = []
    invalid_paths = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating images"):
        image_path = row["image_path"]
        if Path(image_path).exists() and is_image_valid(image_path):
            valid_indices.append(idx)
        else:
            invalid_paths.append(image_path)

    print(f"Total images checked: {len(df)}")
    print(f"Valid images: {len(valid_indices)}")
    print(f"Invalid images: {len(invalid_paths)}")

    if args.cleaned_csv_path:
        cleaned_df = df.iloc[valid_indices]
        args.cleaned_csv_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(args.cleaned_csv_path, index=False)
        print(f"Saved cleaned CSV to: {args.cleaned_csv_path}")

    if args.invalid_log_path:
        args.invalid_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.invalid_log_path, "w") as f:
            for path in invalid_paths:
                f.write(f"{path}\\n")
        print(f"Saved invalid image log to: {args.invalid_log_path}")


if __name__ == "__main__":
    main()
