from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import pandas as pd

from src.data.flive_utils import (
    parse_patch_filename,
    find_source_image,
    make_patch_image,
)


def build_argparser():
    parser = argparse.ArgumentParser(description="Reconstruct FLIVE patches from source images.")
    parser.add_argument("--images_dir", required=True, help="Directory containing source FLIVE images folders")
    parser.add_argument("--meta_dir", required=True, help="Directory containing labels_image.csv / labels_patch.csv")
    parser.add_argument("--output_dir", required=True, help="Output raw FLIVE directory")
    parser.add_argument("--csv_out", required=True, help="Output patch csv path")
    parser.add_argument("--patch_size_ratio", type=float, default=0.5)
    parser.add_argument("--resize_to", type=int, default=224)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    meta_dir = Path(args.meta_dir)
    output_dir = Path(args.output_dir)
    csv_out = Path(args.csv_out)

    labels_image_csv = meta_dir / "labels_image.csv"
    labels_patch_csv = meta_dir / "labels_patch.csv"

    if not labels_patch_csv.exists():
        raise FileNotFoundError(f"Missing: {labels_patch_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    patches_dir = output_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    patch_df = pd.read_csv(labels_patch_csv)
    image_df = pd.read_csv(labels_image_csv) if labels_image_csv.exists() else None

    stats: Dict[str, int] = {
        "total_patch_rows": 0,
        "source_found": 0,
        "created": 0,
        "skipped_existing": 0,
        "failed": 0,
        "missing_source": 0,
    }

    csv_out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    errors = []

    for _, row in patch_df.iterrows():
        stats["total_patch_rows"] += 1

        patch_rel = row["name"]
        score = float(row["mos"])

        try:
            base_name, patch_idx = parse_patch_filename(patch_rel)
        except Exception as e:
            stats["failed"] += 1
            errors.append((patch_rel, f"parse_error: {e}"))
            continue

        src_path = find_source_image(base_name, images_dir)
        if src_path is None:
            stats["missing_source"] += 1
            errors.append((patch_rel, f"source_not_found for base_name={base_name}"))
            continue

        stats["source_found"] += 1

        dst_path = patches_dir / Path(patch_rel).name

        try:
            if dst_path.exists() and not args.overwrite:
                stats["skipped_existing"] += 1
            else:
                make_patch_image(
                    src_path=src_path,
                    dst_path=dst_path,
                    patch_idx=patch_idx,
                    patch_size_ratio=args.patch_size_ratio,
                    resize_to=args.resize_to,
                )
                stats["created"] += 1

            rows.append(
                {
                    "patch_path": str(dst_path),
                    "image_id": base_name,
                    "split": "train",  # FLIVE patch split 정보가 따로 없으므로 기본 train으로 둠
                    "score": score,
                    "relative_patch_name": Path(patch_rel).name,
                    "source_image_path": str(src_path),
                }
            )
        except Exception as e:
            stats["failed"] += 1
            errors.append((patch_rel, f"patch_create_error: {e}"))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(csv_out, index=False)

    # sanity check
    existing_patch_count = sum(1 for _ in patches_dir.glob("*.jpg"))
    csv_count = len(out_df)

    print("\n=== FLIVE PATCH PREP SUMMARY ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"existing_patch_files: {existing_patch_count}")
    print(f"csv_rows: {csv_count}")

    if csv_count > 0:
        print("\nscore summary:")
        print(out_df["score"].describe())

    if errors:
        err_path = csv_out.with_suffix(".errors.csv")
        with err_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["patch_rel", "error"])
            writer.writerows(errors)
        print(f"\nerror log saved: {err_path}")

    if csv_count != existing_patch_count:
        print("\n[WARN] CSV row count and actual patch file count differ.")
    else:
        print("\nSanity check passed: CSV rows == actual patch files")


if __name__ == "__main__":
    main()
