from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def build_argparser():
    parser = argparse.ArgumentParser(description="Build train/val/test csv for FLIVE image/patch data.")
    parser.add_argument("--raw_dir", required=True, help="FLIVE raw directory")
    parser.add_argument("--processed_dir", required=True, help="output processed directory")
    return parser


def split_and_save(df: pd.DataFrame, out_dir: Path, prefix: str):
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_df.to_csv(out_dir / f"{prefix}_train.csv", index=False)
    val_df.to_csv(out_dir / f"{prefix}_val.csv", index=False)
    test_df.to_csv(out_dir / f"{prefix}_test.csv", index=False)

    print(f"[{prefix}] train={len(train_df)} val={len(val_df)} test={len(test_df)}")


def main():
    parser = build_argparser()
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # image-level
    labels_image = pd.read_csv(raw_dir / "labels_image.csv")
    labels_image = labels_image.rename(columns={"name": "relative_path", "mos": "mos"})
    labels_image["image_path"] = labels_image["relative_path"].apply(lambda x: str(raw_dir / x))
    labels_image["dataset"] = "flive_image"
    labels_image["is_patch"] = 0
    labels_image["exists"] = labels_image["image_path"].apply(lambda x: Path(x).exists())
    labels_image = labels_image[labels_image["exists"]].copy()
    labels_image = labels_image[["image_path", "mos", "dataset", "is_patch", "relative_path"]]
    labels_image.to_csv(processed_dir / "labels_image_all.csv", index=False)
    split_and_save(labels_image, processed_dir, "image")

    # patch-level
    patch_csv = raw_dir / "patches_rebuilt.csv"
    if patch_csv.exists():
        patch_df = pd.read_csv(patch_csv)
        patch_df = patch_df.rename(columns={"patch_path": "image_path"})
        patch_df["dataset"] = "flive_patch"
        patch_df["is_patch"] = 1
        patch_df["relative_path"] = patch_df["relative_patch_name"]
        patch_df = patch_df[["image_path", "score", "dataset", "is_patch", "relative_path"]]
        patch_df = patch_df.rename(columns={"score": "mos"})
        patch_df.to_csv(processed_dir / "labels_patch_all.csv", index=False)
        split_and_save(patch_df, processed_dir, "patch")

        all_df = pd.concat([labels_image, patch_df], ignore_index=True)
        all_df.to_csv(processed_dir / "labels_all.csv", index=False)
        split_and_save(all_df, processed_dir, "all")
    else:
        print("No patches_rebuilt.csv found. Skipping patch/all csv generation.")
        labels_image.to_csv(processed_dir / "labels_all.csv", index=False)

    print("saved to:", processed_dir)


if __name__ == "__main__":
    main()
