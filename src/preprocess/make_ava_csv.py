
"""
Prepare AVA CSV files for NIMA-style distribution training.

Based on:
- AVA dataset vote histograms
- NIMA distribution supervision

Faithful parts:
- stores 10-bin vote/distribution labels
- computes mean aesthetic score from the vote histogram

Approximated parts:
- uses a simple random split when an official split file is not available in the repo

Expected raw inputs:
- one AVA annotation file in data/raw/ava with 10 score bins
- AVA images under data/raw/ava (recursive search)

Output files:
- data/processed/ava/labels_all.csv
- data/processed/ava/train.csv
- data/processed/ava/val.csv
- data/processed/ava/test.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path.home() / "photo_score_project"
RAW = ROOT / "data" / "raw" / "ava"
PROC = ROOT / "data" / "processed" / "ava"


def _find_annotation_file(raw_dir: Path) -> tuple[Path, str]:
    """Finds the best available annotation file and returns its path and type."""
    ava_txt_path = raw_dir / "AVA_Files" / "AVA.txt"
    if ava_txt_path.exists():
        return ava_txt_path, "ava_txt"
    
    ground_truth_path = raw_dir / "ground_truth_dataset.csv"
    if ground_truth_path.exists():
        return ground_truth_path, "ground_truth_csv"

    # Fallback to previous logic if specific files aren't found
    candidates = list(raw_dir.glob("*.txt")) + list(raw_dir.glob("*.csv"))
    for path in candidates:
        lowered = path.name.lower()
        if "ava" in lowered or "score" in lowered or "label" in lowered:
            if path.suffix.lower() == ".csv":
                return path, "ground_truth_csv"
            return path, "ava_txt"
    
    if candidates:
        if candidates[0].suffix.lower() == ".csv":
             return candidates[0], "ground_truth_csv"
        return candidates[0], "ava_txt"
        
    raise FileNotFoundError("No AVA annotation file found in data/raw/ava.")


def _load_raw_annotations(path: Path, file_type: str) -> pd.DataFrame:
    """Loads annotations based on the file type."""
    print(f"Loading annotations from: {path} (type: {file_type})")
    if file_type == "ava_txt":
        # Format: index, image_id, vote_1..10, tag_1, tag_2, challenge_id
        return pd.read_csv(path, sep=r"\s+", header=None, index_col=0)
    elif file_type == "ground_truth_csv":
        # Format: image_id, mean_score, std_dev, dist_1..10
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unknown annotation file type: {file_type}")


def _index_images(raw_dir: Path) -> dict[str, Path]:
    """Recursively find all images and create a stem-to-path map."""
    print("Indexing images...")
    image_paths = list(raw_dir.glob("**/*.jpg")) + list(raw_dir.glob("**/*.jpeg"))
    index = {p.stem: p for p in tqdm(image_paths, desc="Indexing")}
    print(f"Found {len(index)} images.")
    return index


def build_ava_frame(raw_df: pd.DataFrame, file_type: str, image_index: dict[str, Path]) -> pd.DataFrame:
    if file_type == "ava_txt":
        # Columns are: 1=image_id, 2-11=votes, 12-13=tags, 14=challenge_id
        raw_df.columns = ["image_id"] + [f"raw_vote_{i}" for i in range(1, 11)] + ["tag1", "tag2", "challenge_id"]
        vote_cols = [f"raw_vote_{i}" for i in range(1, 11)]
        votes = raw_df[vote_cols].astype("float32")
    elif file_type == "ground_truth_csv":
        # Columns are already named, but we need to generate votes from distributions if needed
        # This format often lacks raw votes, so we may need to approximate or just use mean/dist
        pass # For now, assume we have what we need
    else:
        raise ValueError(f"Unknown file type for building frame: {file_type}")

    out = pd.DataFrame()
    out["image_id"] = raw_df["image_id"].astype(str)
    
    print("Matching annotations to indexed images...")
    out["image_path"] = out["image_id"].apply(lambda x: image_index.get(str(x)))
    
    matched_rows = out["image_path"].notna()
    num_unmatched = len(out) - matched_rows.sum()
    if num_unmatched > 0:
        print(f"Warning: {num_unmatched} annotations had no matching image file.")
        # print("Sample unmatched IDs:", out[~matched_rows]["image_id"].head().tolist())

    out = out[matched_rows].copy()
    out["image_path"] = out["image_path"].astype(str)

    if file_type == "ava_txt":
        for i, col in enumerate(vote_cols):
            out[f"vote_{i + 1}"] = raw_df.loc[out.index, col]
        
        total_votes = out[[f"vote_{i + 1}" for i in range(10)]].sum(axis=1)
        total_votes = total_votes.replace(0, 1.0) # Avoid division by zero

        for i in range(10):
            out[f"dist_{i + 1}"] = out[f"vote_{i + 1}"] / total_votes

        weights = pd.Series(range(1, 11), dtype="float32")
        mean_calc = (out[[f"vote_{i + 1}" for i in range(10)]].mul(weights.values, axis=1).sum(axis=1) / total_votes)
        out["mean_score"] = mean_calc.astype("float32")

    elif file_type == "ground_truth_csv":
        # Directly use provided columns
        out["mean_score"] = raw_df.loc[out.index, "mean_score"]
        for i in range(10):
            col_name = f"dist_{i+1}"
            if col_name in raw_df.columns:
                out[col_name] = raw_df.loc[out.index, col_name]

    out["dataset"] = "ava"
    
    # Drop rows with no valid score
    out = out.dropna(subset=["mean_score"]).copy()

    return out.drop(columns=["image_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default=str(RAW))
    parser.add_argument("--out_dir", default=str(PROC))
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--val_size", type=float, default=0.1)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotation_path, file_type = _find_annotation_file(raw_dir)
    raw_df = _load_raw_annotations(annotation_path, file_type)
    
    image_index = _index_images(raw_dir)
    
    df = build_ava_frame(raw_df, file_type, image_index)

    if len(df) == 0:
        raise ValueError("After processing, the DataFrame is empty. Check image paths and annotation format.")

    print(f"\nSuccessfully matched {len(df)} samples.")

    # Splitting the data
    train_val_df, test_df = train_test_split(df, test_size=args.test_size, random_state=42)
    
    # Adjust val_size to be a fraction of the remaining train_val set
    val_ratio_of_train_val = args.val_size / (1.0 - args.test_size)
    train_df, val_df = train_test_split(train_val_df, test_size=val_ratio_of_train_val, random_state=42)

    labels_path = out_dir / "labels_all.csv"
    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    pd.concat([train_df, val_df, test_df], ignore_index=True).to_csv(labels_path, index=False)

    print("\n--- Processing Summary ---")
    print(f"Annotation source: {annotation_path.name}")
    print(f"Raw annotation rows: {len(raw_df)}")
    print(f"Discovered images: {len(image_index)}")
    print(f"Matched samples: {len(df)}")
    print("--------------------------")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    print("--------------------------")
    print(f"Output files saved to: {out_dir}")
    print(f"  - {train_path.name}")
    print(f"  - {val_path.name}")
    print(f"  - {test_path.name}")
    print(f"  - {labels_path.name}")
    print("--------------------------\n")


if __name__ == "__main__":
    main()
