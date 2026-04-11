from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path.home() / "photo_score_project"
RAW = ROOT / "data" / "raw" / "paq2piq"
PROC = ROOT / "data" / "processed" / "paq2piq"
IMG_ROOT = RAW   # 중요: 현재 구조에 맞게 RAW로 둔다

PROC.mkdir(parents=True, exist_ok=True)

def build_image_csv():
    df = pd.read_csv(RAW / "labels_image.csv")
    df = df.rename(columns={"name": "relative_path", "mos": "mos"})
    df["image_path"] = df["relative_path"].apply(lambda x: str(IMG_ROOT / x))
    df["dataset"] = "flive_image"
    df["is_patch"] = 0
    df["exists"] = df["image_path"].apply(lambda x: Path(x).exists())

    print("image rows:", len(df))
    print("image exists:", df["exists"].sum(), "/", len(df))

    df = df[df["exists"]].copy()
    return df[["image_path", "mos", "dataset", "is_patch", "relative_path"]]

def build_patch_csv():
    df = pd.read_csv(RAW / "labels_patch.csv")
    df = df.rename(columns={"name": "relative_path", "mos": "mos"})
    df["image_path"] = df["relative_path"].apply(lambda x: str(IMG_ROOT / x))
    df["dataset"] = "flive_patch"
    df["is_patch"] = 1
    df["exists"] = df["image_path"].apply(lambda x: Path(x).exists())

    print("patch rows:", len(df))
    print("patch exists:", df["exists"].sum(), "/", len(df))

    df = df[df["exists"]].copy()
    return df[["image_path", "mos", "dataset", "is_patch", "relative_path"]]

def split_and_save(df, prefix):
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_df.to_csv(PROC / f"{prefix}_train.csv", index=False)
    val_df.to_csv(PROC / f"{prefix}_val.csv", index=False)
    test_df.to_csv(PROC / f"{prefix}_test.csv", index=False)

    print(f"[{prefix}] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

def main():
    image_df = build_image_csv()
    patch_df = build_patch_csv()

    image_df.to_csv(PROC / "labels_image_all.csv", index=False)
    patch_df.to_csv(PROC / "labels_patch_all.csv", index=False)

    split_and_save(image_df, "image")
    split_and_save(patch_df, "patch")

    all_df = pd.concat([image_df, patch_df], ignore_index=True)
    all_df.to_csv(PROC / "labels_all.csv", index=False)
    split_and_save(all_df, "all")

    print("saved to:", PROC)

if __name__ == "__main__":
    main()
