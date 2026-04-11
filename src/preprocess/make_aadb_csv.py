from pathlib import Path
from scipy.io import loadmat
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path.home() / "photo_score_project"
RAW = ROOT / "data" / "raw" / "aadb"
PROC = ROOT / "data" / "processed" / "aadb"
IMG_DIR = RAW / "images"

PROC.mkdir(parents=True, exist_ok=True)

def flatten_name(x):
    """
    AADBinfo.mat의 name entry는
    array(['filename.jpg'], dtype='<U38')
    같은 중첩 구조라 문자열로 꺼내야 한다.
    """
    while hasattr(x, "__len__") and not isinstance(x, str):
        try:
            x = x[0]
        except Exception:
            break
    return str(x)

def build_df(name_arr, score_arr, split_name):
    names = [flatten_name(x) for x in name_arr[0]]
    scores = [float(x) for x in score_arr[0]]

    df = pd.DataFrame({
        "relative_path": names,
        "score": scores,
    })

    df["image_path"] = df["relative_path"].apply(lambda x: str(IMG_DIR / x))
    df["dataset"] = "aadb"
    df["split"] = split_name
    df["exists"] = df["image_path"].apply(lambda x: Path(x).exists())

    print(f"[{split_name}] rows:", len(df))
    print(f"[{split_name}] exists:", df['exists'].sum(), "/", len(df))

    missing = df[~df["exists"]]
    if len(missing) > 0:
        print(f"\n[{split_name}] missing samples:")
        print(missing.head(10)[["relative_path", "image_path"]])

    df = df[df["exists"]].copy()
    df = df[["image_path", "score", "dataset", "split", "relative_path"]]
    return df

def main():
    mat = loadmat(RAW / "AADBinfo.mat")

    train_df = build_df(mat["trainNameList"], mat["trainScore"], "train")
    test_df  = build_df(mat["testNameList"],  mat["testScore"],  "test")

    # train 내부에서 val 분리
    train_df_final, val_df = train_test_split(
        train_df,
        test_size=0.1,
        random_state=42
    )

    all_df = pd.concat([train_df_final, val_df, test_df], ignore_index=True)

    train_df_final.to_csv(PROC / "train.csv", index=False)
    val_df.to_csv(PROC / "val.csv", index=False)
    test_df.to_csv(PROC / "test.csv", index=False)
    all_df.to_csv(PROC / "labels_all.csv", index=False)

    print("\nSaved files:")
    print(PROC / "train.csv", len(train_df_final))
    print(PROC / "val.csv", len(val_df))
    print(PROC / "test.csv", len(test_df))
    print(PROC / "labels_all.csv", len(all_df))

if __name__ == "__main__":
    main()
