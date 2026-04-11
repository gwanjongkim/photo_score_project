from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path.home() / "photo_score_project"
RAW = ROOT / "data" / "raw" / "koniq10k"
PROC = ROOT / "data" / "processed" / "koniq10k"
IMG_DIR = RAW / "images"

PROC.mkdir(parents=True, exist_ok=True)

# 파일명은 실제 raw 폴더에 있는 이름으로 수정 가능
score_file = None
for p in RAW.glob("*.csv"):
    if "score" in p.name.lower():
        score_file = p
        break

if score_file is None:
    raise FileNotFoundError("KonIQ score csv not found in raw/koniq10k")

df = pd.read_csv(score_file)

print("columns:", df.columns.tolist())
print(df.head())

# KonIQ 공식 csv는 대개 image_name / MOS 형태를 포함
img_col = None
mos_col = None

for c in df.columns:
    cl = c.lower()
    if "image" in cl and "name" in cl:
        img_col = c
    if cl == "mos" or "mos" in cl:
        mos_col = c

if img_col is None:
    raise ValueError("image name column not found")
if mos_col is None:
    raise ValueError("MOS column not found")

out = pd.DataFrame()
out["relative_path"] = df[img_col].astype(str)
out["image_path"] = out["relative_path"].apply(lambda x: str(IMG_DIR / x))
out["mos"] = pd.to_numeric(df[mos_col], errors="coerce")
out["dataset"] = "koniq10k"
out["is_patch"] = 0
out["exists"] = out["image_path"].apply(lambda x: Path(x).exists())

print("rows:", len(out))
print("exists:", out["exists"].sum(), "/", len(out))

out = out[out["exists"] & out["mos"].notna()].copy()
out = out[["image_path", "mos", "dataset", "is_patch", "relative_path"]]
out.to_csv(PROC / "labels_all.csv", index=False)

train_df, temp_df = train_test_split(out, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_df.to_csv(PROC / "train.csv", index=False)
val_df.to_csv(PROC / "val.csv", index=False)
test_df.to_csv(PROC / "test.csv", index=False)

print("train:", len(train_df))
print("val  :", len(val_df))
print("test :", len(test_df))
print("saved to:", PROC)
