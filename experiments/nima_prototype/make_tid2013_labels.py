import os
import numpy as np
import pandas as pd

# --------------------
# Paths
# --------------------
TID_ROOT = os.path.expanduser("~/tid2013")
IMG_DIR = os.path.join(TID_ROOT, "distorted_images")
MOS_PATH = os.path.join(TID_ROOT, "mos_with_names.txt")
OUT_CSV = os.path.expanduser("~/nima_project/data/labels_technical.csv")

# --------------------
# Read MOS file
# mos_with_names.txt format:
#   mos_value   filename
# --------------------
raw = pd.read_csv(MOS_PATH, sep=r"\s+|,", engine="python", header=None)
raw.columns = ["mos", "filename"]

print("raw shape:", raw.shape)
print(raw.head())

# --------------------
# TID2013 MOS is already roughly in [0, 9] or [1, 9]-like range depending on file
# We normalize to [1, 10] for NIMA-style scoring
# --------------------
mos_min = raw["mos"].min()
mos_max = raw["mos"].max()

def normalize_mos_to_1_10(mos, old_min, old_max):
    if old_max == old_min:
        return 5.5
    return 1.0 + (mos - old_min) * 9.0 / (old_max - old_min)

raw["mos_1_10"] = raw["mos"].apply(lambda x: normalize_mos_to_1_10(x, mos_min, mos_max))

print("\nMOS min/max:", mos_min, mos_max)
print(raw[["mos", "mos_1_10"]].head())

# --------------------
# Convert MOS to 10-bin Gaussian-like count distribution
# --------------------
def mos_to_counts(mos_1_10, sigma=1.0, scale=100):
    bins = np.arange(1, 11, dtype=np.float32)
    probs = np.exp(-0.5 * ((bins - mos_1_10) / sigma) ** 2)
    probs = probs / probs.sum()
    counts = np.round(probs * scale).astype(int)

    # avoid all-zero after rounding
    if counts.sum() == 0:
        counts[np.argmin(np.abs(bins - mos_1_10))] = scale

    # rounding can slightly change total
    diff = scale - counts.sum()
    if diff != 0:
        idx = np.argmin(np.abs(bins - mos_1_10))
        counts[idx] += diff

    return counts

rows = []
existing = set(os.listdir(IMG_DIR))

for _, row in raw.iterrows():
    filename = str(row["filename"])
    if filename not in existing:
        continue

    mos_1_10 = float(row["mos_1_10"])
    counts = mos_to_counts(mos_1_10, sigma=1.0, scale=100)

    entry = {"filename": filename}
    for i in range(10):
        entry[f"c{i+1}"] = int(counts[i])
    rows.append(entry)

labels = pd.DataFrame(rows)

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
labels.to_csv(OUT_CSV, index=False)

print("\nSaved:", OUT_CSV)
print("rows:", len(labels))
print(labels.head())

# sanity check
tot = labels[[f"c{i}" for i in range(1, 11)]].sum(axis=1)
print("vote_total min/median/max:", int(tot.min()), float(np.median(tot)), int(tot.max()))
