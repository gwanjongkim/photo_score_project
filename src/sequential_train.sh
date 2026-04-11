#!/bin/bash
# Lightweight sequential training runner for photo_score_project
# Uses .venv_gpu and skips if complete checkpoints exist.

set -e

PYTHON_EXEC="./.venv_gpu/bin/python"
BASE_OUT_DIR="outputs"

# 1. NIMA (AVA)
echo "--- STAGE 1: NIMA ---"
NIMA_DIR="$BASE_OUT_DIR/nima_ava_restart"
if [ -f "$NIMA_DIR/final_model.keras" ]; then
    echo "NIMA already complete. Skipping."
else
    PYTHONPATH=. $PYTHON_EXEC src/train/train_nima.py \
        --train_csv data/processed/ava/train_cleaned.csv \
        --val_csv data/processed/ava/val_cleaned.csv \
        --out_dir "$NIMA_DIR" \
        --batch_size 64 \
        --epochs 10
fi

# 2. MUSIQ (AADB)
echo "--- STAGE 2: MUSIQ ---"
MUSIQ_DIR="$BASE_OUT_DIR/musiq_restart"
if [ -f "$MUSIQ_DIR/final_model.keras" ]; then
    echo "MUSIQ already complete. Skipping."
else
    PYTHONPATH=. $PYTHON_EXEC src/train/train_musiq.py \
        --train_csv data/processed/aadb/train.csv \
        --val_csv data/processed/aadb/val.csv \
        --target_col score \
        --out_dir "$MUSIQ_DIR" \
        --batch_size 4 \
        --epochs 10
fi

# 3. A-Lamp (AADB)
echo "--- STAGE 3: A-Lamp ---"
ALAMP_DIR="$BASE_OUT_DIR/alamp_restart"
if [ -f "$ALAMP_DIR/final_model.keras" ]; then
    echo "A-Lamp already complete. Skipping."
else
    PYTHONPATH=. $PYTHON_EXEC src/train/train_alamp.py \
        --train_csv data/processed/aadb/train.csv \
        --val_csv data/processed/aadb/val.csv \
        --target_col score \
        --out_dir "$ALAMP_DIR" \
        --batch_size 8 \
        --epochs 10
fi

# 4. RGNet (AADB)
echo "--- STAGE 4: RGNet ---"
RGNET_DIR="$BASE_OUT_DIR/rgnet_restart"
if [ -f "$RGNET_DIR/final_model.keras" ]; then
    echo "RGNet already complete. Skipping."
else
    PYTHONPATH=. $PYTHON_EXEC src/train/train_rgnet.py \
        --train_csv data/processed/aadb/train.csv \
        --val_csv data/processed/aadb/val.csv \
        --target_col score \
        --out_dir "$RGNET_DIR" \
        --batch_size 16 \
        --epochs 10
fi

# 5. Pairwise (AADB)
echo "--- STAGE 5: Pairwise ---"
PAIRWISE_DIR="$BASE_OUT_DIR/pairwise_restart"
if [ -f "$PAIRWISE_DIR/final_model.keras" ]; then
    echo "Pairwise already complete. Skipping."
else
    PYTHONPATH=. $PYTHON_EXEC src/train/train_pairwise.py \
        --train_csv data/processed/aadb/train.csv \
        --val_csv data/processed/aadb/val.csv \
        --target_col score \
        --out_dir "$PAIRWISE_DIR" \
        --batch_size 16 \
        --epochs 10
fi

echo "All training stages complete."
