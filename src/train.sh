#!/bin/bash
# Repeatable training workflow for photo_score_project
# Strictly uses .venv_gpu and handles both train/val cleaning.

set -e

# Configuration
PYTHON_EXEC="./.venv_gpu/bin/python"
DATA_DIR="data/processed/ava"
TRAIN_RAW="$DATA_DIR/train.csv"
VAL_RAW="$DATA_DIR/val.csv"
TRAIN_CLEANED="$DATA_DIR/train_cleaned.csv"
VAL_CLEANED="$DATA_DIR/val_cleaned.csv"
OUT_DIR="outputs/nima_ava_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=64
EPOCHS=10

# 1. Environment Check
echo "Checking environment with $PYTHON_EXEC..."
$PYTHON_EXEC src/utils/env_check.py

# 2. Dataset Validation / Cleaning
# We clean both train and val to ensure the pipeline never sees a bad image record.
# The dataset loader also has a runtime filter as a second layer of defense.

if [ ! -f "$TRAIN_CLEANED" ]; then
    echo "Train cleaned CSV not found. Validating $TRAIN_RAW..."
    PYTHONPATH=. $PYTHON_EXEC src/utils/validate_images.py "$TRAIN_RAW" \
        --cleaned_csv_path "$TRAIN_CLEANED" \
        --invalid_log_path "$DATA_DIR/bad_images_train.txt"
fi

if [ ! -f "$VAL_CLEANED" ]; then
    echo "Validation cleaned CSV not found. Validating $VAL_RAW..."
    PYTHONPATH=. $PYTHON_EXEC src/utils/validate_images.py "$VAL_RAW" \
        --cleaned_csv_path "$VAL_CLEANED" \
        --invalid_log_path "$DATA_DIR/bad_images_val.txt"
fi

# 3. Training Launch
echo "Starting NIMA training..."
echo "Output directory: $OUT_DIR"
PYTHONPATH=. $PYTHON_EXEC src/train/train_nima.py \
    --train_csv "$TRAIN_CLEANED" \
    --val_csv "$VAL_CLEANED" \
    --out_dir "$OUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS"

echo "Training workflow complete."
