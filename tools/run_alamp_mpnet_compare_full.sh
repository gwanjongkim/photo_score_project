#!/bin/bash
set -e

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=""
BASE_OUT="outputs/alamp_paper_mpnet_selector_compare_20260512"

# 1. Strict Selector Training (1024)
echo "=== Training Strict Selector MPNet ==="
./.venv_gpu/bin/python src/train/train_alamp_paper_mpnet_ava.py \
    --config configs/paper_benchmarks/alamp_paper_mpnet_compare_selectors.yaml \
    --train_jsonl outputs/alamp_paper_strict_patch_selector_20260512/train_patch_boxes_1024_strict.jsonl \
    --val_jsonl outputs/alamp_paper_strict_patch_selector_20260512/val_patch_boxes_1024_strict.jsonl \
    --out_dir "$BASE_OUT/train" \
    --selector_name strict \
    --epochs 3 \
    --batch_size 16 \
    --max_train_samples 1024 \
    --max_val_samples 256

# 2. Strict Selector Evaluation
echo "=== Evaluating Strict Selector MPNet ==="
./.venv_gpu/bin/python src/eval/evaluate_alamp_paper_mpnet_ava.py \
    --config configs/paper_benchmarks/alamp_paper_mpnet_compare_selectors.yaml \
    --model_path "$BASE_OUT/train/strict/final_model.keras" \
    --test_jsonl outputs/alamp_paper_strict_patch_selector_20260512/test_patch_boxes_1024_strict.jsonl \
    --val_jsonl outputs/alamp_paper_strict_patch_selector_20260512/val_patch_boxes_1024_strict.jsonl \
    --output_dir "$BASE_OUT/eval" \
    --selector_name strict

# 3. V4 Selector Training (1024)
echo "=== Training V4 Selector MPNet ==="
./.venv_gpu/bin/python src/train/train_alamp_paper_mpnet_ava.py \
    --config configs/paper_benchmarks/alamp_paper_mpnet_compare_selectors.yaml \
    --train_jsonl outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/train_patch_boxes_1024_v4.jsonl \
    --val_jsonl outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/val_patch_boxes_1024_v4.jsonl \
    --out_dir "$BASE_OUT/train" \
    --selector_name v4 \
    --epochs 3 \
    --batch_size 16 \
    --max_train_samples 1024 \
    --max_val_samples 256

# 4. V4 Selector Evaluation
echo "=== Evaluating V4 Selector MPNet ==="
./.venv_gpu/bin/python src/eval/evaluate_alamp_paper_mpnet_ava.py \
    --config configs/paper_benchmarks/alamp_paper_mpnet_compare_selectors.yaml \
    --model_path "$BASE_OUT/train/v4/final_model.keras" \
    --test_jsonl outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/test_patch_boxes_1024_v4.jsonl \
    --val_jsonl outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/val_patch_boxes_1024_v4.jsonl \
    --output_dir "$BASE_OUT/eval" \
    --selector_name v4
