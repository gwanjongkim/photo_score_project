#!/bin/bash
set -e

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=""

# Forward Smoke
echo "=== Forward Smoke Test ==="
./.venv_gpu/bin/python src/train/train_alamp_paper_mpnet_ava.py \
    --config configs/paper_benchmarks/alamp_paper_mpnet_compare_selectors.yaml \
    --train_jsonl outputs/alamp_paper_strict_patch_selector_20260512/train_patch_boxes_1024_strict.jsonl \
    --val_jsonl outputs/alamp_paper_strict_patch_selector_20260512/val_patch_boxes_1024_strict.jsonl \
    --out_dir outputs/alamp_paper_mpnet_selector_compare_20260512/smoke \
    --backbone_weights none \
    --batch_size 2 \
    --max_train_samples 2 \
    --max_val_samples 2 \
    --epochs 1 \
    --selector_name smoke

# Tiny Train Smoke
echo "=== Tiny Train Smoke (Strict) ==="
./.venv_gpu/bin/python src/train/train_alamp_paper_mpnet_ava.py \
    --config configs/paper_benchmarks/alamp_paper_mpnet_compare_selectors.yaml \
    --train_jsonl outputs/alamp_paper_strict_patch_selector_20260512/train_patch_boxes_1024_strict.jsonl \
    --val_jsonl outputs/alamp_paper_strict_patch_selector_20260512/val_patch_boxes_1024_strict.jsonl \
    --out_dir outputs/alamp_paper_mpnet_selector_compare_20260512/tiny_smoke \
    --backbone_weights none \
    --batch_size 2 \
    --max_train_samples 64 \
    --max_val_samples 32 \
    --epochs 1 \
    --selector_name strict_tiny
