#!/bin/bash
set -e

OUT_DIR="outputs/alamp_paper_mpnet_patch_selector_v2_20260511"
mkdir -p "$OUT_DIR/subsets"

echo "=== Compiling ==="
./.venv_gpu/bin/python -m py_compile src/datasets/alamp_paper_patch_selector.py

echo "=== V2 Smoke Test (10 images) ==="
./.venv_gpu/bin/python src/datasets/alamp_paper_patch_selector.py \
    --csv outputs/alamp_object_graph_subset_20260511/subsets/train_ultra_valid_1024.csv \
    --output_jsonl "$OUT_DIR/smoke_patches_v2.jsonl" \
    --max_images 10 \
    --selector_version v2 \
    --grid_size 8

echo "=== Generating Train Subset V2 ==="
./.venv_gpu/bin/python src/datasets/alamp_paper_patch_selector.py \
    --csv outputs/alamp_object_graph_subset_20260511/subsets/train_ultra_valid_1024.csv \
    --output_jsonl "$OUT_DIR/subsets/train_patch_boxes_1024_v2.jsonl" \
    --selector_version v2 \
    --grid_size 8

echo "=== Generating Val Subset V2 ==="
./.venv_gpu/bin/python src/datasets/alamp_paper_patch_selector.py \
    --csv outputs/alamp_object_graph_subset_20260511/subsets/val_ultra_valid_1024.csv \
    --output_jsonl "$OUT_DIR/subsets/val_patch_boxes_1024_v2.jsonl" \
    --selector_version v2 \
    --grid_size 8

echo "=== Generating Test Subset V2 ==="
./.venv_gpu/bin/python src/datasets/alamp_paper_patch_selector.py \
    --csv outputs/alamp_object_graph_subset_20260511/subsets/test_ultra_valid_1024.csv \
    --output_jsonl "$OUT_DIR/subsets/test_patch_boxes_1024_v2.jsonl" \
    --selector_version v2 \
    --grid_size 8

echo "=== Generating Visualizations V2 (50 images) ==="
./.venv_gpu/bin/python tools/visualize_alamp_patch_boxes.py \
  --input_jsonl "$OUT_DIR/subsets/train_patch_boxes_1024_v2.jsonl" \
  --output_dir "$OUT_DIR/patch_visualizations/train" \
  --max_images 50

echo "=== Done ==="
