#!/bin/bash
set -e

BASE_DIR="outputs/alamp_paper_mpnet_patch_selector_v4_20260512"
mkdir -p "$BASE_DIR/subsets"

SALIENCY_ROOT="outputs/alamp_paper_mpnet_patch_selector_v3_20260512/saliency_maps"

echo "=== V4 Smoke Test (10 images) ==="
./.venv_gpu/bin/python src/datasets/alamp_paper_patch_selector.py \
    --csv outputs/alamp_object_graph_subset_20260511/subsets/train_ultra_valid_1024.csv \
    --output_jsonl "$BASE_DIR/smoke_patches_v4.jsonl" \
    --max_images 10 \
    --selector_version v4 \
    --saliency_map_jsonl "$SALIENCY_ROOT/train/saliency_metadata.jsonl"

echo "=== Generating Train Subset V4 ==="
./.venv_gpu/bin/python src/datasets/alamp_paper_patch_selector.py \
    --csv outputs/alamp_object_graph_subset_20260511/subsets/train_ultra_valid_1024.csv \
    --output_jsonl "$BASE_DIR/subsets/train_patch_boxes_1024_v4.jsonl" \
    --selector_version v4 \
    --saliency_map_jsonl "$SALIENCY_ROOT/train/saliency_metadata.jsonl"

echo "=== Generating Val Subset V4 ==="
./.venv_gpu/bin/python src/datasets/alamp_paper_patch_selector.py \
    --csv outputs/alamp_object_graph_subset_20260511/subsets/val_ultra_valid_1024.csv \
    --output_jsonl "$BASE_DIR/subsets/val_patch_boxes_1024_v4.jsonl" \
    --selector_version v4 \
    --saliency_map_jsonl "$SALIENCY_ROOT/val/saliency_metadata.jsonl"

echo "=== Generating Test Subset V4 ==="
./.venv_gpu/bin/python src/datasets/alamp_paper_patch_selector.py \
    --csv outputs/alamp_object_graph_subset_20260511/subsets/test_ultra_valid_1024.csv \
    --output_jsonl "$BASE_DIR/subsets/test_patch_boxes_1024_v4.jsonl" \
    --selector_version v4 \
    --saliency_map_jsonl "$SALIENCY_ROOT/test/saliency_metadata.jsonl"

echo "=== Generating Visualizations V4 (50 images) ==="
./.venv_gpu/bin/python tools/visualize_alamp_patch_boxes.py \
  --input_jsonl "$BASE_DIR/subsets/train_patch_boxes_1024_v4.jsonl" \
  --output_dir "$BASE_DIR/patch_visualizations/train" \
  --max_images 50

echo "=== Done V4 Patch Selection ==="
