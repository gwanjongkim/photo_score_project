#!/bin/bash
set -e

BASE_DIR="outputs/alamp_paper_mpnet_patch_selector_v3_20260512"
mkdir -p "$BASE_DIR/subsets"

echo "=== V3 Smoke Test (10 images) ==="
./.venv_gpu/bin/python src/datasets/alamp_paper_patch_selector.py \
    --csv outputs/alamp_object_graph_subset_20260511/subsets/train_ultra_valid_1024.csv \
    --output_jsonl "$BASE_DIR/smoke_patches_v3.jsonl" \
    --max_images 10 \
    --selector_version v3 \
    --saliency_map_jsonl "$BASE_DIR/saliency_maps/train/saliency_metadata.jsonl" \
    --grid_size 8

echo "=== Generating Train Subset V3 ==="
./.venv_gpu/bin/python src/datasets/alamp_paper_patch_selector.py \
    --csv outputs/alamp_object_graph_subset_20260511/subsets/train_ultra_valid_1024.csv \
    --output_jsonl "$BASE_DIR/subsets/train_patch_boxes_1024_v3.jsonl" \
    --selector_version v3 \
    --saliency_map_jsonl "$BASE_DIR/saliency_maps/train/saliency_metadata.jsonl" \
    --grid_size 8

echo "=== Generating Val Subset V3 ==="
./.venv_gpu/bin/python src/datasets/alamp_paper_patch_selector.py \
    --csv outputs/alamp_object_graph_subset_20260511/subsets/val_ultra_valid_1024.csv \
    --output_jsonl "$BASE_DIR/subsets/val_patch_boxes_1024_v3.jsonl" \
    --selector_version v3 \
    --saliency_map_jsonl "$BASE_DIR/saliency_maps/val/saliency_metadata.jsonl" \
    --grid_size 8

echo "=== Generating Test Subset V3 ==="
./.venv_gpu/bin/python src/datasets/alamp_paper_patch_selector.py \
    --csv outputs/alamp_object_graph_subset_20260511/subsets/test_ultra_valid_1024.csv \
    --output_jsonl "$BASE_DIR/subsets/test_patch_boxes_1024_v3.jsonl" \
    --selector_version v3 \
    --saliency_map_jsonl "$BASE_DIR/saliency_maps/test/saliency_metadata.jsonl" \
    --grid_size 8

echo "=== Generating Visualizations V3 (50 images) ==="
./.venv_gpu/bin/python tools/visualize_alamp_patch_boxes.py \
  --input_jsonl "$BASE_DIR/subsets/train_patch_boxes_1024_v3.jsonl" \
  --output_dir "$BASE_DIR/patch_visualizations/train" \
  --max_images 50

echo "=== Done V3 Patch Selection ==="
