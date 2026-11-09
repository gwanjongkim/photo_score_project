#!/bin/bash
set -e

BASE_DIR="outputs/alamp_paper_mpnet_patch_selector_v3_20260512"
mkdir -p "$BASE_DIR/saliency_maps/train"
mkdir -p "$BASE_DIR/saliency_maps/val"
mkdir -p "$BASE_DIR/saliency_maps/test"

echo "=== Generating Saliency Maps (Train) ==="
./.venv_gpu/bin/python tools/generate_u2net_saliency_maps.py \
    --csv outputs/alamp_object_graph_subset_20260511/subsets/train_ultra_valid_1024.csv \
    --output_dir "$BASE_DIR/saliency_maps/train"

echo "=== Generating Saliency Maps (Val) ==="
./.venv_gpu/bin/python tools/generate_u2net_saliency_maps.py \
    --csv outputs/alamp_object_graph_subset_20260511/subsets/val_ultra_valid_1024.csv \
    --output_dir "$BASE_DIR/saliency_maps/val"

echo "=== Generating Saliency Maps (Test) ==="
./.venv_gpu/bin/python tools/generate_u2net_saliency_maps.py \
    --csv outputs/alamp_object_graph_subset_20260511/subsets/test_ultra_valid_1024.csv \
    --output_dir "$BASE_DIR/saliency_maps/test"

echo "=== Done Saliency Generation ==="
