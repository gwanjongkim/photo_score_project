#!/bin/bash
# A-LAMP 논문식 엄격 패치 선택 검증을 실행하는 스크립트
set -euo pipefail

BASE_DIR="outputs/alamp_paper_strict_patch_selector_20260512"
SALIENCY_ROOT="outputs/alamp_paper_mpnet_patch_selector_v3_20260512/saliency_maps"
PYTHON_BIN="./.venv_gpu/bin/python"
SELECTOR="src/datasets/alamp_paper_strict_patch_selector.py"

mkdir -p "$BASE_DIR"
LOG_PATH="$BASE_DIR/command_log.txt"

exec > >(tee "$LOG_PATH") 2>&1

echo "=== A-LAMP Paper Strict Patch Selector ==="
date

echo "=== py_compile ==="
"$PYTHON_BIN" -m py_compile "$SELECTOR"

echo "=== Smoke Test (10 images) ==="
"$PYTHON_BIN" "$SELECTOR" generate \
  --csv outputs/alamp_object_graph_subset_20260511/subsets/train_ultra_valid_1024.csv \
  --saliency_map_jsonl "$SALIENCY_ROOT/train/saliency_metadata.jsonl" \
  --output_jsonl "$BASE_DIR/smoke_patch_boxes_10_strict.jsonl" \
  --max_images 10

echo "=== Generating Train 1024 Strict Patches ==="
"$PYTHON_BIN" "$SELECTOR" generate \
  --csv outputs/alamp_object_graph_subset_20260511/subsets/train_ultra_valid_1024.csv \
  --saliency_map_jsonl "$SALIENCY_ROOT/train/saliency_metadata.jsonl" \
  --output_jsonl "$BASE_DIR/train_patch_boxes_1024_strict.jsonl"

echo "=== Generating Val 1024 Strict Patches ==="
"$PYTHON_BIN" "$SELECTOR" generate \
  --csv outputs/alamp_object_graph_subset_20260511/subsets/val_ultra_valid_1024.csv \
  --saliency_map_jsonl "$SALIENCY_ROOT/val/saliency_metadata.jsonl" \
  --output_jsonl "$BASE_DIR/val_patch_boxes_1024_strict.jsonl"

echo "=== Generating Test 1024 Strict Patches ==="
"$PYTHON_BIN" "$SELECTOR" generate \
  --csv outputs/alamp_object_graph_subset_20260511/subsets/test_ultra_valid_1024.csv \
  --saliency_map_jsonl "$SALIENCY_ROOT/test/saliency_metadata.jsonl" \
  --output_jsonl "$BASE_DIR/test_patch_boxes_1024_strict.jsonl"

echo "=== Generating 50 Overlay Visualizations ==="
"$PYTHON_BIN" tools/visualize_alamp_patch_boxes.py \
  --input_jsonl "$BASE_DIR/train_patch_boxes_1024_strict.jsonl" \
  --output_dir "$BASE_DIR/overlay_visualizations" \
  --max_images 50

echo "=== Writing Report and Comparing Against V4 ==="
"$PYTHON_BIN" "$SELECTOR" report \
  --train_jsonl "$BASE_DIR/train_patch_boxes_1024_strict.jsonl" \
  --val_jsonl "$BASE_DIR/val_patch_boxes_1024_strict.jsonl" \
  --test_jsonl "$BASE_DIR/test_patch_boxes_1024_strict.jsonl" \
  --v4_train_jsonl outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/train_patch_boxes_1024_v4.jsonl \
  --v4_val_jsonl outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/val_patch_boxes_1024_v4.jsonl \
  --v4_test_jsonl outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/test_patch_boxes_1024_v4.jsonl \
  --visualization_dir "$BASE_DIR/overlay_visualizations" \
  --output_md "$BASE_DIR/patch_selector_strict_report.md" \
  --output_json "$BASE_DIR/patch_selector_strict_summary.json"

echo "=== Done ==="
date
