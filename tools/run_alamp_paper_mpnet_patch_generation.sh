#!/bin/bash
set -e

OUT_DIR="outputs/alamp_paper_mpnet_design_audit_20260511"
mkdir -p "$OUT_DIR/subsets"

echo "=== Compiling ==="
./.venv_gpu/bin/python -m py_compile src/datasets/alamp_paper_patch_selector.py

echo "=== Smoke Test (10 images) ==="
./.venv_gpu/bin/python src/datasets/alamp_paper_patch_selector.py \
    --csv outputs/alamp_object_graph_subset_20260511/subsets/train_ultra_valid_1024.csv \
    --output_jsonl "$OUT_DIR/smoke_patches.jsonl" \
    --max_images 10 \
    --patch_count 5

echo "=== Validating Smoke JSONL ==="
./.venv_gpu/bin/python -c "
import json, sys
count = 0
for line in open('$OUT_DIR/smoke_patches.jsonl'):
    data = json.loads(line)
    if data.get('skipped'): continue
    count += 1
    assert len(data['boxes_norm_xyxy']) == 5, 'Must have 5 boxes'
    for b in data['boxes_norm_xyxy']:
        assert all(0.0 <= v <= 1.0 for v in b), 'Norm coords must be in [0,1]'
        assert b[0] < b[2] and b[1] < b[3], 'x1<x2 and y1<y2'
print(f'Smoke validation passed on {count} valid images.')
"

echo "=== Generating Train Subset ==="
./.venv_gpu/bin/python src/datasets/alamp_paper_patch_selector.py \
    --csv outputs/alamp_object_graph_subset_20260511/subsets/train_ultra_valid_1024.csv \
    --output_jsonl "$OUT_DIR/subsets/train_patch_boxes_1024.jsonl"

echo "=== Generating Val Subset ==="
./.venv_gpu/bin/python src/datasets/alamp_paper_patch_selector.py \
    --csv outputs/alamp_object_graph_subset_20260511/subsets/val_ultra_valid_1024.csv \
    --output_jsonl "$OUT_DIR/subsets/val_patch_boxes_1024.jsonl"

echo "=== Generating Test Subset ==="
./.venv_gpu/bin/python src/datasets/alamp_paper_patch_selector.py \
    --csv outputs/alamp_object_graph_subset_20260511/subsets/test_ultra_valid_1024.csv \
    --output_jsonl "$OUT_DIR/subsets/test_patch_boxes_1024.jsonl"

echo "=== Validation complete ==="