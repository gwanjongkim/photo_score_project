# Plan - A-LAMP Dual-Branch GCN Teacher Prototype

## Goal
Implement a small-scale VGG16 Multi-Patch + YOLO graph GCN teacher prototype to test whether a layout-aware branch improves over the Multi-Patch-only teacher.

## Scope
- Add isolated dataset, model, train, eval, and config files.
- Reuse the validated external adaptive patch JSONLs and existing YOLO graph JSONL subsets.
- Do not generate full AVA graphs, run full AVA training, export TFLite, touch Flutter, or claim full A-LAMP reproduction.

## Preflight
- Valid 4096 graph JSONLs exist under `outputs/alamp_object_graph_subset_20260511/graphs_conf010/`.
- Train/val/test graph rows are fixed-shape `max_objects=4` records.
- Overlap with external patch manifests is slightly below 4096 and must be recorded.

## Steps
1. Build a dataset that joins patch and graph rows by image ID/path stem, uses only matched samples, and emits VGG16-preprocessed patches plus graph tensors.
2. Build a dual-branch model with shared VGG16 patch encoding, orderless mean+max patch aggregation, and a 2-layer masked GCN branch.
3. Add train/eval CLIs with bounded sample limits, repeat-safe training data, summaries, and required metrics.
4. Validate with `py_compile`, tiny smoke training, then a 4096 subset small run only.

## Success Criteria
- All new files compile.
- Tiny smoke training writes `best.weights.h5`, `final_model.keras`, `training_history.csv`, and `train_summary.json`.
- 4096 small run is limited to existing graph subsets and produces enough evidence to decide whether full graph generation is justified.
