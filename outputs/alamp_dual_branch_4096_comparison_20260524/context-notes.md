# Context Notes - A-LAMP 4096 Fair Comparison

## Decisions
- The graph JSONL lacks an explicit `image_id`, so the comparison ID set is filename-stem based from graph `image_path`, matched against external patch `image_id` and patch `image_path` stem.
- Use the existing Multi-Patch evaluator and Dual-Branch evaluator rather than changing evaluator logic.
- Use threshold sweep from `0.05` to `0.95` in `0.01` increments and emphasize balanced accuracy / ROC-AUC because the subset is positive-heavy.
- Do not call external patch boxes ground truth and do not claim full A-LAMP reproduction.

## Located Inputs
- Graph JSONL: `outputs/alamp_object_graph_subset_20260511/graphs_conf010/test_graphs_4096.jsonl`
- Full external patch test JSONL: `outputs/alamp_external_patch_full_conversion_20260524/alamp_external_patches_test.jsonl`
- Multi-Patch weights: `outputs/alamp_multipatch_teacher_full_ava_20260524/best.weights.h5`
- Dual-Branch weights: `outputs/smallrun_alamp_dual_branch_teacher_4096_20260524/best.weights.h5`

## Validation Log
- Created `multipatch_test_4096_matched.jsonl` with `4081` rows.
- Graph records: `4096`; unique filename-stem graph IDs: `4096`; unmatched graph IDs in the external patch test manifest: `15`.
- Multi-Patch-only eval completed on `4081` samples with no skipped samples: accuracy `0.7625581965204606`, F1 `0.848`, ROC-AUC `0.7913719866672455`, AP `0.9002650001826908`.
- Dual-Branch GCN eval was rerun into this comparison directory on the same `4081` labels: accuracy `0.7120803724577309`, F1 `0.8281409975135293`, ROC-AUC `0.6635004698845645`, AP `0.8222186081753107`.
- `python -m py_compile scripts/analyze_alamp_4096_thresholds.py` passed.
- Threshold artifacts written: `threshold_sweep_multipatch.csv`, `threshold_sweep_dual_branch.csv`, `summary.json`, and `report.md`.
- Best balanced accuracy: Multi-Patch-only `0.7209095404751876` at threshold `0.72`; Dual-Branch GCN `0.6216119621698929` at threshold `0.75`.
- Final recommendation: stop this GCN branch for now and improve the Multi-Patch teacher instead.
