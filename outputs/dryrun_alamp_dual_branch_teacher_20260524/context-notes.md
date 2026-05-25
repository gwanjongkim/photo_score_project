# Context Notes - A-LAMP Dual-Branch GCN Teacher Prototype

## Decisions
- This is a teacher-strengthening prototype, not a full A-LAMP reproduction.
- External patch boxes are adaptive selections, not ground-truth labels.
- Labels remain `mean_score > 5.0 -> 1 else 0`.
- Graph rows come from YOLO object detections at confidence `0.10`, not from full A-LAMP Layout-Aware ground truth.
- Use graph JSONL `object_mask`, boxes, centers, areas, class IDs, confidences, local edges, and global edges to form fixed node features.
- Use the reusable `GraphConvolutionLayer` and `MaskedMeanPoolingLayer` from `src/models/alamp_paper_fusion.py`.
- Training and validation datasets must repeat when explicit step counts are used.

## Located Graph Artifacts
- `outputs/alamp_object_graph_subset_20260511/graphs_conf010/train_graphs_4096.jsonl`
- `outputs/alamp_object_graph_subset_20260511/graphs_conf010/val_graphs_4096.jsonl`
- `outputs/alamp_object_graph_subset_20260511/graphs_conf010/test_graphs_4096.jsonl`
- Matching external-patch rows: train `4085`, val `4082`, test `4081`.

## Validation Log
- Added isolated source/config files:
  `src/datasets/alamp_dual_branch_dataset.py`,
  `src/models/alamp_dual_branch_teacher.py`,
  `src/train/train_alamp_dual_branch_teacher.py`,
  `src/eval/evaluate_alamp_dual_branch_teacher.py`, and
  `configs/paper_benchmarks/alamp_dual_branch_teacher_ava_4096.yaml`.
- `py_compile` passed for all new source files.
- Tiny smoke training passed with 16 matched train samples and 8 matched val samples.
- Smoke artifacts were written under `outputs/dryrun_alamp_dual_branch_teacher_20260524/smoke_train/`.
- Smoke metrics: train accuracy `0.3125`, train AUC `0.34375`, val accuracy `0.375`, val AUC `0.46666663885116577`.
- Save/load verification passed with `save_load_max_abs_diff: 0.0`.
- 4096 subset small training completed for 3 epochs in `outputs/smallrun_alamp_dual_branch_teacher_4096_20260524`.
- 4096 train/val matched counts were train `4085` and val `4082`; explicit repeat was enabled for both.
- Best validation AUC was `0.6761755347251892` at epoch 3.
- Matching 4096 test-subset evaluation used `best.weights.h5` and wrote `outputs/smallrun_alamp_dual_branch_teacher_4096_20260524/eval_test/evaluation_summary.json`.
- Test-subset metrics: accuracy `0.7120803724577309`, F1 `0.8281409975135293`, ROC-AUC `0.6635004698845645`, AP `0.8222186081753107`.
- The prediction distribution is strongly positive-biased: positive prediction ratio at 0.5 was `0.9625091889242833`, with confusion matrix `tn=75`, `fp=1097`, `fn=78`, `tp=2831`.
- Recommendation: this prototype does not yet justify 30-40 hour full AVA graph generation.
