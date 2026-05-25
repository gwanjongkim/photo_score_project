# A-LAMP 4096 Fair Comparison: Multi-Patch-only vs Dual-Branch GCN

## 1. Summary

- Multi-Patch-only ROC-AUC: `0.791372`.
- Dual-Branch GCN ROC-AUC: `0.663500`.
- Multi-Patch-only best balanced accuracy: `0.720910` at threshold `0.720000`.
- Dual-Branch GCN best balanced accuracy: `0.621612` at threshold `0.750000`.
- Judgment: Stop this GCN branch for now; improve the Multi-Patch teacher instead.

## 2. Matched 4096 Test Subset

- Graph JSONL: `outputs/alamp_object_graph_subset_20260511/graphs_conf010/test_graphs_4096.jsonl`.
- Graph records: `4096`.
- Matched test image IDs: `4081`.
- ID basis: filename stem from graph image_path matched to patch image_id or patch image_path stem.
- Filtered JSONL: `outputs/alamp_dual_branch_4096_comparison_20260524/multipatch_test_4096_matched.jsonl`.

## 3. Actual Positive Ratio

- Positive labels: `2909`.
- Negative labels: `1172`.
- Actual positive ratio: `0.712815`.

## 4. Multi-Patch-only Results at Threshold 0.5

- Accuracy: `0.762558`.
- Balanced accuracy: `0.639081`.
- F1: `0.848000`.
- Precision: `0.779862`.
- Recall: `0.929185`.
- Specificity: `0.348976`.
- FPR: `0.651024`.
- Predicted positive ratio: `0.849302`.
- Confusion matrix: `tn=409`, `fp=763`, `fn=206`, `tp=2703`.

## 5. Dual-Branch GCN Results at Threshold 0.5

- Accuracy: `0.712080`.
- Balanced accuracy: `0.518590`.
- F1: `0.828141`.
- Precision: `0.720723`.
- Recall: `0.973187`.
- Specificity: `0.063993`.
- FPR: `0.936007`.
- Predicted positive ratio: `0.962509`.
- Confusion matrix: `tn=75`, `fp=1097`, `fn=78`, `tp=2831`.

## 6. Balanced Metrics

| Model | ROC-AUC | AP | Balanced Acc @0.5 | Specificity @0.5 | FPR @0.5 |
|---|---:|---:|---:|---:|---:|
| Multi-Patch-only | 0.791372 | 0.900265 | 0.639081 | 0.348976 | 0.651024 |
| Dual-Branch GCN | 0.663500 | 0.822219 | 0.518590 | 0.063993 | 0.936007 |

## 7. Threshold Sweep Results

- Multi-Patch sweep CSV: `outputs/alamp_dual_branch_4096_comparison_20260524/threshold_sweep_multipatch.csv`.
- Dual-Branch sweep CSV: `outputs/alamp_dual_branch_4096_comparison_20260524/threshold_sweep_dual_branch.csv`.

## 8. Best Threshold Comparison

| Model | Best balanced acc | Threshold | Best F1 | F1 threshold | Best accuracy | Accuracy threshold |
|---|---:|---:|---:|---:|---:|---:|
| Multi-Patch-only | 0.720910 | 0.720000 | 0.850952 | 0.470000 | 0.766724 | 0.560000 |
| Dual-Branch GCN | 0.621612 | 0.750000 | 0.833142 | 0.400000 | 0.715756 | 0.400000 |

## 9. Positive Bias Analysis

- Multi-Patch predicted positive ratio at 0.5: `0.849302`.
- Dual-Branch predicted positive ratio at 0.5: `0.962509`.
- Actual positive ratio: `0.712815`.
- Dual-Branch specificity at 0.5 is `0.063993`, so its F1 is inflated by predicting nearly everything positive.

## 10. Final Judgment

- Dual-Branch GCN does not beat Multi-Patch-only on ROC-AUC or best balanced accuracy. Its threshold-0.5 F1 is driven by very high recall and severe positive bias, not better ranking.
