# MPNet Threshold Calibration

## Inputs
- Validation predictions: `outputs/alamp_paper_mpnet_selector_compare_20260512/eval/v4_1024_gpu_tuned_cw22_p2short/v4_1024_gpu_tuned_cw22_p2short/val_predictions.csv`
- Test predictions: `outputs/alamp_paper_mpnet_selector_compare_20260512/eval/v4_1024_gpu_tuned_cw22_p2short/v4_1024_gpu_tuned_cw22_p2short/test_predictions.csv`
- Rows: validation 1024, test 1024

## Selected Thresholds
| Selection | Threshold | Val Balanced Acc | Val Specificity | Val Recall | Val F1 | Test Balanced Acc | Test Specificity | Test Recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| threshold_0_50_baseline | 0.50 | 0.596237 | 0.440397 | 0.752078 | 0.757322 | 0.581078 | 0.425170 | 0.736986 | 0.748782 |
| best_validation_f1 | 0.03 | 0.506742 | 0.023179 | 0.990305 | 0.825635 | 0.507441 | 0.027211 | 0.987671 | 0.830167 |
| best_validation_balanced_accuracy | 0.71 | 0.635381 | 0.682119 | 0.588643 | 0.683829 | 0.617366 | 0.629252 | 0.605479 | 0.690086 |
| best_validation_youden_j | 0.71 | 0.635381 | 0.682119 | 0.588643 | 0.683829 | 0.617366 | 0.629252 | 0.605479 | 0.690086 |
| lowest_validation_specificity_ge_0_30 | 0.35 | 0.583488 | 0.301325 | 0.865651 | 0.802311 | 0.576279 | 0.326531 | 0.826027 | 0.787720 |
| lowest_validation_specificity_ge_0_50 | 0.55 | 0.613573 | 0.500000 | 0.727147 | 0.751073 | 0.595299 | 0.472789 | 0.717808 | 0.743790 |
| lowest_validation_specificity_ge_0_70 | 0.75 | 0.633120 | 0.705298 | 0.560942 | 0.666118 | 0.612785 | 0.666667 | 0.558904 | 0.660194 |
