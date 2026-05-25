# MPNet Threshold Calibration

## Inputs
- Validation predictions: `outputs/alamp_paper_mpnet_selector_compare_20260512/eval/v4_1024_gpu_tuned_cw23_p2mid/v4_1024_gpu_tuned_cw23_p2mid/val_predictions.csv`
- Test predictions: `outputs/alamp_paper_mpnet_selector_compare_20260512/eval/v4_1024_gpu_tuned_cw23_p2mid/v4_1024_gpu_tuned_cw23_p2mid/test_predictions.csv`
- Rows: validation 1024, test 1024

## Selected Thresholds
| Selection | Threshold | Val Balanced Acc | Val Specificity | Val Recall | Val F1 | Test Balanced Acc | Test Specificity | Test Recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| threshold_0_50_baseline | 0.50 | 0.596962 | 0.423841 | 0.770083 | 0.765840 | 0.596808 | 0.431973 | 0.761644 | 0.765313 |
| best_validation_f1 | 0.02 | 0.501926 | 0.006623 | 0.997230 | 0.826636 | 0.502362 | 0.010204 | 0.994521 | 0.831139 |
| best_validation_balanced_accuracy | 0.82 | 0.631185 | 0.774834 | 0.487535 | 0.616462 | 0.598877 | 0.727891 | 0.469863 | 0.594970 |
| best_validation_youden_j | 0.82 | 0.631185 | 0.774834 | 0.487535 | 0.616462 | 0.598877 | 0.727891 | 0.469863 | 0.594970 |
| lowest_validation_specificity_ge_0_30 | 0.36 | 0.578911 | 0.304636 | 0.853186 | 0.795866 | 0.581758 | 0.326531 | 0.836986 | 0.794022 |
| lowest_validation_specificity_ge_0_50 | 0.56 | 0.610803 | 0.500000 | 0.721607 | 0.747489 | 0.596338 | 0.469388 | 0.723288 | 0.746818 |
| lowest_validation_specificity_ge_0_70 | 0.77 | 0.621769 | 0.701987 | 0.541551 | 0.650042 | 0.605559 | 0.676871 | 0.534247 | 0.641975 |
