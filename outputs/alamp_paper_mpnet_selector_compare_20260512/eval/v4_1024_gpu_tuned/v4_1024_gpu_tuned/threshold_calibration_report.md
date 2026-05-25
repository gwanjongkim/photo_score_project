# MPNet Threshold Calibration

## Inputs
- Validation predictions: `outputs/alamp_paper_mpnet_selector_compare_20260512/eval/v4_1024_gpu_tuned/v4_1024_gpu_tuned/val_predictions.csv`
- Test predictions: `outputs/alamp_paper_mpnet_selector_compare_20260512/eval/v4_1024_gpu_tuned/v4_1024_gpu_tuned/test_predictions.csv`
- Rows: validation 1024, test 1024

## Selected Thresholds
| Selection | Threshold | Val Balanced Acc | Val Specificity | Val Recall | Val F1 | Test Balanced Acc | Test Specificity | Test Recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| threshold_0_50_baseline | 0.50 | 0.593564 | 0.390728 | 0.796399 | 0.776502 | 0.587052 | 0.380952 | 0.793151 | 0.776660 |
| best_validation_f1 | 0.03 | 0.502889 | 0.009934 | 0.995845 | 0.826437 | 0.504063 | 0.013605 | 0.994521 | 0.831615 |
| best_validation_balanced_accuracy | 0.81 | 0.640673 | 0.738411 | 0.542936 | 0.657167 | 0.620557 | 0.697279 | 0.543836 | 0.652961 |
| best_validation_youden_j | 0.81 | 0.640673 | 0.738411 | 0.542936 | 0.657167 | 0.620557 | 0.697279 | 0.543836 | 0.652961 |
| lowest_validation_specificity_ge_0_30 | 0.41 | 0.577255 | 0.301325 | 0.853186 | 0.795352 | 0.582798 | 0.323129 | 0.842466 | 0.796632 |
| lowest_validation_specificity_ge_0_50 | 0.61 | 0.615229 | 0.503311 | 0.727147 | 0.751611 | 0.624611 | 0.513605 | 0.735616 | 0.761702 |
| lowest_validation_specificity_ge_0_70 | 0.79 | 0.636583 | 0.705298 | 0.567867 | 0.671581 | 0.619257 | 0.676871 | 0.561644 | 0.663968 |
