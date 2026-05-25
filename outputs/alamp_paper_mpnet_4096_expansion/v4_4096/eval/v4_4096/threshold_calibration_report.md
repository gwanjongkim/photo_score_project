# MPNet Threshold Calibration

## Inputs
- Validation predictions: `outputs/alamp_paper_mpnet_4096_expansion/v4_4096/eval/v4_4096/val_predictions.csv`
- Test predictions: `outputs/alamp_paper_mpnet_4096_expansion/v4_4096/eval/v4_4096/test_predictions.csv`
- Rows: validation 4096, test 4096

## Selected Thresholds
| Selection | Threshold | Val Balanced Acc | Val Specificity | Val Recall | Val F1 | Test Balanced Acc | Test Specificity | Test Recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| threshold_0_50_baseline | 0.50 | 0.655832 | 0.507525 | 0.804138 | 0.801237 | 0.630288 | 0.463042 | 0.797533 | 0.791971 |
| best_validation_f1 | 0.15 | 0.543219 | 0.103679 | 0.982759 | 0.835532 | 0.532779 | 0.090909 | 0.974649 | 0.832602 |
| best_validation_balanced_accuracy | 0.73 | 0.658670 | 0.719064 | 0.598276 | 0.698049 | 0.648789 | 0.696686 | 0.600891 | 0.697416 |
| best_validation_youden_j | 0.73 | 0.658670 | 0.719064 | 0.598276 | 0.698049 | 0.648789 | 0.696686 | 0.600891 | 0.697416 |
| lowest_validation_specificity_ge_0_30 | 0.32 | 0.606066 | 0.303512 | 0.908621 | 0.827575 | 0.580399 | 0.260833 | 0.899966 | 0.818890 |
| lowest_validation_specificity_ge_0_50 | 0.50 | 0.655832 | 0.507525 | 0.804138 | 0.801237 | 0.630288 | 0.463042 | 0.797533 | 0.791971 |
| lowest_validation_specificity_ge_0_70 | 0.72 | 0.657226 | 0.706522 | 0.607931 | 0.703231 | 0.649254 | 0.687341 | 0.611168 | 0.703609 |
