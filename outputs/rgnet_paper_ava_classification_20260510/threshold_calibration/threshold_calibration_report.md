# RGNet AVA Threshold Calibration

## Scope
This calibration uses existing full AVA lse_r4 validation and test prediction CSVs only.
No model training or evaluation rerun was performed by this script.

## Inputs
- Validation CSV: `outputs/rgnet_paper_ava_classification_20260510/eval/lse_r4/val_predictions.csv`
- Test CSV: `outputs/rgnet_paper_ava_classification_20260510/eval/lse_r4/test_predictions.csv`
- Label columns: validation `label`, test `label`
- Probability columns: validation `prediction_prob`, test `prediction_prob`
- Rows: validation 25551, test 25549

## Selected Thresholds
| Selection | Threshold | Val F1 | Val Balanced Acc | Val Precision | Val Recall | Val Specificity | Val FP | Test F1 | Test Balanced Acc | Test Precision | Test Recall | Test Specificity | Test FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 baseline | 0.500000 | 0.847343 | 0.683065 | 0.800890 | 0.899517 | 0.466614 | 4026 | 0.846781 | 0.678403 | 0.805728 | 0.892242 | 0.464564 | 3921 |
| Best validation F1 | 0.410000 | 0.850860 | 0.651827 | 0.779179 | 0.937066 | 0.366587 | 4781 | 0.851918 | 0.644876 | 0.782985 | 0.934160 | 0.355592 | 4719 |
| Best validation balanced accuracy | 0.700000 | 0.777484 | 0.727189 | 0.871166 | 0.701994 | 0.752385 | 1869 | 0.776942 | 0.723243 | 0.872990 | 0.699934 | 0.746552 | 1856 |
| Best validation Youden J | 0.700000 | 0.777484 | 0.727189 | 0.871166 | 0.701994 | 0.752385 | 1869 | 0.776942 | 0.723243 | 0.872990 | 0.699934 | 0.746552 | 1856 |

## Answers
- Is threshold 0.5 already good? Yes, threshold 0.50 is a strong baseline, but validation F1 improves by 0.003517 at threshold 0.41.
- Does a higher threshold reduce false positives? Yes, selected thresholds above 0.50 reduce false positives relative to 0.50.
- Best validation F1 threshold: 0.41, with validation F1 0.850860 and test F1 0.851918.
- Best validation balanced accuracy threshold: 0.70, with validation balanced accuracy 0.727189 and test balanced accuracy 0.723243.
- Applying best validation F1 to test changes F1 from 0.846781 to 0.851918, and test false positives from 3921 to 4719.
- Applying best validation balanced accuracy to test changes balanced accuracy from 0.678403 to 0.723243, and test false positives from 3921 to 1856.

## Confusion Matrices
- 0.50 validation: {'tn': 3522, 'fp': 4026, 'fn': 1809, 'tp': 16194}
- 0.50 test: {'tn': 3402, 'fp': 3921, 'fn': 1964, 'tp': 16262}
- Best validation F1 applied to test: {'tn': 2604, 'fp': 4719, 'fn': 1200, 'tp': 17026}
- Best validation balanced accuracy applied to test: {'tn': 5467, 'fp': 1856, 'fn': 5469, 'tp': 12757}
