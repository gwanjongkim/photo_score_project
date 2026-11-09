# Mobile Student IQA v3 Training Report

## 1. Summary
- Final decision: `NEEDS_MORE_TUNING`
- Input manifest: `outputs/student_training_manifest_50k_20260601/student_training_manifest.csv`
- Output directory: `outputs/mobile_student_iqa_v3_20260601`

## 2. Dataset
- Split counts after image validation: `{'train': 34998, 'val': 7507, 'test': 7495}`
- Image load failures: `0`
- Target: `target_norm = guarded_target_0_100 / 100.0`

## 3. Model
- Backbone: `efficientnetv2b0`
- Input size: `256x256x3`
- Head: `GAP -> Dense(256, relu) -> Dropout(0.3) -> Dense(64, relu) -> Dropout(0.2) -> Dense(1, sigmoid)`
- Include preprocessing in Keras application: `True`
- Preprocessing behavior: Input generator emits RGB float32 0..255. Keras application preprocessing is included when supported.
- ImageNet weights loaded: `True`

## 4. Training Configuration
- Stage A head epochs: `5`
- Stage B fine-tune epochs: `20`
- Stage B requested unfreeze last N: `40`
- Stage B trainable backbone layers: `32`
- Stage B trainable BatchNorm layers: `0`
- Batch size: `32`
- Loss: `huber` with huber_delta `0.05` when applicable

## 5. Training History Summary
- History rows: `25`
- Best validation MAE norm: `0.04572773724794388`

## 6. Overall Metrics Against Guarded Target
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 34998 | 34998 | 3.152188 | 4.639573 | 0.817717 | 0.931444 | 11.656751 | 80.711824 | 66.423369 | 11.481825 |
| val | overall | 7507 | 7507 | 4.572775 | 6.714674 | 0.667474 | 0.842053 | 13.329400 | 80.375803 | 66.321262 | 10.918976 |
| test | overall | 7495 | 7495 | 4.543416 | 6.587053 | 0.658204 | 0.848766 | 12.366852 | 78.467983 | 66.230478 | 10.996612 |

## 7. Dataset-wise Metrics Against Guarded Target
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | flive | 4799 | 4799 | 3.300102 | 4.577548 | 0.514385 | 0.528198 | 31.065497 | 78.467983 | 69.992619 | 4.576795 |
| test | koniq10k | 1348 | 1348 | 5.944221 | 8.230735 | 0.643786 | 0.678792 | 28.353325 | 78.035688 | 66.785376 | 7.558552 |
| test | spaq | 1348 | 1348 | 7.568918 | 9.945111 | 0.840971 | 0.851651 | 12.366852 | 78.213418 | 52.282022 | 17.187074 |

## 8. MOS Reference Metrics
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | overall | 7495 | 7495 | 6.229262 | 8.554435 | 0.582835 | 0.790085 | 12.366852 | 78.467983 | 66.230478 | 10.996612 |
| test | flive | 4799 | 4799 | 5.132270 | 6.949717 | 0.397023 | 0.391463 | 31.065497 | 78.467983 | 69.992619 | 4.576795 |
| test | koniq10k | 1348 | 1348 | 6.593854 | 8.958396 | 0.628047 | 0.679478 | 28.353325 | 78.035688 | 66.785376 | 7.558552 |
| test | spaq | 1348 | 1348 | 9.770059 | 12.436966 | 0.801587 | 0.809943 | 12.366852 | 78.213418 | 52.282022 | 17.187074 |

## 9. Safety Metrics
| split | dataset | n | finite_n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- |
| val | overall | 7507 | 7507 | 131 | 18 | 1 |
| val | flive | 4801 | 4801 | 45 | 2 | 1 |
| val | koniq10k | 1353 | 1353 | 40 | 4 | 0 |
| val | spaq | 1353 | 1353 | 46 | 12 | 0 |
| test | overall | 7495 | 7495 | 145 | 28 | 7 |
| test | flive | 4799 | 4799 | 48 | 2 | 3 |
| test | koniq10k | 1348 | 1348 | 53 | 7 | 0 |
| test | spaq | 1348 | 1348 | 44 | 19 | 4 |

## 10. Severe False Positive Cases
- Severe/standard FP case rows saved: `276`
- CSV: `outputs/mobile_student_iqa_v3_20260601/severe_false_positive_cases.csv`

## 11. v2 Comparison
| metric | v2 | v3 | improvement |
| --- | --- | --- | --- |
| test_srcc | 0.666845 | 0.658204 | -0.008641 |
| test_mae | 7.234272 | 4.543416 | 2.690856 |
| test_severe_fp | 5.000000 | 28.000000 | -23.000000 |
| flive_test_srcc | 0.361239 | 0.514385 | 0.153146 |

## 12. Risks
- This run is not a TFLite export and does not modify app runtime scoring.
- The guarded target is a teacher-label design target, not a Flutter runtime formula.
- If FLIVE SRCC remains low, inspect data balance, resolution, and target behavior before deployment work.

## 13. Final Decision
- `NEEDS_MORE_TUNING`
- Test SRCC 0.6582 is below 0.85.
- Test severe FP count is 28, expected 0.
