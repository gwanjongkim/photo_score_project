# Mobile Student IQA v3 Training Report

## 1. Summary
- Final decision: `NEEDS_MORE_TUNING`
- Input manifest: `outputs/student_training_manifest_50k_20260601/student_training_manifest.csv`
- Output directory: `outputs/mobile_student_iqa_v3_20260601_smoke`

## 2. Dataset
- Split counts after image validation: `{'train': 1024, 'val': 512, 'test': 512}`
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
- Stage A head epochs: `1`
- Stage B fine-tune epochs: `1`
- Stage B requested unfreeze last N: `40`
- Stage B trainable backbone layers: `32`
- Stage B trainable BatchNorm layers: `0`
- Batch size: `32`
- Loss: `huber` with huber_delta `0.05` when applicable

## 5. Training History Summary
- History rows: `2`
- Best validation MAE norm: `0.07929568737745285`

## 6. Overall Metrics Against Guarded Target
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 1024 | 1024 | 7.779202 | 11.058979 | 0.487952 | 0.583486 | 45.324546 | 85.168201 | 67.556628 | 6.480594 |
| val | overall | 512 | 512 | 7.929568 | 11.292974 | 0.282000 | 0.374924 | 50.770044 | 82.319576 | 67.911456 | 5.676871 |
| test | overall | 512 | 512 | 8.492888 | 11.502405 | 0.198098 | 0.315290 | 44.032630 | 84.948093 | 67.086323 | 6.128131 |

## 7. Dataset-wise Metrics Against Guarded Target
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | flive | 321 | 321 | 5.997377 | 7.575717 | 0.001669 | 0.040461 | 44.032630 | 80.999041 | 68.315746 | 5.422461 |
| test | koniq10k | 96 | 96 | 8.806087 | 11.192181 | 0.142323 | 0.177624 | 54.453784 | 84.948093 | 67.156930 | 6.122967 |
| test | spaq | 95 | 95 | 16.608595 | 19.812870 | 0.295048 | 0.299772 | 48.144576 | 81.943363 | 62.860819 | 6.495561 |

## 8. MOS Reference Metrics
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | overall | 512 | 512 | 9.686744 | 12.857001 | 0.174357 | 0.279190 | 44.032630 | 84.948093 | 67.086323 | 6.128131 |
| test | flive | 321 | 321 | 7.444917 | 9.350614 | 0.019691 | 0.066802 | 44.032630 | 80.999041 | 68.315746 | 5.422461 |
| test | koniq10k | 96 | 96 | 9.379029 | 12.000583 | 0.153434 | 0.187166 | 54.453784 | 84.948093 | 67.156930 | 6.122967 |
| test | spaq | 95 | 95 | 17.572715 | 21.211522 | 0.244871 | 0.249825 | 48.144576 | 81.943363 | 62.860819 | 6.495561 |

## 9. Safety Metrics
| split | dataset | n | finite_n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- |
| val | overall | 512 | 512 | 23 | 9 | 0 |
| val | flive | 329 | 329 | 3 | 0 | 0 |
| val | koniq10k | 95 | 95 | 6 | 1 | 0 |
| val | spaq | 88 | 88 | 14 | 8 | 0 |
| test | overall | 512 | 512 | 15 | 8 | 0 |
| test | flive | 321 | 321 | 4 | 0 | 0 |
| test | koniq10k | 96 | 96 | 3 | 1 | 0 |
| test | spaq | 95 | 95 | 8 | 7 | 0 |

## 10. Severe False Positive Cases
- Severe/standard FP case rows saved: `38`
- CSV: `outputs/mobile_student_iqa_v3_20260601_smoke/severe_false_positive_cases.csv`

## 11. v2 Comparison
| metric | v2 | v3 | improvement |
| --- | --- | --- | --- |
| test_srcc | 0.666845 | 0.198098 | -0.468747 |
| test_mae | 7.234272 | 8.492888 | -1.258616 |
| test_severe_fp | 5.000000 | 8.000000 | -3.000000 |
| flive_test_srcc | 0.361239 | 0.001669 | -0.359570 |

## 12. Risks
- This run is not a TFLite export and does not modify app runtime scoring.
- The guarded target is a teacher-label design target, not a Flutter runtime formula.
- If FLIVE SRCC remains low, inspect data balance, resolution, and target behavior before deployment work.

## 13. Final Decision
- `NEEDS_MORE_TUNING`
- Test SRCC 0.1981 is below 0.85.
- Test severe FP count is 8, expected 0.
- At least one test dataset SRCC is below 0.50.
