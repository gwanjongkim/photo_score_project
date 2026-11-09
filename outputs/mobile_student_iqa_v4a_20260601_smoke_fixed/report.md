# Mobile Student IQA v4a Training Report

## 1. Summary
- Final decision: `NEEDS_MORE_TUNING`
- Input manifest: `outputs/student_training_manifest_50k_20260601/student_training_manifest.csv`
- Output directory: `outputs/mobile_student_iqa_v4a_20260601_smoke_fixed`

## 2. Dataset
- Split counts after image validation: `{'train': 900, 'val': 450, 'test': 450}`
- Image load failures: `0`

## 3. Balanced Batch Sampling
- balanced_batches: `True`
- batch_size: `30`
- train dataset distribution before balancing: `{'flive': 567, 'spaq': 179, 'koniq10k': 154}`
- per-dataset samples per balanced batch: `10`

## 4. Model
- Backbone: `efficientnetv2b0`
- Input size: `256x256x3`
- Head: `GAP -> Dense(256, relu) -> Dropout(0.3) -> Dense(64, relu) -> Dropout(0.2) -> Dense(1, sigmoid)`
- Include preprocessing in Keras application: `True`
- Preprocessing behavior: Input generator emits RGB float32 0..255. Keras application preprocessing is included when supported.

## 5. Training Configuration
- Stage A head epochs: `1`
- Stage B fine-tune epochs: `1`
- Stage B trainable backbone layers: `32`
- Stage B trainable BatchNorm layers: `0`
- Training history rows: `2`

## 6. Loss and Safety Penalty
- base loss: `huber`
- huber_delta: `0.05`
- safety_lambda: `2.0`
- safety_threshold: `0.65`
- target_safety_cutoff: `55.0`
- mos_safety_cutoff: `50.0`
- sample weights enabled: `True`
- low_mos_weight: `3.0`
- severe_low_mos_weight: `5.0`

## 7. Checkpoint Selection
- Composite checkpointing used: `True`
- checkpoint metric rows: `2`
- severe_fp_penalty: `10.0`
- mae_penalty: `0.01`

## 8. Training History Summary
- best validation composite: `0.20585379327334108`
- best validation SRCC: `0.3174330737435741`
- best validation severe FP count: `0`

## 9. Overall Metrics Against Guarded Target
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 900 | 900 | 10.530770 | 12.999826 | 0.449334 | 0.632296 | 21.801789 | 84.970832 | 58.692751 | 11.784783 |
| val | overall | 450 | 450 | 11.157928 | 13.631193 | 0.317433 | 0.454127 | 27.157113 | 86.279535 | 59.431183 | 10.625809 |
| test | overall | 450 | 450 | 12.087251 | 14.462402 | 0.250497 | 0.433977 | 25.647578 | 85.816193 | 58.148415 | 10.628369 |

## 10. Dataset-wise Metrics Against Guarded Target
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | flive | 277 | 277 | 11.910372 | 14.237946 | -0.024748 | 0.010377 | 31.231213 | 85.816193 | 61.082286 | 9.179293 |
| test | koniq10k | 85 | 85 | 11.345933 | 13.390214 | 0.131592 | 0.201496 | 29.999527 | 78.642291 | 57.646677 | 9.372957 |
| test | spaq | 88 | 88 | 13.360062 | 16.071211 | 0.508784 | 0.541557 | 25.647578 | 75.625384 | 49.398022 | 11.120153 |

## 11. MOS Reference Metrics
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | overall | 450 | 450 | 12.955272 | 15.660621 | 0.228624 | 0.390067 | 25.647578 | 85.816193 | 58.148415 | 10.628369 |
| test | flive | 277 | 277 | 12.638723 | 15.228662 | -0.013159 | 0.029052 | 31.231213 | 85.816193 | 61.082286 | 9.179293 |
| test | koniq10k | 85 | 85 | 11.756962 | 13.899432 | 0.169048 | 0.222776 | 29.999527 | 78.642291 | 57.646677 | 9.372957 |
| test | spaq | 88 | 88 | 15.109138 | 18.372287 | 0.477719 | 0.514953 | 25.647578 | 75.625384 | 49.398022 | 11.120153 |

## 12. Safety Metrics
| split | dataset | n | finite_n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- |
| val | overall | 450 | 450 | 1 | 0 | 2 |
| val | flive | 301 | 301 | 0 | 0 | 0 |
| val | koniq10k | 80 | 80 | 1 | 0 | 0 |
| val | spaq | 69 | 69 | 0 | 0 | 2 |
| test | overall | 450 | 450 | 1 | 0 | 4 |
| test | flive | 277 | 277 | 0 | 0 | 2 |
| test | koniq10k | 85 | 85 | 1 | 0 | 0 |
| test | spaq | 88 | 88 | 0 | 0 | 2 |

## 13. Severe False Positive Cases
- Severe/standard FP case rows saved: `2`
- CSV: `outputs/mobile_student_iqa_v4a_20260601_smoke_fixed/severe_false_positive_cases.csv`

## 14. v3 Comparison
| metric | v3 | v4a | improvement |
| --- | --- | --- | --- |
| test_srcc | 0.658204 | 0.250497 | -0.407707 |
| test_mae | 4.543416 | 12.087251 | -7.543835 |
| test_severe_fp | 28.000000 | 0.000000 | 28.000000 |
| flive_test_srcc | 0.514385 | -0.024748 | -0.539133 |
| koniq10k_test_srcc | 0.643786 | 0.131592 | -0.512194 |
| spaq_test_srcc | 0.840971 | 0.508784 | -0.332187 |

## 15. Risks
- v4a does not add ranking loss yet; ranking can still lag even if safety improves.
- Balanced sampling changes the effective training distribution and should be judged against unbiased val/test metrics.
- The guarded target is a teacher-label design target, not a Flutter runtime formula.

## 16. Final Decision
- `NEEDS_MORE_TUNING`
- Test SRCC 0.2505 is below 0.85.
- Test MAE 12.0873 is above 10.
- At least one test dataset SRCC is below 0.50.
