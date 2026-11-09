# Mobile Student IQA v4a Training Report

## 1. Summary
- Final decision: `NEEDS_MORE_TUNING`
- Input manifest: `outputs/student_training_manifest_50k_20260601/student_training_manifest.csv`
- Output directory: `outputs/mobile_student_iqa_v4a_20260601`

## 2. Dataset
- Split counts after image validation: `{'train': 34998, 'val': 7507, 'test': 7495}`
- Image load failures: `0`

## 3. Balanced Batch Sampling
- balanced_batches: `True`
- batch_size: `30`
- train dataset distribution before balancing: `{'flive': 22400, 'spaq': 6299, 'koniq10k': 6299}`
- per-dataset samples per balanced batch: `10`

## 4. Model
- Backbone: `efficientnetv2b0`
- Input size: `256x256x3`
- Head: `GAP -> Dense(256, relu) -> Dropout(0.3) -> Dense(64, relu) -> Dropout(0.2) -> Dense(1, sigmoid)`
- Include preprocessing in Keras application: `True`
- Preprocessing behavior: Input generator emits RGB float32 0..255. Keras application preprocessing is included when supported.

## 5. Training Configuration
- Stage A head epochs: `5`
- Stage B fine-tune epochs: `20`
- Stage B trainable backbone layers: `32`
- Stage B trainable BatchNorm layers: `0`
- Training history rows: `25`

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
- checkpoint metric rows: `25`
- severe_fp_penalty: `10.0`
- mae_penalty: `0.01`

## 8. Training History Summary
- best validation composite: `0.5573196366869587`
- best validation SRCC: `0.6239599666490568`
- best validation severe FP count: `4`

## 9. Overall Metrics Against Guarded Target
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 34998 | 34998 | 4.165768 | 5.822549 | 0.716667 | 0.905468 | 16.016580 | 83.579361 | 64.447161 | 11.982656 |
| val | overall | 7507 | 7507 | 5.465152 | 7.526380 | 0.623960 | 0.815924 | 15.270592 | 81.773847 | 64.159320 | 11.547219 |
| test | overall | 7495 | 7495 | 5.410899 | 7.489498 | 0.630522 | 0.821060 | 19.668272 | 81.836408 | 63.980348 | 11.629926 |

## 10. Dataset-wise Metrics Against Guarded Target
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | flive | 4799 | 4799 | 4.762332 | 6.623077 | 0.449674 | 0.502397 | 19.788915 | 81.836408 | 68.048798 | 6.705120 |
| test | koniq10k | 1348 | 1348 | 5.574837 | 7.519884 | 0.639833 | 0.695841 | 29.749677 | 80.566138 | 64.118216 | 8.701517 |
| test | spaq | 1348 | 1348 | 7.555918 | 9.958275 | 0.832751 | 0.843422 | 19.668272 | 79.273367 | 49.358435 | 15.604829 |

## 11. MOS Reference Metrics
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | overall | 7495 | 7495 | 6.905811 | 9.255360 | 0.560922 | 0.763633 | 19.668272 | 81.836408 | 63.980348 | 11.629926 |
| test | flive | 4799 | 4799 | 6.268655 | 8.349163 | 0.352967 | 0.370074 | 19.788915 | 81.836408 | 68.048798 | 6.705120 |
| test | koniq10k | 1348 | 1348 | 6.112580 | 8.166238 | 0.625700 | 0.692199 | 29.749677 | 80.566138 | 64.118216 | 8.701517 |
| test | spaq | 1348 | 1348 | 9.967376 | 12.705535 | 0.793781 | 0.801979 | 19.668272 | 79.273367 | 49.358435 | 15.604829 |

## 12. Safety Metrics
| split | dataset | n | finite_n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- |
| val | overall | 7507 | 7507 | 68 | 9 | 4 |
| val | flive | 4801 | 4801 | 28 | 2 | 3 |
| val | koniq10k | 1353 | 1353 | 22 | 4 | 0 |
| val | spaq | 1353 | 1353 | 18 | 3 | 1 |
| test | overall | 7495 | 7495 | 71 | 13 | 14 |
| test | flive | 4799 | 4799 | 26 | 1 | 7 |
| test | koniq10k | 1348 | 1348 | 28 | 4 | 0 |
| test | spaq | 1348 | 1348 | 17 | 8 | 7 |

## 13. Severe False Positive Cases
- Severe/standard FP case rows saved: `139`
- CSV: `outputs/mobile_student_iqa_v4a_20260601/severe_false_positive_cases.csv`

## 14. v3 Comparison
| metric | v3 | v4a | improvement |
| --- | --- | --- | --- |
| test_srcc | 0.658204 | 0.630522 | -0.027682 |
| test_mae | 4.543416 | 5.410899 | -0.867483 |
| test_severe_fp | 28.000000 | 13.000000 | 15.000000 |
| flive_test_srcc | 0.514385 | 0.449674 | -0.064711 |
| koniq10k_test_srcc | 0.643786 | 0.639833 | -0.003953 |
| spaq_test_srcc | 0.840971 | 0.832751 | -0.008220 |

## 15. Risks
- v4a does not add ranking loss yet; ranking can still lag even if safety improves.
- Balanced sampling changes the effective training distribution and should be judged against unbiased val/test metrics.
- The guarded target is a teacher-label design target, not a Flutter runtime formula.

## 16. Final Decision
- `NEEDS_MORE_TUNING`
- Test SRCC 0.6305 is below 0.85.
- Test severe FP count is 13, expected 0.
- At least one test dataset SRCC is below 0.50.
