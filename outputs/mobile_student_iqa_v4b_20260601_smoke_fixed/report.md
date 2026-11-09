# Mobile Student IQA v4b Training Report

## 1. Summary
- Final decision: `NEEDS_MORE_TUNING`
- Input manifest: `outputs/student_training_manifest_50k_20260601/student_training_manifest.csv`
- Output directory: `outputs/mobile_student_iqa_v4b_20260601_smoke_fixed`

## 2. Dataset
- Split counts after image validation: `{'train': 900, 'val': 450, 'test': 450}`
- Image load failures: `0`

## 3. Sampling and Weights
- dataset_sampling_mode: `balanced`
- balanced_batches_v4a_logic: `True`
- batch_size: `30`
- train dataset distribution before balancing: `{'flive': 567, 'spaq': 179, 'koniq10k': 154}`
- sample weights enabled: `True`
- low_mos_weight: `2.0`
- severe_low_mos_weight: `3.0`

## 4. Model
- Backbone: `efficientnetv2b0`
- Input size: `256x256x3`
- Head: `GAP -> Dense(256, relu) -> Dropout(0.3) -> Dense(64, relu) -> Dropout(0.2) -> Dense(1, sigmoid)`
- Include preprocessing in Keras application: `True`

## 5. Training Configuration
- Stage A head epochs: `1`
- Stage B fine-tune epochs: `1`
- Stage B trainable backbone layers: `32`
- Training history rows: `2`

## 6. Loss Components
- base loss: `huber`
- huber_delta: `0.05`
- safety_lambda: `0.75`
- safety_threshold: `0.65` (prediction cutoff)
- target_safety_cutoff: `55.0` (guarded target cutoff)
- mos_safety_cutoff: `50.0` (MOS cutoff)

## 7. Ranking Loss
- ranking_lambda: `0.25`
- rank_margin_target: `0.1`
- rank_margin_pred: `0.02`
- max_rank_pairs_per_batch: `256`

## 8. Checkpoint Selection
- Composite checkpointing used: `True`
- mae_penalty: `0.01`
- severe_fp_count_penalty: `0.02`
- dataset_collapse_penalty: `0.5`

## 9. Training History Summary
- best validation composite: `0.04593288593945166`
- best validation SRCC: `0.32969861579563364`
- best validation severe FP count: `0`

## 10. Overall Metrics Against Guarded Target
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 900 | 900 | 10.010480 | 12.074055 | 0.494157 | 0.630148 | 30.227128 | 83.002925 | 59.699635 | 9.070564 |
| val | overall | 450 | 450 | 10.329467 | 12.436479 | 0.329699 | 0.436390 | 34.055379 | 83.466023 | 60.349292 | 7.938990 |
| test | overall | 450 | 450 | 11.129103 | 13.259598 | 0.264642 | 0.397601 | 32.222414 | 79.443717 | 59.332185 | 7.862944 |

## 11. Dataset-wise Metrics Against Guarded Target
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | flive | 277 | 277 | 10.844099 | 12.896747 | 0.018472 | 0.057956 | 32.222414 | 79.443717 | 61.186757 | 7.282606 |
| test | koniq10k | 85 | 85 | 9.741013 | 11.538549 | 0.213154 | 0.285059 | 38.039991 | 78.670079 | 58.742513 | 7.664823 |
| test | spaq | 88 | 88 | 13.366987 | 15.713542 | 0.503879 | 0.523419 | 38.322052 | 73.697966 | 54.064069 | 7.317730 |

## 12. MOS Reference Metrics
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | overall | 450 | 450 | 12.073603 | 14.453996 | 0.240774 | 0.364478 | 32.222414 | 79.443717 | 59.332185 | 7.862944 |
| test | flive | 277 | 277 | 11.715725 | 13.970460 | 0.022485 | 0.068747 | 32.222414 | 79.443717 | 61.186757 | 7.282606 |
| test | koniq10k | 85 | 85 | 10.193681 | 12.111303 | 0.245197 | 0.310023 | 38.039991 | 78.670079 | 58.742513 | 7.664823 |
| test | spaq | 88 | 88 | 15.015941 | 17.671837 | 0.466606 | 0.489886 | 38.322052 | 73.697966 | 54.064069 | 7.317730 |

## 13. Safety Metrics
| split | dataset | n | finite_n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- |
| val | overall | 450 | 450 | 3 | 0 | 0 |
| test | overall | 450 | 450 | 1 | 0 | 1 |
| val | flive | 301 | 301 | 0 | 0 | 0 |
| val | koniq10k | 80 | 80 | 2 | 0 | 0 |
| val | spaq | 69 | 69 | 1 | 0 | 0 |
| test | flive | 277 | 277 | 0 | 0 | 1 |
| test | koniq10k | 85 | 85 | 1 | 0 | 0 |
| test | spaq | 88 | 88 | 0 | 0 | 0 |

## 14. Post-hoc Calibration Evaluation
- method: `isotonic`
- cal_mae: `7.824967`
- cal_srcc: `0.265444`
- cal_plcc: `0.443498`

Note: Calibration is for analysis and does not change the raw saved model.

## 15. Severe False Positive Cases
- Severe/standard FP case rows saved: `4`

## 16. v3/v4a Comparison
| metric | v3 | v4a | v4b |
| --- | --- | --- | --- |
| test_srcc | 0.658204 | 0.630522 | 0.264642 |
| test_mae | 4.543416 | 5.410899 | 11.129103 |
| test_severe_fp | 28.000000 | 13.000000 | 0.000000 |
| flive_test_srcc | 0.514385 | 0.449674 | 0.018472 |
| koniq10k_test_srcc | 0.643786 | 0.639833 | 0.213154 |
| spaq_test_srcc | 0.840971 | 0.832751 | 0.503879 |

## 17. Risks
- Ranking loss might increase MAE while improving SRCC.
- EfficientNetV2B0 at 256 might still be resolution-limited for some teacher features.

## 18. Final Decision
- `NEEDS_MORE_TUNING`
- Test SRCC 0.2646 is below 0.85.
- Test MAE 11.1291 is above 10.
- At least one test dataset SRCC is below 0.50.
