# Mobile Student IQA v4b Training Report

## 1. Summary
- Final decision: `NEEDS_MORE_TUNING`
- Input manifest: `outputs/student_training_manifest_50k_20260601/student_training_manifest.csv`
- Output directory: `outputs/mobile_student_iqa_v4b_20260601_sanity`

## 2. Dataset
- Split counts after image validation: `{'train': 3000, 'val': 1000, 'test': 1000}`
- Image load failures: `0`

## 3. Sampling and Weights
- dataset_sampling_mode: `balanced`
- balanced_batches_v4a_logic: `True`
- batch_size: `30`
- train dataset distribution before balancing: `{'flive': 1899, 'spaq': 555, 'koniq10k': 546}`
- sample weights enabled: `True`
- low_mos_weight: `2.0`
- severe_low_mos_weight: `3.0`

## 4. Model
- Backbone: `efficientnetv2b0`
- Input size: `256x256x3`
- Head: `GAP -> Dense(256, relu) -> Dropout(0.3) -> Dense(64, relu) -> Dropout(0.2) -> Dense(1, sigmoid)`
- Include preprocessing in Keras application: `True`

## 5. Training Configuration
- Stage A head epochs: `2`
- Stage B fine-tune epochs: `3`
- Stage B trainable backbone layers: `32`
- Training history rows: `5`

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
- best validation composite: `0.29944299407422204`
- best validation SRCC: `0.49281676873056995`
- best validation severe FP count: `0`

## 10. Overall Metrics Against Guarded Target
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 3000 | 3000 | 6.876588 | 8.684714 | 0.633675 | 0.782146 | 19.542016 | 88.480866 | 62.833131 | 10.421037 |
| val | overall | 1000 | 1000 | 7.988596 | 10.110832 | 0.492817 | 0.639087 | 26.612124 | 84.946448 | 62.667435 | 9.563159 |
| test | overall | 1000 | 1000 | 8.148915 | 10.249936 | 0.442709 | 0.627662 | 19.058649 | 82.981831 | 62.398891 | 9.516505 |

## 11. Dataset-wise Metrics Against Guarded Target
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | flive | 627 | 627 | 7.674987 | 9.531395 | 0.233097 | 0.296867 | 40.464613 | 82.981831 | 65.044186 | 7.356508 |
| test | koniq10k | 189 | 189 | 7.035509 | 9.054483 | 0.521618 | 0.562935 | 38.083276 | 79.859287 | 63.171780 | 8.175217 |
| test | spaq | 184 | 184 | 10.907538 | 13.311671 | 0.656377 | 0.675816 | 19.058649 | 74.051046 | 52.590872 | 10.884626 |

## 12. MOS Reference Metrics
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | overall | 1000 | 1000 | 9.241453 | 11.574654 | 0.415465 | 0.595962 | 19.058649 | 82.981831 | 62.398891 | 9.516505 |
| test | flive | 627 | 627 | 8.745178 | 10.828363 | 0.228186 | 0.274843 | 40.464613 | 82.981831 | 65.044186 | 7.356508 |
| test | koniq10k | 189 | 189 | 7.564413 | 9.681761 | 0.537066 | 0.580111 | 38.083276 | 79.859287 | 63.171780 | 8.175217 |
| test | spaq | 184 | 184 | 12.655176 | 15.240554 | 0.635208 | 0.667641 | 19.058649 | 74.051046 | 52.590872 | 10.884626 |

## 13. Safety Metrics
| split | dataset | n | finite_n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- |
| val | overall | 1000 | 1000 | 12 | 1 | 0 |
| test | overall | 1000 | 1000 | 12 | 1 | 0 |
| val | flive | 640 | 640 | 4 | 1 | 0 |
| val | koniq10k | 177 | 177 | 6 | 0 | 0 |
| val | spaq | 183 | 183 | 2 | 0 | 0 |
| test | flive | 627 | 627 | 3 | 0 | 0 |
| test | koniq10k | 189 | 189 | 8 | 1 | 0 |
| test | spaq | 184 | 184 | 1 | 0 | 0 |

## 14. Post-hoc Calibration Evaluation
- method: `isotonic`
- cal_mae: `6.466467`
- cal_srcc: `0.443417`
- cal_plcc: `0.686974`

Note: Calibration is for analysis and does not change the raw saved model.

## 15. Severe False Positive Cases
- Severe/standard FP case rows saved: `24`

## 16. v3/v4a Comparison
| metric | v3 | v4a | v4b |
| --- | --- | --- | --- |
| test_srcc | 0.658204 | 0.630522 | 0.442709 |
| test_mae | 4.543416 | 5.410899 | 8.148915 |
| test_severe_fp | 28.000000 | 13.000000 | 1.000000 |
| flive_test_srcc | 0.514385 | 0.449674 | 0.233097 |
| koniq10k_test_srcc | 0.643786 | 0.639833 | 0.521618 |
| spaq_test_srcc | 0.840971 | 0.832751 | 0.656377 |

## 17. Risks
- Ranking loss might increase MAE while improving SRCC.
- EfficientNetV2B0 at 256 might still be resolution-limited for some teacher features.

## 18. Final Decision
- `NEEDS_MORE_TUNING`
- Test SRCC 0.4427 is below 0.85.
- Test severe FP count is 1, expected 0.
- At least one test dataset SRCC is below 0.50.
