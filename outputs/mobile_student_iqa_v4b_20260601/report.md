# Mobile Student IQA v4b Training Report

## 1. Summary
- Final decision: `NEEDS_MORE_TUNING`
- Input manifest: `outputs/student_training_manifest_50k_20260601/student_training_manifest.csv`
- Output directory: `outputs/mobile_student_iqa_v4b_20260601`

## 2. Dataset
- Split counts after image validation: `{'train': 34998, 'val': 7507, 'test': 7495}`
- Image load failures: `0`

## 3. Sampling and Weights
- dataset_sampling_mode: `balanced`
- balanced_batches_v4a_logic: `True`
- batch_size: `30`
- train dataset distribution before balancing: `{'flive': 22400, 'spaq': 6299, 'koniq10k': 6299}`
- sample weights enabled: `True`
- low_mos_weight: `2.0`
- severe_low_mos_weight: `3.0`

## 4. Model
- Backbone: `efficientnetv2b0`
- Input size: `256x256x3`
- Head: `GAP -> Dense(256, relu) -> Dropout(0.3) -> Dense(64, relu) -> Dropout(0.2) -> Dense(1, sigmoid)`
- Include preprocessing in Keras application: `True`

## 5. Training Configuration
- Stage A head epochs: `5`
- Stage B fine-tune epochs: `20`
- Stage B trainable backbone layers: `32`
- Training history rows: `25`

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
- best validation composite: `0.4063713496862521`
- best validation SRCC: `0.6514222793917089`
- best validation severe FP count: `5`

## 10. Overall Metrics Against Guarded Target
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 34998 | 34998 | 3.882830 | 4.986075 | 0.749548 | 0.930246 | 9.708494 | 85.233551 | 64.858772 | 11.463851 |
| val | overall | 7507 | 7507 | 5.168786 | 6.982225 | 0.647633 | 0.832710 | 11.492904 | 82.354385 | 64.801377 | 10.870468 |
| test | overall | 7495 | 7495 | 5.108165 | 6.895487 | 0.647898 | 0.839575 | 12.922792 | 81.727993 | 64.606682 | 10.980872 |

## 11. Dataset-wise Metrics Against Guarded Target
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | flive | 4799 | 4799 | 4.373789 | 5.790304 | 0.469730 | 0.504979 | 19.255899 | 81.727993 | 68.349532 | 5.592332 |
| test | koniq10k | 1348 | 1348 | 5.503073 | 7.407689 | 0.654136 | 0.698404 | 27.236053 | 81.306326 | 64.727716 | 7.373235 |
| test | spaq | 1348 | 1348 | 7.327700 | 9.493920 | 0.853897 | 0.858860 | 12.922792 | 78.767818 | 51.160771 | 16.554439 |

## 12. MOS Reference Metrics
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | overall | 7495 | 7495 | 6.601223 | 8.749955 | 0.583011 | 0.784034 | 12.922792 | 81.727993 | 64.606682 | 10.980872 |
| test | flive | 4799 | 4799 | 5.876487 | 7.691317 | 0.381412 | 0.384826 | 19.255899 | 81.727993 | 68.349532 | 5.592332 |
| test | koniq10k | 1348 | 1348 | 6.130709 | 8.178394 | 0.640854 | 0.695383 | 27.236053 | 81.306326 | 64.727716 | 7.373235 |
| test | spaq | 1348 | 1348 | 9.651859 | 12.173821 | 0.812013 | 0.816363 | 12.922792 | 78.767818 | 51.160771 | 16.554439 |

## 13. Safety Metrics
| split | dataset | n | finite_n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- |
| val | overall | 7507 | 7507 | 72 | 8 | 1 |
| test | overall | 7495 | 7495 | 75 | 12 | 7 |
| val | flive | 4801 | 4801 | 30 | 1 | 1 |
| val | koniq10k | 1353 | 1353 | 20 | 3 | 0 |
| val | spaq | 1353 | 1353 | 22 | 4 | 0 |
| test | flive | 4799 | 4799 | 30 | 1 | 4 |
| test | koniq10k | 1348 | 1348 | 27 | 3 | 0 |
| test | spaq | 1348 | 1348 | 18 | 8 | 3 |

## 14. Post-hoc Calibration Evaluation
- method: `isotonic`
- cal_mae: `4.643048`
- cal_srcc: `0.647332`
- cal_plcc: `0.845891`

Note: Calibration is for analysis and does not change the raw saved model.

## 15. Severe False Positive Cases
- Severe/standard FP case rows saved: `147`

## 16. v3/v4a Comparison
| metric | v3 | v4a | v4b |
| --- | --- | --- | --- |
| test_srcc | 0.658204 | 0.630522 | 0.647898 |
| test_mae | 4.543416 | 5.410899 | 5.108165 |
| test_severe_fp | 28.000000 | 13.000000 | 12.000000 |
| flive_test_srcc | 0.514385 | 0.449674 | 0.469730 |
| koniq10k_test_srcc | 0.643786 | 0.639833 | 0.654136 |
| spaq_test_srcc | 0.840971 | 0.832751 | 0.853897 |

## 17. Risks
- Ranking loss might increase MAE while improving SRCC.
- EfficientNetV2B0 at 256 might still be resolution-limited for some teacher features.

## 18. Final Decision
- `NEEDS_MORE_TUNING`
- Test SRCC 0.6479 is below 0.85.
- Test severe FP count is 12, expected 0.
- At least one test dataset SRCC is below 0.50.
