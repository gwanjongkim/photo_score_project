# Mobile Student IQA 14105 Longer Controlled Training Report

## 1. Executive Summary
- Final decision: `STUDENT_14105_LONGER_READY_FOR_FULL_CONTROLLED_TRAINING`.
- Run type: longer controlled training only, not final full distillation.
- Runtime seconds: `1446.73`.
- Train/val/test samples used: `{'train': 11283, 'val': 1409, 'test': 1413}`.
- Final train loss: `0.000865`.
- Final val loss: `0.001571`.
- Best val loss: `0.001557`.

## 2. Sources Inspected
- `outputs/student_distillation_manifest_14105_20260603/student_distillation_manifest.csv`
- `/home/omen_pc1/photo_score_project/outputs/mobile_student_iqa_14105_short_train_20260604/report.md`
- `/home/omen_pc1/photo_score_project/outputs/mobile_student_iqa_14105_short_train_20260604/history.csv`
- `/home/omen_pc1/photo_score_project/outputs/mobile_student_iqa_14105_short_train_20260604/metrics.csv`
- `/home/omen_pc1/photo_score_project/outputs/student_distillation_manifest_14105_20260603/report.md`
- `/home/omen_pc1/photo_score_project/outputs/known_dataset_full_label_coverage_audit_20260603/report.md`
- `/home/omen_pc1/photo_score_project/outputs/known_dataset_full_label_dryrun_20260603/report.md`
- Reused smoke trainer functions from `scripts/technical_iqa/train_mobile_student_iqa_14105_smoke.py`.

## 3. Manifest and Scope
- Scope: validated known-dataset teacher-cache subset, not MDIQA official full-dataset relabeling.
- Unknown/fallback excluded; c6_parity holdout only; Test_weight/product-gold validation only.
- Target: `target_norm`; metrics use 0..100 scale.
- Image load failure rows: `0`.

## 4. Environment / GPU Preflight
- TensorFlow: `2.21.0`.
- Built with CUDA: `True`.
- GPUs visible: `["PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')"]`.
- GPU count: `1`.

## 5. Training Configuration
- Backbone: `efficientnetv2b0`.
- Image size: `256`.
- Batch size: `16`.
- Head epochs: `5`.
- Finetune epochs: `10`.
- Loss: `huber`.
- ImageNet weights requested: `imagenet`.
- ImageNet weights loaded: `True`.
- Include preprocessing: `True`.
- Requested unfreeze_last_n: `24`.
- Finetune trainable backbone layers: `19`.

## 6. Training History
| phase | phase_epoch | global_epoch | loss | mae | rmse | val_loss | val_mae | val_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| head | 1 | 1 | 0.001999 | 0.059971 | 0.085278 | 0.002410 | 0.068540 | 0.091278 |
| head | 2 | 2 | 0.001317 | 0.044903 | 0.062959 | 0.001948 | 0.058362 | 0.078758 |
| head | 3 | 3 | 0.001144 | 0.040918 | 0.057567 | 0.001917 | 0.057720 | 0.077152 |
| head | 4 | 4 | 0.001027 | 0.038088 | 0.053430 | 0.001902 | 0.057209 | 0.077356 |
| head | 5 | 5 | 0.000957 | 0.036391 | 0.051218 | 0.001724 | 0.053305 | 0.071793 |
| finetune | 1 | 6 | 0.001098 | 0.039761 | 0.054238 | 0.001800 | 0.055340 | 0.073441 |
| finetune | 2 | 7 | 0.001047 | 0.038594 | 0.052501 | 0.001730 | 0.053903 | 0.071461 |
| finetune | 3 | 8 | 0.000999 | 0.037451 | 0.051399 | 0.001724 | 0.053854 | 0.071195 |
| finetune | 4 | 9 | 0.000968 | 0.036691 | 0.050531 | 0.001693 | 0.053212 | 0.070347 |
| finetune | 5 | 10 | 0.000961 | 0.036597 | 0.049765 | 0.001618 | 0.051539 | 0.068302 |
| finetune | 6 | 11 | 0.000930 | 0.035786 | 0.048992 | 0.001623 | 0.051711 | 0.068346 |
| finetune | 7 | 12 | 0.000911 | 0.035372 | 0.048432 | 0.001604 | 0.051316 | 0.067823 |
| finetune | 8 | 13 | 0.000869 | 0.034322 | 0.047089 | 0.001557 | 0.050205 | 0.066681 |
| finetune | 9 | 14 | 0.000865 | 0.034268 | 0.047098 | 0.001612 | 0.051462 | 0.068048 |
| finetune | 10 | 15 | 0.000865 | 0.034255 | 0.046833 | 0.001571 | 0.050504 | 0.067063 |

## 7. Best Checkpoint Summary
| monitor | best_epoch | best_value | checkpoint |
| --- | --- | --- | --- |
| val_loss | 13 | 0.001557 | not_saved_separately |
| val_mae | 13 | 0.050205 | outputs/mobile_student_iqa_14105_longer_controlled_20260604/best_by_val_mae_model.keras |
- `best_by_val_srcc_model.keras` not implemented; SRCC is computed after final prediction pass to keep training bounded.

## 8. Prediction and Metric Summary
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 11283 | 11283 | 4.238775 | 5.606422 | 0.822106 | 0.897516 | 10.785113 | 75.568849 | 61.067376 | 7.885754 | 14.804595 | 88.805878 | 64.516098 | 9.806214 | 0 |
| val | overall | 1409 | 1409 | 5.050375 | 6.706289 | 0.687595 | 0.815275 | 13.569139 | 71.110111 | 61.139733 | 7.636955 | 18.707634 | 84.982681 | 64.481513 | 10.001077 | 0 |
| test | overall | 1413 | 1413 | 5.058221 | 6.727101 | 0.674326 | 0.828291 | 12.347898 | 74.520892 | 60.829291 | 8.223310 | 17.632763 | 81.683090 | 64.163819 | 10.403744 | 0 |

## 9. Dataset-wise Metrics
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | flive | 398 | 398 | 9.548560 | 10.198205 | 0.521285 | 0.592774 | 41.458884 | 71.110111 | 63.944644 | 4.204537 |
| val | koniq10k | 899 | 899 | 2.748414 | 3.493802 | 0.752804 | 0.761912 | 26.528758 | 70.570076 | 61.569679 | 4.481842 |
| val | spaq | 112 | 112 | 7.543100 | 9.911071 | 0.860295 | 0.856887 | 13.569139 | 69.943643 | 47.721190 | 17.325488 |
| test | flive | 399 | 399 | 9.427264 | 10.143023 | 0.546987 | 0.524462 | 41.844895 | 70.730948 | 63.903044 | 3.934473 |
| test | koniq10k | 901 | 901 | 2.804017 | 3.618780 | 0.722305 | 0.748814 | 33.607247 | 71.281177 | 61.534031 | 4.646891 |
| test | spaq | 113 | 113 | 7.605027 | 9.908941 | 0.881712 | 0.854926 | 12.347898 | 74.520892 | 44.356737 | 17.609194 |

## 10. Comparison Against Short Controlled Training
| split | dataset | metric | short | longer | delta | improved |
| --- | --- | --- | --- | --- | --- | --- |
| val | overall | mae_0_100 | 5.099313 | 5.050375 | -0.048938 | True |
| val | overall | srcc | 0.643581 | 0.687595 | 0.044014 | True |
| val | overall | plcc | 0.730197 | 0.815275 | 0.085078 | True |
| val | overall | severe_fp_count | 1.000000 | 1.000000 | 0.000000 | False |
| val | overall | pred_std | 6.711669 | 7.636955 | 0.925286 | True |
| val | flive | mae_0_100 | 7.135755 | 9.548560 | 2.412805 | False |
| val | flive | srcc | 0.324147 | 0.521285 | 0.197138 | True |
| val | flive | plcc | 0.444457 | 0.592774 | 0.148317 | True |
| val | flive | severe_fp_count | 0.000000 | 0.000000 | 0.000000 | False |
| val | flive | pred_std | 4.706282 | 4.204537 | -0.501745 | False |
| val | koniq10k | mae_0_100 | 3.272865 | 2.748414 | -0.524451 | True |
| val | koniq10k | srcc | 0.598899 | 0.752804 | 0.153905 | True |
| val | koniq10k | plcc | 0.640944 | 0.761912 | 0.120969 | True |
| val | koniq10k | severe_fp_count | 0.000000 | 0.000000 | 0.000000 | False |
| val | koniq10k | pred_std | 4.811435 | 4.481842 | -0.329593 | False |
| val | spaq | mae_0_100 | 12.603411 | 7.543100 | -5.060311 | True |
| val | spaq | srcc | 0.775292 | 0.860295 | 0.085003 | True |
| val | spaq | plcc | 0.712797 | 0.856887 | 0.144090 | True |
| val | spaq | severe_fp_count | 1.000000 | 1.000000 | 0.000000 | False |
| val | spaq | pred_std | 13.084322 | 17.325488 | 4.241166 | True |
| test | overall | mae_0_100 | 5.008367 | 5.058221 | 0.049854 | False |
| test | overall | srcc | 0.610541 | 0.674326 | 0.063785 | True |
| test | overall | plcc | 0.751056 | 0.828291 | 0.077235 | True |
| test | overall | severe_fp_count | 1.000000 | 0.000000 | -1.000000 | True |
| test | overall | pred_std | 7.382373 | 8.223310 | 0.840937 | True |
| test | flive | mae_0_100 | 7.030894 | 9.427264 | 2.396370 | False |
| test | flive | srcc | 0.287497 | 0.546987 | 0.259489 | True |
| test | flive | plcc | 0.329384 | 0.524462 | 0.195078 | True |
| test | flive | severe_fp_count | 0.000000 | 0.000000 | 0.000000 | False |
| test | flive | pred_std | 4.635648 | 3.934473 | -0.701175 | False |
| test | koniq10k | mae_0_100 | 3.300183 | 2.804017 | -0.496165 | True |
| test | koniq10k | srcc | 0.564519 | 0.722305 | 0.157786 | True |
| test | koniq10k | plcc | 0.604393 | 0.748814 | 0.144420 | True |
| test | koniq10k | severe_fp_count | 0.000000 | 0.000000 | 0.000000 | False |
| test | koniq10k | pred_std | 4.655618 | 4.646891 | -0.008727 | False |
| test | spaq | mae_0_100 | 11.605539 | 7.605027 | -4.000511 | True |
| test | spaq | srcc | 0.782181 | 0.881712 | 0.099531 | True |
| test | spaq | plcc | 0.736443 | 0.854926 | 0.118482 | True |
| test | spaq | severe_fp_count | 1.000000 | 0.000000 | -1.000000 | True |
| test | spaq | pred_std | 15.334057 | 17.609194 | 2.275137 | True |

## 11. FLIVE Weakness Analysis
- val FLIVE: SRCC `0.5213` (`+0.1971` vs short), MAE `9.5486`.
- test FLIVE: SRCC `0.5470` (`+0.2595` vs short), MAE `9.4273`.

## 12. Failure Cases
- Failure CSV: `outputs/mobile_student_iqa_14105_longer_controlled_20260604/failure_cases.csv`.
- Failure rows: `0`.

## 13. Collapse / Safety Checks
- Collapse check: `{'pred_std_min_overall': 7.636955216123753, 'pred_min_overall': 10.785112529993057, 'pred_max_overall': 75.5688488483429, 'nan_inf_count': 0, 'pred_std_too_low': False, 'pred_outside_0_100': False}`.

| split | dataset | n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- |
| train | overall | 11283 | 10 | 0 | 0 |
| train | flive | 3184 | 1 | 0 | 0 |
| train | koniq10k | 7199 | 4 | 0 | 0 |
| train | spaq | 900 | 5 | 0 | 0 |
| val | overall | 1409 | 5 | 1 | 0 |
| val | flive | 398 | 0 | 0 | 0 |
| val | koniq10k | 899 | 2 | 0 | 0 |
| val | spaq | 112 | 3 | 1 | 0 |
| test | overall | 1413 | 1 | 0 | 0 |
| test | flive | 399 | 0 | 0 | 0 |
| test | koniq10k | 901 | 0 | 0 | 0 |
| test | spaq | 113 | 1 | 0 | 0 |

## 14. Final Decision
STUDENT_14105_LONGER_READY_FOR_FULL_CONTROLLED_TRAINING

## 15. Recommended Next Step
- Prepare a full controlled training plan with explicit stop criteria; do not run final full distillation yet.
