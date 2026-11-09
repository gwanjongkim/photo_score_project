# Mobile Student IQA 14105 Short Controlled Training Report

## 1. Executive Summary
- Final decision: `STUDENT_14105_SHORT_READY_FOR_LONGER_TRAINING`.
- Run type: short controlled training only, not full distillation.
- Runtime seconds: `291.76`.
- Train/val/test samples used: `{'train': 4000, 'val': 1000, 'test': 1000}`.
- Final train loss: `0.001394`.
- Final val loss: `0.001579`.

## 2. Sources Inspected
- `outputs/student_distillation_manifest_14105_20260603/student_distillation_manifest.csv`
- `/home/omen_pc1/photo_score_project/outputs/mobile_student_iqa_14105_smoke_gpu_imagenet_20260604/report.md`
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
- Head epochs: `3`.
- Finetune epochs: `3`.
- Loss: `huber`.
- ImageNet weights requested: `imagenet`.
- ImageNet weights loaded: `True`.
- Include preprocessing: `True`.
- Finetune trainable backbone layers: `13`.

## 6. Training History
| phase | phase_epoch | global_epoch | loss | mae | rmse | val_loss | val_mae | val_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| head | 1 | 1 | 0.002912 | 0.079484 | 0.109828 | 0.001983 | 0.059245 | 0.087853 |
| head | 2 | 2 | 0.002012 | 0.060408 | 0.084505 | 0.001810 | 0.055793 | 0.079074 |
| head | 3 | 3 | 0.001743 | 0.054683 | 0.074517 | 0.001662 | 0.052749 | 0.073857 |
| finetune | 1 | 4 | 0.001482 | 0.048881 | 0.067777 | 0.001625 | 0.051950 | 0.072809 |
| finetune | 2 | 5 | 0.001441 | 0.047931 | 0.066148 | 0.001628 | 0.052124 | 0.072381 |
| finetune | 3 | 6 | 0.001394 | 0.046540 | 0.064730 | 0.001579 | 0.050993 | 0.071440 |

## 7. Prediction and Metric Summary
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 4000 | 4000 | 3.701646 | 5.225171 | 0.830483 | 0.865057 | 11.128011 | 81.317633 | 63.260608 | 7.424553 | 14.804595 | 86.531998 | 64.576533 | 9.842804 | 0 |
| val | overall | 1000 | 1000 | 5.099313 | 7.144013 | 0.643581 | 0.730197 | 17.300208 | 76.760340 | 63.292752 | 6.711669 | 18.707634 | 84.982681 | 64.412599 | 10.263173 | 0 |
| test | overall | 1000 | 1000 | 5.008367 | 6.952443 | 0.610541 | 0.751056 | 14.900930 | 78.380632 | 63.065234 | 7.382373 | 17.632763 | 81.683090 | 64.241839 | 10.360682 | 0 |

## 8. Dataset-wise Metrics
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | flive | 282 | 282 | 7.135755 | 8.379048 | 0.324147 | 0.444457 | 50.194931 | 76.760340 | 66.737462 | 4.706282 |
| val | koniq10k | 639 | 639 | 3.272865 | 4.129418 | 0.598899 | 0.640944 | 44.884962 | 75.750828 | 62.941090 | 4.811435 |
| val | spaq | 79 | 79 | 12.603411 | 16.046550 | 0.775292 | 0.712797 | 17.300208 | 70.400077 | 53.840896 | 13.084322 |
| test | flive | 282 | 282 | 7.030894 | 8.272840 | 0.287497 | 0.329384 | 50.317711 | 78.380632 | 66.629347 | 4.635648 |
| test | koniq10k | 639 | 639 | 3.300183 | 4.183110 | 0.564519 | 0.604393 | 40.774673 | 75.072640 | 63.043189 | 4.655618 |
| test | spaq | 79 | 79 | 11.605539 | 15.033686 | 0.782181 | 0.736443 | 14.900930 | 74.332750 | 50.521023 | 15.334057 |

## 9. Comparison Against Smoke
| split | metric | smoke | short | delta | improved |
| --- | --- | --- | --- | --- | --- |
| train | mae_0_100 | 6.543346 | 3.701646 | -2.841701 | True |
| train | srcc | 0.359402 | 0.830483 | 0.471081 | True |
| train | plcc | 0.355718 | 0.865057 | 0.509339 | True |
| train | severe_fp_count | 2.000000 | 0.000000 | -2.000000 | True |
| train | pred_std | 5.449367 | 7.424553 | 1.975187 | True |
| val | mae_0_100 | 7.459097 | 5.099313 | -2.359784 | True |
| val | srcc | 0.140179 | 0.643581 | 0.503402 | True |
| val | plcc | 0.136771 | 0.730197 | 0.593426 | True |
| val | severe_fp_count | 6.000000 | 1.000000 | -5.000000 | True |
| val | pred_std | 5.625630 | 6.711669 | 1.086040 | True |
| test | mae_0_100 | 8.244832 | 5.008367 | -3.236466 | True |
| test | srcc | 0.188285 | 0.610541 | 0.422256 | True |
| test | plcc | 0.156409 | 0.751056 | 0.594647 | True |
| test | severe_fp_count | 2.000000 | 1.000000 | -1.000000 | True |
| test | pred_std | 5.961262 | 7.382373 | 1.421111 | True |

## 10. Failure Cases
- Failure CSV: `outputs/mobile_student_iqa_14105_short_train_20260604/failure_cases.csv`.
- Failure rows: `0`.

## 11. Collapse / Safety Checks
- Collapse check: `{'pred_std_min_overall': 6.711669312506952, 'pred_min_overall': 11.128010600805283, 'pred_max_overall': 81.31763339042664, 'nan_inf_count': 0, 'pred_std_too_low': False, 'pred_outside_0_100': False}`.

| split | dataset | n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- |
| train | overall | 4000 | 12 | 0 | 0 |
| train | flive | 1160 | 1 | 0 | 0 |
| train | koniq10k | 2531 | 7 | 0 | 0 |
| train | spaq | 309 | 4 | 0 | 0 |
| val | overall | 1000 | 10 | 1 | 0 |
| val | flive | 282 | 1 | 0 | 0 |
| val | koniq10k | 639 | 4 | 0 | 0 |
| val | spaq | 79 | 5 | 1 | 0 |
| test | overall | 1000 | 5 | 1 | 0 |
| test | flive | 282 | 0 | 0 | 0 |
| test | koniq10k | 639 | 2 | 0 | 0 |
| test | spaq | 79 | 3 | 1 | 0 |

## 12. Final Decision
STUDENT_14105_SHORT_READY_FOR_LONGER_TRAINING

## 13. Recommended Next Step
- Prepare a longer controlled GPU training run; do not start full distillation yet.
