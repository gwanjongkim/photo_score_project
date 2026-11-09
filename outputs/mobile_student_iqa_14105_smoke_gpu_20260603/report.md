# Mobile Student IQA 14105 Smoke Training Report

## 1. Executive Summary
- Final decision: `STUDENT_14105_SMOKE_READY_FOR_SHORT_TRAINING`.
- Run type: bounded smoke only, not full training.
- Runtime seconds: `62.20`.
- Train/val/test samples used: `{'train': 512, 'val': 256, 'test': 256}`.
- Final train loss: `0.003771`.
- Final val loss: `0.003326`.

## 2. Sources Inspected
- `/home/omen_pc1/photo_score_project/outputs/student_distillation_manifest_14105_20260603/student_distillation_manifest.csv`
- `/home/omen_pc1/photo_score_project/outputs/student_distillation_manifest_14105_20260603/report.md`
- `/home/omen_pc1/photo_score_project/outputs/known_dataset_full_label_coverage_audit_20260603/report.md`
- `/home/omen_pc1/photo_score_project/outputs/known_dataset_full_label_dryrun_20260603/report.md`
- Existing trainers inspected: `train_mobile_student_iqa_v3.py`, `train_mobile_student_iqa_v4a.py`, `train_mobile_student_iqa_v4b.py`.

## 3. Manifest and Scope
- Scope: validated known-dataset teacher-cache subset, not MDIQA official full-dataset relabeling.
- Unknown/fallback excluded; c6_parity holdout only; Test_weight/product-gold validation only.
- Target: `target_norm`; report metrics also use 0..100 scale.
- Image load failure rows: `0`.

## 4. Environment / GPU Preflight
- TensorFlow: `2.21.0`.
- GPUs visible: `["PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')"]`.
- GPU count: `1`.
- Execution mode: `GPU`.

## 5. Smoke Training Configuration
- Backbone: `efficientnetv2b0`.
- Image size: `256`.
- Batch size: `16`.
- Head epochs: `1`.
- Finetune epochs: `1`.
- Loss: `huber`.
- ImageNet weights requested: `none`.
- ImageNet weights loaded: `False`.
- Backbone weights error: ``.
- Include preprocessing: `True`.
- Finetune trainable backbone layers: `13`.

## 6. Training History
| phase | phase_epoch | global_epoch | loss | mae | rmse | val_loss | val_mae | val_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| head | 1 | 1 | 0.005127 | 0.126303 | 0.144832 | 0.003369 | 0.089480 | 0.110129 |
| finetune | 1 | 2 | 0.003771 | 0.098044 | 0.120901 | 0.003326 | 0.088603 | 0.109219 |

## 7. Prediction and Metric Summary
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 512 | 512 | 9.401220 | 11.575769 | -0.177675 | -0.031278 | 53.737450 | 85.114491 | 61.366032 | 5.533916 | 16.697649 | 81.521065 | 65.038269 | 9.309467 | 0 |
| val | overall | 256 | 256 | 8.860291 | 10.921896 | -0.176036 | -0.061119 | 54.046708 | 89.376980 | 61.051950 | 4.781705 | 26.901156 | 84.982681 | 64.393710 | 8.945778 | 0 |
| test | overall | 256 | 256 | 9.264618 | 11.734224 | -0.015724 | 0.070206 | 54.481155 | 83.545500 | 61.529074 | 5.328574 | 17.632763 | 78.933716 | 64.194809 | 10.490031 | 0 |

## 8. Dataset-wise Metrics
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | flive | 143 | 143 | 14.394110 | 15.391595 | -0.216410 | -0.106820 | 54.172605 | 75.616992 | 59.629229 | 3.467449 |
| train | koniq10k | 331 | 331 | 6.358189 | 7.575333 | -0.064081 | 0.042990 | 53.737450 | 85.114491 | 62.333129 | 6.181732 |
| train | spaq | 38 | 38 | 17.118592 | 20.349362 | 0.137761 | 0.212575 | 54.924250 | 70.681769 | 59.477970 | 3.383867 |
| val | flive | 75 | 75 | 13.633588 | 14.475588 | 0.020541 | 0.127063 | 54.561043 | 89.376980 | 60.051754 | 4.768115 |
| val | koniq10k | 169 | 169 | 5.984738 | 7.162348 | -0.156076 | -0.119609 | 54.046708 | 78.248781 | 61.557949 | 4.676136 |
| val | spaq | 12 | 12 | 19.524560 | 22.642905 | 0.188811 | -0.062025 | 55.229855 | 72.067893 | 60.177033 | 5.207074 |
| test | flive | 75 | 75 | 12.134295 | 13.170766 | -0.064922 | -0.024100 | 54.692686 | 78.433496 | 60.856603 | 4.304793 |
| test | koniq10k | 164 | 164 | 6.384135 | 7.549083 | -0.058357 | 0.026694 | 54.481155 | 83.545500 | 62.129779 | 5.799644 |
| test | spaq | 17 | 17 | 24.392466 | 27.539099 | 0.139706 | 0.223789 | 54.775137 | 65.721685 | 58.700821 | 2.803180 |

## 9. Failure Cases
- Failure CSV: `outputs/mobile_student_iqa_14105_smoke_gpu_20260603/failure_cases.csv`.
- Failure rows: `0`.

## 10. Collapse / Safety Checks
- Collapse check: `{'pred_std_min_overall': 4.7817046345988565, 'pred_min_overall': 53.737449645996094, 'pred_max_overall': 89.37698006629944, 'nan_inf_count': 0, 'pred_std_too_low': False, 'pred_outside_0_100': False}`.

| split | dataset | n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- |
| train | overall | 512 | 12 | 3 | 0 |
| train | flive | 143 | 1 | 0 | 0 |
| train | koniq10k | 331 | 8 | 3 | 0 |
| train | spaq | 38 | 3 | 0 | 0 |
| val | overall | 256 | 7 | 4 | 0 |
| val | flive | 75 | 1 | 0 | 0 |
| val | koniq10k | 169 | 4 | 2 | 0 |
| val | spaq | 12 | 2 | 2 | 0 |
| test | overall | 256 | 7 | 2 | 0 |
| test | flive | 75 | 0 | 0 | 0 |
| test | koniq10k | 164 | 5 | 1 | 0 |
| test | spaq | 17 | 2 | 1 | 0 |

## 11. Final Decision
STUDENT_14105_SMOKE_READY_FOR_SHORT_TRAINING

## 12. Recommended Next Step
- Proceed to a short controlled GPU training run; do not start full distillation yet.
