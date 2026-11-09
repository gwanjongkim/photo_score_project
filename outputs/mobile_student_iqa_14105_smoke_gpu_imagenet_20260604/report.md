# Mobile Student IQA 14105 Smoke Training Report

## 1. Executive Summary
- Final decision: `STUDENT_14105_SMOKE_READY_FOR_SHORT_TRAINING`.
- Run type: bounded smoke only, not full training.
- Runtime seconds: `60.86`.
- Train/val/test samples used: `{'train': 512, 'val': 256, 'test': 256}`.
- Final train loss: `0.003081`.
- Final val loss: `0.002697`.

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
- ImageNet weights requested: `imagenet`.
- ImageNet weights loaded: `True`.
- Backbone weights error: ``.
- Include preprocessing: `True`.
- Finetune trainable backbone layers: `13`.

## 6. Training History
| phase | phase_epoch | global_epoch | loss | mae | rmse | val_loss | val_mae | val_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| head | 1 | 1 | 0.004980 | 0.122552 | 0.156035 | 0.002847 | 0.077854 | 0.102628 |
| finetune | 1 | 2 | 0.003081 | 0.083248 | 0.110560 | 0.002697 | 0.074591 | 0.099914 |

## 7. Prediction and Metric Summary
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 512 | 512 | 6.543346 | 9.062300 | 0.359402 | 0.355718 | 46.167612 | 81.481576 | 63.676231 | 5.449367 | 16.697649 | 81.521065 | 65.038269 | 9.309467 | 0 |
| val | overall | 256 | 256 | 7.459097 | 9.991384 | 0.140179 | 0.136771 | 44.300997 | 80.343056 | 63.008313 | 5.625630 | 26.901156 | 84.982681 | 64.393710 | 8.945778 | 0 |
| test | overall | 256 | 256 | 8.244832 | 11.296240 | 0.188285 | 0.156409 | 45.397118 | 85.576081 | 62.934134 | 5.961262 | 17.632763 | 78.933716 | 64.194809 | 10.490031 | 0 |

## 8. Dataset-wise Metrics
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | flive | 143 | 143 | 8.249262 | 10.459129 | 0.107911 | 0.131135 | 50.360149 | 81.481576 | 66.172929 | 5.631194 |
| train | koniq10k | 331 | 331 | 4.720012 | 5.955704 | 0.222996 | 0.233821 | 46.167612 | 79.680115 | 62.991852 | 5.054358 |
| train | spaq | 38 | 38 | 16.005921 | 19.644326 | 0.469745 | 0.500373 | 51.662213 | 72.099286 | 60.242070 | 4.350098 |
| val | flive | 75 | 75 | 9.926281 | 12.313568 | 0.031920 | -0.012206 | 49.681616 | 74.779916 | 64.270527 | 5.750707 |
| val | koniq10k | 169 | 169 | 5.481237 | 7.037898 | 0.025941 | 0.050923 | 44.300997 | 80.343056 | 62.414680 | 5.382987 |
| val | spaq | 12 | 12 | 19.894052 | 22.009843 | 0.587413 | 0.590298 | 48.118845 | 73.999071 | 63.479792 | 6.725377 |
| test | flive | 75 | 75 | 8.816753 | 10.246138 | 0.087255 | 0.159504 | 51.453149 | 76.101846 | 64.374094 | 5.269543 |
| test | koniq10k | 164 | 164 | 6.180591 | 7.991493 | 0.033352 | 0.002584 | 45.397118 | 85.576081 | 62.430021 | 6.294652 |
| test | spaq | 17 | 17 | 25.635513 | 29.022755 | 0.436275 | 0.439135 | 54.664940 | 68.318701 | 61.444578 | 4.071087 |

## 9. Failure Cases
- Failure CSV: `outputs/mobile_student_iqa_14105_smoke_gpu_imagenet_20260604/failure_cases.csv`.
- Failure rows: `0`.

## 10. Collapse / Safety Checks
- Collapse check: `{'pred_std_min_overall': 5.449366504138738, 'pred_min_overall': 44.300997257232666, 'pred_max_overall': 85.57608127593994, 'nan_inf_count': 0, 'pred_std_too_low': False, 'pred_outside_0_100': False}`.

| split | dataset | n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- |
| train | overall | 512 | 10 | 2 | 0 |
| train | flive | 143 | 1 | 0 | 0 |
| train | koniq10k | 331 | 3 | 2 | 0 |
| train | spaq | 38 | 6 | 0 | 0 |
| val | overall | 256 | 10 | 6 | 0 |
| val | flive | 75 | 2 | 2 | 0 |
| val | koniq10k | 169 | 4 | 3 | 0 |
| val | spaq | 12 | 4 | 1 | 0 |
| test | overall | 256 | 11 | 2 | 0 |
| test | flive | 75 | 0 | 0 | 0 |
| test | koniq10k | 164 | 5 | 1 | 0 |
| test | spaq | 17 | 6 | 1 | 0 |

## 11. Final Decision
STUDENT_14105_SMOKE_READY_FOR_SHORT_TRAINING

## 12. Recommended Next Step
- Proceed to a short controlled GPU training run; do not start full distillation yet.
