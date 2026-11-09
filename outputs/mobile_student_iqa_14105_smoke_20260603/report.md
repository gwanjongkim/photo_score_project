# Mobile Student IQA 14105 Smoke Training Report

## 1. Executive Summary
- Final decision: `STUDENT_14105_SMOKE_NEEDS_TF_GPU_FIX`.
- Run type: bounded smoke only, not full training.
- Runtime seconds: `22.92`.
- Train/val/test samples used: `{'train': 128, 'val': 64, 'test': 64}`.
- Final train loss: `0.005784`.
- Final val loss: `0.005941`.

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
- GPUs visible: `[]`.
- GPU count: `0`.
- Execution mode: `CPU smoke capped <=128 train rows`.

## 5. Smoke Training Configuration
- Backbone: `efficientnetv2b0`.
- Image size: `256`.
- Batch size: `16`.
- Head epochs: `1`.
- Finetune epochs: `1`.
- Loss: `huber`.
- ImageNet weights requested: `imagenet`.
- ImageNet weights loaded: `False`.
- Backbone weights error: `Exception('URL fetch failure on https://storage.googleapis.com/tensorflow/keras-applications/efficientnet_v2/efficientnetv2-b0_notop.h5: None -- [Errno -3] Temporary failure in name resolution')`.
- Include preprocessing: `True`.
- Finetune trainable backbone layers: `13`.

## 6. Training History
| phase | phase_epoch | global_epoch | loss | mae | rmse | val_loss | val_mae | val_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| head | 1 | 1 | 0.006272 | 0.150090 | 0.163613 | 0.005978 | 0.143661 | 0.159353 |
| finetune | 1 | 2 | 0.005784 | 0.140116 | 0.154979 | 0.005941 | 0.142916 | 0.158709 |

## 7. Prediction and Metric Summary
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 128 | 128 | 13.987073 | 15.490427 | -0.126402 | -0.070848 | 50.801063 | 57.994175 | 51.910750 | 1.145793 | 23.717338 | 80.030823 | 64.804162 | 8.428054 | 0 |
| val | overall | 64 | 64 | 14.291581 | 15.870873 | -0.031410 | -0.080367 | 50.807047 | 55.746537 | 51.816822 | 0.931907 | 27.490243 | 84.982681 | 62.988067 | 11.160097 | 0 |
| test | overall | 64 | 64 | 14.588823 | 16.016638 | -0.038095 | -0.090142 | 50.844967 | 57.078880 | 52.048832 | 1.132261 | 23.448595 | 78.824104 | 65.490353 | 8.534290 | 0 |

## 8. Dataset-wise Metrics
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | flive | 27 | 27 | 22.636592 | 22.968544 | -0.057387 | -0.117126 | 50.907069 | 52.914375 | 51.446395 | 0.427764 |
| train | koniq10k | 90 | 90 | 11.388666 | 12.171790 | 0.070470 | 0.105807 | 50.801063 | 57.994175 | 52.093667 | 1.286543 |
| train | spaq | 11 | 11 | 14.016129 | 16.885563 | -0.490909 | -0.397074 | 51.002872 | 53.186876 | 51.553941 | 0.595608 |
| val | flive | 17 | 17 | 21.396158 | 21.978867 | 0.186275 | 0.169273 | 50.969541 | 53.272414 | 51.666742 | 0.637557 |
| val | koniq10k | 38 | 38 | 10.643583 | 11.450558 | -0.087209 | -0.071054 | 50.807047 | 55.746537 | 51.854807 | 0.969490 |
| val | spaq | 9 | 9 | 16.274487 | 18.030942 | 0.166667 | -0.123857 | 51.121378 | 55.009145 | 51.939925 | 1.172840 |
| test | flive | 21 | 21 | 20.562620 | 21.078661 | 0.171429 | -0.153761 | 51.046884 | 53.691030 | 51.783523 | 0.749576 |
| test | koniq10k | 41 | 41 | 11.229323 | 12.162068 | 0.195470 | 0.184688 | 50.844967 | 57.078880 | 52.161053 | 1.276916 |
| test | spaq | 2 | 2 | 20.733702 | 22.616715 | -1.000000 | -1.000000 | 51.850885 | 53.217196 | 52.534041 | 0.683156 |

## 9. Failure Cases
- Failure CSV: `/home/omen_pc1/photo_score_project/outputs/mobile_student_iqa_14105_smoke_20260603/failure_cases.csv`.
- Failure rows: `0`.

## 10. Collapse / Safety Checks
- Collapse check: `{'pred_std_min_overall': 0.931907201164994, 'pred_min_overall': 50.80106258392334, 'pred_max_overall': 57.99417495727539, 'nan_inf_count': 0, 'pred_std_too_low': True, 'pred_outside_0_100': False}`.

| split | dataset | n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- |
| train | overall | 128 | 0 | 0 | 0 |
| train | flive | 27 | 0 | 0 | 0 |
| train | koniq10k | 90 | 0 | 0 | 0 |
| train | spaq | 11 | 0 | 0 | 0 |
| val | overall | 64 | 0 | 0 | 0 |
| val | flive | 17 | 0 | 0 | 0 |
| val | koniq10k | 38 | 0 | 0 | 0 |
| val | spaq | 9 | 0 | 0 | 0 |
| test | overall | 64 | 0 | 0 | 0 |
| test | flive | 21 | 0 | 0 | 0 |
| test | koniq10k | 41 | 0 | 0 | 0 |
| test | spaq | 2 | 0 | 0 | 0 |

## 11. Final Decision
STUDENT_14105_SMOKE_NEEDS_TF_GPU_FIX

## 12. Recommended Next Step
- Fix TensorFlow GPU visibility before any short controlled training run. CPU smoke evidence is recorded.
