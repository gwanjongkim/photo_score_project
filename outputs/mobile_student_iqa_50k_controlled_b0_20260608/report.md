# Mobile Student IQA 50k Controlled B0 Training Report

## 1. Executive Summary
- Final decision: `STUDENT_50K_B0_NEEDS_TUNING`.
- Run type: controlled B0 baseline; not final production training.
- Output directory: `outputs/mobile_student_iqa_50k_controlled_b0_20260608`.
- Runtime seconds: `3242.59`.
- Train/val/test counts: `{'train': 40000, 'val': 5004, 'test': 4996}`.
- Final train loss: `0.001288`.
- Final val loss: `0.001486`.

## 2. Sources Inspected
- `/home/omen_pc1/photo_score_project/outputs/mdiqa_distillation_50k_full_targets_20260608/final_50k_mdiqa_distillation_manifest.csv`
- `/home/omen_pc1/photo_score_project/outputs/official_protocol_benchmark_manifest_20260604/official_benchmark_test_manifest.csv`
- `/home/omen_pc1/photo_score_project/outputs/mdiqa_distillation_50k_full_targets_20260608/report.md`
- `/home/omen_pc1/photo_score_project/outputs/mobile_student_iqa_50k_gpu_visibility_fix_20260608/gpu_preflight_report.md`
- `/home/omen_pc1/photo_score_project/outputs/mobile_student_iqa_50k_smoke_gpu_fixed_venv_tf_20260608/report.md`
- `scripts/technical_iqa/train_mobile_student_iqa_50k_smoke.py`
- `scripts/technical_iqa/train_mobile_student_iqa_14105_smoke.py`
- `scripts/technical_iqa/train_mobile_student_iqa_v4a.py`
- `scripts/technical_iqa/train_mobile_student_iqa_v4b.py`
- `scripts/technical_iqa/evaluate_mobile_student_iqa_checkpoints.py`
- `scripts/technical_iqa/evaluate_student_iqa_14105_checkpoints.py`
- `scripts/technical_iqa/run_mdiqa_official_image_only_benchmark.py`

## 3. Manifest and Split Construction
- Input rows: `50000`.
- Full split counts before caps: `{'train': 40000, 'val': 5004, 'test': 4996}`.
- Capped counts before validation: `{'train': 40000, 'val': 5004, 'test': 4996}`.
- Usable counts: `{'train': 40000, 'val': 5004, 'test': 4996}`.
- Duplicate image paths in input: `0`.
- Image-path overlap across splits: `0`.
- Source-group overlap across splits: `0`.
- Official Track B rows in train: `0`.

## 4. Environment / GPU Preflight
- TensorFlow: `2.21.0`.
- CUDA build: `True`.
- GPUs visible: `["PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')"]`.
- GPU count: `1`.
- Correct environment: `.venv_tf`.

## 5. Training Configuration
- Backbone: `efficientnetv2b0`.
- Image size: `256`.
- Batch size: `16`.
- Head epochs: `5`.
- Finetune epochs: `10`.
- Unfreeze last N: `16`.
- Loss: `huber`.
- Target: `teacher_target_0_100 / 100.0`.
- ImageNet weights requested: `imagenet`.
- ImageNet weights loaded: `True`.
- Finetune trainable backbone layers: `13`.

## 6. Training History
| phase | loss | mae | rmse | val_loss | val_mae | val_rmse | phase_epoch | global_epoch | saved_best_by_val_mae | best_val_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| head | 0.002627 | 0.073579 | 0.097895 | 0.002046 | 0.061210 | 0.082713 | 1 | 1 | True | 0.061210 |
| head | 0.002103 | 0.062490 | 0.083068 | 0.001918 | 0.058386 | 0.079104 | 2 | 2 | True | 0.058386 |
| head | 0.001935 | 0.058823 | 0.078522 | 0.001837 | 0.056501 | 0.077046 | 3 | 3 | True | 0.056501 |
| head | 0.001828 | 0.056527 | 0.075760 | 0.001814 | 0.055850 | 0.076712 | 4 | 4 | True | 0.055850 |
| head | 0.001746 | 0.054683 | 0.073511 | 0.001786 | 0.055343 | 0.075708 | 5 | 5 | True | 0.055343 |
| finetune | 0.001568 | 0.050710 | 0.068440 | 0.001682 | 0.053083 | 0.072083 | 1 | 6 | True | 0.053083 |
| finetune | 0.001488 | 0.048958 | 0.066014 | 0.001635 | 0.051868 | 0.071359 | 2 | 7 | True | 0.051868 |
| finetune | 0.001453 | 0.048220 | 0.064894 | 0.001597 | 0.051190 | 0.069627 | 3 | 8 | True | 0.051190 |
| finetune | 0.001413 | 0.047282 | 0.063617 | 0.001583 | 0.050801 | 0.069556 | 4 | 9 | True | 0.050801 |
| finetune | 0.001399 | 0.046965 | 0.063058 | 0.001574 | 0.050534 | 0.069408 | 5 | 10 | True | 0.050534 |
| finetune | 0.001364 | 0.046172 | 0.062008 | 0.001541 | 0.049815 | 0.068348 | 6 | 11 | True | 0.049815 |
| finetune | 0.001351 | 0.045846 | 0.061632 | 0.001531 | 0.049653 | 0.067773 | 7 | 12 | True | 0.049653 |
| finetune | 0.001324 | 0.045260 | 0.060702 | 0.001525 | 0.049467 | 0.067926 | 8 | 13 | True | 0.049467 |
| finetune | 0.001303 | 0.044781 | 0.060106 | 0.001519 | 0.049300 | 0.067776 | 9 | 14 | True | 0.049300 |
| finetune | 0.001288 | 0.044455 | 0.059574 | 0.001486 | 0.048773 | 0.066081 | 10 | 15 | True | 0.048773 |

## 7. Best Checkpoint Summary
- Best checkpoint path: `outputs/mobile_student_iqa_50k_controlled_b0_20260608/best_by_val_mae_model.keras`.
- Best checkpoint row: `{'phase': 'finetune', 'phase_epoch': 10, 'global_epoch': 15, 'val_mae': 0.04877326264977455, 'val_loss': 0.001486034831032157, 'saved_best_by_val_mae': True, 'best_val_mae': 0.04877326264977455}`.
- Final checkpoint path: `outputs/mobile_student_iqa_50k_controlled_b0_20260608/final_model.keras`.

## 8. Prediction and Metric Summary
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 40000 | 40000 | 3.645044 | 5.003598 | 0.888983 | 0.933370 | 7.743755 | 82.496184 | 59.954482 | 12.262247 | 9.462190 | 87.818849 | 59.985600 | 13.825416 | 0 |
| val | overall | 5004 | 5004 | 4.877327 | 6.608054 | 0.803352 | 0.876693 | 8.660454 | 80.497766 | 60.487078 | 11.931697 | 12.248955 | 83.539933 | 60.042876 | 13.704552 | 0 |
| test | overall | 4996 | 4996 | 4.813891 | 6.488826 | 0.801720 | 0.884311 | 10.214819 | 80.476230 | 60.493539 | 12.123209 | 12.483929 | 83.644652 | 59.992393 | 13.853520 | 0 |

## 9. Dataset-wise Metrics
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | flive | 27114 | 27114 | 3.880458 | 5.260252 | 0.856384 | 0.875874 | 12.481052 | 82.496184 | 63.589850 | 8.097452 | 12.927923 | 87.818849 | 63.825576 | 10.606236 | 0 |
| train | koniq10k | 5754 | 5754 | 2.412587 | 3.207204 | 0.852216 | 0.852052 | 23.370148 | 79.487759 | 60.222242 | 6.051795 | 43.978918 | 69.057435 | 59.817607 | 4.861244 | 0 |
| train | spaq | 7132 | 7132 | 3.744392 | 5.188565 | 0.956250 | 0.962407 | 7.743755 | 75.543374 | 45.917740 | 17.808944 | 9.462190 | 79.486680 | 45.522546 | 18.974527 | 0 |
| val | flive | 3390 | 3390 | 5.071611 | 6.737931 | 0.744097 | 0.774218 | 21.716352 | 80.497766 | 64.188334 | 7.585508 | 18.319198 | 83.539933 | 63.841344 | 10.587671 | 0 |
| val | koniq10k | 720 | 720 | 3.365493 | 4.629865 | 0.812473 | 0.799116 | 20.467173 | 77.284926 | 60.143609 | 7.495812 | 43.457001 | 69.008762 | 59.779824 | 4.991793 | 0 |
| val | spaq | 894 | 894 | 5.358198 | 7.416039 | 0.912937 | 0.919278 | 8.660454 | 73.068875 | 46.728732 | 17.090643 | 12.248955 | 79.557705 | 45.851145 | 18.706821 | 0 |
| test | flive | 3387 | 3387 | 4.956118 | 6.566689 | 0.744300 | 0.784323 | 17.969862 | 80.476230 | 64.206063 | 7.690981 | 14.259478 | 83.644652 | 63.867670 | 10.531465 | 0 |
| test | koniq10k | 719 | 719 | 3.444227 | 4.592374 | 0.752153 | 0.769996 | 24.314038 | 77.096230 | 60.417687 | 7.083745 | 44.500118 | 68.831193 | 59.873890 | 4.849619 | 0 |
| test | spaq | 890 | 890 | 5.379132 | 7.430584 | 0.910158 | 0.922889 | 10.214819 | 73.888820 | 46.426369 | 17.579499 | 12.483929 | 78.212756 | 45.340303 | 19.089394 | 0 |

## 10. Official Benchmark Evaluation
- Official evaluation failures: `0`.
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_track_b | overall | 4967 | 4967 | 9.623013 | 11.757923 | 0.703290 | 0.778297 | 11.295193 | 78.952408 | 57.435712 | 12.465917 | 4.167411 | 91.700000 | 61.111534 | 17.669026 | 0 |
| official_track_b | flive | 1827 | 1827 | 10.555370 | 11.997310 | 0.507083 | 0.590562 | 19.464216 | 78.060234 | 61.920596 | 6.859424 | 13.520276 | 90.534363 | 71.876490 | 7.815734 | 0 |
| official_track_b | koniq10k | 2015 | 2015 | 8.759144 | 11.386179 | 0.739097 | 0.737586 | 25.266504 | 78.952408 | 60.464051 | 7.126409 | 4.167411 | 88.243750 | 58.700369 | 15.423829 | 0 |
| official_track_b | spaq | 1125 | 1125 | 9.656148 | 12.017640 | 0.854897 | 0.850860 | 11.295193 | 77.006108 | 44.728170 | 17.554383 | 4.800000 | 91.700000 | 47.947911 | 21.935921 | 0 |

| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_track_b | flive | 1827 | 1827 | 10.555370 | 11.997310 | 0.507083 | 0.590562 | 19.464216 | 78.060234 | 61.920596 | 6.859424 | 13.520276 | 90.534363 | 71.876490 | 7.815734 | 0 |
| official_track_b | koniq10k | 2015 | 2015 | 8.759144 | 11.386179 | 0.739097 | 0.737586 | 25.266504 | 78.952408 | 60.464051 | 7.126409 | 4.167411 | 88.243750 | 58.700369 | 15.423829 | 0 |
| official_track_b | spaq | 1125 | 1125 | 9.656148 | 12.017640 | 0.854897 | 0.850860 | 11.295193 | 77.006108 | 44.728170 | 17.554383 | 4.800000 | 91.700000 | 47.947911 | 21.935921 | 0 |

| split | dataset | n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- |
| official_track_b | overall | 4967 | 22 | 5 | 1 |
| official_track_b | flive | 1827 | 0 | 0 | 0 |
| official_track_b | koniq10k | 2015 | 18 | 5 | 0 |
| official_track_b | spaq | 1125 | 4 | 0 | 1 |

## 11. Comparison Against 14k Student and MDIQA Teacher
| system | official_srcc | delta_vs_14k | note |
| --- | --- | --- | --- |
| 14k student official | 0.743600 | 0.000000 | previous student |
| MDIQA teacher official | 0.822700 | 0.079100 | teacher ceiling reference |
| 50k B0 best_by_val_mae | 0.703290 | -0.040310 | this run |
- TOPIQ+FLIVE log-only note: topiq_existing_flive_57_43 was log-only and not used as a direct checkpoint baseline.

## 12. Failure Cases
- Training/input failure rows: `0`.
- Official failure rows: `0`.

## 13. Collapse / Safety Checks
- Collapse check: `{'pred_std_min_overall': 11.93169667399368, 'pred_min_overall': 7.743754982948303, 'pred_max_overall': 82.49618411064148, 'nan_inf_count': 0, 'pred_std_too_low': False, 'pred_outside_0_100': False}`.

| split | dataset | n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- |
| train | overall | 40000 | 21 | 2 | 25 |
| train | flive | 27114 | 2 | 0 | 25 |
| train | koniq10k | 5754 | 2 | 1 | 0 |
| train | spaq | 7132 | 17 | 1 | 0 |
| val | overall | 5004 | 1 | 0 | 6 |
| val | flive | 3390 | 0 | 0 | 3 |
| val | koniq10k | 720 | 0 | 0 | 0 |
| val | spaq | 894 | 1 | 0 | 3 |
| test | overall | 4996 | 7 | 1 | 7 |
| test | flive | 3387 | 1 | 0 | 5 |
| test | koniq10k | 719 | 0 | 0 | 0 |
| test | spaq | 890 | 6 | 1 | 2 |

## 14. Final Decision
STUDENT_50K_B0_NEEDS_TUNING

## 15. Recommended Next Step
- Analyze official benchmark errors and scale compression before B1/B2.
