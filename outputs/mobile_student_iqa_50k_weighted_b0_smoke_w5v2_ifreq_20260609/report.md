# Mobile Student IQA 50k Weighted B0 Smoke W5-v2 Inverse-Frequency Report

## 1. Executive Summary
- Final decision: `WEIGHTED_B0_W5V2_SMOKE_NEEDS_LABEL_SCALING_OR_RANKING`.
- Run type: inverse-frequency corrected weighted B0 smoke only; no B1/B2, no full controlled training, no TFLite export, no Flutter edits, no production `technical_score` edits.
- Output directory: `outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609`.
- Runtime seconds: `239.38`.
- Train/val/test counts: `{'train': 4000, 'val': 1000, 'test': 1000}`.
- Final train loss: `0.002789`.
- Final val loss: `0.002665`.

## 2. Sources Inspected
- `/home/omen_pc1/photo_score_project/outputs/mdiqa_distillation_50k_full_targets_20260608/final_50k_mdiqa_distillation_manifest.csv`
- `/home/omen_pc1/photo_score_project/outputs/official_protocol_benchmark_manifest_20260604/official_benchmark_test_manifest.csv`
- `scripts/technical_iqa/train_mobile_student_iqa_50k_controlled.py`
- `scripts/technical_iqa/train_mobile_student_iqa_50k_smoke.py`
- `outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/report.md`
- `outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609/report.md`

## 3. Why W5-lite Failed
- W5-lite used raw per-sample weights `0.30/0.35/0.35`.
- Because sample weights multiply each sample loss, FLIVE still dominated due to higher raw sample count.
- Phase 18 effective train contribution remained about FLIVE `64.4%`, KonIQ-10k `15.9%`, SPAQ `19.7%`.
- Phase 18 official Track B SRCC was `0.530639`.

## 4. W5-v2 Weighting Policy
- Full 50k observed shares: FLIVE `0.67782`, KonIQ-10k `0.14386`, SPAQ `0.17832`.
- Target effective shares: FLIVE `0.30`, KonIQ-10k `0.35`, SPAQ `0.35`.
- Applied per-sample weights: `{'flive': 0.443, 'koniq10k': 2.433, 'spaq': 1.963}`.
- Validation, test prediction, and official benchmark evaluation are unweighted.
- Target values and model architecture are unchanged.

## 5. Effective Contribution Audit
| split | dataset | raw_count | raw_share | configured_sample_weight | effective_weighted_count | effective_share | target_effective_share | abs_delta_from_target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | flive | 2712 | 0.678000 | 0.443000 | 1201.415970 | 0.300353 | 0.300000 | 0.000353 |
| train | koniq10k | 575 | 0.143750 | 2.433000 | 1398.975050 | 0.349743 | 0.350000 | 0.000257 |
| train | spaq | 713 | 0.178250 | 1.963000 | 1399.619042 | 0.349904 | 0.350000 | 0.000096 |
| val | flive | 677 | 0.677000 | 0.443000 | 299.910992 | 0.299420 | 0.300000 | 0.000580 |
| val | koniq10k | 144 | 0.144000 | 2.433000 | 350.352013 | 0.349778 | 0.350000 | 0.000222 |
| val | spaq | 179 | 0.179000 | 1.963000 | 351.377011 | 0.350802 | 0.350000 | 0.000802 |
| test | flive | 678 | 0.678000 | 0.443000 | 300.353992 | 0.300318 | 0.300000 | 0.000318 |
| test | koniq10k | 144 | 0.144000 | 2.433000 | 350.352013 | 0.350310 | 0.350000 | 0.000310 |
| test | spaq | 178 | 0.178000 | 1.963000 | 349.414011 | 0.349372 | 0.350000 | 0.000628 |

## 6. Code Changes
- Added `dataset_weight_effective_contribution.csv` audit for train/val/test manifests.
- Added W5-v2-specific report sections and decision labels.
- Reused Phase 18 train-only sample-weight path; no architecture or target-value change.

## 7. Environment / GPU Preflight
- TensorFlow: `2.21.0`.
- CUDA build: `True`.
- GPUs visible: `["PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')"]`.
- GPU count: `1`.
- Correct environment: `.venv_tf`.

## 8. Smoke Training Configuration
- Backbone: `efficientnetv2b0`.
- Image size: `256`.
- Batch size: `16`.
- Head epochs: `1`.
- Finetune epochs: `2`.
- Unfreeze last N: `16`.
- Loss: `huber`.
- Target: `teacher_target_0_100 / 100.0`.
- ImageNet weights requested: `imagenet`.
- ImageNet weights loaded: `True`.
- Finetune trainable backbone layers: `13`.

## 9. Training History
| phase | loss | mae | rmse | val_loss | val_mae | val_rmse | phase_epoch | global_epoch | saved_best_by_val_mae | best_val_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| head | 0.003902 | 0.099110 | 0.127938 | 0.002660 | 0.074663 | 0.097199 | 1 | 1 | True | 0.074663 |
| finetune | 0.002818 | 0.080084 | 0.104395 | 0.002673 | 0.075026 | 0.096305 | 1 | 2 | False | 0.074663 |
| finetune | 0.002789 | 0.079904 | 0.103334 | 0.002665 | 0.074862 | 0.095713 | 2 | 3 | False | 0.074663 |

## 10. Prediction and Metric Summary
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 4000 | 4000 | 6.917140 | 9.181850 | 0.602325 | 0.746830 | 7.249527 | 79.122072 | 60.249032 | 9.941451 | 10.749967 | 87.818849 | 59.978032 | 13.790484 | 0 |
| val | overall | 1000 | 1000 | 7.466329 | 9.719863 | 0.508759 | 0.678164 | 15.874235 | 79.181570 | 60.258747 | 9.325503 | 14.501648 | 79.914570 | 60.067715 | 13.214069 | 0 |
| test | overall | 1000 | 1000 | 8.013639 | 10.379937 | 0.422850 | 0.617235 | 17.243001 | 75.820547 | 60.477454 | 8.411957 | 14.627834 | 83.644652 | 60.163288 | 13.182265 | 0 |

## 11. Dataset-wise Metrics
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | flive | 2712 | 2712 | 7.117037 | 9.192279 | 0.464232 | 0.511522 | 24.109152 | 79.122072 | 62.884653 | 6.130798 | 15.932895 | 87.818849 | 63.685449 | 10.626349 | 0 |
| train | koniq10k | 575 | 575 | 3.823573 | 5.025878 | 0.644256 | 0.648948 | 28.226280 | 75.821632 | 60.521382 | 6.448570 | 44.442070 | 68.092257 | 59.699285 | 4.900317 | 0 |
| train | spaq | 713 | 713 | 8.651616 | 11.454083 | 0.841625 | 0.830381 | 7.249527 | 74.630785 | 50.004428 | 15.701082 | 10.749967 | 79.486680 | 46.101122 | 19.316609 | 0 |
| val | flive | 677 | 677 | 7.656221 | 9.739597 | 0.336585 | 0.375909 | 36.204317 | 79.181570 | 62.703237 | 5.787519 | 18.996963 | 79.914570 | 63.660596 | 10.249056 | 0 |
| val | koniq10k | 144 | 144 | 4.288879 | 5.737398 | 0.632887 | 0.646961 | 34.125262 | 73.302400 | 60.078977 | 7.419055 | 44.806111 | 67.099279 | 59.557322 | 5.600893 | 0 |
| val | spaq | 179 | 179 | 9.304295 | 11.939216 | 0.787768 | 0.789686 | 15.874235 | 77.504241 | 51.158008 | 14.471520 | 14.501648 | 78.360319 | 46.889595 | 18.173238 | 0 |
| test | flive | 678 | 678 | 8.119753 | 10.188668 | 0.305256 | 0.339052 | 35.870355 | 75.820547 | 62.302071 | 5.461033 | 23.296875 | 83.644652 | 63.373657 | 10.584621 | 0 |
| test | koniq10k | 144 | 144 | 4.560331 | 5.847191 | 0.387947 | 0.470159 | 36.767003 | 74.368465 | 61.267736 | 6.428934 | 46.197146 | 67.445320 | 60.551394 | 4.239804 | 0 |
| test | spaq | 178 | 178 | 10.403142 | 13.499337 | 0.704839 | 0.737143 | 17.243001 | 70.465773 | 52.888182 | 13.405931 | 14.627834 | 75.627726 | 47.621057 | 18.392440 | 0 |

## 12. Official Benchmark Evaluation
- Official evaluation failures: `0`.
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_track_b | overall | 4967 | 4967 | 11.427810 | 13.862564 | 0.516337 | 0.647968 | 8.859096 | 78.012609 | 58.149931 | 9.937741 | 4.167411 | 91.700000 | 61.111534 | 17.669026 | 0 |
| official_track_b | flive | 1827 | 1827 | 11.800636 | 13.446873 | 0.309128 | 0.387486 | 28.552863 | 76.120806 | 60.983144 | 6.228405 | 13.520276 | 90.534363 | 71.876490 | 7.815734 | 0 |
| official_track_b | koniq10k | 2015 | 2015 | 10.171607 | 13.086837 | 0.519221 | 0.549472 | 17.956664 | 78.012609 | 60.269346 | 6.821449 | 4.167411 | 88.243750 | 58.700369 | 15.423829 | 0 |
| official_track_b | spaq | 1125 | 1125 | 13.072341 | 15.749623 | 0.725192 | 0.703645 | 8.859096 | 74.686134 | 49.752687 | 14.074488 | 4.800000 | 91.700000 | 47.947911 | 21.935921 | 0 |

| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_track_b | flive | 1827 | 1827 | 11.800636 | 13.446873 | 0.309128 | 0.387486 | 28.552863 | 76.120806 | 60.983144 | 6.228405 | 13.520276 | 90.534363 | 71.876490 | 7.815734 | 0 |
| official_track_b | koniq10k | 2015 | 2015 | 10.171607 | 13.086837 | 0.519221 | 0.549472 | 17.956664 | 78.012609 | 60.269346 | 6.821449 | 4.167411 | 88.243750 | 58.700369 | 15.423829 | 0 |
| official_track_b | spaq | 1125 | 1125 | 13.072341 | 15.749623 | 0.725192 | 0.703645 | 8.859096 | 74.686134 | 49.752687 | 14.074488 | 4.800000 | 91.700000 | 47.947911 | 21.935921 | 0 |

| split | dataset | n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- |
| official_track_b | overall | 4967 | 80 | 13 | 5 |
| official_track_b | flive | 1827 | 1 | 0 | 2 |
| official_track_b | koniq10k | 2015 | 54 | 10 | 1 |
| official_track_b | spaq | 1125 | 25 | 3 | 2 |

## 13. Comparison Against W5-lite, Unweighted B0, and 14k Student
| system | official_srcc | delta_vs_14k | note |
| --- | --- | --- | --- |
| unweighted 50k B0 smoke |  |  | prior smoke had no official Track B table |
| Phase 18 W5-lite smoke | 0.530639 | -0.212961 | raw sample weights |
| unweighted 50k B0 controlled | 0.703290 | -0.040310 | Phase 16 |
| 14k student official | 0.743600 | 0.000000 | previous student |
| MDIQA teacher official | 0.822700 | 0.079100 | teacher ceiling reference |
| W5-v2 inverse-frequency smoke | 0.516337 | -0.227263 | this run |
- MDIQA teacher official SRCC: `0.8227`.

## 14. Collapse / Safety Checks
- Training/input failure rows: `0`.
- Official failure rows: `0`.
- Collapse check: `{'pred_std_min_overall': 8.411956706099119, 'pred_min_overall': 7.249526679515839, 'pred_max_overall': 79.181569814682, 'nan_inf_count': 0, 'pred_std_too_low': False, 'pred_outside_0_100': False}`.

| split | dataset | n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- |
| train | overall | 4000 | 10 | 1 | 2 |
| train | flive | 2712 | 0 | 0 | 2 |
| train | koniq10k | 575 | 2 | 1 | 0 |
| train | spaq | 713 | 8 | 0 | 0 |
| val | overall | 1000 | 3 | 0 | 0 |
| val | flive | 677 | 0 | 0 | 0 |
| val | koniq10k | 144 | 0 | 0 | 0 |
| val | spaq | 179 | 3 | 0 | 0 |
| test | overall | 1000 | 6 | 1 | 1 |
| test | flive | 678 | 1 | 1 | 1 |
| test | koniq10k | 144 | 0 | 0 | 0 |
| test | spaq | 178 | 5 | 0 | 0 |

## 15. Final Decision
WEIGHTED_B0_W5V2_SMOKE_NEEDS_LABEL_SCALING_OR_RANKING

## 16. Recommended Next Step
- Keep B1/B2 paused and investigate KonIQ label scaling or ranking loss.
