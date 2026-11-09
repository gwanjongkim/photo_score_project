# Mobile Student IQA 50k Weighted B0 Smoke W5-lite Report

## 1. Executive Summary
- Final decision: `WEIGHTED_B0_SMOKE_NEEDS_WEIGHT_TUNING`.
- Run type: weighted B0 smoke only; no B1/B2, no TFLite export, no Flutter edits, no production `technical_score` edits.
- Output directory: `outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609`.
- Runtime seconds: `239.80`.
- Train/val/test counts: `{'train': 4000, 'val': 1000, 'test': 1000}`.
- Final train loss: `0.000853`.
- Final val loss: `0.002547`.

## 2. Sources Inspected
- `/home/omen_pc1/photo_score_project/outputs/mdiqa_distillation_50k_full_targets_20260608/final_50k_mdiqa_distillation_manifest.csv`
- `/home/omen_pc1/photo_score_project/outputs/official_protocol_benchmark_manifest_20260604/official_benchmark_test_manifest.csv`
- `scripts/technical_iqa/train_mobile_student_iqa_50k_controlled.py`
- `scripts/technical_iqa/train_mobile_student_iqa_50k_smoke.py`
- `outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/recommendation.json`
- `outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/report.md`

## 3. Weighting Policy
- Dataset weights: `{'flive': 0.3, 'koniq10k': 0.35, 'spaq': 0.35}`.
- Training dataset emits Keras sample weights from `dataset_sample_weight`.
- Validation, test prediction, and official benchmark evaluation are unweighted.
- Target values, model architecture, and official MOS labels are unchanged.

## 4. Code Changes
- Added dataset-weight CLI args to 50k smoke and controlled runners with default `1.0` values.
- Added `dataset_sample_weight` to split manifests for auditability.
- Applied sample weights only when building the training dataset.

## 5. Environment / GPU Preflight
- TensorFlow: `2.21.0`.
- CUDA build: `True`.
- GPUs visible: `["PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')"]`.
- GPU count: `1`.
- Correct environment: `.venv_tf`.

## 6. Smoke Training Configuration
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

## 7. Training History
| phase | loss | mae | rmse | val_loss | val_mae | val_rmse | phase_epoch | global_epoch | saved_best_by_val_mae | best_val_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| head | 0.001183 | 0.096014 | 0.127650 | 0.002654 | 0.073994 | 0.099187 | 1 | 1 | True | 0.073994 |
| finetune | 0.000868 | 0.075676 | 0.100900 | 0.002588 | 0.073031 | 0.095810 | 1 | 2 | True | 0.073031 |
| finetune | 0.000853 | 0.074858 | 0.099210 | 0.002547 | 0.072154 | 0.094805 | 2 | 3 | True | 0.072154 |

## 8. Prediction and Metric Summary
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 4000 | 4000 | 6.295864 | 8.553702 | 0.713990 | 0.787531 | 7.286029 | 79.098761 | 60.294222 | 9.945320 | 10.749967 | 87.818849 | 59.978032 | 13.790484 | 0 |
| val | overall | 1000 | 1000 | 7.215405 | 9.480487 | 0.565171 | 0.698407 | 14.312291 | 79.999405 | 60.724925 | 9.310860 | 14.501648 | 79.914570 | 60.067715 | 13.214069 | 0 |
| test | overall | 1000 | 1000 | 7.436169 | 9.776834 | 0.538100 | 0.673147 | 18.070497 | 77.297819 | 60.704466 | 8.361466 | 14.627834 | 83.644652 | 60.163288 | 13.182265 | 0 |

## 9. Dataset-wise Metrics
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | flive | 2712 | 2712 | 6.104415 | 8.014757 | 0.640925 | 0.661564 | 21.178776 | 79.098761 | 62.985588 | 6.531712 | 15.932895 | 87.818849 | 63.685449 | 10.626349 | 0 |
| train | koniq10k | 575 | 575 | 3.673859 | 4.769332 | 0.689738 | 0.695054 | 24.977629 | 75.912517 | 60.110577 | 6.594241 | 44.442070 | 68.092257 | 59.699285 | 4.900317 | 0 |
| train | spaq | 713 | 713 | 9.138588 | 12.156939 | 0.832498 | 0.806000 | 7.286029 | 75.266266 | 50.205314 | 15.105725 | 10.749967 | 79.486680 | 46.101122 | 19.316609 | 0 |
| val | flive | 677 | 677 | 7.115554 | 9.197302 | 0.427629 | 0.456869 | 36.594194 | 79.999405 | 63.317564 | 5.846556 | 18.996963 | 79.914570 | 63.660596 | 10.249056 | 0 |
| val | koniq10k | 144 | 144 | 4.722905 | 6.333246 | 0.683599 | 0.704246 | 25.926805 | 75.104743 | 59.976531 | 8.855888 | 44.806111 | 67.099279 | 59.557322 | 5.600893 | 0 |
| val | spaq | 179 | 179 | 9.598193 | 12.244281 | 0.776697 | 0.782787 | 14.312291 | 75.170618 | 51.521306 | 13.467016 | 14.501648 | 78.360319 | 46.889595 | 18.173238 | 0 |
| test | flive | 678 | 678 | 7.278267 | 9.346238 | 0.456621 | 0.474491 | 38.024563 | 77.297819 | 62.758414 | 5.426003 | 23.296875 | 83.644652 | 63.373657 | 10.584621 | 0 |
| test | koniq10k | 144 | 144 | 4.431183 | 5.620326 | 0.500257 | 0.582623 | 29.527432 | 76.664943 | 60.959390 | 6.891474 | 46.197146 | 67.445320 | 60.551394 | 4.239804 | 0 |
| test | spaq | 178 | 178 | 10.468614 | 13.368802 | 0.722345 | 0.741410 | 18.070497 | 70.290112 | 52.674772 | 12.712794 | 14.627834 | 75.627726 | 47.621057 | 18.392440 | 0 |

## 10. Official Benchmark Evaluation
- Official evaluation failures: `0`.
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_track_b | overall | 4967 | 4967 | 11.504630 | 13.876769 | 0.530639 | 0.650211 | 9.294223 | 78.296745 | 57.943045 | 9.965885 | 4.167411 | 91.700000 | 61.111534 | 17.669026 | 0 |
| official_track_b | flive | 1827 | 1827 | 12.296926 | 13.834658 | 0.386997 | 0.449846 | 24.417584 | 76.263636 | 60.322148 | 6.544182 | 13.520276 | 90.534363 | 71.876490 | 7.815734 | 0 |
| official_track_b | koniq10k | 2015 | 2015 | 9.736237 | 12.572820 | 0.563700 | 0.597420 | 14.639781 | 78.296745 | 60.251210 | 7.576399 | 4.167411 | 88.243750 | 58.700369 | 15.423829 | 0 |
| official_track_b | spaq | 1125 | 1125 | 13.385331 | 16.007288 | 0.720993 | 0.694009 | 9.294223 | 74.404323 | 49.945203 | 13.544037 | 4.800000 | 91.700000 | 47.947911 | 21.935921 | 0 |

| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_track_b | flive | 1827 | 1827 | 12.296926 | 13.834658 | 0.386997 | 0.449846 | 24.417584 | 76.263636 | 60.322148 | 6.544182 | 13.520276 | 90.534363 | 71.876490 | 7.815734 | 0 |
| official_track_b | koniq10k | 2015 | 2015 | 9.736237 | 12.572820 | 0.563700 | 0.597420 | 14.639781 | 78.296745 | 60.251210 | 7.576399 | 4.167411 | 88.243750 | 58.700369 | 15.423829 | 0 |
| official_track_b | spaq | 1125 | 1125 | 13.385331 | 16.007288 | 0.720993 | 0.694009 | 9.294223 | 74.404323 | 49.945203 | 13.544037 | 4.800000 | 91.700000 | 47.947911 | 21.935921 | 0 |

| split | dataset | n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- |
| official_track_b | overall | 4967 | 83 | 11 | 6 |
| official_track_b | flive | 1827 | 2 | 1 | 2 |
| official_track_b | koniq10k | 2015 | 55 | 6 | 2 |
| official_track_b | spaq | 1125 | 26 | 4 | 2 |

## 11. Comparison Against Unweighted B0 and 14k Student
| system | official_srcc | delta_vs_14k | note |
| --- | --- | --- | --- |
| unweighted 50k B0 smoke |  |  | prior smoke had no official Track B table |
| unweighted 50k B0 controlled | 0.703290 | -0.040310 | Phase 16 |
| 14k student official | 0.743600 | 0.000000 | previous student |
| MDIQA teacher official | 0.822700 | 0.079100 | teacher ceiling reference |
| weighted 50k B0 smoke best_by_val_mae | 0.530639 | -0.212961 | this run |
- MDIQA teacher official SRCC: `0.8227`.

## 12. Collapse / Safety Checks
- Training/input failure rows: `0`.
- Official failure rows: `0`.
- Collapse check: `{'pred_std_min_overall': 8.361465563313491, 'pred_min_overall': 7.2860293090343475, 'pred_max_overall': 79.99940514564514, 'nan_inf_count': 0, 'pred_std_too_low': False, 'pred_outside_0_100': False}`.

| split | dataset | n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- |
| train | overall | 4000 | 7 | 2 | 1 |
| train | flive | 2712 | 0 | 0 | 1 |
| train | koniq10k | 575 | 1 | 1 | 0 |
| train | spaq | 713 | 6 | 1 | 0 |
| val | overall | 1000 | 3 | 0 | 1 |
| val | flive | 677 | 0 | 0 | 1 |
| val | koniq10k | 144 | 0 | 0 | 0 |
| val | spaq | 179 | 3 | 0 | 0 |
| test | overall | 1000 | 4 | 1 | 0 |
| test | flive | 678 | 1 | 1 | 0 |
| test | koniq10k | 144 | 1 | 0 | 0 |
| test | spaq | 178 | 2 | 0 | 0 |

## 13. Final Decision
WEIGHTED_B0_SMOKE_NEEDS_WEIGHT_TUNING

## 14. Recommended Next Step
- Tune W5-lite weights and rerun smoke; keep B1/B2 paused.
