# Mobile Student IQA 50k Ranking B0 Smoke R2 Report

## 1. Executive Summary
- Final decision: `RANKING_B0_SMOKE_NEEDS_LABEL_SCALING`.
- Run type: one bounded B0 ranking smoke only; no B1/B2, no full controlled training, no TFLite export, no Flutter edits, no production `technical_score` edits.
- Output directory: `outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609`.
- Runtime seconds: `242.86`.
- Train/val/test counts: `{'train': 4000, 'val': 1000, 'test': 1000}`.
- Final total/huber/rank loss: `0.06609393656253815` / `0.004943629261106253` / `0.6115031242370605`.

## 2. Sources Inspected
- `scripts/technical_iqa/train_mobile_student_iqa_50k_smoke.py`
- `scripts/technical_iqa/train_mobile_student_iqa_50k_controlled.py`
- `outputs/mobile_student_iqa_50k_label_scaling_ranking_plan_20260609/report.md`
- `outputs/mobile_student_iqa_50k_controlled_b0_20260608/report.md`
- `outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/report.md`

## 3. Implementation Summary
- Created focused ranking smoke runner to avoid changing existing default smoke/controlled behavior.
- Training dataset emits image tensor plus target, dataset id, and neutral sample weight.
- Custom `train_step` computes normalized Huber loss plus within-dataset pairwise logistic rank loss.
- Validation, prediction, safety metrics, and official Track B evaluation stay unweighted.

## 4. Environment / GPU Preflight
- TensorFlow: `2.21.0`.
- CUDA build: `True`.
- GPUs visible: `["PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')"]`.
- GPU count: `1`.
- Correct environment: `.venv_tf`.

## 5. Ranking Loss Configuration
- Enabled: `True`.
- Rank loss type: `logistic`.
- Rank margin: `0.05`.
- Lambda rank: `0.1`.
- Pair mask: same dataset, target abs diff >= margin, diagonal excluded.
- Dataset sample weights: neutral/default `1.0`.

## 6. Smoke Training Configuration
- Backbone: `efficientnetv2b0`.
- Image size: `256`.
- Batch size: `16`.
- Head epochs: `1`.
- Finetune epochs: `2`.
- Unfreeze last N: `16`.
- ImageNet weights requested: `imagenet`.
- ImageNet weights loaded: `True`.
- Finetune trainable backbone layers: `13`.

## 7. Training History
| phase | phase_epoch | global_epoch | huber_loss | mae | rank_loss | rmse | total_loss | val_huber_loss | val_mae | val_rank_loss | val_rmse | val_total_loss | val_valid_pair_count | val_valid_pair_rate | valid_pair_count | valid_pair_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| head | 1 | 1 | 0.004863 | 0.119899 | 0.652251 | 0.155079 | 0.070088 | 0.004786 | 0.118232 | 0.000000 | 0.149116 | 0.004786 | 0.000000 | 0.000000 | 85.375999 | 0.355733 |
| finetune | 1 | 2 | 0.004916 | 0.120911 | 0.614241 | 0.155685 | 0.066340 | 0.004781 | 0.118344 | 0.000000 | 0.149518 | 0.004781 | 0.000000 | 0.000000 | 85.248001 | 0.355200 |
| finetune | 2 | 3 | 0.004944 | 0.121680 | 0.611503 | 0.156490 | 0.066094 | 0.004844 | 0.119535 | 0.000000 | 0.151785 | 0.004844 | 0.000000 | 0.000000 | 84.528000 | 0.352200 |

## 8. Ranking Pair Statistics
| phase | epochs | mean_valid_pair_count | min_valid_pair_count | mean_valid_pair_rate | final_rank_loss | final_total_loss | final_huber_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| head | 1 | 85.375999 | 85.375999 | 0.355733 | 0.652251 | 0.070088 | 0.004863 |
| finetune | 2 | 84.888000 | 84.528000 | 0.353700 | 0.611503 | 0.066094 | 0.004944 |
| overall | 3 | 85.050667 | 84.528000 | 0.354378 | 0.611503 | 0.066094 | 0.004944 |

## 9. Prediction and Metric Summary
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 4000 | 4000 | 10.555568 | 13.669269 | 0.682660 | 0.698281 | 3.034523 | 95.226163 | 58.310576 | 18.936659 | 10.749967 | 87.818849 | 59.978032 | 13.790484 | 0 |
| val | overall | 1000 | 1000 | 11.823243 | 14.911606 | 0.551214 | 0.598567 | 6.333329 | 93.378323 | 59.539392 | 18.398821 | 14.501648 | 79.914570 | 60.067715 | 13.214069 | 0 |
| test | overall | 1000 | 1000 | 11.047387 | 14.183811 | 0.525126 | 0.587059 | 5.856764 | 92.950708 | 59.481062 | 17.057107 | 14.627834 | 83.644652 | 60.163288 | 13.182265 | 0 |

## 10. Dataset-wise Metrics
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | flive | 2712 | 2712 | 10.507680 | 13.620312 | 0.651614 | 0.662023 | 5.287462 | 95.226163 | 61.256329 | 17.813694 | 15.932895 | 87.818849 | 63.685449 | 10.626349 | 0 |
| train | koniq10k | 575 | 575 | 10.731939 | 14.054738 | 0.672455 | 0.682155 | 3.034523 | 90.464884 | 57.297800 | 16.719230 | 44.442070 | 68.092257 | 59.699285 | 4.900317 | 0 |
| train | spaq | 713 | 713 | 10.595486 | 13.539291 | 0.778675 | 0.780454 | 4.049157 | 87.194926 | 47.922729 | 20.918704 | 10.749967 | 79.486680 | 46.101122 | 19.316609 | 0 |
| val | flive | 677 | 677 | 11.491318 | 14.507405 | 0.449967 | 0.476232 | 6.764019 | 93.195242 | 62.909228 | 16.224667 | 18.996963 | 79.914570 | 63.660596 | 10.249056 | 0 |
| val | koniq10k | 144 | 144 | 13.880063 | 17.693380 | 0.677550 | 0.724934 | 8.631032 | 93.378323 | 56.303032 | 21.018494 | 44.806111 | 67.099279 | 59.557322 | 5.600893 | 0 |
| val | spaq | 179 | 179 | 11.423977 | 13.941477 | 0.735955 | 0.739635 | 6.333329 | 91.769421 | 49.397808 | 19.644998 | 14.501648 | 78.360319 | 46.889595 | 18.173238 | 0 |
| test | flive | 678 | 678 | 11.153493 | 14.144747 | 0.477088 | 0.499601 | 12.063377 | 92.950708 | 61.748117 | 15.935209 | 23.296875 | 83.644652 | 63.373657 | 10.584621 | 0 |
| test | koniq10k | 144 | 144 | 10.508182 | 14.449149 | 0.539603 | 0.577771 | 8.187918 | 87.788588 | 59.334287 | 16.425374 | 46.197146 | 67.445320 | 60.551394 | 4.239804 | 0 |
| test | spaq | 178 | 178 | 11.079441 | 14.115572 | 0.696503 | 0.729736 | 5.856764 | 83.630222 | 50.964618 | 18.892298 | 14.627834 | 75.627726 | 47.621057 | 18.392440 | 0 |

## 11. Official Benchmark Evaluation
- Official evaluation failures: `0`.
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_track_b | overall | 4967 | 4967 | 14.852354 | 19.079430 | 0.480260 | 0.529617 | 1.689007 | 93.632436 | 53.759959 | 18.596494 | 4.167411 | 91.700000 | 61.111534 | 17.669026 | 0 |
| official_track_b | flive | 1827 | 1827 | 19.744488 | 24.287822 | 0.440312 | 0.456943 | 2.759331 | 93.632436 | 53.688926 | 18.089597 | 13.520276 | 90.534363 | 71.876490 | 7.815734 | 0 |
| official_track_b | koniq10k | 2015 | 2015 | 11.319880 | 14.552965 | 0.596420 | 0.627136 | 1.689007 | 92.888927 | 57.547088 | 17.804761 | 4.167411 | 88.243750 | 58.700369 | 15.423829 | 0 |
| official_track_b | spaq | 1125 | 1125 | 13.234585 | 16.427937 | 0.695659 | 0.686593 | 3.692081 | 92.107666 | 47.092145 | 18.907483 | 4.800000 | 91.700000 | 47.947911 | 21.935921 | 0 |

| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_track_b | flive | 1827 | 1827 | 19.744488 | 24.287822 | 0.440312 | 0.456943 | 2.759331 | 93.632436 | 53.688926 | 18.089597 | 13.520276 | 90.534363 | 71.876490 | 7.815734 | 0 |
| official_track_b | koniq10k | 2015 | 2015 | 11.319880 | 14.552965 | 0.596420 | 0.627136 | 1.689007 | 92.888927 | 57.547088 | 17.804761 | 4.167411 | 88.243750 | 58.700369 | 15.423829 | 0 |
| official_track_b | spaq | 1125 | 1125 | 13.234585 | 16.427937 | 0.695659 | 0.686593 | 3.692081 | 92.107666 | 47.092145 | 18.907483 | 4.800000 | 91.700000 | 47.947911 | 21.935921 | 0 |

| split | dataset | n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- |
| official_track_b | overall | 4967 | 74 | 37 | 94 |
| official_track_b | flive | 1827 | 1 | 1 | 81 |
| official_track_b | koniq10k | 2015 | 40 | 18 | 5 |
| official_track_b | spaq | 1125 | 33 | 18 | 8 |

## 12. Comparison Against Phase 16, Phase 19, and 14k Student
| system | official_srcc | delta_vs_14k | note |
| --- | --- | --- | --- |
| Phase 16 unweighted B0 controlled | 0.703290 | -0.040310 | controlled reference |
| Phase 19 W5-v2 weighted smoke | 0.516337 | -0.227263 | weighted smoke reference |
| 14k student official | 0.743600 | 0.000000 | student baseline |
| MDIQA teacher official | 0.822700 | 0.079100 | teacher reference |
| Phase 21 ranking B0 smoke | 0.480260 | -0.263340 | this run |

## 13. Collapse / Safety Checks
- Training/input failure rows: `0`.
- Official failure rows: `0`.
- Collapse check: `{'pred_std_min_overall': 17.05710730883583, 'pred_min_overall': 3.034522570669651, 'pred_max_overall': 95.22616267204285, 'nan_inf_count': 0, 'pred_std_too_low': False, 'pred_outside_0_100': False}`.

| split | dataset | n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- |
| train | overall | 4000 | 17 | 9 | 49 |
| train | flive | 2712 | 0 | 0 | 46 |
| train | koniq10k | 575 | 2 | 1 | 0 |
| train | spaq | 713 | 15 | 8 | 3 |
| val | overall | 1000 | 3 | 3 | 13 |
| val | flive | 677 | 0 | 0 | 10 |
| val | koniq10k | 144 | 1 | 1 | 0 |
| val | spaq | 179 | 2 | 2 | 3 |
| test | overall | 1000 | 7 | 5 | 14 |
| test | flive | 678 | 1 | 1 | 11 |
| test | koniq10k | 144 | 1 | 0 | 0 |
| test | spaq | 178 | 5 | 4 | 3 |

## 14. Final Decision
RANKING_B0_SMOKE_NEEDS_LABEL_SCALING

## 15. Recommended Next Step
- Test KonIQ label scaling as a separate smoke before more ranking variants.
