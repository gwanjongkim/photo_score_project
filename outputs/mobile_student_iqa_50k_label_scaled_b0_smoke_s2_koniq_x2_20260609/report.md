# Mobile Student IQA 50k Label-Scaled B0 Smoke S2 KonIQ x2 Report

## 1. Executive Summary
- Final decision: `LABEL_SCALED_B0_SMOKE_NO_GO_RESEARCH_REFERENCE_ONLY`.
- Run type: final bounded B0 label-scaling smoke only; no B1/B2, no full controlled training, no TFLite export, no Flutter edits, no production `technical_score` edits, no ranking loss, no extra label-scaling variants.
- Output directory: `/home/omen_pc1/photo_score_project/outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609`.
- Runtime seconds: `244.18`.
- Train/val/test counts: `{'train': 4000, 'val': 1000, 'test': 1000}`.
- Official Track B SRCC: `0.540411`.

## 2. Sources Inspected
- `/home/omen_pc1/photo_score_project/outputs/mdiqa_distillation_50k_full_targets_20260608/final_50k_mdiqa_distillation_manifest.csv`
- `/home/omen_pc1/photo_score_project/outputs/official_protocol_benchmark_manifest_20260604/official_benchmark_test_manifest.csv`
- `scripts/technical_iqa/train_mobile_student_iqa_50k_smoke.py`
- `scripts/technical_iqa/train_mobile_student_iqa_50k_controlled.py`
- `scripts/technical_iqa/train_mobile_student_iqa_50k_ranking_smoke.py`
- `outputs/mobile_student_iqa_50k_controlled_b0_20260608/report.md`
- `outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/report.md`
- `outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609/report.md`
- `outputs/mdiqa_student_pre_final_audit_20260609/report.md`
- `/home/omen_pc1/photo_score_project/outputs/official_protocol_technical_signal_benchmark_20260604/official_signal_metrics_overall.csv`

## 3. Why This Is the Final Smoke
- Phase 16 proved the 50k B0 line can train, but official SRCC stayed below the 14k student.
- Phase 19 W5-v2 weighting and Phase 21 ranking both regressed, leaving KonIQ target compression as the last narrow bottleneck to test.
- This run tests exactly one target scaling policy; a weak result terminates the MDIQA mobile student smoke line.

## 4. Target Scaling Policy
- Policy: `s2_koniq_x2`.
- KonIQ scale factor: `2.0`.
- KonIQ rows use `mean_k + 2.0 * (teacher_target_0_100 - mean_k)`, clipped to `[0, 100]`.
- FLIVE and SPAQ targets remain unchanged.
- Training uses `target_norm = teacher_target_0_100_scaled / 100.0`.
- Official benchmark uses official MOS labels, not scaled teacher targets.

## 5. Target Scaling Audit
- Target scaling audit valid: `True`.
| dataset | count | original_min | original_max | original_mean | original_std | scaled_min | scaled_max | scaled_mean | scaled_std | std_ratio_scaled_over_original | delta_mean | delta_std | clipped_count | clipped_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flive | 4067 | 15.932895 | 87.818849 | 63.629334 | 10.558117 | 15.932895 | 87.818849 | 63.629334 | 10.558117 | 1.000000 | 0.000000 | 0.000000 | 0 | 0.000000 |
| koniq10k | 863 | 44.442070 | 68.092257 | 59.817780 | 4.933943 | 29.064688 | 76.365063 | 59.816109 | 9.867886 | 2.000000 | -0.001671 | 4.933943 | 0 | 0.000000 |
| spaq | 1070 | 10.749967 | 79.486680 | 46.485874 | 18.986928 | 10.749967 | 79.486680 | 46.485874 | 18.986928 | 1.000000 | 0.000000 | 0.000000 | 0 | 0.000000 |

## 6. Implementation Summary
- Added focused phase runner instead of changing existing smoke/controlled/ranking defaults.
- Reused smoke helper Module for manifest splits, image loading, model creation, prediction, metrics, and collapse checks.
- Reused controlled helper Module for official Track B image-only evaluation and safety metrics.
- Scaling columns are saved in each split manifest for auditability.

## 7. Environment / GPU Preflight
- TensorFlow: `2.21.0`.
- CUDA build: `True`.
- GPUs visible: `['PhysicalDevice(name=/physical_device:GPU:0, device_type=GPU)']`.
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
- ImageNet weights requested: `imagenet`.
- ImageNet weights loaded: `True`.
- Backbone weights error: ``.
- Include preprocessing: `True`.
- Finetune trainable backbone layers: `13`.
- Dataset sample weights: `{'flive': 1.0, 'koniq10k': 1.0, 'spaq': 1.0}`.
- Ranking loss disabled: `True`.

## 9. Training History
| phase | loss | mae | rmse | val_loss | val_mae | val_rmse | phase_epoch | global_epoch | saved_best_by_val_mae | best_val_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| head | 0.003804 | 0.098066 | 0.130267 | 0.002736 | 0.075826 | 0.100794 | 1 | 1 | True | 0.075826 |
| finetune | 0.002829 | 0.078001 | 0.102638 | 0.002680 | 0.074851 | 0.098098 | 1 | 2 | True | 0.074851 |
| finetune | 0.002757 | 0.076305 | 0.100944 | 0.002630 | 0.073857 | 0.096631 | 2 | 3 | True | 0.073857 |

## 10. Prediction and Metric Summary
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 4000 | 4000 | 6.444682 | 8.625725 | 0.731671 | 0.797132 | 7.825762 | 79.264694 | 60.318130 | 10.211118 | 10.749967 | 87.818849 | 59.960758 | 14.161374 | 0 |
| val | overall | 1000 | 1000 | 7.385696 | 9.663050 | 0.584428 | 0.711726 | 14.438675 | 81.269258 | 60.730644 | 9.654649 | 14.501648 | 79.914570 | 60.029969 | 13.718982 | 0 |
| test | overall | 1000 | 1000 | 7.479871 | 9.778685 | 0.566202 | 0.690348 | 18.516976 | 78.496850 | 60.703520 | 8.725808 | 14.627834 | 83.644652 | 60.268688 | 13.479079 | 0 |

## 11. Dataset-wise Metrics
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | flive | 2712 | 2712 | 6.023254 | 7.862862 | 0.653882 | 0.676738 | 19.513083 | 79.264694 | 62.982633 | 6.837018 | 15.932895 | 87.818849 | 63.685449 | 10.626349 | 0 |
| train | koniq10k | 575 | 575 | 5.020238 | 6.466656 | 0.755120 | 0.758645 | 24.911834 | 77.349406 | 60.194574 | 8.252910 | 29.064688 | 76.365063 | 59.579119 | 9.800634 | 0 |
| train | spaq | 713 | 713 | 9.196390 | 12.187127 | 0.829497 | 0.806063 | 7.825762 | 76.120526 | 50.282943 | 14.983244 | 10.749967 | 79.486680 | 46.101122 | 19.316609 | 0 |
| val | flive | 677 | 677 | 7.120119 | 9.202241 | 0.430116 | 0.463249 | 30.831808 | 81.269258 | 63.346785 | 6.190718 | 18.996963 | 79.914570 | 63.660596 | 10.249056 | 0 |
| val | koniq10k | 144 | 144 | 5.820213 | 7.837221 | 0.712881 | 0.739678 | 21.041927 | 74.975884 | 59.710387 | 10.388734 | 29.792771 | 74.379108 | 59.295194 | 11.201786 | 0 |
| val | spaq | 179 | 179 | 9.649523 | 12.327144 | 0.774243 | 0.781398 | 14.438675 | 77.657568 | 51.656841 | 13.413020 | 14.501648 | 78.360319 | 46.889595 | 18.173238 | 0 |
| test | flive | 678 | 678 | 7.157039 | 9.210913 | 0.476622 | 0.498948 | 35.815465 | 78.496850 | 62.755240 | 5.842509 | 23.296875 | 83.644652 | 63.373657 | 10.584621 | 0 |
| test | koniq10k | 144 | 144 | 5.711349 | 7.399657 | 0.528028 | 0.597538 | 26.417449 | 78.395289 | 61.069255 | 7.978712 | 32.574842 | 75.071189 | 61.283337 | 8.479609 | 0 |
| test | spaq | 178 | 178 | 10.140248 | 13.028887 | 0.730492 | 0.758306 | 18.516976 | 70.350063 | 52.592664 | 12.818084 | 14.627834 | 75.627726 | 47.621057 | 18.392440 | 0 |

## 12. Official Benchmark Evaluation
- Official evaluation failures: `0`.
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_track_b | overall | 4967 | 4967 | 11.318940 | 13.723468 | 0.540411 | 0.656752 | 9.855867 | 78.666562 | 58.044873 | 10.424658 | 4.167411 | 91.700000 | 61.111534 | 17.669026 | 0 |
| official_track_b | flive | 1827 | 1827 | 12.200186 | 13.856593 | 0.398079 | 0.462100 | 24.008439 | 76.141548 | 60.422693 | 7.185722 | 13.520276 | 90.534363 | 71.876490 | 7.815734 | 0 |
| official_track_b | koniq10k | 2015 | 2015 | 9.357232 | 12.145313 | 0.589104 | 0.628293 | 11.100834 | 78.666562 | 60.309101 | 8.723532 | 4.167411 | 88.243750 | 58.700369 | 15.423829 | 0 |
| official_track_b | spaq | 1125 | 1125 | 13.401433 | 15.984144 | 0.722845 | 0.697658 | 9.855867 | 75.567675 | 50.127812 | 13.364239 | 4.800000 | 91.700000 | 47.947911 | 21.935921 | 0 |

| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_track_b | flive | 1827 | 1827 | 12.200186 | 13.856593 | 0.398079 | 0.462100 | 24.008439 | 76.141548 | 60.422693 | 7.185722 | 13.520276 | 90.534363 | 71.876490 | 7.815734 | 0 |
| official_track_b | koniq10k | 2015 | 2015 | 9.357232 | 12.145313 | 0.589104 | 0.628293 | 11.100834 | 78.666562 | 60.309101 | 8.723532 | 4.167411 | 88.243750 | 58.700369 | 15.423829 | 0 |
| official_track_b | spaq | 1125 | 1125 | 13.401433 | 15.984144 | 0.722845 | 0.697658 | 9.855867 | 75.567675 | 50.127812 | 13.364239 | 4.800000 | 91.700000 | 47.947911 | 21.935921 | 0 |

| split | dataset | n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- |
| official_track_b | overall | 4967 | 74 | 14 | 7 |
| official_track_b | flive | 1827 | 2 | 1 | 4 |
| official_track_b | koniq10k | 2015 | 48 | 11 | 2 |
| official_track_b | spaq | 1125 | 24 | 2 | 1 |

## 13. Comparison Against Phase 16, Phase 19, Phase 21, 14k Student, and MDIQA Teacher
| system | official_srcc | delta_vs_this | note |
| --- | --- | --- | --- |
| Phase 16 unweighted 50k B0 controlled | 0.703290 | -0.162879 | controlled reference |
| Phase 19 W5-v2 weighted smoke | 0.516337 | 0.024074 | weighted smoke reference |
| Phase 21 ranking B0 smoke | 0.480260 | 0.060151 | ranking reference |
| 14k student official | 0.743600 | -0.203189 | student baseline |
| MDIQA teacher official | 0.822700 | -0.282289 | teacher reference |
| production KonIQ+FLIVE existing | 0.716815 | -0.176403 | /home/omen_pc1/photo_score_project/outputs/official_protocol_technical_signal_benchmark_20260604/official_signal_metrics_overall.csv |
| Phase 23 S2 KonIQ x2 label-scaled B0 smoke | 0.540411 | 0.000000 | this run |

## 14. Collapse / Safety Checks
- Training/input failure rows: `0`.
- Official failure rows: `0`.
- Collapse check: `{'pred_std_min_overall': 8.725807785826905, 'pred_min_overall': 7.825762033462524, 'pred_max_overall': 81.26925826072693, 'nan_inf_count': 0, 'pred_std_too_low': False, 'pred_outside_0_100': False}`.

| split | dataset | n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- |
| train | overall | 4000 | 9 | 1 | 1 |
| train | flive | 2712 | 0 | 0 | 1 |
| train | koniq10k | 575 | 1 | 1 | 0 |
| train | spaq | 713 | 8 | 0 | 0 |
| val | overall | 1000 | 3 | 0 | 1 |
| val | flive | 677 | 0 | 0 | 1 |
| val | koniq10k | 144 | 0 | 0 | 0 |
| val | spaq | 179 | 3 | 0 | 0 |
| test | overall | 1000 | 4 | 1 | 0 |
| test | flive | 678 | 1 | 1 | 0 |
| test | koniq10k | 144 | 1 | 0 | 0 |
| test | spaq | 178 | 2 | 0 | 0 |

## 15. Final Decision
LABEL_SCALED_B0_SMOKE_NO_GO_RESEARCH_REFERENCE_ONLY

## 16. Recommended Next Step
- Terminate the MDIQA mobile student smoke line; keep existing KonIQ+FLIVE and archive MDIQA teacher as research reference.

## Checkpoint Summary
- Best checkpoint path: `/home/omen_pc1/photo_score_project/outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609/best_by_val_mae_model.keras`.
- Final checkpoint path: `/home/omen_pc1/photo_score_project/outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609/final_model.keras`.
- Checkpoint metric rows: `3`.
- Final loss row: `{'phase': 'finetune', 'loss': 0.0027568486984819, 'mae': 0.0763052254915237, 'rmse': 0.1009443998336792, 'val_loss': 0.0026299431920051, 'val_mae': 0.0738569721579551, 'val_rmse': 0.0966305136680603, 'phase_epoch': 2, 'global_epoch': 3, 'saved_best_by_val_mae': True, 'best_val_mae': 0.0738569721579551}`.
