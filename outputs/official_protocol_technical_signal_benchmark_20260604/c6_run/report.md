# Technical IQA Signal Comparison with C6 Report

## 1. Executive Summary
- Final decision: `TOPIQ_FLIVE_REMAINS_BEST_LOG_ONLY_CANDIDATE`.
- Benchmark type: offline WSL CSV computation only.
- Candidate status: log-only only; not production replacement.
- C6 overall SRCC/MAE: `0.7314` / `13.6706`.
- TOPIQ+FLIVE 57/43 overall SRCC/MAE: `0.8066` / `7.7143`.

## 2. Sources Inspected
- `outputs/technical_iqa_signal_comparison_20260604/`
- `scripts/technical_iqa/benchmark_technical_iqa_signals_20260604.py`
- `outputs/technical_iqa_topiq_koniq_cap_benchmark_20260524/`
- `outputs/technical_iqa_topiq_koniq_cap_ranking_audit_20260524/`
- `outputs/c6_android_wsl_parity_20260527/`
- `outputs/c6_android_wsl_parity_exif_fixed_20260529/`
- `outputs/c6_topiq_preprocess_backend_ablation_20260529/`
- `outputs/topiq_mixed112_artifact_recovery_20260527/`
- `outputs/final_topiq_lite_mixed112_export_20260517/`
- `exports/tflite/`

## 3. Benchmark Scope and Coverage
| dataset | n | topiq_non_null | koniq_non_null | c6_non_null | coverage_pct |
| --- | --- | --- | --- | --- | --- |
| overall | 4967 | 4967 | 4967 | 4967 | 1.000000 |
| flive | 1827 | 1827 | 1827 | 1827 | 1.000000 |
| koniq10k | 2015 | 2015 | 2015 | 2015 | 1.000000 |
| spaq | 1125 | 1125 | 1125 | 1125 | 1.000000 |

## 4. C6 Formula and Inputs
- Main formula: `min(0.7 * topiq_existing + 0.3 * koniq_existing, koniq_existing + 8)`.
- Inputs: `topiq_existing_0_100`, `koniq_existing_0_100` from current 1413-row common predictions.
- Sensitivity variants are experimental.

## 5. C6 Cap Diagnostics
| signal | dataset | n | cap_activation_count | cap_activation_pct | mean_clip_amount_0_100 | max_clip_amount_0_100 |
| --- | --- | --- | --- | --- | --- | --- |
| c6_topiq_koniq_cap | overall | 4967 | 3231 | 0.650493 | 5.830596 | 25.099161 |
| c6_topiq_koniq_cap | flive | 1827 | 1467 | 0.802956 | 6.788326 | 25.099161 |
| c6_topiq_koniq_cap | koniq10k | 2015 | 1213 | 0.601985 | 5.141309 | 22.022972 |
| c6_topiq_koniq_cap | spaq | 1125 | 551 | 0.489778 | 4.798140 | 19.049362 |
| c6_60_40_cap8 | overall | 4967 | 2776 | 0.558889 | 4.584592 | 20.370710 |
| c6_60_40_cap8 | flive | 1827 | 1328 | 0.726875 | 5.224525 | 20.370710 |
| c6_60_40_cap8 | koniq10k | 2015 | 998 | 0.495285 | 4.099982 | 17.733976 |
| c6_60_40_cap8 | spaq | 1125 | 450 | 0.400000 | 3.770838 | 15.185168 |
| c6_80_20_cap8 | overall | 4967 | 3508 | 0.706261 | 7.235138 | 29.827613 |
| c6_80_20_cap8 | flive | 1827 | 1545 | 0.845649 | 8.482046 | 29.827613 |
| c6_80_20_cap8 | koniq10k | 2015 | 1335 | 0.662531 | 6.426905 | 26.311968 |
| c6_80_20_cap8 | spaq | 1125 | 628 | 0.558222 | 5.885644 | 22.913557 |
| c6_70_30_cap5 | overall | 4967 | 4001 | 0.805516 | 7.433308 | 28.099161 |
| c6_70_30_cap5 | flive | 1827 | 1680 | 0.919540 | 8.750713 | 28.099161 |
| c6_70_30_cap5 | koniq10k | 2015 | 1564 | 0.776179 | 6.665277 | 25.022972 |
| c6_70_30_cap5 | spaq | 1125 | 757 | 0.672889 | 6.096396 | 22.049362 |
| c6_70_30_cap10 | overall | 4967 | 2560 | 0.515402 | 5.103505 | 23.099161 |
| c6_70_30_cap10 | flive | 1827 | 1260 | 0.689655 | 5.736751 | 23.099161 |
| c6_70_30_cap10 | koniq10k | 2015 | 891 | 0.442184 | 4.651571 | 20.022972 |
| c6_70_30_cap10 | spaq | 1125 | 409 | 0.363556 | 4.137204 | 17.049362 |

## 6. Overall Metrics
| signal | dataset | n | mae | rmse | srcc | plcc | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production_existing | overall | 4967 | 11.995573 | 14.019543 | 0.716815 | 0.787413 | 1 | 0 | 3 |
| topiq_existing | overall | 4967 | 7.674801 | 9.784870 | 0.797461 | 0.871728 | 5 | 2 | 6 |
| koniq_existing | overall | 4967 | 19.683427 | 22.419346 | 0.690079 | 0.758218 | 0 | 0 | 231 |
| flive_existing | overall | 4967 | 11.028784 | 15.712899 | 0.654318 | 0.680082 | 458 | 189 | 0 |
| student_14105 | overall | 4967 | 9.393720 | 11.574483 | 0.743636 | 0.778156 | 34 | 3 | 0 |
| topiq_existing_flive_57_43 | overall | 4967 | 7.714334 | 9.997076 | 0.806553 | 0.867664 | 21 | 3 | 0 |
| topiq_existing_flive_50_50 | overall | 4967 | 7.981696 | 10.463076 | 0.804413 | 0.861521 | 25 | 3 | 0 |
| topiq_existing_flive_grid_s06 | overall | 4967 | 7.623836 | 9.840454 | 0.806929 | 0.869447 | 18 | 3 | 0 |
| student_14105_flive_50_50 | overall | 4967 | 9.299458 | 12.218644 | 0.760066 | 0.805276 | 114 | 25 | 0 |
| c6_topiq_koniq_cap | overall | 4967 | 13.670634 | 16.398569 | 0.731365 | 0.810032 | 1 | 1 | 68 |
| c6_60_40_cap8 | overall | 4967 | 13.847814 | 16.523244 | 0.736750 | 0.815351 | 1 | 0 | 69 |
| c6_80_20_cap8 | overall | 4967 | 13.566700 | 16.326883 | 0.727639 | 0.805967 | 1 | 1 | 67 |
| c6_70_30_cap5 | overall | 4967 | 15.611737 | 18.437070 | 0.712456 | 0.787178 | 1 | 0 | 112 |
| c6_70_30_cap10 | overall | 4967 | 12.650082 | 15.263007 | 0.747381 | 0.826851 | 1 | 1 | 45 |

## 7. Dataset-wise Metrics
| signal | dataset | n | mae | rmse | srcc | plcc | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c6_topiq_koniq_cap | flive | 1827 | 17.984899 | 19.899860 | 0.463433 | 0.545037 | 0 | 0 | 42 |
| topiq_existing | flive | 1827 | 8.169910 | 10.166529 | 0.476589 | 0.588271 | 0 | 0 | 1 |
| topiq_existing_flive_57_43 | flive | 1827 | 5.970117 | 7.396128 | 0.589428 | 0.705114 | 0 | 0 | 0 |
| c6_topiq_koniq_cap | koniq10k | 2015 | 10.446995 | 12.662649 | 0.798817 | 0.845401 | 1 | 1 | 6 |
| topiq_existing | koniq10k | 2015 | 6.096501 | 7.644477 | 0.865489 | 0.886510 | 2 | 2 | 0 |
| topiq_existing_flive_57_43 | koniq10k | 2015 | 6.866727 | 9.231834 | 0.862697 | 0.884667 | 14 | 2 | 0 |
| c6_topiq_koniq_cap | spaq | 1125 | 12.438163 | 16.030541 | 0.858451 | 0.852403 | 0 | 0 | 20 |
| topiq_existing | spaq | 1125 | 9.697656 | 12.255452 | 0.872801 | 0.868314 | 3 | 0 | 5 |
| topiq_existing_flive_57_43 | spaq | 1125 | 12.065099 | 14.133830 | 0.879334 | 0.848873 | 7 | 1 | 0 |

## 8. Comparison Against TOPIQ+FLIVE
- C6 SRCC delta vs TOPIQ+FLIVE 57/43: `-0.0752`.
- C6 MAE delta vs TOPIQ+FLIVE 57/43: `5.9563`.
- Dataset deltas: `[{"dataset": "flive", "c6_srcc_delta_vs_topiq_flive": -0.12599571076351507}, {"dataset": "koniq10k", "c6_srcc_delta_vs_topiq_flive": -0.06387990208824457}, {"dataset": "spaq", "c6_srcc_delta_vs_topiq_flive": -0.02088342444044644}]`.

## 9. Safety Cases
- Safety rows: `1513`.
- C6 severe FP: `1`.
- TOPIQ+FLIVE severe FP: `3`.

## 10. Final Ranking
| rank | signal | overall_srcc | overall_mae | min_dataset_srcc | fp_count | severe_fp_count | composite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | topiq_existing | 0.797461 | 7.674801 | 0.476589 | 5 | 2 | 0.886608 |
| 2 | topiq_existing_flive_grid_s06 | 0.806929 | 7.623836 | 0.581879 | 18 | 3 | 0.886399 |
| 3 | topiq_existing_flive_57_43 | 0.806553 | 7.714334 | 0.589428 | 21 | 3 | 0.881910 |
| 4 | topiq_existing_flive_50_50 | 0.804413 | 7.981696 | 0.607605 | 25 | 3 | 0.876315 |
| 5 | c6_70_30_cap10 | 0.747381 | 12.650082 | 0.472368 | 1 | 1 | 0.853473 |
| 6 | c6_60_40_cap8 | 0.736750 | 13.847814 | 0.467262 | 1 | 0 | 0.851566 |
| 7 | production_existing | 0.716815 | 11.995573 | 0.537230 | 1 | 0 | 0.849122 |
| 8 | c6_topiq_koniq_cap | 0.731365 | 13.670634 | 0.463433 | 1 | 1 | 0.835224 |
| 9 | c6_80_20_cap8 | 0.727639 | 13.566700 | 0.460880 | 1 | 1 | 0.830859 |
| 10 | c6_70_30_cap5 | 0.712456 | 15.611737 | 0.454631 | 1 | 0 | 0.824114 |
| 11 | koniq_existing | 0.690079 | 19.683427 | 0.448477 | 0 | 0 | 0.802198 |
| 12 | student_14105 | 0.743636 | 9.393720 | 0.432862 | 34 | 3 | 0.753851 |
| 13 | student_14105_flive_50_50 | 0.760066 | 9.299458 | 0.634079 | 114 | 25 | 0.440586 |
| 14 | flive_existing | 0.654318 | 11.028784 | 0.654439 | 458 | 189 | -1.988073 |

## 11. Final Decision
TOPIQ_FLIVE_REMAINS_BEST_LOG_ONLY_CANDIDATE

## 12. Recommended Next Step
- Keep production unchanged. If desired, carry only the stronger log-only candidate forward; C6 remains offline/log-only evidence, not production replacement.
