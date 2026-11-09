# Technical IQA Signal Comparison with C6 Report

## 1. Executive Summary
- Final decision: `TOPIQ_FLIVE_REMAINS_BEST_LOG_ONLY_CANDIDATE`.
- Benchmark type: offline WSL CSV computation only.
- Candidate status: log-only only; not production replacement.
- C6 overall SRCC/MAE: `0.7036` / `15.3872`.
- TOPIQ+FLIVE 57/43 overall SRCC/MAE: `0.8222` / `5.0395`.

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
| overall | 1413 | 1413 | 1413 | 1413 | 1.000000 |
| flive | 399 | 399 | 399 | 399 | 1.000000 |
| koniq10k | 901 | 901 | 901 | 901 | 1.000000 |
| spaq | 113 | 113 | 113 | 113 | 1.000000 |

## 4. C6 Formula and Inputs
- Main formula: `min(0.7 * topiq_existing + 0.3 * koniq_existing, koniq_existing + 8)`.
- Inputs: `topiq_existing_0_100`, `koniq_existing_0_100` from current 1413-row common predictions.
- Sensitivity variants are experimental.

## 5. C6 Cap Diagnostics
| signal | dataset | n | cap_activation_count | cap_activation_pct | mean_clip_amount_0_100 | max_clip_amount_0_100 |
| --- | --- | --- | --- | --- | --- | --- |
| c6_topiq_koniq_cap | overall | 1413 | 973 | 0.688606 | 6.143833 | 23.242083 |
| c6_topiq_koniq_cap | flive | 399 | 362 | 0.907268 | 8.226774 | 23.242083 |
| c6_topiq_koniq_cap | koniq10k | 901 | 557 | 0.618202 | 4.971281 | 22.677416 |
| c6_topiq_koniq_cap | spaq | 113 | 54 | 0.477876 | 4.275066 | 14.195980 |
| c6_60_40_cap8 | overall | 1413 | 832 | 0.588818 | 4.921436 | 18.778928 |
| c6_60_40_cap8 | flive | 399 | 337 | 0.844612 | 6.389950 | 18.778928 |
| c6_60_40_cap8 | koniq10k | 901 | 450 | 0.499445 | 3.996315 | 18.294928 |
| c6_60_40_cap8 | spaq | 113 | 45 | 0.398230 | 3.175101 | 11.025126 |
| c6_80_20_cap8 | overall | 1413 | 1048 | 0.741684 | 7.618447 | 27.705237 |
| c6_80_20_cap8 | flive | 399 | 365 | 0.914787 | 10.461136 | 27.705237 |
| c6_80_20_cap8 | koniq10k | 901 | 623 | 0.691454 | 6.158831 | 27.059904 |
| c6_80_20_cap8 | spaq | 113 | 60 | 0.530973 | 5.481110 | 17.366835 |
| c6_70_30_cap5 | overall | 1413 | 1177 | 0.832979 | 7.825202 | 26.242083 |
| c6_70_30_cap5 | flive | 399 | 382 | 0.957393 | 10.712456 | 26.242083 |
| c6_70_30_cap5 | koniq10k | 901 | 721 | 0.800222 | 6.511883 | 25.677416 |
| c6_70_30_cap5 | spaq | 113 | 74 | 0.654867 | 5.716711 | 17.195980 |
| c6_70_30_cap10 | overall | 1413 | 761 | 0.538570 | 5.576800 | 21.242083 |
| c6_70_30_cap10 | flive | 399 | 325 | 0.814536 | 7.049441 | 21.242083 |
| c6_70_30_cap10 | koniq10k | 901 | 399 | 0.442841 | 4.544006 | 20.677416 |
| c6_70_30_cap10 | spaq | 113 | 37 | 0.327434 | 3.778870 | 12.195980 |

## 6. Overall Metrics
| signal | dataset | n | mae | rmse | srcc | plcc | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production_existing | overall | 1413 | 11.491342 | 13.220654 | 0.691359 | 0.775555 | 0 | 0 | 0 |
| topiq_existing | overall | 1413 | 7.423350 | 9.079668 | 0.814808 | 0.873160 | 0 | 0 | 1 |
| koniq_existing | overall | 1413 | 21.921748 | 23.683875 | 0.657893 | 0.730980 | 0 | 0 | 58 |
| flive_existing | overall | 1413 | 7.663446 | 11.331054 | 0.638799 | 0.709334 | 54 | 19 | 0 |
| student_14105 | overall | 1413 | 7.560970 | 9.171451 | 0.677100 | 0.777833 | 1 | 0 | 0 |
| topiq_existing_flive_57_43 | overall | 1413 | 5.039497 | 6.741511 | 0.822178 | 0.878604 | 0 | 0 | 0 |
| topiq_existing_flive_50_50 | overall | 1413 | 5.037330 | 6.895602 | 0.818939 | 0.875522 | 1 | 0 | 0 |
| topiq_existing_flive_grid_s06 | overall | 1413 | 5.082671 | 6.728207 | 0.822871 | 0.879300 | 0 | 0 | 0 |
| student_14105_flive_50_50 | overall | 1413 | 6.635737 | 8.800708 | 0.719986 | 0.832624 | 8 | 0 | 0 |
| c6_topiq_koniq_cap | overall | 1413 | 15.387165 | 17.197735 | 0.703615 | 0.788584 | 0 | 0 | 14 |
| c6_60_40_cap8 | overall | 1413 | 15.596650 | 17.340930 | 0.709978 | 0.794576 | 0 | 0 | 14 |
| c6_80_20_cap8 | overall | 1413 | 15.258910 | 17.116407 | 0.699433 | 0.784073 | 0 | 0 | 14 |
| c6_70_30_cap5 | overall | 1413 | 17.612816 | 19.437238 | 0.683077 | 0.762758 | 0 | 0 | 23 |
| c6_70_30_cap10 | overall | 1413 | 14.195189 | 15.945223 | 0.722591 | 0.807852 | 0 | 0 | 9 |

## 7. Dataset-wise Metrics
| signal | dataset | n | mae | rmse | srcc | plcc | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c6_topiq_koniq_cap | flive | 399 | 19.158656 | 21.026843 | 0.369157 | 0.433867 | 0 | 0 | 11 |
| topiq_existing | flive | 399 | 6.560740 | 8.430203 | 0.419861 | 0.473589 | 0 | 0 | 0 |
| topiq_existing_flive_57_43 | flive | 399 | 4.800907 | 6.054776 | 0.518827 | 0.628403 | 0 | 0 | 0 |
| c6_topiq_koniq_cap | koniq10k | 901 | 14.244412 | 15.546327 | 0.795265 | 0.850835 | 0 | 0 | 1 |
| topiq_existing | koniq10k | 901 | 7.586039 | 9.037915 | 0.861878 | 0.895037 | 0 | 0 | 0 |
| topiq_existing_flive_57_43 | koniq10k | 901 | 4.136586 | 5.150404 | 0.854731 | 0.891573 | 0 | 0 | 0 |
| c6_topiq_koniq_cap | spaq | 113 | 11.181818 | 14.494912 | 0.876084 | 0.880749 | 0 | 0 | 2 |
| topiq_existing | spaq | 113 | 9.172007 | 11.341359 | 0.867263 | 0.891295 | 0 | 0 | 1 |
| topiq_existing_flive_57_43 | spaq | 113 | 13.081270 | 15.077969 | 0.876679 | 0.879081 | 0 | 0 | 0 |

## 8. Comparison Against TOPIQ+FLIVE
- C6 SRCC delta vs TOPIQ+FLIVE 57/43: `-0.1186`.
- C6 MAE delta vs TOPIQ+FLIVE 57/43: `10.3477`.
- Dataset deltas: `[{"dataset": "flive", "c6_srcc_delta_vs_topiq_flive": -0.1496699663732195}, {"dataset": "koniq10k", "c6_srcc_delta_vs_topiq_flive": -0.05946563763492496}, {"dataset": "spaq", "c6_srcc_delta_vs_topiq_flive": -0.0005946910310242748}]`.

## 9. Safety Cases
- Safety rows: `216`.
- C6 severe FP: `0`.
- TOPIQ+FLIVE severe FP: `0`.

## 10. Final Ranking
| rank | signal | overall_srcc | overall_mae | min_dataset_srcc | fp_count | severe_fp_count | composite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | topiq_existing_flive_57_43 | 0.822178 | 5.039497 | 0.518827 | 0 | 0 | 0.951885 |
| 2 | topiq_existing_flive_grid_s06 | 0.822871 | 5.082671 | 0.512233 | 0 | 0 | 0.950929 |
| 3 | topiq_existing_flive_50_50 | 0.818939 | 5.037330 | 0.534567 | 1 | 0 | 0.950580 |
| 4 | topiq_existing | 0.814808 | 7.423350 | 0.419861 | 0 | 0 | 0.919773 |
| 5 | student_14105_flive_50_50 | 0.719986 | 6.635737 | 0.543093 | 8 | 0 | 0.839760 |
| 6 | c6_70_30_cap10 | 0.722591 | 14.195189 | 0.372991 | 0 | 0 | 0.815839 |
| 7 | production_existing | 0.691359 | 11.491342 | 0.464870 | 0 | 0 | 0.807576 |
| 8 | c6_60_40_cap8 | 0.709978 | 15.596650 | 0.372248 | 0 | 0 | 0.803040 |
| 9 | c6_topiq_koniq_cap | 0.703615 | 15.387165 | 0.369157 | 0 | 0 | 0.795904 |
| 10 | c6_80_20_cap8 | 0.699433 | 15.258910 | 0.367612 | 0 | 0 | 0.791336 |
| 11 | c6_70_30_cap5 | 0.683077 | 17.612816 | 0.369144 | 0 | 0 | 0.775363 |
| 12 | student_14105 | 0.677100 | 7.560970 | 0.346686 | 1 | 0 | 0.761772 |
| 13 | koniq_existing | 0.657893 | 21.921748 | 0.372075 | 0 | 0 | 0.750912 |
| 14 | flive_existing | 0.638799 | 7.663446 | 0.618878 | 54 | 19 | 0.495518 |

## 11. Final Decision
TOPIQ_FLIVE_REMAINS_BEST_LOG_ONLY_CANDIDATE

## 12. Recommended Next Step
- Keep production unchanged. If desired, carry only the stronger log-only candidate forward; C6 remains offline/log-only evidence, not production replacement.
