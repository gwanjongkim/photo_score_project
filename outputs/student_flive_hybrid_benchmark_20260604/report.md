# Student + FLIVE Hybrid Benchmark Report

## 1. Executive Summary
- Final decision: `HYBRID_STUDENT_FLIVE_NEEDS_RATIO_TUNING`.
- Benchmark type: WSL offline only.
- Candidate status: log-only if accepted; not production replacement.
- Common rows: `596`.
- Selected ratio: `hybrid_4_6` with student weight `0.4000` and FLIVE weight `0.6000`.
- Runtime seconds: `8.61`.

## 2. Sources Inspected
- `outputs/mobile_student_iqa_14105_checkpoint_eval_20260604/report.md`
- `outputs/mobile_student_iqa_14105_tflite_export_20260604/report.md`
- `outputs/student_distillation_manifest_14105_20260603/test_manifest.csv`
- `outputs/technical_iqa_combo_benchmark_20260524/fusion_predictions_flive.csv`
- `outputs/technical_iqa_combo_benchmark_20260524/fusion_predictions_koniq.csv`
- `outputs/technical_iqa_combo_benchmark_20260524/fusion_predictions_spaq.csv`
- `outputs/research_protocol_production_technical_eval_20260530/report.md`
- `src/infer/select_best_shots.py`

## 3. Signal Coverage
| dataset | test_manifest_rows | cached_fusion_rows | common_rows | common_pct_of_test |
| --- | --- | --- | --- | --- |
| overall | 1413 | 6114 | 596 | 0.421798 |
| flive | 399 | 3981 | 399 | 1.000000 |
| koniq10k | 901 | 1008 | 84 | 0.093230 |
| spaq | 113 | 1125 | 113 | 1.000000 |

## 4. Existing Production vs Student vs FLIVE
| signal | dataset | n | mae | rmse | srcc | plcc | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production_existing | overall | 596 | 6.684858 | 9.015445 | 0.673147 | 0.830178 | 4 | 1 | 0 |
| student | overall | 596 | 8.430324 | 10.082840 | 0.598076 | 0.842830 | 1 | 0 | 0 |
| flive | overall | 596 | 8.009892 | 13.386201 | 0.690684 | 0.734949 | 44 | 24 | 0 |

## 5. Hybrid Ratio Candidates
| signal | dataset | n | mae | rmse | srcc | plcc | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_4_3 | overall | 596 | 6.827776 | 9.286248 | 0.680284 | 0.872421 | 5 | 0 | 0 |
| hybrid_5_5 | overall | 596 | 6.758426 | 9.526719 | 0.689956 | 0.872334 | 5 | 0 | 0 |
| hybrid_6_4 | overall | 596 | 6.869359 | 9.219135 | 0.676628 | 0.871788 | 5 | 0 | 0 |
| hybrid_4_6 | overall | 596 | 6.745608 | 10.024858 | 0.700312 | 0.866979 | 8 | 1 | 0 |

## 6. Dataset-wise Results
| signal | dataset | n | mae | rmse | srcc | plcc | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flive | flive | 399 | 3.195328 | 4.246690 | 0.630415 | 0.734161 | 0 | 0 | 0 |
| hybrid_4_6 | flive | 399 | 3.879833 | 4.901759 | 0.573884 | 0.712213 | 0 | 0 | 0 |
| production_existing | flive | 399 | 5.349589 | 7.067032 | 0.505036 | 0.608896 | 0 | 0 | 0 |
| student | flive | 399 | 8.256740 | 9.471187 | 0.346509 | 0.436850 | 0 | 0 | 0 |
| flive | koniq10k | 84 | 9.864039 | 12.549521 | 0.630475 | 0.610225 | 2 | 2 | 0 |
| hybrid_4_6 | koniq10k | 84 | 7.044902 | 9.524363 | 0.699160 | 0.691385 | 2 | 1 | 0 |
| production_existing | koniq10k | 84 | 5.347490 | 6.697348 | 0.802936 | 0.821819 | 1 | 0 | 0 |
| student | koniq10k | 84 | 7.142470 | 8.393871 | 0.697681 | 0.693029 | 0 | 0 | 0 |
| flive | spaq | 113 | 23.631685 | 27.647039 | 0.818923 | 0.715159 | 42 | 22 | 0 |
| hybrid_4_6 | spaq | 113 | 16.642099 | 19.436731 | 0.859191 | 0.838446 | 6 | 0 | 0 |
| production_existing | spaq | 113 | 12.393804 | 14.798596 | 0.878259 | 0.884775 | 3 | 1 | 0 |
| student | spaq | 113 | 10.000587 | 12.926444 | 0.808838 | 0.805162 | 1 | 0 | 0 |

## 7. Safety Cases
- Safety case rows: `98`.
- Selected severe FP total: `1`.
- Production severe FP total: `1`.

## 8. Best Ratio Selection
- Best by overall SRCC: `hybrid_grid_s02`.
- Best by overall MAE: `hybrid_grid_s04`.
- Best by composite: `hybrid_grid_s04`.
- FLIVE constraint pass: `True`.
- KonIQ/SPAQ stability pass: `False`.
| signal | student_weight | flive_weight | overall_mae | overall_srcc | min_dataset_srcc | fp_count | severe_fp_count | composite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_grid_s04 | 0.400000 | 0.600000 | 6.745608 | 0.700312 | 0.573884 | 8 | 1 | 0.817783 |
| hybrid_4_6 | 0.400000 | 0.600000 | 6.745608 | 0.700312 | 0.573884 | 8 | 1 | 0.817783 |
| hybrid_grid_s05 | 0.500000 | 0.500000 | 6.758426 | 0.689956 | 0.543881 | 5 | 0 | 0.815926 |
| hybrid_5_5 | 0.500000 | 0.500000 | 6.758426 | 0.689956 | 0.543881 | 5 | 0 | 0.815926 |
| hybrid_4_3 | 0.571429 | 0.428571 | 6.827776 | 0.680284 | 0.519360 | 5 | 0 | 0.800124 |
| hybrid_grid_s06 | 0.600000 | 0.400000 | 6.869359 | 0.676628 | 0.509132 | 5 | 0 | 0.793911 |
| hybrid_6_4 | 0.600000 | 0.400000 | 6.869359 | 0.676628 | 0.509132 | 5 | 0 | 0.793911 |
| hybrid_grid_s03 | 0.300000 | 0.700000 | 6.839109 | 0.707859 | 0.596995 | 12 | 4 | 0.793108 |
| hybrid_grid_s02 | 0.200000 | 0.800000 | 7.071377 | 0.710623 | 0.614443 | 21 | 5 | 0.772234 |
| hybrid_grid_s07 | 0.700000 | 0.300000 | 7.063855 | 0.658960 | 0.468587 | 4 | 0 | 0.768107 |

## 9. FLIVE Weakness Analysis
- Student alone remains weaker on FLIVE than FLIVE component and production image-level blend on this common-row benchmark.
- Hybrid improves KonIQ/SPAQ exposure by injecting student signal, but selected candidate must not regress FLIVE beyond constraint.
- This benchmark uses cached image-level FLIVE component, not Flutter production runtime.

## 10. Final Decision
HYBRID_STUDENT_FLIVE_NEEDS_RATIO_TUNING

## 11. Recommended Next Step
- Do not profile yet; increase KonIQ coverage and tune ratio/gating because selected hybrid regresses KonIQ/SPAQ stability.
