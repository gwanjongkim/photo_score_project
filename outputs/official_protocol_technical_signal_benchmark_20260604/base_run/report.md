# Technical IQA Signal Comparison and Coverage Recovery Report

## 1. Executive Summary
- Final decision: `TECHNICAL_SIGNAL_COMPARISON_NEEDS_GATING_DESIGN`.
- Common rows after recovery: `4967`.
- Best overall SRCC signal: `topiq_existing_flive_grid_s06`.
- Best overall MAE signal: `topiq_existing_flive_grid_s08`.
- Best hybrid composite: `topiq_existing_flive_grid_s08`.
- Runtime seconds: `360.68`.

## 2. Sources Inspected
- `exports/tflite/`
- `outputs/mobile_student_iqa_14105_checkpoint_eval_20260604/`
- `outputs/mobile_student_iqa_14105_tflite_export_20260604/`
- `outputs/student_distillation_manifest_14105_20260603/test_manifest.csv`
- `outputs/student_flive_hybrid_benchmark_20260604/`
- `outputs/technical_iqa_combo_benchmark_20260524/`
- `outputs/research_protocol_production_technical_eval_20260530/`
- `outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/`
- `outputs/eval_tflite_technical_compare/`
- `src/infer/select_best_shots.py`
- `src/infer/predict_musiq.py`
- `scripts/eval_final_topiq_candidates_vs_existing_technical.py`

## 3. Model and Signal Inventory
- See `model_signal_inventory.csv`.

## 4. Prediction Cache Inventory
- See `prediction_cache_inventory.csv`.

## 5. Coverage Recovery
| dataset | test_rows | before_cached_common_rows | common_rows | common_pct |
| --- | --- | --- | --- | --- |
| overall | 4967 | 596 | 4967 | 1.000000 |
| flive | 1827 | 399 | 1827 | 1.000000 |
| koniq10k | 2015 | 84 | 2015 | 1.000000 |
| spaq | 1125 | 113 | 1125 | 1.000000 |

## 6. Benchmark Scope
- Primary split: student 14,105 `test_manifest.csv` only.
- Recovered full test coverage by running existing local TFLite/Keras-compatible assets in WSL.
- Production score is reconstructed offline from recovered KonIQ/FLIVE image-level scores; production code unchanged.

## 7. Overall Signal Metrics
| signal | dataset | n | mae | rmse | srcc | plcc | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production_existing | overall | 4967 | 11.995573 | 14.019543 | 0.716815 | 0.787413 | 1 | 0 | 3 |
| koniq_existing | overall | 4967 | 19.683427 | 22.419346 | 0.690079 | 0.758218 | 0 | 0 | 231 |
| flive_existing | overall | 4967 | 11.028784 | 15.712899 | 0.654318 | 0.680082 | 458 | 189 | 0 |
| topiq_existing | overall | 4967 | 7.674801 | 9.784870 | 0.797461 | 0.871728 | 5 | 2 | 6 |
| student_14105 | overall | 4967 | 9.393720 | 11.574483 | 0.743636 | 0.778156 | 34 | 3 | 0 |
| student_14105_flive_40_60 | overall | 4967 | 9.526695 | 12.737087 | 0.749384 | 0.798297 | 139 | 33 | 0 |
| student_14105_flive_50_50 | overall | 4967 | 9.299458 | 12.218644 | 0.760066 | 0.805276 | 114 | 25 | 0 |
| student_14105_flive_57_43 | overall | 4967 | 9.167492 | 11.919253 | 0.764994 | 0.806482 | 94 | 15 | 0 |
| student_14105_flive_grid_s02 | overall | 4967 | 10.157645 | 14.070623 | 0.714819 | 0.759502 | 243 | 67 | 0 |
| student_14105_flive_grid_s03 | overall | 4967 | 9.810230 | 13.358983 | 0.734432 | 0.783569 | 185 | 48 | 0 |
| student_14105_flive_grid_s04 | overall | 4967 | 9.526695 | 12.737087 | 0.749384 | 0.798297 | 139 | 33 | 0 |
| student_14105_flive_grid_s05 | overall | 4967 | 9.299458 | 12.218644 | 0.760066 | 0.805276 | 114 | 25 | 0 |
| student_14105_flive_grid_s06 | overall | 4967 | 9.124063 | 11.817278 | 0.766329 | 0.806242 | 89 | 15 | 0 |
| student_14105_flive_grid_s07 | overall | 4967 | 9.022488 | 11.545207 | 0.767611 | 0.802806 | 69 | 9 | 0 |
| student_14105_flive_grid_s08 | overall | 4967 | 9.014686 | 11.411682 | 0.764107 | 0.796317 | 61 | 5 | 0 |
| topiq_existing_flive_40_60 | overall | 4967 | 8.434778 | 11.272304 | 0.797422 | 0.847753 | 42 | 7 | 0 |
| topiq_existing_flive_50_50 | overall | 4967 | 7.981696 | 10.463076 | 0.804413 | 0.861521 | 25 | 3 | 0 |
| topiq_existing_flive_57_43 | overall | 4967 | 7.714334 | 9.997076 | 0.806553 | 0.867664 | 21 | 3 | 0 |
| topiq_existing_flive_grid_s02 | overall | 4967 | 9.591127 | 13.307331 | 0.759188 | 0.792579 | 168 | 37 | 0 |
| topiq_existing_flive_grid_s03 | overall | 4967 | 8.973648 | 12.231158 | 0.783595 | 0.825760 | 80 | 15 | 0 |
| topiq_existing_flive_grid_s04 | overall | 4967 | 8.434778 | 11.272304 | 0.797422 | 0.847753 | 42 | 7 | 0 |
| topiq_existing_flive_grid_s05 | overall | 4967 | 7.981696 | 10.463076 | 0.804413 | 0.861521 | 25 | 3 | 0 |
| topiq_existing_flive_grid_s06 | overall | 4967 | 7.623836 | 9.840454 | 0.806929 | 0.869447 | 18 | 3 | 0 |
| topiq_existing_flive_grid_s07 | overall | 4967 | 7.398330 | 9.441430 | 0.806498 | 0.873320 | 11 | 3 | 1 |
| topiq_existing_flive_grid_s08 | overall | 4967 | 7.323232 | 9.294846 | 0.804443 | 0.874430 | 8 | 2 | 2 |

## 8. Dataset-wise Signal Metrics
- Best flive SRCC: `student_14105_flive_grid_s02` SRCC `0.6607`.
- Best koniq10k SRCC: `topiq_existing_flive_grid_s08` SRCC `0.8679`.
- Best spaq SRCC: `student_14105_flive_40_60` SRCC `0.8825`.
| signal | dataset | n | mae | rmse | srcc | plcc | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student_14105_flive_grid_s02 | flive | 1827 | 4.758556 | 5.965902 | 0.660720 | 0.764521 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s02 | flive | 1827 | 4.540995 | 5.741721 | 0.660325 | 0.769047 | 0 | 0 | 0 |
| student_14105_flive_grid_s03 | flive | 1827 | 5.189721 | 6.385184 | 0.657684 | 0.762101 | 0 | 0 | 0 |
| flive_existing | flive | 1827 | 4.158391 | 5.396526 | 0.654439 | 0.759788 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s03 | flive | 1827 | 4.854314 | 6.078875 | 0.649252 | 0.760323 | 0 | 0 | 0 |
| student_14105_flive_40_60 | flive | 1827 | 5.689938 | 6.873503 | 0.649158 | 0.754745 | 0 | 0 | 0 |
| student_14105_flive_grid_s04 | flive | 1827 | 5.689938 | 6.873503 | 0.649158 | 0.754745 | 0 | 0 | 0 |
| student_14105_flive_50_50 | flive | 1827 | 6.238560 | 7.417236 | 0.634079 | 0.740833 | 0 | 0 | 0 |
| student_14105_flive_grid_s05 | flive | 1827 | 6.238560 | 7.417236 | 0.634079 | 0.740833 | 0 | 0 | 0 |
| topiq_existing_flive_40_60 | flive | 1827 | 5.228956 | 6.504873 | 0.630560 | 0.744306 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s04 | flive | 1827 | 5.228956 | 6.504873 | 0.630560 | 0.744306 | 0 | 0 | 0 |
| student_14105_flive_57_43 | flive | 1827 | 6.650816 | 7.833202 | 0.619084 | 0.725929 | 0 | 0 | 0 |
| student_14105_flive_grid_s06 | flive | 1827 | 6.819572 | 8.005101 | 0.611551 | 0.718654 | 0 | 0 | 0 |
| topiq_existing_flive_50_50 | flive | 1827 | 5.648749 | 7.003522 | 0.607605 | 0.722802 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s05 | flive | 1827 | 5.648749 | 7.003522 | 0.607605 | 0.722802 | 0 | 0 | 0 |
| topiq_existing_flive_57_43 | flive | 1827 | 5.970117 | 7.396128 | 0.589428 | 0.705114 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s06 | flive | 1827 | 6.103974 | 7.560459 | 0.581879 | 0.697660 | 0 | 0 | 0 |
| student_14105_flive_grid_s07 | flive | 1827 | 7.427526 | 8.628080 | 0.579618 | 0.686710 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s07 | flive | 1827 | 6.593013 | 8.163765 | 0.554599 | 0.670514 | 0 | 0 | 0 |
| student_14105_flive_grid_s08 | flive | 1827 | 8.053801 | 9.279105 | 0.538791 | 0.644148 | 0 | 0 | 0 |
| production_existing | flive | 1827 | 15.305637 | 16.586883 | 0.537230 | 0.642332 | 0 | 0 | 1 |
| topiq_existing_flive_grid_s08 | flive | 1827 | 7.103928 | 8.803911 | 0.527864 | 0.642658 | 0 | 0 | 1 |
| topiq_existing | flive | 1827 | 8.169910 | 10.166529 | 0.476589 | 0.588271 | 0 | 0 | 1 |
| koniq_existing | flive | 1827 | 25.272244 | 26.901925 | 0.448477 | 0.518921 | 0 | 0 | 142 |
| student_14105 | flive | 1827 | 9.347689 | 10.644505 | 0.432862 | 0.529382 | 3 | 0 | 0 |
| topiq_existing_flive_grid_s08 | koniq10k | 2015 | 5.937152 | 7.675308 | 0.867899 | 0.888975 | 4 | 2 | 0 |
| topiq_existing_flive_grid_s07 | koniq10k | 2015 | 6.192719 | 8.187832 | 0.867212 | 0.888466 | 5 | 2 | 0 |
| topiq_existing | koniq10k | 2015 | 6.096501 | 7.644477 | 0.865489 | 0.886510 | 2 | 2 | 0 |
| topiq_existing_flive_grid_s06 | koniq10k | 2015 | 6.687253 | 8.968477 | 0.864188 | 0.885912 | 12 | 2 | 0 |
| topiq_existing_flive_57_43 | koniq10k | 2015 | 6.866727 | 9.231834 | 0.862697 | 0.884667 | 14 | 2 | 0 |
| topiq_existing_flive_50_50 | koniq10k | 2015 | 7.376758 | 9.954360 | 0.857568 | 0.880228 | 17 | 2 | 0 |
| topiq_existing_flive_grid_s05 | koniq10k | 2015 | 7.376758 | 9.954360 | 0.857568 | 0.880228 | 17 | 2 | 0 |
| topiq_existing_flive_40_60 | koniq10k | 2015 | 8.211225 | 11.090884 | 0.845699 | 0.869790 | 25 | 5 | 0 |
| topiq_existing_flive_grid_s04 | koniq10k | 2015 | 8.211225 | 11.090884 | 0.845699 | 0.869790 | 25 | 5 | 0 |
| topiq_existing_flive_grid_s03 | koniq10k | 2015 | 9.153758 | 12.336485 | 0.825773 | 0.852212 | 45 | 12 | 0 |
| student_14105_flive_grid_s08 | koniq10k | 2015 | 9.558555 | 12.802822 | 0.807809 | 0.826235 | 47 | 4 | 0 |
| student_14105_flive_grid_s07 | koniq10k | 2015 | 9.672223 | 13.083424 | 0.806400 | 0.828367 | 54 | 8 | 0 |
| student_14105_flive_grid_s06 | koniq10k | 2015 | 9.867762 | 13.423858 | 0.799667 | 0.824314 | 70 | 14 | 0 |
| student_14105_flive_57_43 | koniq10k | 2015 | 9.939730 | 13.531502 | 0.796664 | 0.822105 | 75 | 14 | 0 |
| topiq_existing_flive_grid_s02 | koniq10k | 2015 | 10.185065 | 13.661359 | 0.794105 | 0.824103 | 80 | 23 | 0 |
| student_14105 | koniq10k | 2015 | 9.542131 | 12.436457 | 0.793329 | 0.802688 | 24 | 2 | 0 |
| student_14105_flive_50_50 | koniq10k | 2015 | 10.144810 | 13.819702 | 0.787066 | 0.814740 | 91 | 22 | 0 |
| student_14105_flive_grid_s05 | koniq10k | 2015 | 10.144810 | 13.819702 | 0.787066 | 0.814740 | 91 | 22 | 0 |
| production_existing | koniq10k | 2015 | 8.758891 | 10.569756 | 0.772383 | 0.823152 | 1 | 0 | 0 |
| student_14105_flive_40_60 | koniq10k | 2015 | 10.483375 | 14.266345 | 0.769628 | 0.800560 | 107 | 27 | 0 |
| student_14105_flive_grid_s04 | koniq10k | 2015 | 10.483375 | 14.266345 | 0.769628 | 0.800560 | 107 | 27 | 0 |
| student_14105_flive_grid_s03 | koniq10k | 2015 | 10.882026 | 14.759176 | 0.748235 | 0.782802 | 135 | 41 | 0 |
| koniq_existing | koniq10k | 2015 | 16.172213 | 18.479278 | 0.743401 | 0.791765 | 0 | 0 | 26 |
| student_14105_flive_grid_s02 | koniq10k | 2015 | 11.344297 | 15.293730 | 0.722825 | 0.762479 | 148 | 52 | 0 |
| flive_existing | koniq10k | 2015 | 12.432307 | 16.471450 | 0.667866 | 0.717656 | 175 | 84 | 0 |
| student_14105_flive_40_60 | spaq | 1125 | 14.044067 | 16.583429 | 0.882534 | 0.831485 | 32 | 6 | 0 |
| student_14105_flive_grid_s04 | spaq | 1125 | 14.044067 | 16.583429 | 0.882534 | 0.831485 | 32 | 6 | 0 |
| student_14105_flive_50_50 | spaq | 1125 | 12.756237 | 15.090934 | 0.882515 | 0.847098 | 23 | 3 | 0 |
| student_14105_flive_grid_s05 | spaq | 1125 | 12.756237 | 15.090934 | 0.882515 | 0.847098 | 23 | 3 | 0 |
| student_14105_flive_57_43 | spaq | 1125 | 11.871410 | 14.129626 | 0.880987 | 0.853986 | 19 | 1 | 0 |
| student_14105_flive_grid_s06 | spaq | 1125 | 11.534511 | 13.774433 | 0.880638 | 0.855950 | 19 | 1 | 0 |
| student_14105_flive_grid_s03 | spaq | 1125 | 15.394231 | 18.208693 | 0.879820 | 0.806635 | 50 | 7 | 0 |
| topiq_existing_flive_grid_s06 | spaq | 1125 | 11.769616 | 13.807284 | 0.879495 | 0.852257 | 6 | 1 | 0 |
| topiq_existing_flive_57_43 | spaq | 1125 | 12.065099 | 14.133830 | 0.879334 | 0.848873 | 7 | 1 | 0 |
| topiq_existing_flive_grid_s07 | spaq | 1125 | 10.865547 | 12.855105 | 0.878842 | 0.861102 | 6 | 1 | 1 |

## 9. Hybrid Ratio Results
| signal | overall_n | overall_mae | overall_srcc | overall_plcc | min_dataset_srcc | fp_count | severe_fp_count | composite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| topiq_existing_flive_grid_s08 | 4967 | 7.323232 | 0.804443 | 0.874430 | 0.527864 | 8 | 2 | 0.900409 |
| topiq_existing_flive_grid_s07 | 4967 | 7.398330 | 0.806498 | 0.873320 | 0.554599 | 11 | 3 | 0.893147 |
| topiq_existing_flive_grid_s06 | 4967 | 7.623836 | 0.806929 | 0.869447 | 0.581879 | 18 | 3 | 0.886399 |
| topiq_existing_flive_57_43 | 4967 | 7.714334 | 0.806553 | 0.867664 | 0.589428 | 21 | 3 | 0.881910 |
| topiq_existing_flive_50_50 | 4967 | 7.981696 | 0.804413 | 0.861521 | 0.607605 | 25 | 3 | 0.876315 |
| topiq_existing_flive_grid_s05 | 4967 | 7.981696 | 0.804413 | 0.861521 | 0.607605 | 25 | 3 | 0.876315 |
| topiq_existing_flive_40_60 | 4967 | 8.434778 | 0.797422 | 0.847753 | 0.630560 | 42 | 7 | 0.801062 |
| topiq_existing_flive_grid_s04 | 4967 | 8.434778 | 0.797422 | 0.847753 | 0.630560 | 42 | 7 | 0.801062 |
| student_14105_flive_grid_s08 | 4967 | 9.014686 | 0.764107 | 0.796317 | 0.538791 | 61 | 5 | 0.726805 |
| student_14105_flive_grid_s07 | 4967 | 9.022488 | 0.767611 | 0.802806 | 0.579618 | 69 | 9 | 0.684515 |
| topiq_existing_flive_grid_s03 | 4967 | 8.973648 | 0.783595 | 0.825760 | 0.649252 | 80 | 15 | 0.635907 |
| student_14105_flive_grid_s06 | 4967 | 9.124063 | 0.766329 | 0.806242 | 0.611551 | 89 | 15 | 0.591216 |
| student_14105_flive_57_43 | 4967 | 9.167492 | 0.764994 | 0.806482 | 0.619084 | 94 | 15 | 0.581765 |
| student_14105_flive_50_50 | 4967 | 9.299458 | 0.760066 | 0.805276 | 0.634079 | 114 | 25 | 0.440586 |
| student_14105_flive_grid_s05 | 4967 | 9.299458 | 0.760066 | 0.805276 | 0.634079 | 114 | 25 | 0.440586 |
| student_14105_flive_40_60 | 4967 | 9.526695 | 0.749384 | 0.798297 | 0.649158 | 139 | 33 | 0.303673 |
| student_14105_flive_grid_s04 | 4967 | 9.526695 | 0.749384 | 0.798297 | 0.649158 | 139 | 33 | 0.303673 |
| topiq_existing_flive_grid_s02 | 4967 | 9.591127 | 0.759188 | 0.792579 | 0.660325 | 168 | 37 | 0.218269 |
| student_14105_flive_grid_s03 | 4967 | 9.810230 | 0.734432 | 0.783569 | 0.657684 | 185 | 48 | 0.048853 |
| student_14105_flive_grid_s02 | 4967 | 10.157645 | 0.714819 | 0.759502 | 0.660720 | 243 | 67 | -0.276001 |

## 10. Safety Cases
- Safety case rows: `2980`.
- Production severe FP total: `0`.
- Selected candidate severe FP total: `2`.

## 11. Best Signal / Hybrid Recommendation
- Selected candidate: `topiq_existing_flive_grid_s08`.
- Dataset stability pass: `True`.
- Dataset deltas vs production: `[{"dataset": "flive", "srcc_delta_vs_production": -0.009366324925702663}, {"dataset": "koniq10k", "srcc_delta_vs_production": 0.0955163292855743}, {"dataset": "spaq", "srcc_delta_vs_production": 0.036161141238093886}]`.
- Recommendation status: log-only only if decision says ready; never production replacement from this phase.

## 12. Remaining Coverage Gaps
- Primary test coverage target recovered for KonIQ, FLIVE, SPAQ, and overall.
- MUSIQ signals are local probe models, not official Kaggle MUSIQ; interpret separately.

## 13. Final Decision
TECHNICAL_SIGNAL_COMPARISON_NEEDS_GATING_DESIGN

## 14. Recommended Next Step
- Keep production unchanged; design gated/log-only experiment only after reviewing dataset-specific winners.
