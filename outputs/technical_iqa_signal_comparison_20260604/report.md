# Technical IQA Signal Comparison and Coverage Recovery Report

## 1. Executive Summary
- Final decision: `TECHNICAL_SIGNAL_COMPARISON_READY_FOR_LOG_ONLY_PROFILING`.
- Common rows after recovery: `1413`.
- Best overall SRCC signal: `topiq_existing_flive_grid_s07`.
- Best overall MAE signal: `topiq_existing_flive_50_50`.
- Best hybrid composite: `topiq_existing_flive_57_43`.
- Runtime seconds: `152.96`.

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
| overall | 1413 | 596 | 1413 | 1.000000 |
| flive | 399 | 399 | 399 | 1.000000 |
| koniq10k | 901 | 84 | 901 | 1.000000 |
| spaq | 113 | 113 | 113 | 1.000000 |

## 6. Benchmark Scope
- Primary split: student 14,105 `test_manifest.csv` only.
- Recovered full test coverage by running existing local TFLite/Keras-compatible assets in WSL.
- Production score is reconstructed offline from recovered KonIQ/FLIVE image-level scores; production code unchanged.

## 7. Overall Signal Metrics
| signal | dataset | n | mae | rmse | srcc | plcc | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production_existing | overall | 1413 | 11.491342 | 13.220654 | 0.691359 | 0.775555 | 0 | 0 | 0 |
| koniq_existing | overall | 1413 | 21.921748 | 23.683875 | 0.657893 | 0.730980 | 0 | 0 | 58 |
| flive_existing | overall | 1413 | 7.663446 | 11.331054 | 0.638799 | 0.709334 | 54 | 19 | 0 |
| topiq_existing | overall | 1413 | 7.423350 | 9.079668 | 0.814808 | 0.873160 | 0 | 0 | 1 |
| student_14105 | overall | 1413 | 7.560970 | 9.171451 | 0.677100 | 0.777833 | 1 | 0 | 0 |
| musiq_koniq_probe | overall | 1413 | 16.243879 | 17.937830 | 0.046822 | 0.141784 | 0 | 0 | 0 |
| musiq_flive_probe | overall | 1413 | 17.149579 | 18.849668 | 0.217765 | 0.219620 | 0 | 0 | 0 |
| musiq_spaq_probe | overall | 1413 | 21.298982 | 23.114447 | 0.020650 | 0.141251 | 0 | 0 | 0 |
| musiq_flive_probe_flive_40_60 | overall | 1413 | 8.998368 | 11.222177 | 0.638757 | 0.709338 | 11 | 0 | 0 |
| musiq_flive_probe_flive_50_50 | overall | 1413 | 10.060031 | 12.030990 | 0.638725 | 0.709339 | 4 | 0 | 0 |
| musiq_flive_probe_flive_57_43 | overall | 1413 | 10.920146 | 12.764594 | 0.638705 | 0.709340 | 2 | 0 | 0 |
| musiq_flive_probe_flive_grid_s02 | overall | 1413 | 7.611002 | 10.565413 | 0.638758 | 0.709335 | 45 | 5 | 0 |
| musiq_flive_probe_flive_grid_s03 | overall | 1413 | 8.162085 | 10.719064 | 0.638771 | 0.709336 | 25 | 3 | 0 |
| musiq_flive_probe_flive_grid_s04 | overall | 1413 | 8.998368 | 11.222177 | 0.638757 | 0.709338 | 11 | 0 | 0 |
| musiq_flive_probe_flive_grid_s05 | overall | 1413 | 10.060031 | 12.030990 | 0.638725 | 0.709339 | 4 | 0 | 0 |
| musiq_flive_probe_flive_grid_s06 | overall | 1413 | 11.288235 | 13.088953 | 0.638685 | 0.709341 | 0 | 0 | 0 |
| musiq_flive_probe_flive_grid_s07 | overall | 1413 | 12.643151 | 14.341033 | 0.638702 | 0.709343 | 0 | 0 | 0 |
| musiq_flive_probe_flive_grid_s08 | overall | 1413 | 14.094783 | 15.740974 | 0.638533 | 0.709343 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_40_60 | overall | 1413 | 8.800581 | 11.132997 | 0.638446 | 0.709396 | 17 | 0 | 0 |
| musiq_koniq_probe_flive_50_50 | overall | 1413 | 9.761102 | 11.826777 | 0.638287 | 0.709420 | 6 | 0 | 0 |
| musiq_koniq_probe_flive_57_43 | overall | 1413 | 10.534829 | 12.466913 | 0.638103 | 0.709437 | 3 | 0 | 0 |
| musiq_koniq_probe_flive_grid_s02 | overall | 1413 | 7.583959 | 10.608258 | 0.638701 | 0.709360 | 47 | 5 | 0 |
| musiq_koniq_probe_flive_grid_s03 | overall | 1413 | 8.067443 | 10.715928 | 0.638580 | 0.709377 | 31 | 3 | 0 |
| musiq_koniq_probe_flive_grid_s04 | overall | 1413 | 8.800581 | 11.132997 | 0.638446 | 0.709396 | 17 | 0 | 0 |
| musiq_koniq_probe_flive_grid_s05 | overall | 1413 | 9.761102 | 11.826777 | 0.638287 | 0.709420 | 6 | 0 | 0 |
| musiq_koniq_probe_flive_grid_s06 | overall | 1413 | 10.865291 | 12.752184 | 0.638014 | 0.709444 | 2 | 0 | 0 |
| musiq_koniq_probe_flive_grid_s07 | overall | 1413 | 12.097639 | 13.862911 | 0.637489 | 0.709458 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_grid_s08 | overall | 1413 | 13.428685 | 15.118165 | 0.636158 | 0.709386 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_40_60 | overall | 1413 | 9.971249 | 11.843463 | 0.638769 | 0.709357 | 3 | 0 | 0 |
| musiq_spaq_probe_flive_50_50 | overall | 1413 | 11.531998 | 13.213105 | 0.638720 | 0.709368 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_57_43 | overall | 1413 | 12.773650 | 14.379739 | 0.638696 | 0.709379 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s02 | overall | 1413 | 7.783680 | 10.452289 | 0.638774 | 0.709343 | 28 | 3 | 0 |
| musiq_spaq_probe_flive_grid_s03 | overall | 1413 | 8.677882 | 10.885266 | 0.638771 | 0.709349 | 10 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s04 | overall | 1413 | 9.971249 | 11.843463 | 0.638769 | 0.709357 | 3 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s05 | overall | 1413 | 11.531998 | 13.213105 | 0.638720 | 0.709368 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s06 | overall | 1413 | 13.294804 | 14.881015 | 0.638700 | 0.709385 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s07 | overall | 1413 | 15.171622 | 16.758370 | 0.638635 | 0.709411 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s08 | overall | 1413 | 17.142213 | 18.782473 | 0.638520 | 0.709458 | 0 | 0 | 0 |
| student_14105_flive_40_60 | overall | 1413 | 6.680838 | 9.112203 | 0.714672 | 0.827214 | 11 | 1 | 0 |
| student_14105_flive_50_50 | overall | 1413 | 6.635737 | 8.800708 | 0.719986 | 0.832624 | 8 | 0 | 0 |
| student_14105_flive_57_43 | overall | 1413 | 6.643717 | 8.653183 | 0.720622 | 0.831673 | 8 | 0 | 0 |
| student_14105_flive_grid_s02 | overall | 1413 | 7.005265 | 10.055061 | 0.687710 | 0.787813 | 22 | 3 | 0 |
| student_14105_flive_grid_s03 | overall | 1413 | 6.800317 | 9.534998 | 0.703579 | 0.812575 | 13 | 1 | 0 |
| student_14105_flive_grid_s04 | overall | 1413 | 6.680838 | 9.112203 | 0.714672 | 0.827214 | 11 | 1 | 0 |
| student_14105_flive_grid_s05 | overall | 1413 | 6.635737 | 8.800708 | 0.719986 | 0.832624 | 8 | 0 | 0 |
| student_14105_flive_grid_s06 | overall | 1413 | 6.658906 | 8.612595 | 0.720119 | 0.830336 | 8 | 0 | 0 |
| student_14105_flive_grid_s07 | overall | 1413 | 6.748297 | 8.556008 | 0.715612 | 0.822127 | 6 | 0 | 0 |
| student_14105_flive_grid_s08 | overall | 1413 | 6.924262 | 8.633533 | 0.707325 | 0.809711 | 3 | 0 | 0 |
| topiq_existing_flive_40_60 | overall | 1413 | 5.222828 | 7.379652 | 0.810007 | 0.866805 | 1 | 0 | 0 |
| topiq_existing_flive_50_50 | overall | 1413 | 5.037330 | 6.895602 | 0.818939 | 0.875522 | 1 | 0 | 0 |
| topiq_existing_flive_57_43 | overall | 1413 | 5.039497 | 6.741511 | 0.822178 | 0.878604 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s02 | overall | 1413 | 6.189151 | 9.064616 | 0.763401 | 0.822308 | 16 | 1 | 0 |
| topiq_existing_flive_grid_s03 | overall | 1413 | 5.629452 | 8.123950 | 0.792773 | 0.850395 | 4 | 1 | 0 |
| topiq_existing_flive_grid_s04 | overall | 1413 | 5.222828 | 7.379652 | 0.810007 | 0.866805 | 1 | 0 | 0 |
| topiq_existing_flive_grid_s05 | overall | 1413 | 5.037330 | 6.895602 | 0.818939 | 0.875522 | 1 | 0 | 0 |
| topiq_existing_flive_grid_s06 | overall | 1413 | 5.082671 | 6.728207 | 0.822871 | 0.879300 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s07 | overall | 1413 | 5.396024 | 6.900551 | 0.823029 | 0.879964 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s08 | overall | 1413 | 5.922089 | 7.388898 | 0.821567 | 0.878699 | 0 | 0 | 0 |

## 8. Dataset-wise Signal Metrics
- Best flive SRCC: `musiq_koniq_probe_flive_grid_s08` SRCC `0.6282`.
- Best koniq10k SRCC: `topiq_existing_flive_grid_s08` SRCC `0.8628`.
- Best spaq SRCC: `topiq_existing_flive_50_50` SRCC `0.8772`.
| signal | dataset | n | mae | rmse | srcc | plcc | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| musiq_koniq_probe_flive_grid_s08 | flive | 399 | 16.274934 | 17.058508 | 0.628234 | 0.743617 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_grid_s07 | flive | 399 | 14.418663 | 15.205237 | 0.625953 | 0.743745 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_grid_s06 | flive | 399 | 12.583564 | 13.369472 | 0.623424 | 0.743660 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_57_43 | flive | 399 | 12.062144 | 12.849235 | 0.623170 | 0.743631 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_50_50 | flive | 399 | 10.770669 | 11.559558 | 0.622035 | 0.743562 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_grid_s05 | flive | 399 | 10.770669 | 11.559558 | 0.622035 | 0.743562 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s08 | flive | 399 | 21.658939 | 22.308830 | 0.621905 | 0.743534 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_40_60 | flive | 399 | 8.976759 | 9.789841 | 0.620918 | 0.743476 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_grid_s04 | flive | 399 | 8.976759 | 9.789841 | 0.620918 | 0.743476 | 0 | 0 | 0 |
| musiq_flive_probe_flive_grid_s08 | flive | 399 | 17.291335 | 18.047032 | 0.620744 | 0.743310 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s07 | flive | 399 | 19.119872 | 19.774349 | 0.620546 | 0.743430 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_grid_s03 | flive | 399 | 7.218804 | 8.086758 | 0.620254 | 0.743404 | 0 | 0 | 0 |
| musiq_flive_probe_flive_grid_s07 | flive | 399 | 15.302407 | 16.064550 | 0.620137 | 0.743297 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s06 | flive | 399 | 16.586014 | 17.254311 | 0.619941 | 0.743372 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_57_43 | flive | 399 | 15.863210 | 16.537877 | 0.619750 | 0.743360 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_grid_s02 | flive | 399 | 5.556077 | 6.502873 | 0.619592 | 0.743346 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_50_50 | flive | 399 | 14.061847 | 14.756116 | 0.619513 | 0.743335 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s05 | flive | 399 | 14.061847 | 14.756116 | 0.619513 | 0.743335 | 0 | 0 | 0 |
| musiq_flive_probe_flive_grid_s06 | flive | 399 | 13.330475 | 14.098921 | 0.619505 | 0.743286 | 0 | 0 | 0 |
| musiq_flive_probe_flive_57_43 | flive | 399 | 12.770594 | 13.541437 | 0.619323 | 0.743283 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_40_60 | flive | 399 | 11.567381 | 12.293088 | 0.619219 | 0.743309 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s04 | flive | 399 | 11.567381 | 12.293088 | 0.619219 | 0.743309 | 0 | 0 | 0 |
| musiq_flive_probe_flive_50_50 | flive | 399 | 11.383124 | 12.158322 | 0.619199 | 0.743277 | 0 | 0 | 0 |
| musiq_flive_probe_flive_grid_s05 | flive | 399 | 11.383124 | 12.158322 | 0.619199 | 0.743277 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s03 | flive | 399 | 9.112175 | 9.891534 | 0.619166 | 0.743290 | 0 | 0 | 0 |
| musiq_flive_probe_flive_40_60 | flive | 399 | 9.464069 | 10.256969 | 0.619156 | 0.743271 | 0 | 0 | 0 |
| musiq_flive_probe_flive_grid_s04 | flive | 399 | 9.464069 | 10.256969 | 0.619156 | 0.743271 | 0 | 0 | 0 |
| musiq_flive_probe_flive_grid_s03 | flive | 399 | 7.571404 | 8.421486 | 0.619100 | 0.743266 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s02 | flive | 399 | 6.731048 | 7.609876 | 0.619003 | 0.743276 | 0 | 0 | 0 |
| musiq_flive_probe_flive_grid_s02 | flive | 399 | 5.770751 | 6.706178 | 0.618978 | 0.743262 | 0 | 0 | 0 |
| flive_existing | flive | 399 | 3.275994 | 4.252275 | 0.618878 | 0.743256 | 0 | 0 | 0 |
| student_14105_flive_grid_s02 | flive | 399 | 3.832264 | 4.813656 | 0.611501 | 0.742521 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s02 | flive | 399 | 3.638059 | 4.586442 | 0.602252 | 0.736852 | 0 | 0 | 0 |
| student_14105_flive_grid_s03 | flive | 399 | 4.232692 | 5.239682 | 0.594969 | 0.733560 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s03 | flive | 399 | 3.905745 | 4.893409 | 0.582598 | 0.717135 | 0 | 0 | 0 |
| student_14105_flive_40_60 | flive | 399 | 4.721539 | 5.735510 | 0.571913 | 0.716752 | 0 | 0 | 0 |
| student_14105_flive_grid_s04 | flive | 399 | 4.721539 | 5.735510 | 0.571913 | 0.716752 | 0 | 0 | 0 |
| topiq_existing_flive_40_60 | flive | 399 | 4.199821 | 5.273604 | 0.559612 | 0.688852 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s04 | flive | 399 | 4.199821 | 5.273604 | 0.559612 | 0.688852 | 0 | 0 | 0 |
| student_14105_flive_50_50 | flive | 399 | 5.262578 | 6.284642 | 0.543093 | 0.690788 | 0 | 0 | 0 |
| student_14105_flive_grid_s05 | flive | 399 | 5.262578 | 6.284642 | 0.543093 | 0.690788 | 0 | 0 | 0 |
| topiq_existing_flive_50_50 | flive | 399 | 4.541711 | 5.712427 | 0.534567 | 0.654772 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s05 | flive | 399 | 4.541711 | 5.712427 | 0.534567 | 0.654772 | 0 | 0 | 0 |
| student_14105_flive_57_43 | flive | 399 | 5.667418 | 6.702310 | 0.519806 | 0.666249 | 0 | 0 | 0 |
| topiq_existing_flive_57_43 | flive | 399 | 4.800907 | 6.054776 | 0.518827 | 0.628403 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s06 | flive | 399 | 4.906977 | 6.197435 | 0.512233 | 0.617613 | 0 | 0 | 0 |
| student_14105_flive_grid_s06 | flive | 399 | 5.833403 | 6.874315 | 0.508860 | 0.655036 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s07 | flive | 399 | 5.291146 | 6.718633 | 0.487189 | 0.579604 | 0 | 0 | 0 |
| student_14105_flive_grid_s07 | flive | 399 | 6.429522 | 7.494966 | 0.472582 | 0.609879 | 0 | 0 | 0 |
| production_existing | flive | 399 | 15.910847 | 16.992212 | 0.464870 | 0.566273 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s08 | flive | 399 | 5.699176 | 7.268242 | 0.463889 | 0.542339 | 0 | 0 | 0 |
| student_14105_flive_grid_s08 | flive | 399 | 7.032814 | 8.139513 | 0.433163 | 0.556824 | 0 | 0 | 0 |
| topiq_existing | flive | 399 | 6.560740 | 8.430203 | 0.419861 | 0.473589 | 0 | 0 | 0 |
| koniq_existing | flive | 399 | 26.811376 | 28.311653 | 0.372075 | 0.426459 | 0 | 0 | 44 |
| student_14105 | flive | 399 | 8.265818 | 9.480637 | 0.346686 | 0.436994 | 0 | 0 | 0 |
| musiq_flive_probe | flive | 399 | 21.281355 | 22.044335 | 0.125559 | 0.065104 | 0 | 0 | 0 |
| musiq_koniq_probe | flive | 399 | 20.009991 | 20.798816 | 0.112731 | 0.039289 | 0 | 0 | 0 |
| musiq_spaq_probe | flive | 399 | 26.748936 | 27.405072 | 0.098744 | 0.046701 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s08 | koniq10k | 901 | 5.458768 | 6.618695 | 0.862785 | 0.896691 | 0 | 0 | 0 |
| topiq_existing | koniq10k | 901 | 7.586039 | 9.037915 | 0.861878 | 0.895037 | 0 | 0 | 0 |

## 9. Hybrid Ratio Results
| signal | overall_n | overall_mae | overall_srcc | overall_plcc | min_dataset_srcc | fp_count | severe_fp_count | composite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| topiq_existing_flive_57_43 | 1413 | 5.039497 | 0.822178 | 0.878604 | 0.518827 | 0 | 0 | 0.951885 |
| topiq_existing_flive_grid_s06 | 1413 | 5.082671 | 0.822871 | 0.879300 | 0.512233 | 0 | 0 | 0.950929 |
| topiq_existing_flive_50_50 | 1413 | 5.037330 | 0.818939 | 0.875522 | 0.534567 | 1 | 0 | 0.950580 |
| topiq_existing_flive_grid_s05 | 1413 | 5.037330 | 0.818939 | 0.875522 | 0.534567 | 1 | 0 | 0.950580 |
| topiq_existing_flive_40_60 | 1413 | 5.222828 | 0.810007 | 0.866805 | 0.559612 | 1 | 0 | 0.947910 |
| topiq_existing_flive_grid_s04 | 1413 | 5.222828 | 0.810007 | 0.866805 | 0.559612 | 1 | 0 | 0.947910 |
| topiq_existing_flive_grid_s07 | 1413 | 5.396024 | 0.823029 | 0.879964 | 0.487189 | 0 | 0 | 0.944826 |
| topiq_existing_flive_grid_s08 | 1413 | 5.922089 | 0.821567 | 0.878699 | 0.463889 | 0 | 0 | 0.937539 |
| topiq_existing_flive_grid_s03 | 1413 | 5.629452 | 0.792773 | 0.850395 | 0.582598 | 4 | 1 | 0.920423 |
| topiq_existing_flive_grid_s02 | 1413 | 6.189151 | 0.763401 | 0.822308 | 0.602252 | 16 | 1 | 0.871964 |
| student_14105_flive_50_50 | 1413 | 6.635737 | 0.719986 | 0.832624 | 0.543093 | 8 | 0 | 0.839760 |
| student_14105_flive_grid_s05 | 1413 | 6.635737 | 0.719986 | 0.832624 | 0.543093 | 8 | 0 | 0.839760 |
| student_14105_flive_57_43 | 1413 | 6.643717 | 0.720622 | 0.831673 | 0.519806 | 8 | 0 | 0.834573 |
| student_14105_flive_grid_s06 | 1413 | 6.658906 | 0.720119 | 0.830336 | 0.508860 | 8 | 0 | 0.831334 |
| student_14105_flive_40_60 | 1413 | 6.680838 | 0.714672 | 0.827214 | 0.571913 | 11 | 1 | 0.825650 |
| student_14105_flive_grid_s04 | 1413 | 6.680838 | 0.714672 | 0.827214 | 0.571913 | 11 | 1 | 0.825650 |
| student_14105_flive_grid_s07 | 1413 | 6.748297 | 0.715612 | 0.822127 | 0.472582 | 6 | 0 | 0.821758 |
| student_14105_flive_grid_s03 | 1413 | 6.800317 | 0.703579 | 0.812575 | 0.594969 | 13 | 1 | 0.816321 |
| student_14105_flive_grid_s08 | 1413 | 6.924262 | 0.707325 | 0.809711 | 0.433163 | 3 | 0 | 0.809616 |
| musiq_spaq_probe_flive_grid_s08 | 1413 | 17.142213 | 0.638520 | 0.709458 | 0.621905 | 0 | 0 | 0.793996 |

## 10. Safety Cases
- Safety case rows: `557`.
- Production severe FP total: `0`.
- Selected candidate severe FP total: `0`.

## 11. Best Signal / Hybrid Recommendation
- Selected candidate: `topiq_existing_flive_57_43`.
- Dataset stability pass: `True`.
- Dataset deltas vs production: `[{"dataset": "flive", "srcc_delta_vs_production": 0.05395719197491222}, {"dataset": "koniq10k", "srcc_delta_vs_production": 0.0853847986166878}, {"dataset": "spaq", "srcc_delta_vs_production": 0.03704134974358919}]`.
- Recommendation status: log-only only if decision says ready; never production replacement from this phase.

## 12. Remaining Coverage Gaps
- Primary test coverage target recovered for KonIQ, FLIVE, SPAQ, and overall.
- MUSIQ signals are local probe models, not official Kaggle MUSIQ; interpret separately.

## 13. Final Decision
TECHNICAL_SIGNAL_COMPARISON_READY_FOR_LOG_ONLY_PROFILING

## 14. Recommended Next Step
- Prepare log-only profiling for selected candidate; keep production `technical_score` unchanged.
