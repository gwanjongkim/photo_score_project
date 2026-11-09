# Technical IQA Signal Comparison and Coverage Recovery Report

## 1. Executive Summary
- Final decision: `TECHNICAL_SIGNAL_COMPARISON_NEEDS_MORE_COVERAGE`.
- Common rows after recovery: `30`.
- Best overall SRCC signal: `topiq_existing_flive_grid_s07`.
- Best overall MAE signal: `topiq_existing_flive_grid_s08`.
- Best hybrid composite: `topiq_existing_flive_grid_s08`.
- Runtime seconds: `4.02`.

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
| overall | 30 | 596 | 30 | 1.000000 |
| flive | 10 | 399 | 10 | 1.000000 |
| koniq10k | 10 | 84 | 10 | 1.000000 |
| spaq | 10 | 113 | 10 | 1.000000 |

## 6. Benchmark Scope
- Primary split: student 14,105 `test_manifest.csv` only.
- Recovered full test coverage by running existing local TFLite/Keras-compatible assets in WSL.
- Production score is reconstructed offline from recovered KonIQ/FLIVE image-level scores; production code unchanged.

## 7. Overall Signal Metrics
| signal | dataset | n | mae | rmse | srcc | plcc | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production_existing | overall | 30 | 14.232808 | 16.244981 | 0.558176 | 0.646920 | 0 | 0 | 0 |
| koniq_existing | overall | 30 | 23.502382 | 26.453700 | 0.531479 | 0.625074 | 0 | 0 | 4 |
| flive_existing | overall | 30 | 10.331772 | 15.118135 | 0.628476 | 0.579021 | 4 | 2 | 0 |
| topiq_existing | overall | 30 | 7.899250 | 9.321060 | 0.801112 | 0.890710 | 0 | 0 | 0 |
| student_14105 | overall | 30 | 9.054733 | 10.516268 | 0.596440 | 0.818710 | 0 | 0 | 0 |
| musiq_koniq_probe | overall | 30 | 17.595537 | 19.281625 | 0.127920 | 0.245981 | 0 | 0 | 0 |
| musiq_flive_probe | overall | 30 | 18.454949 | 20.058219 | 0.241629 | 0.151685 | 0 | 0 | 0 |
| musiq_spaq_probe | overall | 30 | 22.140671 | 23.842620 | 0.136819 | 0.222942 | 0 | 0 | 0 |
| musiq_flive_probe_flive_40_60 | overall | 30 | 11.421074 | 14.412126 | 0.628476 | 0.579052 | 2 | 0 | 0 |
| musiq_flive_probe_flive_50_50 | overall | 30 | 12.245939 | 14.876890 | 0.628476 | 0.579067 | 1 | 0 | 0 |
| musiq_flive_probe_flive_57_43 | overall | 30 | 12.973321 | 15.351064 | 0.628476 | 0.579081 | 0 | 0 | 0 |
| musiq_flive_probe_flive_grid_s02 | overall | 30 | 10.336672 | 14.250403 | 0.628476 | 0.579033 | 2 | 1 | 0 |
| musiq_flive_probe_flive_grid_s03 | overall | 30 | 10.791148 | 14.199533 | 0.628476 | 0.579041 | 2 | 1 | 0 |
| musiq_flive_probe_flive_grid_s04 | overall | 30 | 11.421074 | 14.412126 | 0.628476 | 0.579052 | 2 | 0 | 0 |
| musiq_flive_probe_flive_grid_s05 | overall | 30 | 12.245939 | 14.876890 | 0.628476 | 0.579067 | 1 | 0 | 0 |
| musiq_flive_probe_flive_grid_s06 | overall | 30 | 13.307781 | 15.571262 | 0.628476 | 0.579089 | 0 | 0 | 0 |
| musiq_flive_probe_flive_grid_s07 | overall | 30 | 14.494732 | 16.466220 | 0.628476 | 0.579125 | 0 | 0 | 0 |
| musiq_flive_probe_flive_grid_s08 | overall | 30 | 15.793010 | 17.531071 | 0.628476 | 0.579194 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_40_60 | overall | 30 | 11.274402 | 14.381025 | 0.628476 | 0.580475 | 2 | 0 | 0 |
| musiq_koniq_probe_flive_50_50 | overall | 30 | 11.977917 | 14.759510 | 0.628921 | 0.581192 | 1 | 0 | 0 |
| musiq_koniq_probe_flive_57_43 | overall | 30 | 12.603533 | 15.157621 | 0.628921 | 0.581904 | 1 | 0 | 0 |
| musiq_koniq_probe_flive_grid_s02 | overall | 30 | 10.333910 | 14.300829 | 0.628476 | 0.579569 | 2 | 1 | 0 |
| musiq_koniq_probe_flive_grid_s03 | overall | 30 | 10.699458 | 14.225707 | 0.628476 | 0.579958 | 2 | 1 | 0 |
| musiq_koniq_probe_flive_grid_s04 | overall | 30 | 11.274402 | 14.381025 | 0.628476 | 0.580475 | 2 | 0 | 0 |
| musiq_koniq_probe_flive_grid_s05 | overall | 30 | 11.977917 | 14.759510 | 0.628921 | 0.581192 | 1 | 0 | 0 |
| musiq_koniq_probe_flive_grid_s06 | overall | 30 | 12.884184 | 15.344657 | 0.628921 | 0.582257 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_grid_s07 | overall | 30 | 13.984195 | 16.113968 | 0.628921 | 0.584000 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_grid_s08 | overall | 30 | 15.157787 | 17.042522 | 0.643159 | 0.587370 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_40_60 | overall | 30 | 12.232086 | 14.730184 | 0.628476 | 0.579360 | 1 | 0 | 0 |
| musiq_spaq_probe_flive_50_50 | overall | 30 | 13.611148 | 15.646785 | 0.628476 | 0.579529 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_57_43 | overall | 30 | 14.720565 | 16.499634 | 0.628476 | 0.579698 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s02 | overall | 30 | 10.469538 | 14.077549 | 0.628476 | 0.579148 | 2 | 1 | 0 |
| musiq_spaq_probe_flive_grid_s03 | overall | 30 | 11.232907 | 14.192609 | 0.628476 | 0.579239 | 2 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s04 | overall | 30 | 12.232086 | 14.730184 | 0.628476 | 0.579360 | 1 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s05 | overall | 30 | 13.611148 | 15.646785 | 0.628476 | 0.579529 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s06 | overall | 30 | 15.186254 | 16.880783 | 0.628476 | 0.579782 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s07 | overall | 30 | 16.832267 | 18.368319 | 0.628476 | 0.580202 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s08 | overall | 30 | 18.601735 | 20.053052 | 0.628921 | 0.581037 | 0 | 0 | 0 |
| student_14105_flive_40_60 | overall | 30 | 8.540167 | 11.203579 | 0.758843 | 0.830465 | 1 | 0 | 0 |
| student_14105_flive_50_50 | overall | 30 | 8.296632 | 10.554450 | 0.735706 | 0.840004 | 0 | 0 | 0 |
| student_14105_flive_57_43 | overall | 30 | 8.161825 | 10.205763 | 0.717909 | 0.841568 | 0 | 0 | 0 |
| student_14105_flive_grid_s02 | overall | 30 | 9.299381 | 12.951082 | 0.722358 | 0.762188 | 2 | 1 | 0 |
| student_14105_flive_grid_s03 | overall | 30 | 8.898598 | 12.012419 | 0.748610 | 0.807116 | 1 | 0 | 0 |
| student_14105_flive_grid_s04 | overall | 30 | 8.540167 | 11.203579 | 0.758843 | 0.830465 | 1 | 0 | 0 |
| student_14105_flive_grid_s05 | overall | 30 | 8.296632 | 10.554450 | 0.735706 | 0.840004 | 0 | 0 | 0 |
| student_14105_flive_grid_s06 | overall | 30 | 8.107902 | 10.095888 | 0.702336 | 0.841366 | 0 | 0 | 0 |
| student_14105_flive_grid_s07 | overall | 30 | 8.059788 | 9.854530 | 0.693437 | 0.838159 | 0 | 0 | 0 |
| student_14105_flive_grid_s08 | overall | 30 | 8.217464 | 9.846363 | 0.682759 | 0.832560 | 0 | 0 | 0 |
| topiq_existing_flive_40_60 | overall | 30 | 8.262860 | 11.021594 | 0.801557 | 0.841602 | 0 | 0 | 0 |
| topiq_existing_flive_50_50 | overall | 30 | 8.059878 | 10.257068 | 0.813126 | 0.863132 | 0 | 0 | 0 |
| topiq_existing_flive_57_43 | overall | 30 | 7.939664 | 9.806091 | 0.820690 | 0.873283 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s02 | overall | 30 | 8.935929 | 12.907933 | 0.761513 | 0.756554 | 2 | 0 | 0 |
| topiq_existing_flive_grid_s03 | overall | 30 | 8.590638 | 11.914052 | 0.778420 | 0.807840 | 1 | 0 | 0 |
| topiq_existing_flive_grid_s04 | overall | 30 | 8.262860 | 11.021594 | 0.801557 | 0.841602 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s05 | overall | 30 | 8.059878 | 10.257068 | 0.813126 | 0.863132 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s06 | overall | 30 | 7.891579 | 9.650925 | 0.820245 | 0.876421 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s07 | overall | 30 | 7.723280 | 9.234408 | 0.824694 | 0.884249 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s08 | overall | 30 | 7.593263 | 9.033783 | 0.816685 | 0.888485 | 0 | 0 | 0 |

## 8. Dataset-wise Signal Metrics
- Best flive SRCC: `musiq_flive_probe_flive_50_50` SRCC `0.8303`.
- Best koniq10k SRCC: `student_14105` SRCC `0.5273`.
- Best spaq SRCC: `student_14105_flive_57_43` SRCC `0.8788`.
| signal | dataset | n | mae | rmse | srcc | plcc | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flive_existing | flive | 10 | 3.649619 | 5.067580 | 0.830303 | 0.648661 | 0 | 0 | 0 |
| musiq_flive_probe_flive_40_60 | flive | 10 | 9.818597 | 10.357787 | 0.830303 | 0.648663 | 0 | 0 | 0 |
| musiq_flive_probe_flive_50_50 | flive | 10 | 11.480109 | 12.282612 | 0.830303 | 0.648664 | 0 | 0 | 0 |
| musiq_flive_probe_flive_57_43 | flive | 10 | 12.885761 | 13.699143 | 0.830303 | 0.648665 | 0 | 0 | 0 |
| musiq_flive_probe_flive_grid_s02 | flive | 10 | 6.495573 | 6.923849 | 0.830303 | 0.648661 | 0 | 0 | 0 |
| musiq_flive_probe_flive_grid_s03 | flive | 10 | 8.157085 | 8.542606 | 0.830303 | 0.648662 | 0 | 0 | 0 |
| musiq_flive_probe_flive_grid_s04 | flive | 10 | 9.818597 | 10.357787 | 0.830303 | 0.648663 | 0 | 0 | 0 |
| musiq_flive_probe_flive_grid_s05 | flive | 10 | 11.480109 | 12.282612 | 0.830303 | 0.648664 | 0 | 0 | 0 |
| musiq_flive_probe_flive_grid_s06 | flive | 10 | 13.477031 | 14.272791 | 0.830303 | 0.648666 | 0 | 0 | 0 |
| musiq_flive_probe_flive_grid_s07 | flive | 10 | 15.546474 | 16.304409 | 0.830303 | 0.648669 | 0 | 0 | 0 |
| musiq_flive_probe_flive_grid_s08 | flive | 10 | 17.615918 | 18.363718 | 0.830303 | 0.648674 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_40_60 | flive | 10 | 9.395695 | 9.883641 | 0.830303 | 0.648638 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_50_50 | flive | 10 | 10.951482 | 11.671238 | 0.830303 | 0.648627 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_57_43 | flive | 10 | 12.130139 | 12.991073 | 0.830303 | 0.648616 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_grid_s02 | flive | 10 | 6.284122 | 6.727511 | 0.830303 | 0.648652 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_grid_s03 | flive | 10 | 7.839909 | 8.207734 | 0.830303 | 0.648646 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_grid_s04 | flive | 10 | 9.395695 | 9.883641 | 0.830303 | 0.648638 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_grid_s05 | flive | 10 | 10.951482 | 11.671238 | 0.830303 | 0.648627 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_grid_s06 | flive | 10 | 12.683628 | 13.526315 | 0.830303 | 0.648610 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_grid_s07 | flive | 10 | 14.620838 | 15.424545 | 0.830303 | 0.648583 | 0 | 0 | 0 |
| musiq_koniq_probe_flive_grid_s08 | flive | 10 | 16.558047 | 17.351771 | 0.830303 | 0.648526 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_40_60 | flive | 10 | 11.588056 | 12.388824 | 0.830303 | 0.648573 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_50_50 | flive | 10 | 14.172044 | 14.885683 | 0.830303 | 0.648529 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_57_43 | flive | 10 | 16.045141 | 16.705685 | 0.830303 | 0.648485 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s02 | flive | 10 | 7.380302 | 7.788562 | 0.830303 | 0.648628 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s03 | flive | 10 | 9.484179 | 9.992114 | 0.830303 | 0.648604 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s04 | flive | 10 | 11.588056 | 12.388824 | 0.830303 | 0.648573 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s05 | flive | 10 | 14.172044 | 14.885683 | 0.830303 | 0.648529 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s06 | flive | 10 | 16.794379 | 17.439729 | 0.830303 | 0.648464 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s07 | flive | 10 | 19.416714 | 20.029097 | 0.830303 | 0.648354 | 0 | 0 | 0 |
| musiq_spaq_probe_flive_grid_s08 | flive | 10 | 22.039049 | 22.641672 | 0.830303 | 0.648133 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s02 | flive | 10 | 4.259851 | 5.478757 | 0.769697 | 0.634049 | 0 | 0 | 0 |
| student_14105_flive_grid_s02 | flive | 10 | 4.659910 | 5.545418 | 0.757576 | 0.626527 | 0 | 0 | 0 |
| student_14105_flive_grid_s03 | flive | 10 | 5.403591 | 6.017454 | 0.757576 | 0.609243 | 0 | 0 | 0 |
| topiq_existing_flive_40_60 | flive | 10 | 5.259119 | 6.255929 | 0.709091 | 0.608043 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s03 | flive | 10 | 4.737477 | 5.829132 | 0.709091 | 0.622617 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s04 | flive | 10 | 5.259119 | 6.255929 | 0.709091 | 0.608043 | 0 | 0 | 0 |
| student_14105_flive_40_60 | flive | 10 | 6.147272 | 6.604253 | 0.696970 | 0.585815 | 0 | 0 | 0 |
| student_14105_flive_grid_s04 | flive | 10 | 6.147272 | 6.604253 | 0.696970 | 0.585815 | 0 | 0 | 0 |
| topiq_existing_flive_50_50 | flive | 10 | 5.780762 | 6.744655 | 0.648485 | 0.590128 | 0 | 0 | 0 |
| topiq_existing_flive_57_43 | flive | 10 | 6.153364 | 7.124669 | 0.648485 | 0.575234 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s05 | flive | 10 | 5.780762 | 6.744655 | 0.648485 | 0.590128 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s06 | flive | 10 | 6.302405 | 7.282854 | 0.648485 | 0.568789 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s07 | flive | 10 | 6.824047 | 7.860370 | 0.587879 | 0.544086 | 0 | 0 | 0 |
| student_14105_flive_50_50 | flive | 10 | 6.890953 | 7.278108 | 0.563636 | 0.554203 | 0 | 0 | 0 |
| student_14105_flive_grid_s05 | flive | 10 | 6.890953 | 7.278108 | 0.563636 | 0.554203 | 0 | 0 | 0 |
| production_existing | flive | 10 | 16.950820 | 18.147300 | 0.515152 | 0.577339 | 0 | 0 | 0 |
| topiq_existing_flive_grid_s08 | flive | 10 | 7.345690 | 8.469164 | 0.478788 | 0.516242 | 0 | 0 | 0 |
| student_14105_flive_57_43 | flive | 10 | 7.422153 | 7.800361 | 0.466667 | 0.525286 | 0 | 0 | 0 |
| student_14105_flive_grid_s06 | flive | 10 | 7.634634 | 8.017098 | 0.466667 | 0.512000 | 0 | 0 | 0 |
| student_14105_flive_grid_s07 | flive | 10 | 8.378314 | 8.804838 | 0.442424 | 0.456780 | 0 | 0 | 0 |
| topiq_existing | flive | 10 | 8.388975 | 9.756894 | 0.430303 | 0.452762 | 0 | 0 | 0 |
| koniq_existing | flive | 10 | 28.868657 | 30.207262 | 0.393939 | 0.487443 | 0 | 0 | 2 |
| musiq_flive_probe | flive | 10 | 21.754805 | 22.534961 | 0.321212 | 0.357773 | 0 | 0 | 0 |
| student_14105_flive_grid_s08 | flive | 10 | 9.121995 | 9.629371 | 0.260606 | 0.386870 | 0 | 0 | 0 |
| musiq_koniq_probe | flive | 10 | 20.432466 | 21.261576 | 0.163636 | -0.090468 | 0 | 0 | 0 |
| student_14105 | flive | 10 | 10.609357 | 11.356447 | -0.006061 | 0.206867 | 0 | 0 | 0 |
| musiq_spaq_probe | flive | 10 | 27.283718 | 27.910327 | -0.175758 | -0.370395 | 0 | 0 | 0 |
| student_14105 | koniq10k | 10 | 9.146092 | 11.252965 | 0.527273 | 0.769806 | 0 | 0 | 0 |
| student_14105_flive_grid_s06 | koniq10k | 10 | 7.529240 | 10.776272 | 0.527273 | 0.791092 | 0 | 0 | 0 |

## 9. Hybrid Ratio Results
| signal | overall_n | overall_mae | overall_srcc | overall_plcc | min_dataset_srcc | fp_count | severe_fp_count | composite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| topiq_existing_flive_grid_s08 | 30 | 7.593263 | 0.816685 | 0.888485 | 0.466667 | 0 | 0 | 0.933352 |
| topiq_existing_flive_grid_s07 | 30 | 7.723280 | 0.824694 | 0.884249 | 0.430303 | 0 | 0 | 0.932270 |
| topiq_existing_flive_50_50 | 30 | 8.059878 | 0.813126 | 0.863132 | 0.381818 | 0 | 0 | 0.908580 |
| topiq_existing_flive_grid_s05 | 30 | 8.059878 | 0.813126 | 0.863132 | 0.381818 | 0 | 0 | 0.908580 |
| topiq_existing_flive_57_43 | 30 | 7.939664 | 0.820690 | 0.873283 | 0.345455 | 0 | 0 | 0.907053 |
| topiq_existing_flive_grid_s06 | 30 | 7.891579 | 0.820245 | 0.876421 | 0.345455 | 0 | 0 | 0.906608 |
| topiq_existing_flive_40_60 | 30 | 8.262860 | 0.801557 | 0.841602 | 0.406061 | 0 | 0 | 0.903072 |
| topiq_existing_flive_grid_s04 | 30 | 8.262860 | 0.801557 | 0.841602 | 0.406061 | 0 | 0 | 0.903072 |
| topiq_existing_flive_grid_s03 | 30 | 8.590638 | 0.778420 | 0.807840 | 0.430303 | 1 | 0 | 0.883996 |
| student_14105_flive_40_60 | 30 | 8.540167 | 0.758843 | 0.830465 | 0.393939 | 1 | 0 | 0.855328 |
| student_14105_flive_grid_s04 | 30 | 8.540167 | 0.758843 | 0.830465 | 0.393939 | 1 | 0 | 0.855328 |
| topiq_existing_flive_grid_s02 | 30 | 8.935929 | 0.761513 | 0.756554 | 0.357576 | 2 | 0 | 0.846907 |
| student_14105_flive_grid_s03 | 30 | 8.898598 | 0.748610 | 0.807116 | 0.393939 | 1 | 0 | 0.845094 |
| student_14105_flive_57_43 | 30 | 8.161825 | 0.717909 | 0.841568 | 0.466667 | 0 | 0 | 0.834575 |
| student_14105_flive_50_50 | 30 | 8.296632 | 0.735706 | 0.840004 | 0.393939 | 0 | 0 | 0.834191 |
| student_14105_flive_grid_s05 | 30 | 8.296632 | 0.735706 | 0.840004 | 0.393939 | 0 | 0 | 0.834191 |
| student_14105_flive_grid_s06 | 30 | 8.107902 | 0.702336 | 0.841366 | 0.466667 | 0 | 0 | 0.819003 |
| student_14105_flive_grid_s07 | 30 | 8.059788 | 0.693437 | 0.838159 | 0.442424 | 0 | 0 | 0.804043 |
| student_14105_flive_grid_s02 | 30 | 9.299381 | 0.722358 | 0.762188 | 0.296970 | 2 | 1 | 0.782601 |
| student_14105_flive_grid_s08 | 30 | 8.217464 | 0.682759 | 0.832560 | 0.260606 | 0 | 0 | 0.747910 |

## 10. Safety Cases
- Safety case rows: `51`.
- Production severe FP total: `0`.
- Selected candidate severe FP total: `0`.

## 11. Best Signal / Hybrid Recommendation
- Selected candidate: `topiq_existing_flive_grid_s08`.
- Dataset stability pass: `False`.
- Dataset deltas vs production: `[{"dataset": "flive", "srcc_delta_vs_production": -0.03636363636363643}, {"dataset": "koniq10k", "srcc_delta_vs_production": 0.2545454545454545}, {"dataset": "spaq", "srcc_delta_vs_production": 0.2545454545454545}]`.
- Recommendation status: log-only only if decision says ready; never production replacement from this phase.

## 12. Remaining Coverage Gaps
- Primary test coverage target recovered for KonIQ, FLIVE, SPAQ, and overall.
- MUSIQ signals are local probe models, not official Kaggle MUSIQ; interpret separately.

## 13. Final Decision
TECHNICAL_SIGNAL_COMPARISON_NEEDS_MORE_COVERAGE

## 14. Recommended Next Step
- Resolve remaining coverage or safety gaps before any candidate recommendation.
