# Aggressive Mobile Aesthetic Ensemble Weight Search

## 1. Executive summary

- Research-best: `group_a` / `calibrated` / `combined_70_30` with weights `{'nima': 0.37, 'rgnet': 0.41, 'alamp': 0.22, 'icaa': 0.0}`.
- Research-best held-out combined SRCC: `0.659554`.
- Production candidate: `group_a` / `calibrated` / `combined_70_30` with weights `{'nima': 0.37, 'rgnet': 0.41, 'alamp': 0.22, 'icaa': 0.0}`.
- ICAA decision: `keep_icaa_as_separate_color_subscore`. Group B improves Group A by less than +0.005 combined SRCC.

## 2. Data source and row counts

| dataset | path | rows_after_dropna | train_rows | validation_rows | heldout_test_rows |
| --- | --- | --- | --- | --- | --- |
| ava | outputs/aesthetic_cross_predictions_for_weight_search_20260603/aligned_ava_complete_primary.csv | 25548 | 15329 | 5110 | 5109 |
| aadb | outputs/aesthetic_cross_predictions_for_weight_search_20260603/aligned_aadb_complete_primary.csv | 1000 | 600 | 200 | 200 |

## 3. Score column and scale verification

| dataset | gt_score_min_original | gt_score_max_original | gt_scale_note | gt_score_min_1_10 | gt_score_max_1_10 | nima_min | nima_max | rgnet_min | rgnet_max | alamp_min | alamp_max | icaa_min | icaa_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ava | 1.988764 | 8.090580 | already_1_10 | 1.988764 | 8.090580 | 2.607785 | 6.999500 | 0.369778 | 9.569139 | 0.007083 | 10.000000 | 3.612465 | 7.753280 |
| aadb | 0.050000 | 1.000000 | converted_gt_score_0_1_to_1_10_by_multiply_10 | 0.500000 | 10.000000 | 3.126592 | 5.925374 | 0.555421 | 9.314902 | 0.030743 | 9.937122 | 3.918131 | 7.519248 |

## 4. Baseline single-model metrics on AVA and AADB

| prep | dataset | model | n | srcc | plcc | mae | mse | binary_accuracy_at_5 | f1_at_5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibrated | aadb | alamp | 200 | 0.274131 | 0.288720 | 1.682343 | 4.028873 | 0.580000 | 0.611111 |
| calibrated | aadb | icaa | 200 | 0.284780 | 0.299212 | 1.632441 | 4.006651 | 0.635000 | 0.709163 |
| calibrated | aadb | nima | 200 | 0.439065 | 0.449903 | 1.541697 | 3.514898 | 0.640000 | 0.678571 |
| calibrated | aadb | rgnet | 200 | 0.667876 | 0.667578 | 1.238224 | 2.439205 | 0.755000 | 0.787879 |
| calibrated | ava | alamp | 5109 | 0.579290 | 0.582412 | 0.467831 | 0.350770 | 0.756117 | 0.836996 |
| calibrated | ava | icaa | 5109 | 0.335819 | 0.353169 | 0.539548 | 0.464617 | 0.733607 | 0.837764 |
| calibrated | ava | nima | 5109 | 0.623022 | 0.638247 | 0.440012 | 0.314874 | 0.780388 | 0.856338 |
| calibrated | ava | rgnet | 5109 | 0.195965 | 0.201092 | 0.562176 | 0.509390 | 0.716774 | 0.834798 |
| raw_scaled | aadb | alamp | 200 | 0.274131 | 0.288720 | 2.739288 | 10.934762 | 0.525000 | 0.431138 |
| raw_scaled | aadb | icaa | 200 | 0.284780 | 0.299212 | 1.722690 | 4.399488 | 0.620000 | 0.756410 |
| raw_scaled | aadb | nima | 200 | 0.439065 | 0.449903 | 1.814295 | 4.852191 | 0.460000 | 0.169231 |
| raw_scaled | aadb | rgnet | 200 | 0.667876 | 0.667578 | 1.233816 | 2.448977 | 0.755000 | 0.786026 |
| raw_scaled | ava | alamp | 5109 | 0.579290 | 0.582412 | 2.030200 | 5.702858 | 0.634762 | 0.681896 |
| raw_scaled | ava | icaa | 5109 | 0.335819 | 0.353169 | 0.787437 | 0.910729 | 0.725582 | 0.837355 |
| raw_scaled | ava | nima | 5109 | 0.623022 | 0.638247 | 0.632733 | 0.595467 | 0.620669 | 0.659163 |
| raw_scaled | ava | rgnet | 5109 | 0.195965 | 0.201092 | 1.350500 | 2.748021 | 0.641417 | 0.743345 |

## 5. Group A best weights

| prep | objective | w_nima | w_rgnet | w_alamp | w_icaa | test_ava_srcc | test_aadb_srcc | test_combined_70_30_srcc | zero_weight_models |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibrated | combined_70_30 | 0.370000 | 0.410000 | 0.220000 | 0.000000 | 0.653648 | 0.673337 | 0.659554 |  |
| calibrated | aadb_only | 0.400000 | 0.490000 | 0.110000 | 0.000000 | 0.640922 | 0.692192 | 0.656303 |  |
| calibrated | ava_only | 0.460000 | 0.260000 | 0.280000 | 0.000000 | 0.657218 | 0.612353 | 0.643758 |  |
| raw_scaled | combined_70_30 | 0.790000 | 0.130000 | 0.080000 | 0.000000 | 0.644886 | 0.570909 | 0.622693 |  |
| raw_scaled | ava_only | 0.840000 | 0.050000 | 0.110000 | 0.000000 | 0.657209 | 0.480104 | 0.604077 |  |
| raw_scaled | aadb_only | 0.680000 | 0.300000 | 0.020000 | 0.000000 | 0.515764 | 0.691989 | 0.568632 |  |

## 6. Group B best weights unconstrained

| prep | objective | w_nima | w_rgnet | w_alamp | w_icaa | test_ava_srcc | test_aadb_srcc | test_combined_70_30_srcc | zero_weight_models |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.654102 | 0.661366 | 0.656281 |  |
| calibrated | ava_only | 0.460000 | 0.210000 | 0.270000 | 0.060000 | 0.658082 | 0.584544 | 0.636020 |  |
| calibrated | aadb_only | 0.260000 | 0.450000 | 0.050000 | 0.240000 | 0.602896 | 0.674507 | 0.624379 |  |
| raw_scaled | combined_70_30 | 0.660000 | 0.110000 | 0.080000 | 0.150000 | 0.645290 | 0.551949 | 0.617288 |  |
| raw_scaled | ava_only | 0.800000 | 0.040000 | 0.100000 | 0.060000 | 0.658093 | 0.472477 | 0.602408 |  |
| raw_scaled | aadb_only | 0.440000 | 0.270000 | 0.010000 | 0.280000 | 0.472303 | 0.672308 | 0.532304 |  |

## 7. Group B best weights with ICAA caps

| variant | prep | objective | w_nima | w_rgnet | w_alamp | w_icaa | test_ava_srcc | test_aadb_srcc | test_combined_70_30_srcc | zero_weight_models |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| group_b_icaa_cap_015 | calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.654102 | 0.661366 | 0.656281 |  |
| group_b_icaa_cap_020 | calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.654102 | 0.661366 | 0.656281 |  |
| group_b_icaa_cap_025 | calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.654102 | 0.661366 | 0.656281 |  |
| group_b_icaa_cap_015 | calibrated | aadb_only | 0.400000 | 0.420000 | 0.030000 | 0.150000 | 0.627578 | 0.669184 | 0.640060 |  |
| group_b_icaa_cap_020 | calibrated | aadb_only | 0.370000 | 0.390000 | 0.040000 | 0.200000 | 0.626545 | 0.659863 | 0.636540 |  |
| group_b_icaa_cap_025 | calibrated | ava_only | 0.460000 | 0.210000 | 0.270000 | 0.060000 | 0.658082 | 0.584544 | 0.636020 |  |
| group_b_icaa_cap_020 | calibrated | ava_only | 0.460000 | 0.210000 | 0.270000 | 0.060000 | 0.658082 | 0.584544 | 0.636020 |  |
| group_b_icaa_cap_015 | calibrated | ava_only | 0.460000 | 0.210000 | 0.270000 | 0.060000 | 0.658082 | 0.584544 | 0.636020 |  |
| group_b_icaa_cap_025 | calibrated | aadb_only | 0.260000 | 0.450000 | 0.050000 | 0.240000 | 0.602896 | 0.674507 | 0.624379 |  |
| group_b_icaa_cap_020 | raw_scaled | combined_70_30 | 0.660000 | 0.110000 | 0.080000 | 0.150000 | 0.645290 | 0.551949 | 0.617288 |  |
| group_b_icaa_cap_025 | raw_scaled | combined_70_30 | 0.660000 | 0.110000 | 0.080000 | 0.150000 | 0.645290 | 0.551949 | 0.617288 |  |
| group_b_icaa_cap_015 | raw_scaled | combined_70_30 | 0.660000 | 0.110000 | 0.080000 | 0.150000 | 0.645290 | 0.551949 | 0.617288 |  |
| group_b_icaa_cap_020 | raw_scaled | ava_only | 0.800000 | 0.040000 | 0.100000 | 0.060000 | 0.658093 | 0.472477 | 0.602408 |  |
| group_b_icaa_cap_025 | raw_scaled | ava_only | 0.800000 | 0.040000 | 0.100000 | 0.060000 | 0.658093 | 0.472477 | 0.602408 |  |
| group_b_icaa_cap_015 | raw_scaled | ava_only | 0.800000 | 0.040000 | 0.100000 | 0.060000 | 0.658093 | 0.472477 | 0.602408 |  |
| group_b_icaa_cap_015 | raw_scaled | aadb_only | 0.600000 | 0.240000 | 0.010000 | 0.150000 | 0.527060 | 0.668700 | 0.569552 |  |
| group_b_icaa_cap_025 | raw_scaled | aadb_only | 0.520000 | 0.230000 | 0.010000 | 0.240000 | 0.516021 | 0.662145 | 0.559858 |  |
| group_b_icaa_cap_020 | raw_scaled | aadb_only | 0.570000 | 0.240000 | 0.000000 | 0.190000 | 0.508316 | 0.670032 | 0.556831 | alamp |

## 8. Raw-scaled vs calibrated comparison

| prep | test_combined_70_30_srcc | test_ava_srcc | test_aadb_srcc |
| --- | --- | --- | --- |
| calibrated | 0.659554 | 0.658082 | 0.692192 |
| raw_scaled | 0.623088 | 0.658093 | 0.691989 |

Calibration was fit only on the 60% search/train fold, then applied to validation and held-out test.

## 9. AVA-only, AADB-only, and combined-objective winners

| variant | prep | objective | w_nima | w_rgnet | w_alamp | w_icaa | test_ava_srcc | test_aadb_srcc | test_combined_70_30_srcc | test_combined_60_40_srcc | test_combined_50_50_srcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| group_a | calibrated | combined_70_30 | 0.370000 | 0.410000 | 0.220000 | 0.000000 | 0.653648 | 0.673337 | 0.659554 | 0.661523 | 0.663492 |
| group_a | calibrated | aadb_only | 0.400000 | 0.490000 | 0.110000 | 0.000000 | 0.640922 | 0.692192 | 0.656303 | 0.661430 | 0.666557 |
| group_b_icaa_cap_020 | calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.654102 | 0.661366 | 0.656281 | 0.657007 | 0.657734 |
| group_b_icaa_cap_015 | calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.654102 | 0.661366 | 0.656281 | 0.657007 | 0.657734 |
| group_b_icaa_cap_025 | calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.654102 | 0.661366 | 0.656281 | 0.657007 | 0.657734 |
| group_b_unconstrained | calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.654102 | 0.661366 | 0.656281 | 0.657007 | 0.657734 |
| group_a_nnls_stacking | calibrated | combined_70_30 | 0.422725 | 0.371420 | 0.205854 | 0.000000 | 0.654580 | 0.658642 | 0.655798 | 0.656205 | 0.656611 |
| group_b_unconstrained_nnls_stacking | calibrated | combined_70_30 | 0.395209 | 0.320038 | 0.193629 | 0.091123 | 0.655450 | 0.639574 | 0.650687 | 0.649100 | 0.647512 |
| group_a | calibrated | ava_only | 0.460000 | 0.260000 | 0.280000 | 0.000000 | 0.657218 | 0.612353 | 0.643758 | 0.639272 | 0.634785 |
| group_b_icaa_cap_015 | calibrated | aadb_only | 0.400000 | 0.420000 | 0.030000 | 0.150000 | 0.627578 | 0.669184 | 0.640060 | 0.644221 | 0.648381 |
| group_b_icaa_cap_020 | calibrated | aadb_only | 0.370000 | 0.390000 | 0.040000 | 0.200000 | 0.626545 | 0.659863 | 0.636540 | 0.639872 | 0.643204 |
| group_b_icaa_cap_025 | calibrated | ava_only | 0.460000 | 0.210000 | 0.270000 | 0.060000 | 0.658082 | 0.584544 | 0.636020 | 0.628666 | 0.621313 |
| group_b_icaa_cap_020 | calibrated | ava_only | 0.460000 | 0.210000 | 0.270000 | 0.060000 | 0.658082 | 0.584544 | 0.636020 | 0.628666 | 0.621313 |
| group_b_unconstrained | calibrated | ava_only | 0.460000 | 0.210000 | 0.270000 | 0.060000 | 0.658082 | 0.584544 | 0.636020 | 0.628666 | 0.621313 |
| group_b_icaa_cap_015 | calibrated | ava_only | 0.460000 | 0.210000 | 0.270000 | 0.060000 | 0.658082 | 0.584544 | 0.636020 | 0.628666 | 0.621313 |
| group_b_icaa_cap_025 | calibrated | aadb_only | 0.260000 | 0.450000 | 0.050000 | 0.240000 | 0.602896 | 0.674507 | 0.624379 | 0.631540 | 0.638701 |
| group_b_unconstrained | calibrated | aadb_only | 0.260000 | 0.450000 | 0.050000 | 0.240000 | 0.602896 | 0.674507 | 0.624379 | 0.631540 | 0.638701 |
| group_a_nnls_stacking | raw_scaled | combined_70_30 | 0.764648 | 0.158807 | 0.076545 | 0.000000 | 0.634512 | 0.596431 | 0.623088 | 0.619280 | 0.615471 |
| group_a | raw_scaled | combined_70_30 | 0.790000 | 0.130000 | 0.080000 | 0.000000 | 0.644886 | 0.570909 | 0.622693 | 0.615295 | 0.607897 |
| group_b_unconstrained_nnls_stacking | raw_scaled | combined_70_30 | 0.684543 | 0.141742 | 0.067980 | 0.105735 | 0.633847 | 0.584188 | 0.618949 | 0.613984 | 0.609018 |

## 10. Held-out test metrics for selected weights

| variant | prep | objective | test_ava_n | test_ava_srcc | test_ava_plcc | test_ava_mae | test_ava_mse | test_ava_binary_accuracy_at_5 | test_ava_f1_at_5 | test_ava_positive_prediction_ratio_at_5 | test_ava_positive_ground_truth_ratio_at_5 | test_aadb_n | test_aadb_srcc | test_aadb_plcc | test_aadb_mae | test_aadb_mse | test_aadb_binary_accuracy_at_5 | test_aadb_f1_at_5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| group_a | calibrated | combined_70_30 | 5109 | 0.653648 | 0.665540 | 0.460510 | 0.345610 | 0.760619 | 0.855215 | 0.937170 | 0.716187 | 200 | 0.673337 | 0.668978 | 1.339797 | 2.738002 | 0.740000 | 0.775862 |
| group_a | calibrated | aadb_only | 5109 | 0.640922 | 0.653655 | 0.469598 | 0.358945 | 0.755334 | 0.853114 | 0.949501 | 0.716187 | 200 | 0.692192 | 0.681914 | 1.302304 | 2.607129 | 0.750000 | 0.786325 |
| group_b_icaa_cap_020 | calibrated | combined_70_30 | 5109 | 0.654102 | 0.666386 | 0.463152 | 0.348763 | 0.761401 | 0.855961 | 0.940301 | 0.716187 | 200 | 0.661366 | 0.659974 | 1.363517 | 2.828746 | 0.745000 | 0.784810 |
| group_b_icaa_cap_015 | calibrated | combined_70_30 | 5109 | 0.654102 | 0.666386 | 0.463152 | 0.348763 | 0.761401 | 0.855961 | 0.940301 | 0.716187 | 200 | 0.661366 | 0.659974 | 1.363517 | 2.828746 | 0.745000 | 0.784810 |
| group_b_icaa_cap_025 | calibrated | combined_70_30 | 5109 | 0.654102 | 0.666386 | 0.463152 | 0.348763 | 0.761401 | 0.855961 | 0.940301 | 0.716187 | 200 | 0.661366 | 0.659974 | 1.363517 | 2.828746 | 0.745000 | 0.784810 |
| group_b_unconstrained | calibrated | combined_70_30 | 5109 | 0.654102 | 0.666386 | 0.463152 | 0.348763 | 0.761401 | 0.855961 | 0.940301 | 0.716187 | 200 | 0.661366 | 0.659974 | 1.363517 | 2.828746 | 0.745000 | 0.784810 |
| group_a_nnls_stacking | calibrated | combined_70_30 | 5109 | 0.654580 | 0.666721 | 0.455431 | 0.338327 | 0.766686 | 0.857960 | 0.926404 | 0.716187 | 200 | 0.658642 | 0.657972 | 1.353761 | 2.783504 | 0.740000 | 0.777778 |
| group_b_unconstrained_nnls_stacking | calibrated | combined_70_30 | 5109 | 0.655450 | 0.668043 | 0.457406 | 0.340668 | 0.766099 | 0.857721 | 0.927775 | 0.716187 | 200 | 0.639574 | 0.645939 | 1.377412 | 2.875014 | 0.735000 | 0.778243 |
| group_a | calibrated | ava_only | 5109 | 0.657218 | 0.669374 | 0.444758 | 0.322859 | 0.777256 | 0.861692 | 0.894304 | 0.716187 | 200 | 0.612353 | 0.615740 | 1.411415 | 2.983621 | 0.725000 | 0.771784 |
| group_b_icaa_cap_015 | calibrated | aadb_only | 5109 | 0.627578 | 0.642743 | 0.475221 | 0.366532 | 0.750440 | 0.850580 | 0.954003 | 0.716187 | 200 | 0.669184 | 0.669312 | 1.325527 | 2.703285 | 0.755000 | 0.789700 |
| group_b_icaa_cap_020 | calibrated | aadb_only | 5109 | 0.626545 | 0.641749 | 0.476896 | 0.368599 | 0.749657 | 0.850217 | 0.955177 | 0.716187 | 200 | 0.659863 | 0.662859 | 1.340725 | 2.761026 | 0.745000 | 0.782979 |
| group_b_icaa_cap_025 | calibrated | ava_only | 5109 | 0.658082 | 0.670538 | 0.444150 | 0.321678 | 0.777060 | 0.861486 | 0.893326 | 0.716187 | 200 | 0.584544 | 0.594143 | 1.437038 | 3.079848 | 0.700000 | 0.747899 |
| group_b_icaa_cap_020 | calibrated | ava_only | 5109 | 0.658082 | 0.670538 | 0.444150 | 0.321678 | 0.777060 | 0.861486 | 0.893326 | 0.716187 | 200 | 0.584544 | 0.594143 | 1.437038 | 3.079848 | 0.700000 | 0.747899 |
| group_b_unconstrained | calibrated | ava_only | 5109 | 0.658082 | 0.670538 | 0.444150 | 0.321678 | 0.777060 | 0.861486 | 0.893326 | 0.716187 | 200 | 0.584544 | 0.594143 | 1.437038 | 3.079848 | 0.700000 | 0.747899 |
| group_b_icaa_cap_015 | calibrated | ava_only | 5109 | 0.658082 | 0.670538 | 0.444150 | 0.321678 | 0.777060 | 0.861486 | 0.893326 | 0.716187 | 200 | 0.584544 | 0.594143 | 1.437038 | 3.079848 | 0.700000 | 0.747899 |
| group_b_icaa_cap_025 | calibrated | aadb_only | 5109 | 0.602896 | 0.618372 | 0.491933 | 0.390881 | 0.739088 | 0.845306 | 0.970444 | 0.716187 | 200 | 0.674507 | 0.673809 | 1.326028 | 2.712986 | 0.745000 | 0.784810 |
| group_b_unconstrained | calibrated | aadb_only | 5109 | 0.602896 | 0.618372 | 0.491933 | 0.390881 | 0.739088 | 0.845306 | 0.970444 | 0.716187 | 200 | 0.674507 | 0.673809 | 1.326028 | 2.712986 | 0.745000 | 0.784810 |
| group_a_nnls_stacking | raw_scaled | combined_70_30 | 5109 | 0.634512 | 0.646680 | 0.572291 | 0.512653 | 0.685849 | 0.740249 | 0.493247 | 0.716187 | 200 | 0.596431 | 0.601262 | 1.671806 | 4.145392 | 0.525000 | 0.344828 |
| group_a | raw_scaled | combined_70_30 | 5109 | 0.644886 | 0.656869 | 0.581583 | 0.524369 | 0.678802 | 0.730143 | 0.474065 | 0.716187 | 200 | 0.570909 | 0.578128 | 1.700288 | 4.284862 | 0.515000 | 0.321678 |
| group_b_unconstrained_nnls_stacking | raw_scaled | combined_70_30 | 5109 | 0.633847 | 0.646582 | 0.512111 | 0.421006 | 0.720689 | 0.781904 | 0.564494 | 0.716187 | 200 | 0.584188 | 0.597125 | 1.628434 | 3.917337 | 0.555000 | 0.425806 |
| group_b_icaa_cap_015 | raw_scaled | combined_70_30 | 5109 | 0.645290 | 0.657738 | 0.499362 | 0.400567 | 0.727344 | 0.788072 | 0.570366 | 0.716187 | 200 | 0.551949 | 0.564978 | 1.637460 | 3.959119 | 0.545000 | 0.420382 |
| group_b_icaa_cap_025 | raw_scaled | combined_70_30 | 5109 | 0.645290 | 0.657738 | 0.499362 | 0.400567 | 0.727344 | 0.788072 | 0.570366 | 0.716187 | 200 | 0.551949 | 0.564978 | 1.637460 | 3.959119 | 0.545000 | 0.420382 |
| group_b_unconstrained | raw_scaled | combined_70_30 | 5109 | 0.645290 | 0.657738 | 0.499362 | 0.400567 | 0.727344 | 0.788072 | 0.570366 | 0.716187 | 200 | 0.551949 | 0.564978 | 1.637460 | 3.959119 | 0.545000 | 0.420382 |
| group_b_icaa_cap_020 | raw_scaled | combined_70_30 | 5109 | 0.645290 | 0.657738 | 0.499362 | 0.400567 | 0.727344 | 0.788072 | 0.570366 | 0.716187 | 200 | 0.551949 | 0.564978 | 1.637460 | 3.959119 | 0.545000 | 0.420382 |
| group_a | raw_scaled | ava_only | 5109 | 0.657209 | 0.669396 | 0.637270 | 0.609163 | 0.650029 | 0.694150 | 0.428068 | 0.716187 | 200 | 0.480104 | 0.485219 | 1.785221 | 4.737907 | 0.480000 | 0.257143 |
| group_b_icaa_cap_020 | raw_scaled | ava_only | 5109 | 0.658093 | 0.670472 | 0.590944 | 0.532702 | 0.673909 | 0.723531 | 0.463300 | 0.716187 | 200 | 0.472477 | 0.480068 | 1.756737 | 4.568919 | 0.500000 | 0.305556 |
| group_b_icaa_cap_015 | raw_scaled | ava_only | 5109 | 0.658093 | 0.670472 | 0.590944 | 0.532702 | 0.673909 | 0.723531 | 0.463300 | 0.716187 | 200 | 0.472477 | 0.480068 | 1.756737 | 4.568919 | 0.500000 | 0.305556 |
| group_b_icaa_cap_025 | raw_scaled | ava_only | 5109 | 0.658093 | 0.670472 | 0.590944 | 0.532702 | 0.673909 | 0.723531 | 0.463300 | 0.716187 | 200 | 0.472477 | 0.480068 | 1.756737 | 4.568919 | 0.500000 | 0.305556 |
| group_b_unconstrained | raw_scaled | ava_only | 5109 | 0.658093 | 0.670472 | 0.590944 | 0.532702 | 0.673909 | 0.723531 | 0.463300 | 0.716187 | 200 | 0.472477 | 0.480068 | 1.756737 | 4.568919 | 0.500000 | 0.305556 |
| group_b_icaa_cap_015 | raw_scaled | aadb_only | 5109 | 0.527060 | 0.546018 | 0.516868 | 0.434704 | 0.723429 | 0.800057 | 0.667058 | 0.716187 | 200 | 0.668700 | 0.670213 | 1.520410 | 3.412394 | 0.640000 | 0.586207 |

## 11. Bootstrap stability

| scope | dataset | variant | prep | objective | w_nima | w_rgnet | w_alamp | w_icaa | srcc_mean | srcc_ci_low_95 | srcc_ci_high_95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| combined_70_30 | ava_aadb | group_a | calibrated | combined_70_30 | 0.370000 | 0.410000 | 0.220000 | 0.000000 | 0.658121 | 0.628055 | 0.683700 |
| combined_70_30 | ava_aadb | group_b_unconstrained | calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.655788 | 0.626362 | 0.683423 |
| combined_70_30 | ava_aadb | group_b_icaa_cap_015 | calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.655302 | 0.626547 | 0.681438 |
| combined_70_30 | ava_aadb | group_b_icaa_cap_020 | calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.655133 | 0.623719 | 0.683953 |
| combined_70_30 | ava_aadb | group_a_nnls_stacking | calibrated | combined_70_30 | 0.422725 | 0.371420 | 0.205854 | 0.000000 | 0.654746 | 0.623634 | 0.683040 |
| combined_70_30 | ava_aadb | group_b_icaa_cap_025 | calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.654717 | 0.625858 | 0.681739 |
| combined_70_30 | ava_aadb | group_b_unconstrained_nnls_stacking | calibrated | combined_70_30 | 0.395209 | 0.320038 | 0.193629 | 0.091123 | 0.649704 | 0.615904 | 0.678736 |
| combined_70_30 | ava_aadb | group_a | raw_scaled | combined_70_30 | 0.790000 | 0.130000 | 0.080000 | 0.000000 | 0.622727 | 0.584439 | 0.652629 |
| combined_70_30 | ava_aadb | group_a_nnls_stacking | raw_scaled | combined_70_30 | 0.764648 | 0.158807 | 0.076545 | 0.000000 | 0.622575 | 0.589231 | 0.653475 |
| combined_70_30 | ava_aadb | group_b_unconstrained_nnls_stacking | raw_scaled | combined_70_30 | 0.684543 | 0.141742 | 0.067980 | 0.105735 | 0.617740 | 0.583004 | 0.651846 |
| dataset | aadb | group_a | calibrated | combined_70_30 | 0.370000 | 0.410000 | 0.220000 | 0.000000 | 0.669272 | 0.572948 | 0.747966 |
| dataset | aadb | group_b_unconstrained | calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.659363 | 0.565255 | 0.745220 |
| dataset | aadb | group_b_icaa_cap_020 | calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.658851 | 0.561684 | 0.743019 |
| dataset | aadb | group_b_icaa_cap_015 | calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.658405 | 0.569973 | 0.735244 |
| dataset | aadb | group_b_icaa_cap_025 | calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.656415 | 0.565875 | 0.739440 |
| dataset | aadb | group_a_nnls_stacking | calibrated | combined_70_30 | 0.422725 | 0.371420 | 0.205854 | 0.000000 | 0.655858 | 0.555280 | 0.734655 |
| dataset | ava | group_b_unconstrained_nnls_stacking | calibrated | combined_70_30 | 0.395209 | 0.320038 | 0.193629 | 0.091123 | 0.655156 | 0.638961 | 0.672324 |
| dataset | ava | group_a_nnls_stacking | calibrated | combined_70_30 | 0.422725 | 0.371420 | 0.205854 | 0.000000 | 0.654270 | 0.637334 | 0.672267 |
| dataset | ava | group_b_unconstrained | calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.654256 | 0.636975 | 0.671710 |
| dataset | ava | group_b_icaa_cap_025 | calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.653990 | 0.636355 | 0.669701 |
| dataset | ava | group_b_icaa_cap_015 | calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.653972 | 0.636849 | 0.671360 |
| dataset | ava | group_b_icaa_cap_020 | calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.653540 | 0.636365 | 0.669321 |
| dataset | ava | group_a | calibrated | combined_70_30 | 0.370000 | 0.410000 | 0.220000 | 0.000000 | 0.653342 | 0.635649 | 0.669618 |
| dataset | ava | group_a | raw_scaled | combined_70_30 | 0.790000 | 0.130000 | 0.080000 | 0.000000 | 0.645068 | 0.627806 | 0.662107 |
| dataset | aadb | group_b_unconstrained_nnls_stacking | calibrated | combined_70_30 | 0.395209 | 0.320038 | 0.193629 | 0.091123 | 0.636982 | 0.533520 | 0.722919 |
| dataset | ava | group_a_nnls_stacking | raw_scaled | combined_70_30 | 0.764648 | 0.158807 | 0.076545 | 0.000000 | 0.634802 | 0.616899 | 0.652146 |
| dataset | ava | group_b_unconstrained_nnls_stacking | raw_scaled | combined_70_30 | 0.684543 | 0.141742 | 0.067980 | 0.105735 | 0.633563 | 0.616292 | 0.651107 |
| dataset | aadb | group_a_nnls_stacking | raw_scaled | combined_70_30 | 0.764648 | 0.158807 | 0.076545 | 0.000000 | 0.594044 | 0.488173 | 0.686318 |
| dataset | aadb | group_b_unconstrained_nnls_stacking | raw_scaled | combined_70_30 | 0.684543 | 0.141742 | 0.067980 | 0.105735 | 0.580820 | 0.464364 | 0.683747 |
| dataset | aadb | group_a | raw_scaled | combined_70_30 | 0.790000 | 0.130000 | 0.080000 | 0.000000 | 0.570598 | 0.449704 | 0.662100 |

Multiple top weights should be treated as statistically tied when their 95% bootstrap CIs overlap materially.

5-fold robustness check on selected weights:

| variant | prep | objective | w_nima | w_rgnet | w_alamp | w_icaa | cv5_combined_70_30_srcc_mean | cv5_combined_70_30_srcc_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| group_a | calibrated | combined_70_30 | 0.370000 | 0.410000 | 0.220000 | 0.000000 | 0.654228 | 0.002188 |
| group_b_icaa_cap_015 | calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.652752 | 0.001666 |
| group_b_icaa_cap_020 | calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.652752 | 0.001666 |
| group_b_unconstrained | calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.652752 | 0.001666 |
| group_b_icaa_cap_025 | calibrated | combined_70_30 | 0.330000 | 0.360000 | 0.220000 | 0.090000 | 0.652752 | 0.001666 |
| group_a_nnls_stacking | calibrated | combined_70_30 | 0.422725 | 0.371420 | 0.205854 | 0.000000 | 0.651395 | 0.002717 |
| group_b_unconstrained_nnls_stacking | calibrated | combined_70_30 | 0.395209 | 0.320038 | 0.193629 | 0.091123 | 0.649654 | 0.002627 |
| group_a | calibrated | aadb_only | 0.400000 | 0.490000 | 0.110000 | 0.000000 | 0.649406 | 0.002733 |
| group_a | calibrated | ava_only | 0.460000 | 0.260000 | 0.280000 | 0.000000 | 0.640856 | 0.004415 |
| group_b_icaa_cap_015 | calibrated | aadb_only | 0.400000 | 0.420000 | 0.030000 | 0.150000 | 0.638565 | 0.003944 |
| group_b_icaa_cap_020 | calibrated | aadb_only | 0.370000 | 0.390000 | 0.040000 | 0.200000 | 0.636469 | 0.003757 |
| group_b_icaa_cap_025 | calibrated | ava_only | 0.460000 | 0.210000 | 0.270000 | 0.060000 | 0.636145 | 0.005373 |
| group_b_icaa_cap_020 | calibrated | ava_only | 0.460000 | 0.210000 | 0.270000 | 0.060000 | 0.636145 | 0.005373 |
| group_b_unconstrained | calibrated | ava_only | 0.460000 | 0.210000 | 0.270000 | 0.060000 | 0.636145 | 0.005373 |
| group_b_icaa_cap_015 | calibrated | ava_only | 0.460000 | 0.210000 | 0.270000 | 0.060000 | 0.636145 | 0.005373 |
| group_b_icaa_cap_025 | calibrated | aadb_only | 0.260000 | 0.450000 | 0.050000 | 0.240000 | 0.623219 | 0.005026 |
| group_b_unconstrained | calibrated | aadb_only | 0.260000 | 0.450000 | 0.050000 | 0.240000 | 0.623219 | 0.005026 |
| group_a | raw_scaled | combined_70_30 | 0.790000 | 0.130000 | 0.080000 | 0.000000 | 0.620778 | 0.004716 |
| group_a_nnls_stacking | raw_scaled | combined_70_30 | 0.764648 | 0.158807 | 0.076545 | 0.000000 | 0.619935 | 0.003673 |
| group_b_unconstrained_nnls_stacking | raw_scaled | combined_70_30 | 0.684543 | 0.141742 | 0.067980 | 0.105735 | 0.619769 | 0.004467 |
| group_b_icaa_cap_015 | raw_scaled | combined_70_30 | 0.660000 | 0.110000 | 0.080000 | 0.150000 | 0.619311 | 0.005340 |
| group_b_icaa_cap_025 | raw_scaled | combined_70_30 | 0.660000 | 0.110000 | 0.080000 | 0.150000 | 0.619311 | 0.005340 |
| group_b_unconstrained | raw_scaled | combined_70_30 | 0.660000 | 0.110000 | 0.080000 | 0.150000 | 0.619311 | 0.005340 |
| group_b_icaa_cap_020 | raw_scaled | combined_70_30 | 0.660000 | 0.110000 | 0.080000 | 0.150000 | 0.619311 | 0.005340 |
| group_b_icaa_cap_020 | raw_scaled | ava_only | 0.800000 | 0.040000 | 0.100000 | 0.060000 | 0.604222 | 0.007460 |
| group_b_icaa_cap_025 | raw_scaled | ava_only | 0.800000 | 0.040000 | 0.100000 | 0.060000 | 0.604222 | 0.007460 |
| group_b_icaa_cap_015 | raw_scaled | ava_only | 0.800000 | 0.040000 | 0.100000 | 0.060000 | 0.604222 | 0.007460 |
| group_b_unconstrained | raw_scaled | ava_only | 0.800000 | 0.040000 | 0.100000 | 0.060000 | 0.604222 | 0.007460 |
| group_a | raw_scaled | ava_only | 0.840000 | 0.050000 | 0.110000 | 0.000000 | 0.604068 | 0.007444 |
| group_b_icaa_cap_015 | raw_scaled | aadb_only | 0.600000 | 0.240000 | 0.010000 | 0.150000 | 0.565936 | 0.006655 |

## 12. Final decision

- Best research ratio: `{'nima': 0.37, 'rgnet': 0.41, 'alamp': 0.22, 'icaa': 0.0}` from `group_a` / `calibrated`.
- Safest production ratio: `{'nima': 0.37, 'rgnet': 0.41, 'alamp': 0.22, 'icaa': 0.0}` from `group_a` / `calibrated`.
- ICAA recommendation: `keep_icaa_as_separate_color_subscore`.
- Group B minus Group A combined SRCC: `-0.003274`.
- Group B minus Group A AADB SRCC: `-0.011971`.

## 13. Caveats

- NIMA AVA split is local, not official NIMA paper split.
- A-LAMP is paper-inspired adaptive, not full paper protocol.
- Cross-predictions used forWeights2 local artifacts because assets/models paths were absent.
- No Flutter or production scoring changed.
- No model files were modified, no retraining was run, and no model inference was rerun.
