# A-cut ICAA Skip Weight Search

## 1. Conclusion
- dataset used: `outputs/test_vila_aesthetic_direct_compare_20260504_135316/old_three_model_scores.csv` plus WSL ICAA SavedModel inference on 49 images.
- reference ensemble: NIMA 0.10, RGNet 0.40, A-LAMP 0.30, ICAA 0.20 using ICAA `color` output.
- best candidate: NIMA=0.200, RGNet=0.510, A-LAMP=0.290.
- user proposed 0.2/0.3/0.5 result: MAE 0.020957, Spearman 0.951224, Top-10 overlap 8/10.
- recommendation: Final decision B. Another ratio gives a better preservation tradeoff than the proposed ratio.
- whether ready for Flutter implementation: benchmark-backed prompt is generated, but production implementation should still verify the active Flutter branch because the accessible checkout did not contain an ICAA active ensemble.

## 2. Artifact Inventory
Inventory written to `artifact_inventory.csv`. Existing score cache and ICAA SavedModel were available; no cached four-model NIMA/RGNet/A-LAMP/ICAA score table was found, so ICAA was scored in WSL.

## 3. Score Table Construction
The score table reuses cached 50-image NIMA/RGNet/A-LAMP scores from the selected A-cut cache. NIMA was normalized with the same unit-score convention used by the selector: 1..10 scores map to `(score - 1) / 9`, while RGNet and A-LAMP were treated as unit scores. ICAA was computed from the TF-native SavedModel with RGB 224x224 bilinear resize and ImageNet mean/std normalization.

## 4. Reference Definition
Confirmed fact: WSL has an executable ICAA TF-native SavedModel with two outputs, and the local ICAA report names output index 1 as the active color aesthetic output. Assumption for this benchmark: the ICAA-included teacher uses NIMA 0.10, RGNet 0.40, A-LAMP 0.30, ICAA 0.20. The accessible Flutter checkout inspected in read-only mode did not confirm an active ICAA ensemble.

## 5. Weight Search Method
The first pass searched NIMA/RGNet/A-LAMP weights at step 0.05, constrained to non-negative weights summing to 1.0. The search then refined around the top candidates at step 0.01. The proposed 0.20/0.30/0.50, old no-ICAA renormalized 0.125/0.50/0.375, and equal weights were always included.

## 6. Results
| candidate_name | nima_weight | rgnet_weight | alamp_weight | mae | maxae | rmse | spearman | pearson | top1_match | top3_overlap | top5_overlap | top10_overlap | ranking_inversions | nan_count | range_error_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| refined_0.200_0.510_0.290 | 0.200000 | 0.510000 | 0.290000 | 0.017057 | 0.041709 | 0.019494 | 0.983776 | 0.986023 | True | 3 | 5 | 9 | 51 | 0 | 0 |
| proposed_020_030_050 | 0.200000 | 0.300000 | 0.500000 | 0.020957 | 0.048928 | 0.024578 | 0.951224 | 0.962990 | True | 3 | 4 | 8 | 108 | 0 | 0 |
| old_no_icaa_renormalized | 0.125000 | 0.500000 | 0.375000 | 0.018993 | 0.050745 | 0.021823 | 0.978571 | 0.986379 | True | 3 | 5 | 9 | 63 | 0 | 0 |
| equal | 0.333333 | 0.333333 | 0.333333 | 0.013264 | 0.032589 | 0.016055 | 0.972449 | 0.980397 | True | 3 | 5 | 9 | 76 | 0 | 0 |

Top candidates:

| candidate_name | nima_weight | rgnet_weight | alamp_weight | mae | maxae | rmse | spearman | pearson | top1_match | top3_overlap | top5_overlap | top10_overlap | ranking_inversions | nan_count | range_error_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| refined_0.200_0.510_0.290 | 0.200000 | 0.510000 | 0.290000 | 0.017057 | 0.041709 | 0.019494 | 0.983776 | 0.986023 | True | 3 | 5 | 9 | 51 | 0 | 0 |
| refined_0.230_0.470_0.300 | 0.230000 | 0.470000 | 0.300000 | 0.015795 | 0.038152 | 0.018108 | 0.983469 | 0.986766 | True | 3 | 5 | 9 | 51 | 0 | 0 |
| refined_0.220_0.500_0.280 | 0.220000 | 0.500000 | 0.280000 | 0.016678 | 0.039316 | 0.018951 | 0.983163 | 0.985814 | True | 3 | 5 | 9 | 51 | 0 | 0 |
| refined_0.190_0.510_0.300 | 0.190000 | 0.510000 | 0.300000 | 0.017159 | 0.042912 | 0.019674 | 0.983163 | 0.986366 | True | 3 | 5 | 9 | 53 | 0 | 0 |
| refined_0.220_0.480_0.300 | 0.220000 | 0.480000 | 0.300000 | 0.016082 | 0.039342 | 0.018472 | 0.983061 | 0.986735 | True | 3 | 5 | 9 | 52 | 0 | 0 |
| refined_0.210_0.500_0.290 | 0.210000 | 0.500000 | 0.290000 | 0.016689 | 0.040519 | 0.019082 | 0.983061 | 0.986221 | True | 3 | 5 | 9 | 53 | 0 | 0 |
| refined_0.230_0.480_0.290 | 0.230000 | 0.480000 | 0.290000 | 0.016082 | 0.038139 | 0.018309 | 0.982959 | 0.986485 | True | 3 | 5 | 9 | 53 | 0 | 0 |
| refined_0.220_0.490_0.290 | 0.220000 | 0.490000 | 0.290000 | 0.016376 | 0.039329 | 0.018686 | 0.982959 | 0.986376 | True | 3 | 5 | 9 | 53 | 0 | 0 |
| refined_0.210_0.490_0.300 | 0.210000 | 0.490000 | 0.300000 | 0.016401 | 0.040532 | 0.018856 | 0.982959 | 0.986657 | True | 3 | 5 | 9 | 53 | 0 | 0 |
| refined_0.240_0.470_0.290 | 0.240000 | 0.470000 | 0.290000 | 0.015788 | 0.036949 | 0.017951 | 0.982755 | 0.986546 | True | 3 | 5 | 9 | 53 | 0 | 0 |

## 7. Proposed Ratio Analysis
NIMA=0.200, RGNet=0.300, A-LAMP=0.500 produced MAE 0.020957, MaxAE 0.048928, RMSE 0.024578, Spearman 0.951224, Pearson 0.962990, Top-1 match True, Top-3 overlap 3/3, Top-5 overlap 4/5, and Top-10 overlap 8/10.

## 8. Recommended Ratio
Recommended ratio: NIMA=0.200, RGNet=0.510, A-LAMP=0.290. It was selected by prioritizing Top-K overlap, then Spearman rank correlation, then MAE/MaxAE, with deterministic tie handling.

## 9. Risks
- ICAA removal changes the aesthetic signal; this benchmark only measures preservation of the selected ICAA-included teacher, not human preference.
- Results reflect the available 49-image WSL test set.
- Android speed gain was supplied from Android logs and was not remeasured here.
- The accessible Flutter checkout did not expose the ICAA skip flag or ICAA active ensemble, so implementation must verify the target branch before changing defaults.

## 10. Next Action
Use `flutter_implementation_prompt.md` for a separate Flutter-stage change only after confirming the target Flutter branch contains the ICAA path and skip flag. Preserve technical scoring and run Flutter validation there.

Final decision: B

## Holdout Check
| candidate_name | nima_weight | rgnet_weight | alamp_weight | mae | maxae | rmse | spearman | pearson | top1_match | top3_overlap | top5_overlap | top10_overlap | ranking_inversions | nan_count | range_error_count | subset |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| refined_0.200_0.510_0.290 | 0.200000 | 0.510000 | 0.290000 | 0.018092 | 0.041709 | 0.020970 | 0.970769 | 0.983667 | True | 2 | 5 | 9 | 16 | 0 | 0 | search_even_index |
| refined_0.200_0.510_0.290 | 0.200000 | 0.510000 | 0.290000 | 0.015979 | 0.030598 | 0.017827 | 0.982609 | 0.986868 | True | 3 | 5 | 9 | 9 | 0 | 0 | holdout_odd_index |
| refined_0.230_0.470_0.300 | 0.230000 | 0.470000 | 0.300000 | 0.016646 | 0.038152 | 0.019221 | 0.966923 | 0.985385 | True | 2 | 5 | 9 | 17 | 0 | 0 | search_even_index |
| refined_0.230_0.470_0.300 | 0.230000 | 0.470000 | 0.300000 | 0.014908 | 0.029078 | 0.016870 | 0.980870 | 0.986637 | True | 3 | 5 | 9 | 10 | 0 | 0 | holdout_odd_index |
| refined_0.220_0.500_0.280 | 0.220000 | 0.500000 | 0.280000 | 0.017412 | 0.039316 | 0.020091 | 0.970769 | 0.983370 | True | 2 | 5 | 9 | 16 | 0 | 0 | search_even_index |
| refined_0.220_0.500_0.280 | 0.220000 | 0.500000 | 0.280000 | 0.015912 | 0.029926 | 0.017685 | 0.983478 | 0.986849 | True | 3 | 5 | 9 | 8 | 0 | 0 | holdout_odd_index |
| refined_0.190_0.510_0.300 | 0.190000 | 0.510000 | 0.300000 | 0.018384 | 0.042912 | 0.021330 | 0.970769 | 0.984297 | True | 2 | 5 | 9 | 16 | 0 | 0 | search_even_index |
| refined_0.190_0.510_0.300 | 0.190000 | 0.510000 | 0.300000 | 0.015884 | 0.031682 | 0.017787 | 0.982609 | 0.986865 | True | 3 | 5 | 9 | 9 | 0 | 0 | holdout_odd_index |
| refined_0.220_0.480_0.300 | 0.220000 | 0.480000 | 0.300000 | 0.017064 | 0.039342 | 0.019724 | 0.966923 | 0.985199 | True | 2 | 5 | 9 | 17 | 0 | 0 | search_even_index |
| refined_0.220_0.480_0.300 | 0.220000 | 0.480000 | 0.300000 | 0.015060 | 0.029729 | 0.017071 | 0.980870 | 0.986757 | True | 3 | 5 | 9 | 10 | 0 | 0 | holdout_odd_index |
| refined_0.210_0.500_0.290 | 0.210000 | 0.500000 | 0.290000 | 0.017661 | 0.040519 | 0.020422 | 0.966923 | 0.984053 | True | 2 | 5 | 9 | 17 | 0 | 0 | search_even_index |
| refined_0.210_0.500_0.290 | 0.210000 | 0.500000 | 0.290000 | 0.015677 | 0.029947 | 0.017577 | 0.982609 | 0.986907 | True | 3 | 5 | 9 | 9 | 0 | 0 | holdout_odd_index |
| refined_0.230_0.480_0.290 | 0.230000 | 0.480000 | 0.290000 | 0.016800 | 0.038139 | 0.019371 | 0.966923 | 0.984660 | True | 2 | 5 | 9 | 17 | 0 | 0 | search_even_index |
| refined_0.230_0.480_0.290 | 0.230000 | 0.480000 | 0.290000 | 0.015334 | 0.028948 | 0.017133 | 0.980870 | 0.986863 | True | 3 | 5 | 9 | 10 | 0 | 0 | holdout_odd_index |
| refined_0.220_0.490_0.290 | 0.220000 | 0.490000 | 0.290000 | 0.017231 | 0.039329 | 0.019889 | 0.966923 | 0.984385 | True | 2 | 5 | 9 | 17 | 0 | 0 | search_even_index |
| refined_0.220_0.490_0.290 | 0.220000 | 0.490000 | 0.290000 | 0.015486 | 0.029326 | 0.017346 | 0.980870 | 0.986906 | True | 3 | 5 | 9 | 10 | 0 | 0 | holdout_odd_index |
| refined_0.210_0.490_0.300 | 0.210000 | 0.490000 | 0.300000 | 0.017481 | 0.040532 | 0.020244 | 0.966923 | 0.984954 | True | 2 | 5 | 9 | 17 | 0 | 0 | search_even_index |
| refined_0.210_0.490_0.300 | 0.210000 | 0.490000 | 0.300000 | 0.015276 | 0.030380 | 0.017292 | 0.980870 | 0.986835 | True | 3 | 5 | 9 | 10 | 0 | 0 | holdout_odd_index |
| refined_0.240_0.470_0.290 | 0.240000 | 0.470000 | 0.290000 | 0.016369 | 0.036949 | 0.018871 | 0.966923 | 0.984877 | True | 2 | 5 | 9 | 17 | 0 | 0 | search_even_index |
| refined_0.240_0.470_0.290 | 0.240000 | 0.470000 | 0.290000 | 0.015182 | 0.028569 | 0.016939 | 0.980870 | 0.986776 | True | 3 | 5 | 9 | 10 | 0 | 0 | holdout_odd_index |
