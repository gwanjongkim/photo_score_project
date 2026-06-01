# Best Weight Recommendation

Recommended ratio: NIMA=0.200, RGNet=0.510, A-LAMP=0.290.

| candidate_name | nima_weight | rgnet_weight | alamp_weight | mae | maxae | rmse | spearman | pearson | top1_match | top3_overlap | top5_overlap | top10_overlap | ranking_inversions | nan_count | range_error_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| refined_0.200_0.510_0.290 | 0.200000 | 0.510000 | 0.290000 | 0.017057 | 0.041709 | 0.019494 | 0.983776 | 0.986023 | True | 3 | 5 | 9 | 51 | 0 | 0 |

Compared rows:

| candidate_name | nima_weight | rgnet_weight | alamp_weight | mae | maxae | rmse | spearman | pearson | top1_match | top3_overlap | top5_overlap | top10_overlap | ranking_inversions | nan_count | range_error_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| refined_0.200_0.510_0.290 | 0.200000 | 0.510000 | 0.290000 | 0.017057 | 0.041709 | 0.019494 | 0.983776 | 0.986023 | True | 3 | 5 | 9 | 51 | 0 | 0 |
| proposed_020_030_050 | 0.200000 | 0.300000 | 0.500000 | 0.020957 | 0.048928 | 0.024578 | 0.951224 | 0.962990 | True | 3 | 4 | 8 | 108 | 0 | 0 |
| old_no_icaa_renormalized | 0.125000 | 0.500000 | 0.375000 | 0.018993 | 0.050745 | 0.021823 | 0.978571 | 0.986379 | True | 3 | 5 | 9 | 63 | 0 | 0 |
| equal | 0.333333 | 0.333333 | 0.333333 | 0.013264 | 0.032589 | 0.016055 | 0.972449 | 0.980397 | True | 3 | 5 | 9 | 76 | 0 | 0 |

Final decision: B. Another ratio gives a better preservation tradeoff than the proposed ratio.
