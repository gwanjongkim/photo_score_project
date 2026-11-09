# Product Gold Dataset Evaluation: test_vila

## 1. Executive Summary
The aesthetic ensemble weights were evaluated on the 'test_vila' dataset (49 images). Standalone winner for test_vila: **NIMA 0.70 / RGNet 0.30 / A-LAMP -0.00**.

## 2. Dataset Counts
- Bad: 17
- Soso: 13
- Good: 20
- Failures: 1

## 3. Fixed Candidate Comparison (test_vila)
| candidate_name | w_nima | w_rgnet | w_alamp | srcc | pw_acc | good_minus_bad |
| --- | --- | --- | --- | --- | --- | --- |
| current_v3_alamp_heavy | 0.2500 | 0.3000 | 0.4500 | 0.4199 | 0.7080 | 0.1348 |
| product_gold_winner | 0.4000 | 0.4500 | 0.1500 | 0.4544 | 0.7269 | 0.0982 |
| multi_objective_candidate | 0.4700 | 0.2300 | 0.3000 | 0.4440 | 0.7206 | 0.1132 |
| old_no_icaa_renorm | 0.1250 | 0.5000 | 0.3750 | 0.3985 | 0.6979 | 0.1293 |
| new_uncalibrated | 0.3700 | 0.4100 | 0.2200 | 0.4445 | 0.7206 | 0.1066 |
| rgnet_heavy | 0.2000 | 0.5500 | 0.2500 | 0.4117 | 0.7042 | 0.1138 |
| alamp_very_heavy | 0.2000 | 0.2000 | 0.6000 | 0.3951 | 0.6915 | 0.1526 |
| nima_heavy | 0.5000 | 0.3000 | 0.2000 | 0.4622 | 0.7307 | 0.1015 |

## 4. Night/Sky Caveat Analysis
- Count of night_or_sky: 6
Night/Sky images (especially Good ones) often show different score distributions. If Good night images are under-scored, it may be due to low saliency or lack of ordinary 'beauty' patterns in AVA/AADB training.

| filename | label_name | nima_norm | rgnet_norm | alamp_norm |
| --- | --- | --- | --- | --- |
| IMG_2739(soso-night_.JPG | Soso | 0.3120 | 0.6099 | 0.2415 |
| IMG_2729(soso-night_.JPG | Soso | 0.3548 | 0.6204 | 0.5663 |
| IMG_2738(soso-night_.JPG | Soso | 0.3324 | 0.6008 | 0.1910 |
| IMG_2748(soso-night_.JPG | Soso | 0.3323 | 0.4905 | 0.3848 |
| 1738744324186(good_night).jpg | Good | 0.5676 | 0.4810 | 0.9404 |
| 20250205_000515(good_night).jpg | Good | 0.5170 | 0.3444 | 0.7401 |

## 5. Pooled Results (Test_weight + test_vila)
Pooled winner: **NIMA 0.40 / RGNet 0.50 / A-LAMP 0.10**.

## 6. Final Recommendation
Previous winner (0.40/0.45/0.15) SRCC on test_vila: 0.4544
Current V3 (0.25/0.30/0.45) SRCC on test_vila: 0.4199

**RECOMMENDED**: Use pooled winner **NIMA 0.40 / RGNet 0.50 / A-LAMP 0.10** as it generalizes across both gold sets.
