# TechIQA-Guard v1 Hard-FP Confirmed v2 Evaluation Report

Evaluating models on the expanded confirmed hard-FP set (v2).
- n = 44

## 1. Summary Metrics
| model | n | mean | std | min | max | median |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| techiqa_stage4 | 44 | 68.07 | 8.28 | 44.10 | 78.08 | 69.92 |
| techiqa_stage4b | 44 | 68.82 | 7.97 | 45.84 | 78.90 | 70.70 |
| techiqa_stage5_v2 | 44 | 68.53 | 8.98 | 42.96 | 79.30 | 70.23 |
| techiqa_stage5_v3 | 44 | 68.79 | 8.57 | 44.20 | 78.99 | 70.54 |
| topiq_mixed112 | 44 | 65.35 | 11.05 | 33.09 | 80.14 | 68.42 |
| topiq_ranking | 44 | 58.12 | 13.97 | 21.06 | 78.10 | 62.29 |
| koniq_mobile | 44 | 47.84 | 5.93 | 33.13 | 59.25 | 47.88 |
| flive_mobile | 44 | 74.19 | 2.83 | 64.91 | 79.58 | 74.41 |
| existing_avg | 44 | 61.02 | 3.90 | 50.83 | 67.63 | 61.63 |

## 2. Model Comparison Analysis
- **Stage 4 Mean**: 68.07
- **Stage 4B Mean**: 68.82
- **Absolute Improvement**: -0.75 points lower

Stage 4B regressed (higher scores) compared to Stage 4 on this set.

## 3. Top 10 Over-scored Images (Stage 4)
| filename | techiqa_s4 | mixed112 | existing_avg | delta_s4_avg |
| :--- | :--- | :--- | :--- | :--- |
| AVA__60509.jpg | 78.08 | 69.10 | 58.11 | 19.97 |
| VOC2012__2008_006473.jpg | 78.05 | 73.48 | 57.53 | 20.52 |
| EMOTIC__3ejdmzqxrcxiglybzz.jpg | 77.71 | 75.34 | 65.57 | 12.13 |
| EMOTIC__COCO_train2014_000000440339.jpg | 77.37 | 76.58 | 64.77 | 12.60 |
| JPEGImages__2008_001456.jpg | 77.36 | 75.86 | 60.24 | 17.12 |
| VOC2012__2008_004778.jpg | 76.40 | 73.87 | 56.63 | 19.77 |
| EMOTIC__e6mfk3boq4jnl6tpob.jpg | 75.35 | 70.99 | 65.51 | 9.85 |
| AVA__281607.jpg | 75.34 | 80.14 | 50.83 | 24.51 |
| VOC2012__2011_005579.jpg | 74.80 | 70.50 | 64.20 | 10.60 |
| EMOTIC__2d8m251io0nvtitrlm.jpg | 74.71 | 76.22 | 64.97 | 9.74 |

## 4. Conclusion
All current TechIQA-Guard models (mean ~68.07) are still scoring significantly higher than the existing technical guard baseline (mean 61.02) on this hard false-positive set.

Further training with oversampling of these confirmed samples is required.
