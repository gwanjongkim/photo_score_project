# MDIQA to Student Gap Analysis Report

## 1. Executive Summary
The MDIQA → Student gap analysis on the official 4,967-row image-only benchmark has been completed. The teacher (MDIQA) achieves an overall SRCC of **0.8227** while the mobile student (14k training rows) achieves **0.7436**. The analysis reveals a significant generalization gap: the student performs reasonably well on images seen during its distillation training but suffers a large performance drop on unseen (test/val/absent) splits, whereas MDIQA remains highly robust. This indicates the primary bottlenecks are **training data coverage** and potentially **student backbone capacity**.

**Final Decision: STUDENT_GAP_MAINLY_CAPACITY_AND_COVERAGE**

## 2. Sources Inspected
- `outputs/official_protocol_mdiqa_signal_benchmark_20260604/official_mdiqa_vs_candidates_predictions.csv`
- `outputs/student_distillation_manifest_14105_20260603/student_distillation_manifest.csv`

## 3. Data Join and Coverage
- **Official Benchmark Rows**: 4,967
- **MDIQA Predictions**: 4,967
- **Student Predictions**: 4,967
The predictions were successfully joined row-by-row for accurate paired analysis.

## 4. Overall Gap Summary
- **Overall SRCC Gap**: `-0.0791` (Teacher: 0.8227 vs Student: 0.7436)
- **Overall MAE Gap**: `+0.9853` (Teacher: 8.41 vs Student: 9.39)
- **Teacher-Student Correlation (SRCC)**: `0.8038`

## 5. Dataset-wise Gap Analysis
- **KonIQ-10k**: Teacher SRCC `0.9311` vs Student SRCC `0.7933` (Gap: `-0.1378`)
- **SPAQ**: Teacher SRCC `0.9136` vs Student SRCC `0.8661` (Gap: `-0.0475`)
- **FLIVE**: Teacher SRCC `0.5692` vs Student SRCC `0.4329` (Gap: `-0.1364`)

The student struggles most significantly to replicate the teacher's precision on KonIQ and FLIVE.

## 6. MOS-bin Gap Analysis
- **0-40 (Low Quality)**: Student MAE `14.51` vs Teacher MAE `11.79`. Student heavily overpredicts low-quality images (e.g., scoring 55 on a MOS 4 image).
- **40-60 (Med-Low Quality)**: Student MAE `8.31` vs Teacher MAE `7.08`.
- **60-75 (Med-High Quality)**: Student MAE `6.06` vs Teacher MAE `6.85`. (Student actually has lower absolute error here, heavily clustering predictions in the 55-65 range).
- **75-100 (High Quality)**: Student MAE `13.42` vs Teacher MAE `10.29`. Student severely underpredicts high-quality images.

## 7. Ranking Consistency
- **Student vs MOS**: 0.7436
- **MDIQA vs MOS**: 0.8227
- **Student vs MDIQA**: 0.8038
The student is worse at ranking. It collapses its predictions toward the mean, failing to distinguish fine gradations at the quality extremes.

## 8. Safety Gap Cases
- **MDIQA Severe FPs (MOS < 40, Pred > 65)**: 0
- **Student Severe FPs**: 3
- **Safety Gap**: 3 cases where MDIQA safely scored a bad image low, but the student erroneously scored it high (>65).

## 9. Training Overlap Analysis (The Generalization Gap)
The 4,967 official test images intersect with the 14k student training manifest as follows:
- **`train` split (2479 rows)**: Student SRCC `0.857` vs Teacher `0.894` (Gap: `-0.037`)
- **`test` split (307 rows)**: Student SRCC `0.777` vs Teacher `0.907` (Gap: `-0.130`)
- **`val` split (325 rows)**: Student SRCC `0.718` vs Teacher `0.884` (Gap: `-0.165`)
- **`absent` (1856 rows)**: Student SRCC `0.463` vs Teacher `0.623` (Gap: `-0.160`)

**Finding**: The student overfit the 11k training rows. When evaluated on unseen data, the gap to the teacher quadruples. 

## 10. Top Failure Patterns
- **High-Quality Underprediction**: The student fails to confidently assign scores above 75. Top underpredictions occur almost entirely in the SPAQ dataset.
- **Low-Quality Overprediction**: The student fails to assign very low scores, clustering them near 50-60. Top overpredictions occur in KonIQ.

## 11. Root Cause Hypotheses
1. **Limited Data Scale**: 11k training images (from the 14k manifest) is insufficient to distill a robust general representation of image quality. The teacher was trained on 100k+ images.
2. **Model Capacity**: The lightweight mobile backbone (e.g., EfficientNetV2B0/MobileNetV3) lacks the parameter capacity to match the teacher's complex feature extraction without significantly more diverse supervision.
3. **Regression vs Ranking**: Mean Squared Error (MSE/MAE) distillation encourages the student to predict the mean, penalizing extreme scores and leading to the observed scale compression (underpredicting highs, overpredicting lows).

## 12. Recommended Student Improvement Plan
To close the gap, we must improve the distillation pipeline rather than abandoning the teacher:
1. **Increase Training Data Volume**: Expand the training set from 14k to a much larger scale (e.g., 50k+ or the full 159k dataset) using cached MDIQA teacher labels.
2. **Increase Student Capacity**: Experiment with a slightly larger backbone (e.g., EfficientNetV2-B1 or B2) to provide more representational power.
3. **Implement Pairwise/Ranking Loss**: Introduce a contrastive or pairwise ranking loss during distillation to force the student to maintain relative order, preventing it from collapsing predictions to the mean.

## 13. Final Decision
**STUDENT_GAP_MAINLY_CAPACITY_AND_COVERAGE**

## 14. Next Actions
- Prepare a scaled-up distillation training manifest (50k+ rows).
- Re-run student training with the expanded dataset and evaluate generalization.
