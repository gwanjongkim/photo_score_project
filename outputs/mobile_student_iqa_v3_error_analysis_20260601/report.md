# Mobile Student IQA v3 Error Analysis Report

## 1. Executive Summary
Mobile Student IQA v3 (trained on 50k guarded labels with EfficientNetV2B0) failed to achieve production-grade performance. While the overall Mean Absolute Error (MAE) improved compared to v2 (4.54 vs 7.23), the **ranking correlation (SRCC) stagnated at 0.658** and **severe false positives (MOS < 40, Pred > 65) surged to 28 cases** (vs 5 in v2). 

The primary failure pattern is a **range collapse in the FLIVE dataset**, which constitutes 64% of the training data. The model predicts a very narrow range (~70) for FLIVE images, failing to rank them effectively while simultaneously pulling low-score predictions from KonIQ and SPAQ upward, causing safety violations. All 28 severe FPs were found to be **student-amplified**, meaning the teacher labels were safe, but the student overpredicted by an average of 26.6 points.

## 2. Files Inspected
- `outputs/mobile_student_iqa_v3_20260601/predictions_test.csv`
- `outputs/mobile_student_iqa_v3_20260601/training_history.csv`
- `outputs/mobile_student_iqa_v3_20260601/severe_false_positive_cases.csv`
- `outputs/topiq_teacher_label_expansion_50k_20260601/guarded_labels.csv`

## 3. Integrity Check
- **Prediction Files**: All split files exist (train/val/test).
- **Row Counts**: 34,998 train / 7,507 val / 7,495 test. Matches manifest.
- **Missing Values**: 0.
- **Finite Count**: 100% finite.
- **Prediction Range**: Observed [13.3, 85.6]. Well within the 0..100 scale.
- **Load Errors**: 0 reported.

## 4. Prediction Distribution Analysis
| Split | Target Mean | Pred Mean | Target Std | Pred Std | Std Ratio |
| --- | --- | --- | --- | --- | --- |
| Train | 62.00 | 62.10 | 15.15 | 13.46 | 0.89 |
| Test | 62.06 | 62.41 | 15.02 | 12.75 | 0.85 |

**Decile Analysis (Test Set):**
The model suffers from "regression toward the mean":
- **Decile 0 (Target ~28)**: Mean prediction **39.6** (+11.6 bias).
- **Decile 9 (Target ~78)**: Mean prediction **70.9** (-7.1 bias).
- **Middle Deciles**: Bias is low (~1-2 points), but ranking within the decile is weak.

## 5. Dataset-specific Analysis
| Dataset | Target Mean | Pred Mean | Target Std | Pred Std | SRCC | MAE |
| --- | --- | --- | --- | --- | --- | --- |
| **FLIVE** | 71.0 | 70.0 | 4.6 | **4.6** | 0.514 | 3.30 |
| **KonIQ-10k** | 63.4 | 66.8 | 10.2 | 7.6 | 0.644 | 5.94 |
| **SPAQ** | 50.4 | 52.3 | 18.4 | 17.2 | 0.841 | 7.57 |

- **FLIVE**: Low MAE (3.30) but very low SRCC (0.514). The model predicts the mean for almost all 4,799 FLIVE test samples.
- **SPAQ**: Strongest SRCC (0.841) but highest MAE (7.57). Ranking is preserved but absolute calibration is poor.
- **KonIQ-10k**: Moderate ranking and high upward bias at the low end (+13.4 bias in decile 0).

## 6. Severe False Positive Analysis
- **Total Severe FPs (Test)**: 28 cases (MOS < 40, Pred > 65).
- **Distribution**: 19 SPAQ, 7 KonIQ-10k, 2 FLIVE.
- **Root Cause Categorization**:
  - **Target-Inherited**: 0 cases. (In all cases, the guarded target was <= 65).
  - **Student-Amplified**: 28 cases. (平均 amplification: **+26.6 points**).

**Verdict**: The student is failing to respect the safety guard provided in the teacher labels. It is overpredicting low-quality images, likely because they represent a small fraction of the training data compared to the high-score FLIVE majority.

## 7. Standard FP/FN Analysis
- **FP (MOS < 50, Pred > 65)**: 145 cases. Concentrated in SPAQ and KonIQ.
- **FN (MOS > 75, Pred < 45)**: 7 cases. Model occasionally crashes on high-quality images.

## 8. Teacher Label Safety vs Student Safety
- **Teacher Labels (50k)**: Severe FP = 0.
- **Student v3 (Test 7.5k)**: Severe FP = 28.
The student has significantly degraded the safety of the pipeline. Training on MAE alone allows the student to "ignore" these rare safety violations to reduce overall loss on the dominant FLIVE samples.

## 9. Train/Val/Test Generalization
- **Train SRCC**: 0.817
- **Val SRCC**: 0.667
- **Test SRCC**: 0.658
A significant generalization gap (Delta 0.15-0.16) remains despite using 35k training images. This suggests the features learned from EfficientNetV2B0 at 256x256 are not robust enough for cross-dataset technical IQA or are overfitting to dataset-specific biases.

## 10. Training History and Checkpoint Selection
- **Best Model Selection**: Selected by `val_mae`.
- **Observation**: Training history shows `val_mae` reached a plateau early in Stage B. SRCC and Safety were not used for checkpoint selection, meaning we likely chose a model that predicts the mean well but ranks poorly.

## 11. Root Cause Hypotheses
1. **Dataset Imbalance (High Confidence)**: FLIVE (64%) dominates the training set with high scores, pulling the model's baseline prediction toward ~70.
2. **Loss-Metric Mismatch (High Confidence)**: Huber loss optimizes the mean. Rare low-quality images are treated as outliers and ignored.
3. **Checkpoint Selection Mismatch (Medium Confidence)**: `val_mae` favors range collapse (playing safe) over ranking and safety.
4. **Input Size Limitation (Medium Confidence)**: 256 resolution may still be too low to resolve the fine technical artifacts the TOPIQ teacher uses for ranking.

## 12. Recommended v4 Plan
V4 must move beyond simple regression and incorporate **ranking** and **safety** constraints.

1. **Balanced Batch Sampling**: Ensure every training batch has an equal number of samples from FLIVE, KonIQ, and SPAQ.
2. **Sample Weighting**: Apply higher loss weights to images with MOS < 50.
3. **Ranking Loss**: Add a pairwise ranking loss (e.g., MarginRankingLoss) to force the student to learn order, not just absolute values.
4. **Safety Penalty**: Add an explicit hinge loss penalty for `Pred > 65` when `Guarded_Target < 55`.
5. **Checkpoint Selection**: Save best model based on a composite metric: `SRCC - 5 * Severe_FP_Count`.

## 13. Final Recommendation
`READY_FOR_V4_CODEX_IMPLEMENTATION`

## 14. Next Codex Prompt
"Implement the Mobile Student IQA v4 training script. Use the 50k student manifest. 
Key changes:
1. **Balanced Batch Sampling**: Implement a custom data loader that samples 1/3 of the batch from each dataset (FLIVE, KonIQ, SPAQ).
2. **Sample Weighting**: Assign 3x weight to samples with `quality_score_0_100 < 50`.
3. **Loss Function**: Use a combination of `Huber(0.1)` and a custom `SafetyPenaltyLoss` that penalizes `pred > 0.65` if `target < 0.55`.
4. **Checkpoint Selection**: Implement a custom callback to save the best model using the composite metric: `SRCC - (Severe_FP_Count / 10.0)`.
5. **Backbone**: EfficientNetV2B0 with 320x320 input to better resolve technical artifacts."
