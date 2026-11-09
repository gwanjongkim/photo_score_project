# Mobile Student IQA v2 Error Analysis

## 1. Executive Summary
The Mobile Student IQA v2 model shows significant improvement over v1 but remains below production readiness (Test SRCC 0.67 < 0.85, 5 Severe FPs). A massive generalization gap exists (Train SRCC 0.88 vs Test SRCC 0.67), and the Mean Absolute Error (MAE) nearly doubles from training to test across all datasets. The model struggles most with the **FLIVE** dataset (Test SRCC 0.36), where it collapses into a narrow prediction range. We hypothesize that the primary bottleneck is **data scarcity** (6.3k training images), leading to overfitting and poor generalization.

## 2. Prediction Distribution
The model is not fully collapsed but suffers from "regression toward the mean."
- **Overall Test Mean**: 62.4 (Target Mean: 62.1)
- **Overall Test Std**: 12.7 (Target Std: 15.0)
- **Tails Failure**: For targets > 80, the mean prediction is only 67.9. For targets < 20, the mean prediction is 28.3. The model consistently fails to predict extreme scores.

## 3. Severe False Positives
Cases where `quality_score_0_100` (MOS) < 40 and `pred_0_100` > 65.
| Dataset | Path | MOS | Target | Pred | Abs Error |
| --- | --- | --- | --- | --- | --- |
| spaq | .../06682.jpg | 38.8 | 52.2 | 72.7 | 20.6 |
| koniq10k | .../6172964730.jpg | 38.6 | 40.3 | 68.4 | 28.1 |
| flive | .../motion0144.jpg_patch_3.jpg | 33.9 | 52.8 | 74.9 | 22.2 |
| spaq | .../08414.jpg | 31.0 | 48.3 | 67.0 | 18.7 |
| spaq | .../06786.jpg | 37.6 | 44.1 | 65.4 | 21.4 |

*Note: In 4/5 cases, the guarded target was already > 40, inheriting the teacher's over-estimation, but the student then amplified this error by another 10-20 points.*

## 4. Top Absolute Errors
| Dataset | Path | Target | Pred | Abs Error | Reason |
| --- | --- | --- | --- | --- | --- |
| flive | patches/motion0048.jpg_patch_2.jpg | 65.4 | 20.0 | 45.4 | Massive under-prediction |
| spaq | spaq/09447.jpg | 25.4 | 63.3 | 37.9 | Severe over-prediction |
| spaq | spaq/10210.jpg | 81.6 | 50.4 | 31.2 | Under-prediction |
| spaq | spaq/06386.jpg | 28.9 | 59.5 | 30.6 | Over-prediction |

## 5. Dataset-specific Failure Patterns
- **SPAQ**: Highest error (MAE 9.68). The model sees the widest range here but often misses the target by 20+ points.
- **KonIQ**: Moderate performance (SRCC 0.60).
- **FLIVE**: Lowest correlation (SRCC 0.36). The model predicts almost everything between 60 and 75.

## 6. FLIVE Collapse Analysis
Analysis of FLIVE path types shows a "mean-seeking" behavior:
- **Patches**: Mean 68.5, Std 7.9 (Count 339)
- **VOC Emotic AVA**: Mean 69.2, Std 6.1 (Count 103)
The model treats FLIVE images as a monolithic block of "slightly above average" quality regardless of actual distortions.

## 7. Train/Test Distribution Check
- **Train Mean/Std**: 62.1 / 13.5
- **Test Mean/Std**: 62.4 / 12.7
The distributions are similar, but the model's ability to map features to these scores fails on unseen data, indicating the features learned from 6.3k images are not robust.

## 8. Root Cause Hypotheses
1.  **Overfitting on Small Data**: 6.3k images is insufficient to learn zero-reference IQA for a backbone like MobileNetV3Small when the targets come from a high-capacity teacher.
2.  **Backbone Capacity**: MobileNetV3Small may be too shallow to capture the subtle artifacts (aliasing, noise, compression) the teacher uses.
3.  **Low Input Resolution**: 224x224 may downsample away the very technical defects we are trying to predict.

## 9. Recommended v3 Plan
Expand the training dataset to include at least 50k-100k images with teacher labels. The current 6.3k training set represents only ~4% of the available 159k manifest. Data expansion is the most reliable way to bridge the generalization gap.

## 10. Next Codex Prompt
"Finalize the technical IQA teacher labeling for the full 159k image manifest. Once labels are generated, reconstruct the student training manifest with a group-based split. Then, prepare for a v3 training run that uses a larger backbone (EfficientNetLite0 or V2B0) and higher input resolution (256x256 or 320x320) to better capture technical artifacts."

**Final Recommendation: V3_DATA_EXPANSION_FIRST**
