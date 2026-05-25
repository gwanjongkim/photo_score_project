# DistortionGuard Hard-FP v2 Failure Taxonomy

## 1. Summary
This audit investigated the failure modes of technical IQA models on the Hard-FP v2 dataset (n=44). The primary finding is that **KonIQ-Mobile** remains the most effective technical guard, while **FLIVE-trained models** (including DistortionGuard B2/B3 and TOPIQ Mixed112) significantly over-score images with confirmed technical failures. DistortionGuard B3 worsened the over-scoring issue, particularly for images in the `low_resolution_issue` category.

## 2. Artifacts Inspected
- `data/processed/techiqa_guard/fp_mining_20260522/hard_false_positive_confirmed_v2.csv` (Manifest)
- `outputs/eval_techiqa_guard_v1_hard_fp_confirmed_v2_20260522/` (Baseline scores)
- `outputs/eval_distortionguard_stageB2_partial_unfreeze_b3_from_stageB_e10_20260524/predictions_hard_fp_v2.csv`
- `outputs/eval_distortionguard_stageB3_flive_weighted_b3_from_stageB2_20260524/predictions_hard_fp_v2.csv`

## 3. Merged Hard-FP Score Table
A merged CSV containing scores for all 44 images across 8 model variants was generated at:
`outputs/distortionguard_hard_fp_v2_failure_taxonomy_20260524/merged_hard_fp_scores.csv`

## 4. Per-Model Hard-FP Behavior

| Model | Mean Score | Count > 65 | Behavior |
| :--- | :---: | :---: | :--- |
| **KonIQ-Mobile** | **38.42** | **1** | **Strongest Guard.** Correctily identifies most failures. |
| Existing Avg | 47.96 | 1 | Moderate guard. |
| TechIQA Stage 4 | 62.01 | 21 | Weak guard. Over-scores significantly. |
| **DistortionGuard B2** | **63.68** | **28** | **Weak guard.** Over-scores due to FLIVE bias. |
| **DistortionGuard B3** | **64.23** | **29** | **Worst guard.** FLIVE weighting worsened bias. |
| TOPIQ Mixed112 | 65.35 | 28 | Very weak. Highly semantic-driven. |
| FLIVE-Mobile | 57.50 | 24 | Unreliable. Inconsistent on technical defects. |

## 5. B2 vs B3 Delta Analysis
- **Overall Shift:** B3 mean score increased by +0.55 points on Hard-FP v2.
- **Top Worsened Image:** `AVA__60509.jpg` (+0.78 delta).
- **Categorical Impact:** `low_resolution_issue` saw the highest average increase (+0.56), reaching an average score of **68.82**.

## 6. KonIQ vs FLIVE Guard Comparison
- **Confirmed Fact:** KonIQ Mobile scores are ~19 points lower than FLIVE Mobile scores on average for Hard-FP images.
- **Inference:** FLIVE labels are inherently more "permissive" of technical distortions than KonIQ labels. This makes FLIVE a poor choice for training a technical "guard" model.

## 7. Failure-Type Taxonomy
Based on `user_confirmed_category` in the manifest:

| Category | Count | B3 Avg Score | Analysis |
| :--- | :---: | :---: | :--- |
| **Low Resolution** | 19 | **68.82** | **Highest Failure Rate.** Models fail to detect downscaling artifacts. |
| **Technical Bad** | 11 | 60.62 | Includes blur, noise, lighting. Better than low-res but still over-scored. |
| **Manual Existing** | 14 | 61.68 | User-flagged samples from existing model failures. |

## 8. Contact Sheet Review
Top 5 most over-scored images by B3 relative to KonIQ-Mobile:
1. `EMOTIC__3ytrr4bu33pv3e36ln.jpg` (B3: 71.4, KonIQ: 25.5, **Delta: 45.9**)
2. `AVA__60509.jpg` (B3: 71.3, KonIQ: 35.4, **Delta: 35.9**)
3. `EMOTIC__sun_afkzndehpwgjdxcc.jpg` (B3: 72.0, KonIQ: 35.5, **Delta: 36.5**)
4. `EMOTIC__3ejdmzqxrcxiglybzz.jpg` (B3: 74.6, KonIQ: 44.5, **Delta: 30.1**)
5. `AVA__50690.jpg` (B3: 69.8, KonIQ: 41.2, **Delta: 28.6**)

## 9. Root Cause Hypotheses
1. **FLIVE Label Noise/Permissiveness:** The FLIVE dataset (particularly the VOC/EMOTIC/AVA subset) contains many technically poor images that humans rated highly due to semantic content. Weighting FLIVE in B3 amplified this "semantic bias."
2. **Backbone Semantic Dominance:** EfficientNet/TOPIQ backbones are trained on ImageNet and prioritize semantic features. Without explicit low-level distortion training (like KonIQ's synthetic distortions), they ignore technical artifacts.
3. **Resolution Masking:** The fixed-resolution 384x384 input may be hiding pixel-level low-resolution artifacts that would be more obvious at native scale or via patch-based analysis.

## 10. Recommended Next Experiments

### Option A: KonIQ-Reference Guard Loss (Hard-FP Guard)
- **Purpose:** Force the model to align with KonIQ-Mobile's lower scores on technical failures.
- **Hypothesis:** Adding a loss term that penalizes deviations from KonIQ scores specifically on a "guard" dataset (including Hard-FP) will lower the false-positive rate.
- **Risk:** Might lower valid high scores on authentic images if not carefully balanced.

### Option B: Auxiliary Low-Resolution Task
- **Purpose:** Train the model to explicitly detect downscaling or resolution limits.
- **Hypothesis:** A multi-task head for resolution classification or an auxiliary "pixel-variance" loss will improve detection of `low_resolution_issue`.
- **Expected Improvement:** Significant reduction in over-scoring for the 68.8 avg category.

## 11. No-Go / Go Decision
- **No-Go on B3 export.**
- **No-Go on B3 Flutter integration.**
- **Confirmed:** Do NOT use FLIVE-Mobile as a hard-FP guard reference. Use KonIQ-Mobile instead.
