# Final TechIQA-Guard v1 NO-GO Report

## 1. Summary
The TechIQA-Guard v1 experiment, aimed at creating a single-model replacement for the existing dual-expert technical guard system, is concluded with a **NO-GO** decision. While initial results on a small sample were promising, evaluation on an expanded confirmed hard false-positive set (v2, n=44) revealed that all model candidates significantly over-score technically defective images, failing to provide the necessary safety guard for production.

## 2. Experiment Timeline
- **Stage 1-3**: Architecture selection and initial frozen backbone tests.
- **Stage 4**: Established initial FP guard on 3 manual images (Mean: 49.05).
- **Stage 5**: Introduced pairwise ranking to boost FLIVE performance; resulted in guard regression (Mean: ~63-64).
- **Hard-FP Mining**: Algorithmically identified and visually confirmed 44 hard-FP images (v2).
- **Stage 4B**: Attempted to improve Stage 4 by training specifically on the expanded v2 set.
- **Final Evaluation**: Comparative test of all candidates on the expanded v2 set.

## 3. Stage 4 vs Stage 4B vs Stage 5
Comparative performance on the **expanded hard-FP v2 set (n=44)**:
- **Stage 4**: Mean score **68.07**. Established as the baseline for this experiment.
- **Stage 4B**: Mean score **68.82**. Failed to improve protection; actually **regressed** by +0.75 points.
- **Stage 5**: Mean score **~68.5-68.8**. The addition of ranking loss further destabilized the technical guard behavior.

## 4. Expanded hard-FP v2 Result
The expanded evaluation confirmed that general IQA models and our new TechIQA candidates are not safe enough compared to existing production experts.

| Model | Mean Score (n=44) | Status |
| :--- | :--- | :--- |
| **koniq_mobile** | **47.84** | **Strongest Guard** |
| existing_avg | 61.02 | Target Baseline |
| topiq_mixed112 | 65.35 | Overscoring |
| techiqa_stage4 | 68.07 | Unsafe |
| techiqa_stage4b | 68.82 | Unsafe |
| flive_mobile | 74.19 | Weak Guard |

## 5. Why TechIQA-Guard v1 Failed
1.  **Data Imbalance**: Even with the v2 expansion, 44 images are insufficient to counter the gradient signal from 16,000+ general IQA samples and 10,000 ranking pairs.
2.  **Generalization Failure**: The model learned to suppress scores for the original 3 images but failed to learn the *underlying technical features* (blur, noise, compression) that define a false positive.
3.  **Single-Head Interference**: The single-output head is forced to compromise between IQA MOS regression and technical defect suppression. The "Aesthetic/Quality" signal from large datasets naturally pulls the scores upward for visually "pretty" but technically flawed images.

## 6. What We Learned
- A set of 3 manual images is not a representative test for guard robustness.
- Direct single-head training is prone to "leaking" high scores on technically bad images if those images have high aesthetic content.
- `koniq_mobile` remains a surprisingly robust guard for the specific types of technical defects mined in this experiment.

## 7. Current Production Recommendation
**DO NOT DEPLOY TechIQA-Guard v1.**
The existing production pipeline using `koniq_mobile` and `flive_mobile` averages should remain in place. It provides a safer (lower) score on hard false-positives (61.02) than any TechIQA-Guard iteration (68.07+).

## 8. Future Research Options
- **Multi-Head Architecture**: Separate "Quality" and "Technical Defect" heads to prevent signal interference.
- **Aggressive Oversampling**: 100x+ oversampling of a larger (100-200 image) confirmed FP set.
- **Auxiliary Loss**: Incorporate an explicit blur/noise classification loss into the backbone.

## 9. Final Decision
**NO-GO**
Experiment closed. Revert to Stage 4 weights for internal reference only; do not export for mobile.
