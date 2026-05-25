# DistortionGuard Stage B3 FLIVE-Weighted Diagnosis

## 1. Summary
The DistortionGuard Stage B3 experiment aimed to improve performance on the FLIVE dataset by applying a 2.0x sample weight during training. While validation metrics on a mixed dataset (`val.csv`) showed significant improvement (SRCC 0.6710 -> 0.7208), these gains did not translate to the held-out FLIVE test set (SRCC 0.3985 -> 0.4008). Furthermore, the model's performance on the Hard-FP v2 set worsened (mean score 63.68 -> 64.23), indicating an increase in false positives for technical failure cases.

## 2. Artifact Review
- **Stage B3 Training:** Completed 8 epochs. Best epoch was 8.
- **Validation Metrics (Mixed):** Significant jump in SRCC (+0.05) and PLCC (+0.03).
- **Held-out FLIVE Metrics:** Negligible improvement in SRCC (+0.0023).
- **Hard-FP v2:** Mean score increased (+0.55), and images with score > 65 increased from 28 to 29.
- **Weights:** Stage B2 weights were loaded as the starting point. BatchNorm was successfully frozen.

## 3. Weighting Implementation Check
- **Huber Loss:** Correctly implemented with `sample_weights`.
- **PLCC Loss:** **NOT** implemented with `sample_weights`. The `_plcc_loss` function in `train_distortionguard_stageB_authentic.py` calculates correlation across the entire batch without weighting.
- **Impact:** Since PLCC loss (lambda=0.1) is unweighted, it is dominated by datasets with larger score variance (KonIQ/SPAQ), diluting the effect of FLIVE weighting in the Huber loss (lambda=1.0).

## 4. FLIVE Train/Test Mismatch Analysis
- **Distributions:** Training, validation, and test splits for FLIVE have near-identical distributions (Mean ~0.72, Std ~0.06).
- **Narrow Range:** FLIVE has a very narrow score distribution compared to KonIQ (Std 0.15). This makes SRCC sensitive to small ranking errors and makes it harder to "move the needle" compared to KonIQ.
- **Potential Leakage:** The discrepancy between `val_srcc` (0.72) and `test_flive_srcc` (0.40) is striking. While `val.csv` is a mixed dataset, the improvement in `val_srcc` during B3 suggests either overfitting to the specific image content in `val.csv` or that the aggregate metric on a small mixed set is a poor proxy for single-dataset held-out performance.

## 5. Hard-FP v2 Failure Analysis
- **Top False Positives:** Many top FPs in B3 are images from the `flive/voc_emotic_ava/` directory flagged as `low_resolution_issue` or `technical_bad`.
- **Bias Shift:** Because FLIVE training samples have high MOS (avg 0.72) despite often containing similar technical artifacts, weighting them more forced the model to learn a more "forgiving" bias.
- **Confirmed Case:** Image `VOC2012__2012_000254.jpg` (low-res issue) saw its score increase from 60.06 (B2) to 61.21 (B3).

## 6. Comparison with B2
- Stage B3 starting from B2 weights failed to provide a meaningful delta on any held-out technical metric.
- The slight improvement in FLIVE (+0.002) is statistically insignificant compared to the regression in Hard-FP (+0.55 mean).
- B2 remains the more stable and reliable "technical" model.

## 7. Root Cause Hypotheses
1. **Unweighted Correlation Loss:** The PLCC component of the loss function ignored sample weights, causing the model to prioritize the correlation of KonIQ/SPAQ (which have higher signal variance) over FLIVE.
2. **Label Noise & Permissiveness:** FLIVE labels are biased towards high scores even for images with technical defects. Weighting these labels makes the model ignore the very distortions it is meant to "guard" against.
3. **Metric Sensitivity:** Improving SRCC on a narrow-range dataset like FLIVE is inherently difficult without very precise labels; weighting might just be amplifying label noise.

## 8. Recommended Next Experiments
### Option A: Fully Weighted Loss (Fix Implementation)
- **Purpose:** Ensure all loss components (Huber + PLCC) respect sample weights.
- **Hypothesis:** Weighting the PLCC loss will force the model to prioritize the ranking of FLIVE samples, potentially improving SRCC.
- **Risk:** High risk of further Hard-FP regression if FLIVE labels are indeed noisy.

### Option B: Targeted Hard-FP Guard Audit
- **Purpose:** Stop chasing FLIVE SRCC and focus on penalizing specific failure modes (blur, low-res, overexposure) identified in the Hard-FP set.
- **Hypothesis:** Adding a dedicated "distortion classifier" or a ranking loss between Hard-FP samples and "good" samples will be more effective than general IQA fine-tuning.
- **Recommendation:** Do this after confirming that Stage B2/B3 performance is at a plateau.

### Option C: Stop DistortionGuard Fine-tuning
- **Purpose:** Accept Stage B2 as the current best technical model.
- **Hypothesis:** Current dataset quality/quantity for authentic IQA is insufficient for further gains via simple weighting.
- **Action:** Continue with Stage B2 for deployment or move to Stage C (Aesthetic distillation).

## 9. No-Go / Go Decision
- **B3 should NOT be exported.** It offers no meaningful improvement over B2 and regresses on Hard-FP.
- **B3 should NOT be added to Flutter.**
- **Hard-FP guard should not be added yet** until the failure modes are addressed (Option B).

## 10. Commands for Optional Follow-up
- Compare B2 and B3 predictions on all datasets: `python scripts/eval_distortionguard_stageB_authentic.py --candidate outputs/distortionguard_stageB3_flive_weighted_b3_from_stageB2_20260524`
- Inspect `_plcc_loss` implementation in `src/train/train_distortionguard_stageB_authentic.py` to verify unweighted behavior.
