# TOPIQ Teacher Performance Audit Report

## 1. Executive Summary
The audit of the TOPIQ-based teacher and guarded-label pipeline reveals that the **raw teacher is a performance downgrade compared to the existing production technical score**. While the **Guarded Label** design successfully artificially inflates metrics by blending 50% Ground Truth (MOS), student models (v3, v4a, v4b) fail to significantly surpass the raw teacher's inherent weaknesses, particularly on the FLIVE dataset. 

The current pipeline is **NOT** ready for further student distillation aimed at production replacement without a fundamental revision of the teacher or the label design. The high SRCC of the guarded labels is primarily tautological and hides the raw TOPIQ's failure to rank authentic distortions (FLIVE) effectively.

## 2. Files Inspected
- Teacher validation & holdout reports (20260531, 20260601)
- 50k expansion and guarded label metrics (20260601)
- Student training manifests and reports (v3, v4a, v4b)
- Existing production baseline reports (20260530)
- Student v3 error analysis report (20260601)

## 3. Raw TOPIQ Performance
Raw TOPIQ (topiq_nr) is significantly weaker than the existing production models.

| Dataset | Raw TOPIQ SRCC | Production SRCC | Gap |
| --- | --- | --- | --- |
| **FLIVE** | 0.251 | 0.574 | **-0.323** |
| **KonIQ-10k** | 0.857 | 0.934 | **-0.077** |
| **SPAQ** | 0.803 | 0.873 | **-0.070** |

- **FLIVE Weakness**: Raw TOPIQ's correlation on FLIVE is near-random (0.25-0.30). It fails to capture the technical nuances of the FLIVE dataset.
- **KonIQ/SPAQ**: While better, it still lags behind the existing KonIQ-standalone and SPAQ-benchmark results.

## 4. Calibrated TOPIQ Performance
Dataset-wise isotonic calibration was used to align raw TOPIQ with MOS. While this improves MAE, it does not change SRCC.
- **Range Collapse**: On FLIVE, the calibrated teacher predicts a very narrow range (Std 3.7 vs MOS Std 4.6), effectively "giving up" on ranking FLIVE images and predicting the mean.

## 5. Guarded Label Performance
The guarded label uses `0.5 * calibrated_topiq + 0.5 * MOS`.
- **Inflated Metrics**: Guarded labels show SRCC > 0.92 on all datasets. This is **tautological** because it includes the very MOS it is being evaluated against.
- **Signal Dilution**: On FLIVE, the guarded label is 50% MOS signal and 50% noise/weak-signal (TOPIQ). 

## 6. Leakage and Split Safety
- **Integrity**: 0 leakage was found. Image paths and source groups (patches) are correctly separated across student train/val/test splits.
- **Holdout Safety**: The holdout set remained untouched during training manifest generation.

## 7. Teacher vs Existing Baselines
The user requirement is a "clearly better" replacement.
- **Current Status**: The teacher is a **downgrade**.
- **Student Status**: Student v4b (SRCC 0.47 on FLIVE) is worse than production (SRCC 0.57 on FLIVE).
- **Conclusion**: We are attempting to distill a model into a student that is already worse than what is in production.

## 8. Teacher Signal vs MOS Blending
- **Tautology**: The high correlation of guarded labels is a design artifact, not a performance breakthrough.
- **Validity**: The blending preserves dataset-wise ranking only because MOS is present. Without MOS (at runtime), the student must rely on the weak TOPIQ features.

## 9. Student Failure Implications
Student failures (v3, v4a, v4b) are a combination of:
1. **Teacher Limitation**: The student cannot learn features that the teacher doesn't have (especially for FLIVE).
2. **Safety Degradation**: The student (v3) produced 28 severe FPs where the teacher had 0. The student is "amplifying" errors, possibly due to dataset imbalance (FLIVE dominating the high-score range).
3. **Resolution**: 256x256 might be too low to capture the artifacts TOPIQ uses (if any) on FLIVE.

## 10. Risks
- **Replacement Failure**: Deploying a model based on this teacher will likely regress technical IQA quality in production.
- **Masked Weakness**: The guarded label design makes the pipeline look successful until the student is evaluated against MOS independently.

## 11. Final Decision
**`TEACHER_TARGET_NEEDS_REVISION`**

The current TOPIQ-based teacher is not a valid baseline for production replacement because it is objectively worse than the current production scoring on all benchmark datasets.

## 12. Recommended Next Step
1. **Pivot Teacher**: Consider a stronger teacher (e.g., MUSIQ, LIOP) or a fusion teacher that actually beats production SRCC on FLIVE and KonIQ **before** distillation.
2. **Re-evaluate TOPIQ**: Determine why TOPIQ is so weak on FLIVE. Check if it requires higher resolution (e.g., 512+) or if the `topiq_nr` variant is unsuitable for authentic distortions.
3. **Refine Label Design**: If MOS is available for 50k images, consider using a higher MOS blend (e.g., 80/20) or pure MOS targets for those images, rather than dragging them down with a weak teacher signal.
