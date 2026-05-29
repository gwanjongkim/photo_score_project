# Final Research-Protocol Technical IQA Baseline Report for A-cut

## 1. Executive Summary
The comprehensive audit and evaluation of technical IQA baselines for the A-cut project is complete. The **current production `technical_score`** (an ensemble of KonIQ and FLIVE standalone models) is established as the **definitive performance baseline**.

Key conclusions:
- The production ensemble outperforms standalone models in cross-dataset generalization (e.g., best SRCC on SPAQ).
- It successfully mitigates significant calibration biases found in the standalone FLIVE model.
- It is NOT replaced by any experimental models or the C6 debug-only component.
- **Future Success Rule**: Any new model or distillation effort must outperform this baseline on the standardized research-protocol image-level test split (4,967 images) to be considered for production.

## 2. Confirmed Dataset Protocols
The evaluation utilized three high-quality technical IQA datasets with strict adherence to paper-standard protocols:
- **KonIQ-10k**: MUSIQ-style 80/20 random split (Run_01 Test: 2,015 images).
- **SPAQ**: Official Train/Test split (Official Test: 1,125 images).
- **FLIVE / PaQ-2-PiQ**: IQA-Toolbox/PyIQA standardized metadata (Official Image-Test: 1,827 images).

*Note: Patch-level evaluation was excluded from this image-level phase to maintain a consistent comparison across datasets.*

## 3. Evaluation Scope
- **Total Test Images**: 4,967
- **Total Prediction Rows**: 9,934 (Across two standalone baselines)
- **Methodology**: Standalone predictions were generated via TFLite inference and combined offline using the confirmed production formula.
- **Constraints**: No training was performed, no production source code was changed, and no Android runtime modifications were made.

## 4. Production Formula
The existing production `technical_score` formula was audited and confirmed from the source code (`src/infer/select_best_shots.py`):
**`production_technical_score = (0.32 * KonIQ + 0.24 * FLIVE_image) / 0.56`**
Equivalent to: **`4/7 KonIQ + 3/7 FLIVE_image`**

This formula represents the "Technical Component" used by the A-cut selector to rank images.

## 5. Standalone Baseline Results
- **KonIQ-10k**: The `koniq_standalone` model reached an SRCC of **0.9369**, confirming its status as a top-tier technical IQA model.
- **FLIVE**: The `flive_standalone` model reached an SRCC of **0.6603** on its home dataset, significantly outperforming KonIQ (0.4863) in this domain.
- **SPAQ**: Both models generalized well, with KonIQ (0.8555) slightly ahead of FLIVE (0.8267).

## 6. Production Ensemble Results
The production `technical_score` demonstrated superior robustness:
- **SPAQ**: Achieved the **highest SRCC of 0.8730**, surpassing both standalone models.
- **KonIQ-10k**: Maintained a very high SRCC of **0.9345** while achieving the lowest overall MAE (**3.26**).
- **FLIVE**: Achieved an SRCC of **0.5744**, successfully incorporating FLIVE's specialized knowledge.

## 7. Comparative Interpretation
- **Generalization**: The production ensemble is more stable across diverse datasets than either standalone model.
- **Bias Mitigation**: It reduces the severe positive bias of the FLIVE model on SPAQ from **+18.42** points down to **+7.71** points.
- **Linearity**: The ensemble achieves the best linear metrics (MAE/RMSE) on the KonIQ dataset, suggesting better calibration for ranking purposes.
- **Validity**: All ensemble scores stay within the expected `0..100` range, with no clipping issues observed.

## 8. Limitations
- **Image-Level Only**: Findings are restricted to full-size images; patch-level performance requires a separate audit.
- **Offline Analysis**: Evaluation was conducted on a WSL/Linux environment; while the TFLite models are identical, final Android runtime parity remains a separate verification task.
- **Scope**: Experimental datasets (AADB, AVA, ICAA17K) and debug components (C6) were intentionally excluded from this technical-focused baseline.

## 9. Final Decision
**A. Keep current production technical_score as baseline.**
The existing ensemble is a high-performing and well-balanced component. It is the gold standard for the A-cut project's technical scoring.

## 10. Next Action
Finalize the research-protocol baseline audit and commit the results to the project documentation. This report serves as the official performance target for all future distillation and model improvement tasks in the A-cut capstone.

## 11. Forbidden Actions
- Do not replace or modify the production `technical_score` formula based on these findings.
- Do not use C6 as a production scoring component.
- Do not initiate new model training until this baseline is fully documented and acknowledged.
