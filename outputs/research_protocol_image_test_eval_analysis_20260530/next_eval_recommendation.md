# Next Evaluation Recommendation

The image-level test evaluation has provided a solid baseline for the two standalone models. We now know that the KonIQ model is highly accurate and well-calibrated for KonIQ and SPAQ, while the FLIVE model is necessary for FLIVE but poorly calibrated for other datasets. The next logical step is to evaluate the actual production ensemble to see how these strengths and weaknesses balance out.

## Exact Next Action
**Evaluate current production technical_score combination on the same image-level test split.**

## Reason
We have the metrics for the individual components (KonIQ standalone, FLIVE standalone). However, we do not yet have the "ground truth" performance of the actual ensemble used in the app. Evaluating this combination on the same 4,967 images will provide the definitive baseline against which all future research or training efforts will be measured. It will also reveal whether the current averaging/weighting strategy successfully mitigates the calibration issues found in the standalone FLIVE model.

## Stop Conditions
- Error in implementing the ensemble logic (averaging/weighting) that results in scores outside the 0..100 range.
- Execution time exceeds 5 minutes.
