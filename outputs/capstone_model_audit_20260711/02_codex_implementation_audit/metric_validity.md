# Metric validity

## Independent recomputation

The bounded checker searched `production_technical_predictions.csv` for a ground-truth column and found none. Consequently SRCC, PLCC, MAE and RMSE could not be independently recomputed from that table. Status: **MISSING**, not passed.

NIMA’s distribution-derived MOS implementation is valid: `distribution_mean_score` multiplies the ten probabilities by 1..10 (`src/models/nima_distribution.py:83-85`). The executed legacy TFLite distribution summed to exactly 1.0 and contained only finite values. This validates output semantics for that single input, not dataset accuracy.

## Metric risks

- KonIQ and FLIVE labels are roughly 0–100, while AADB practical scores are 0–1 and NIMA expected scores are 1–10. Every ensemble must show explicit scale conversion before weighting. Metadata documents dividing technical scores by 100; server formulas must be checked per policy.
- FLIVE patch predictions and image predictions represent different annotation units. Treating a patch mean/min as interchangeable with image MOS is an application heuristic, not paper evaluation.
- AADB/SRCC is rank-based and should not be reported as PLCC without paired raw values. Logistic fitting was not demonstrated for local tables.
- Mobile A-LAMP v2 parity metrics are conversion metrics on 1,024 validation samples, not model accuracy. Existing JSON gives FP32 max/mean absolute differences .01110/.001368 and FP16 .02770/.002673.
- Ensemble weight-search reports are leads. Independent recomputation requires aligned ground truth, immutable row IDs, train-only calibration, held-out selection and the exact deployed formula. Active runtime application remains an `UNKNOWN_LINK` for the calibrated aesthetic ensemble.
- Accuracy requires the AVA threshold and exclusion policy. The Mobile A-LAMP export records threshold 5.0, but parity predictions are probabilities and parity is not accuracy.

No positive metric verdict is assigned where raw ground truth was absent.
