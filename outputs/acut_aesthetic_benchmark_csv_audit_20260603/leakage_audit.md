# Leakage Audit

## Split Methodology
- Seed: 42
- Train/Val/Test: 60/20/20 (deterministic permutation)
- Calibration (Linear Fit): Fit on 60% train, applied to all.

## Findings
- Deterministic permutation used.
- Calibration correctly fit on training fold only (verified by run_weight_search.py).
- Selected metrics reported on held-out test (20%).
- Risk: Weight tuning (grid search) was done on the validation fold (20%). The final "test" metrics reported are indeed on the held-out 20% fold which was NOT used for weight tuning or calibration.
- Conclusion: No major leakage found in the split protocol.
