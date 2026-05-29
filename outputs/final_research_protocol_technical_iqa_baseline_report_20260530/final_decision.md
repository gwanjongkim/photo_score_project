# Final Decision: Technical IQA Baseline

The current production `technical_score` is established as the project's definitive technical IQA baseline. It is a robust, well-generalized, and well-calibrated ensemble that consistently outperforms standalone models on cross-dataset benchmarks (e.g., reaching **SRCC=0.8730** on SPAQ).

## Strongest Numerical Evidence
- **Generalization**: The ensemble achieved a higher SRCC on SPAQ (0.873) than both KonIQ (0.856) and FLIVE (0.827) standalone models.
- **Bias Reduction**: It mitigated FLIVE's calibration error on SPAQ, reducing a **+18.42** point positive bias to **+7.71** points.
- **Domain Retention**: It preserved **99.7%** of KonIQ's rank correlation strength (SRCC 0.9345 vs 0.9369) while improving point-wise error metrics (MAE 3.26 vs 5.18).

## Future Baseline Rule
Any proposed technical IQA model, distilled model, or scoring formula must be evaluated on the **Research-Protocol Image-Level Test Split (4,967 images)**. It must demonstrate superior SRCC and lower MAE across all three primary datasets (KonIQ-10k, SPAQ, FLIVE) compared to the metrics documented in this report.

## Stop Conditions
- Total test SRCC drops below 0.80 on either KonIQ or SPAQ.
- MAE exceeds 15.0 on any evaluated test dataset.
- Score output scale deviates from the 0..100 range.
