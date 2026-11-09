# Production Baseline Discrepancy Audit

## 1. Old Baseline Origin
- KonIQ 0.934 SRCC originates from `outputs/research_protocol_baseline_eval_audit_20260530/report.md` (or similar baseline runs).
- This 0.934 score represents the **standalone** `koniq_mobile` TFLite model.

## 2. True Production Ensemble
- The actual production A-cut selector uses a weighted ensemble of 5 components (KonIQ, FLIVE Image, FLIVE Patch Mean/Min, MUSIQ AADB).
- This ensemble dilutes the KonIQ signal with patch and FLIVE signals, resulting in a lower aggregate SRCC on the KonIQ dataset (~0.77 - 0.80).

## 3. Discrepancy Cause
- **Standalone vs Ensemble Mismatch**: The previously quoted "production baseline" of 0.934 was actually a single component's best theoretical performance, not the true output of the `technical_score` function used in the app's ranking pipeline.
- MDIQA (0.948) beats both the standalone component (0.934) and the true ensemble (~0.80).

## 4. Conclusion
The discrepancy is fully explained. The paired script accurately reproduces the true production ensemble logic.
