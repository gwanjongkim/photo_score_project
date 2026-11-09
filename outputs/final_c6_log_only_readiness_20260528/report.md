# Final C6 Log-Only Readiness Report

## 1. Executive Summary
The **C6 (TOPIQ + KonIQ Soft-Cap)** technical quality candidate has completed its comprehensive evaluation cycle, including WSL benchmarking, ranking audits, and Android vs. WSL parity diagnosis. 

**Conclusion:** C6 is **GO** for log-only telemetry in the Flutter application. It demonstrates superior ranking performance and maintains acceptable safety against technical false positives. However, it is **NOT approved** for production `technical_score` replacement at this time due to residual preprocessing parity mismatches (Max AE > 5.0) between Android and WSL platforms.

## 2. Candidate Formula
C6 leverages TOPIQ for fine-grained ranking and KonIQ as a safety guard:
```python
# C6 Formula
base_blend = 0.7 * TOPIQ_mixed112_score_100 + 0.3 * KonIQ_score_100
final_c6 = min(base_blend, KonIQ_score_100 + 8.0)
```

## 3. WSL Benchmark Evidence
Benchmark results across FLIVE, KonIQ, and SPAQ datasets confirm C6 as a superior ranking candidate:
- **Average SRCC**: 0.760 (Improvement: +0.015 over current baseline B5).
- **SPAQ SRCC**: 0.900 (Improvement: +0.039).
- **KonIQ SRCC**: 0.892 (Improvement: +0.025).
- **Selection Precision**: Significant improvement in Top-20% selection on high-quality datasets.

## 4. Hard-FP v2 Safety
Safety against over-scoring technical failures is confirmed on the Hard-FP v2 dataset (N=44):
- **Mean Score**: 54.1 (Safe limit: 60.0).
- **Max Score**: 64.79 (Threshold limit: 65.0).
- **Count > 65**: **0 / 44** (Perfect safety at critical threshold).

## 5. Android Log-Only Execution Evidence
Integration into the Flutter runner has been verified:
- **Model Load**: Successful for TOPIQ mixed112 (FP16 TFLite) and KonIQ Mobile.
- **Runtime Performance**: Average additional latency of ~158ms per image for TOPIQ.
- **Formula Execution**: On-device C6 calculation matches WSL logic.
- **Log-Only Status**: Verified that C6 telemetry is correctly captured without affecting production scores.

## 6. Android Larger Batch Evidence
A 50-image larger batch validation (including HEIF and varying EXIF orientations) confirmed stable execution and successful telemetry logging.

## 7. Android vs WSL Parity Evidence
Parity has been rigorously tracked through multiple diagnostic stages:
- **Pre-EXIF Fix**: C6 MAE 2.3256, Max AE 7.5531.
- **Post-EXIF Fix**: C6 MAE 1.3470, **Max AE 5.2387**, SRCC 0.9690.
- **Residual Mismatch**: KonIQ remains the primary contributor to error (MAE 2.0578, Max AE 11.9568).

## 8. EXIF Fix and KonIQ Residual Mismatch
The EXIF orientation fix (applying `ImageOps.exif_transpose` in WSL) improved C6 SRCC from 0.91 to 0.97. Remaining errors are attributed to platform-specific bilinear resize implementation differences. Diagnostic testing of 10+ preprocessing variants confirmed that the standard **`tf_resize_bilinear_224` (stretch)** remains the best match for the Android stack.

## 9. Risk Assessment
- **High Risk (Production Replacement)**: Max AE of 5.2387 exceeds the strict < 3.0 threshold for production replacement. Switching now could cause unexpected ranking shifts for a subset of users.
- **Low Risk (Log-Only)**: Median AE of 0.82 and SRCC of 0.97 provide high confidence for log-only telemetry and statistical analysis.

## 10. Final Readiness Decision
| Category | Decision | Notes |
| :--- | :--- | :--- |
| **Log-Only Telemetry** | **GO** | Statistically stable, safe for telemetry. |
| **Production Score Replacement**| **NO-GO** | Blocked by platform parity noise. |
| **UI Score Display** | **NO-GO** | Formula not yet user-validated. |
| **B5 Baseline Retirement** | **NO-GO** | Keep existing technical_score (KonIQ/FLIVE). |

## 11. Allowed Next Actions
1.  Enable C6 log-only telemetry for all internal testers.
2.  Aggregate C6 vs. B5 scores from field logs for 1,000+ images.
3.  Perform manual ranking audit using field log data to verify "Better vs. Worse" consistency.
4.  Optionally add a debug mode to Flutter that exports the KonIQ input tensor for bit-exact parity troubleshooting.

## 12. Explicit No-Go Items
- **DO NOT** replace `technical_score` in production.
- **DO NOT** show C6-based scores or rankings to production users.
- **DO NOT** remove the KonIQ-Mobile guard from any candidate pipeline.

## 13. Recommended Next Prompt
"Final parity analysis complete. C6 is stabilized for log-only telemetry. Summarize the A-cut technical quality roadmap for the next sprint, focusing on field log collection and semantic bias monitoring."
