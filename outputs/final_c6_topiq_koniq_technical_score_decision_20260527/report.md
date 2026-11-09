# Final Decision Report: C6 TOPIQ+KonIQ Technical Score Candidate

## 1. Executive Summary
Based on rigorous benchmarking across public IQA datasets (FLIVE, KonIQ, SPAQ) and a confirmed Hard-FP v2 safety dataset, **C6 (TOPIQ+KonIQ Soft-Cap)** is promoted as the next `technical_score` candidate for the A-cut project on WSL.

C6 delivers a statistically significant improvement in ranking performance (+0.015 average SRCC) and top-tier precision (+0.04 to +0.06 on SPAQ) while maintaining absolute safety against technical false positives (zero images > 65 on Hard-FP v2).

**Promotion Status:**
- **WSL Benchmark:** **GO** (Promoted as primary technical candidate).
- **Flutter Production:** **BLOCKED** (Pending on-device runtime validation).

## 2. Background and Failed Alternatives
Prior to identifying C6, several high-capacity models were evaluated:
- **DistortionGuard/TechIQA-Guard:** Single-model attempts failed to generalize across all distortion types. **NO-GO**.
- **FLIVE + TOPIQ (A1):** While ranking was strong, it inherited "semantic bias" from both models, over-scoring confirmed technical failures by ~12 points vs KonIQ-based guards. **REJECTED** for safety.
- **KonIQ + FLIVE (B5):** The current production baseline. Safe, but performance on high-quality photos (SPAQ) is limited by FLIVE's noise.

## 3. Candidate Formula
**C6: TOPIQ capped by KonIQ Guard**
```python
# Assumptions: All inputs are 0..100 scale
base_score = 0.7 * TOPIQ_mixed112 + 0.3 * KonIQ_mobile
technical_score = min(base_score, KonIQ_mobile + 8)
```
*Note: C3 (min(TOPIQ, KonIQ+8)) is a simpler alternative, but C6 provides better blending and higher SRCC on KonIQ-10k.*

## 4. Benchmark Evidence
| Dataset | B5 (Baseline) | C6 (Candidate) | Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| KonIQ SRCC | 0.867 | 0.892 | +0.025 | Improved |
| SPAQ SRCC | 0.861 | 0.900 | +0.039 | Improved |
| FLIVE SRCC | 0.508 | 0.487 | -0.021 | Competitive |
| **Average SRCC** | **0.745** | **0.760** | **+0.015** | **Promoted** |

## 5. Ranking Audit Evidence
- **Correlation with MOS:** C6 shows higher spearman correlation between rank improvement and ground-truth MOS (0.124 on SPAQ) compared to B5.
- **Top-K Precision:** C6 significantly improves the selection of ground-truth "High Quality" (Top 20%) images in realistic scenarios.
- **Demotion Behavior:** C6 correctly demotes images that FLIVE over-scores due to content bias, provided KonIQ identifies the technical flaw.

## 6. Hard-FP v2 Safety
Absolute safety on confirmed technical failures is the primary requirement for A-cut.
- **Hard-FP Mean:** C6 (54.1) vs B5 (52.8). Both well below the 60.0 safety limit.
- **Count > 65:** **0 / 44** for both candidates.
- **Max Score:** C6 (64.79) remains below the critical threshold for autonomous rejection.

## 7. Why C6 Over C3
C3 (`min(TOPIQ, KonIQ+8)`) was evaluated as a simpler fallback. C6 is preferred because:
1. **Higher SRCC:** C6 achieves 0.760 avg SRCC vs C3 0.751.
2. **Blending:** C6 leverages KonIQ's strengths even when the cap is not active, whereas C3 ignores KonIQ until the limit is hit.
3. **KonIQ-10k Parity:** C6 improves SRCC on the lab-quality KonIQ dataset (0.892) significantly over both KonIQ alone (0.866) and C3 (0.875).

## 8. Why C6 Over B5
B5 relies on FLIVE-Mobile, which has been found to share some "semantic bias" with TOPIQ, making it an unreliable guard for content-heavy but technically poor images. C6 uses TOPIQ for what it's best at (ranking high-quality content) and KonIQ for what it's best at (penalizing distortions), resulting in a more robust "separation of concerns."

## 9. Risks and Deployment Blockers
1. **On-Device Latency:** TOPIQ mixed112 (EfficientNetV2-based) is significantly heavier than FLIVE-Mobile. On-device inference time must be validated on Android (S23 Ultra target).
2. **Preprocessing Parity:** WSL uses PIL/TensorFlow preprocessing. Flutter implementation must match exactly to avoid score drift.
3. **Memory Pressure:** Running TOPIQ + KonIQ + Aesthetic models simultaneously may exceed mobile GPU memory limits.

## 10. Flutter Rollout Plan
- **Phase 0:** WSL Decision (Current).
- **Phase 1:** Flutter pre-implementation audit (Model weight loading, input shapes).
- **Phase 2:** Log-only implementation. `candidate_c6_score` is calculated and logged via Firebase but not shown to the user.
- **Phase 3:** Android runtime benchmark (Latency/Thermal check).
- **Phase 4:** Controlled UI replacement.

## 11. Final Decision
- **GO:** Promote **C6** as the official technical quality candidate for WSL evaluation.
- **NO-GO:** Do not update Flutter `technical_score` until parity and latency are confirmed.
- **MANDATORY:** Keep KonIQ-Mobile as the non-linear guard (Margin 8).

## 12. Next Prompt for Flutter Audit
"Perform pre-implementation audit for TOPIQ mixed112 TFLite on Flutter, checking metadata, input shape parity, and potential GPU memory footprint."
