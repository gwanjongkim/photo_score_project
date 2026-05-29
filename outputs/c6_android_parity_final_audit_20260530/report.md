# C6 TOPIQ+KonIQ Android Parity Final Audit

## 1. Executive Summary
**Final Decision: B. Keep log-only; promising but not production-ready.**

The C6 technical quality candidate has shown significant improvement through EXIF orientation correction and preprocessing alignment, reaching an SRCC of >0.97. However, the Maximum Absolute Error (Max AE) remains between 4.6 and 5.2 depending on the backend, which exceeds the required threshold for production score replacement. The underlying tensor fingerprints do not match exactly, and the C6 formula's branch sensitivity amplifies these small model-level differences. C6 should remain in log-only/debug mode to collect more field data without affecting production user experience.

## 2. Confirmed Facts
- **EXIF Orientation:** 100% of the top 10 mismatches in the initial audit were due to EXIF Orientation 6 (rotated 90 deg) being ignored by the WSL reference.
- **Improved Metrics:** Fixing EXIF orientation reduced C6 Max AE from 7.55 to 5.24 and increased SRCC from 0.91 to 0.97.
- **Tensor Discrepancy:** No tested WSL preprocessing backend (PIL bilinear, bicubic, Lanczos, or TF) achieved a perfect match with Android TOPIQ input tensor fingerprints.
- **Source Dimension Impact:** Manually matching WSL source dimensions to Android's logged decoded dimensions actually worsened tensor checksum parity (relative diff 0.013 -> 0.014).
- **Formula Sensitivity:** The C6 formula `min(0.7*T + 0.3*K, K+8)` is sensitive near the cap boundary, where small differences in KonIQ or TOPIQ can cause a branch switch.

## 3. Experiment Timeline
- **WSL benchmark (2026-05-27):** Established baseline on WSL reference.
- **Android smoke (2026-05-27):** Verified basic model execution on Samsung S23 Ultra.
- **Initial Android-vs-WSL parity (2026-05-27):** Result: FAIL (MAE 2.33, Max AE 7.55).
- **EXIF/HEIF diagnosis (2026-05-28):** Identified EXIF orientation as the primary driver of top mismatches.
- **EXIF-fixed parity (2026-05-29):** Applied `ImageOps.exif_transpose`. Result: FAIL (MAE 1.35, Max AE 5.24).
- **TOPIQ Preprocess Backend Ablation (2026-05-29):** Tested 4 backends. `pil_lanczos` yielded best C6 MAE (1.14).
- **Tensor Fingerprint Comparison (2026-05-29):** Compared Android fingerprints to WSL backends. `pil_bilinear` was closest.
- **Source-dimension matched diagnostic (2026-05-29):** Forced dimension match. Result: Parity worsened.

## 4. Key Metrics Table (Latest Best Result)
| Metric | Value | Target | Status |
| :--- | :--- | :--- | :--- |
| Matched Count | 50 | 50 | Pass |
| Failed Count | 0 | 0 | Pass |
| C6 MAE | 1.1397 | < 1.0 | Near-Pass |
| C6 Max AE | 4.6256 | < 3.0 | **Fail** |
| C6 SRCC | 0.9744 | > 0.95 | Pass |
| Rel. Checksum Diff (TOPIQ) | 0.01298 | < 0.001 | **Fail** |
| Patch Mean Diff (TOPIQ) | 0.3662 | < 0.1 | **Fail** |
| Formula Violation Count | 0 | 0 | Pass |
| NaN/Range Error Count | 0 | 0 | Pass |

## 5. Passed Criteria
- **Rank Correlation (SRCC):** 0.9744 is excellent, indicating the relative ordering of images is highly consistent between Android and WSL.
- **Functional Stability:** No NaNs, range errors, or crashes observed in the 50-image batch on S23 Ultra.
- **Formula Compliance:** The Android implementation correctly applies the C6 min() cap formula.

## 6. Failed or Partial Criteria
- **Strict Max AE:** Max AE of 4.6 is still too high for production replacement where scores are visible to users.
- **Tensor Parity:** The discrepancy in tensor fingerprints (checksums and patch means) indicates the Android and WSL preprocessing pipelines are not identical.
- **Source Dimension Matching:** Forcing dimensions to match (simulating Android's downsampled decode) did not resolve the fingerprint gap, suggesting kernel or color-space differences.

## 7. Interpretation
The remaining ~5 point Max AE is likely a "death by a thousand cuts" from floating point rounding, interpolation kernel implementation (Android's `ImageProcessor` vs PIL/TF), and internal decoding pipelines. Because C6 includes a hard cap branch, these small "epsilon" differences in model scores are occasionally magnified into larger C6 differences when one platform caps and the other doesn't.

Source-dimension matching worsened metrics (Rel. Checksum Diff 0.013 -> 0.014), proving that simply matching resolutions is not the solution and might introduce more interpolation artifacts.

## 8. Production Decision
**NO-GO** for production replacement.
Existing `technical_score` (KonIQ+FLIVE) must remain unchanged. C6 is not yet stable enough to replace the primary quality metric.

## 9. Recommended Next Action
**Closure & Documentation.**
Accept that perfect parity is unattainable without a shared C++ preprocessing library. Document the current ~1.2 MAE / ~4.6 Max AE as the "known discrepancy" for C6 and keep it in log-only mode for the duration of the pilot to observe if these differences impact real-world ranking significantly.

## 10. Forbidden Actions
- Do not replace existing `technical_score`.
- Do not show C6 in UI.
- Do not change KonIQ+FLIVE production path.
- Do not adopt source-dimension-matched WSL reference.
- Do not present C6 as production-ready.
