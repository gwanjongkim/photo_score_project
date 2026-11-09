# C6 Android vs WSL Parity Mismatch Diagnosis

## 1. Summary
The parity mismatch is primarily driven by **EXIF orientation handling**. All top 10 mismatches have Orientation 6 (rotated 90 deg), which the current WSL reference script ignores while Android Flutter respects it. Secondary factors include **C6 cap branch sensitivity** (small model diffs causing branch flips) and **resize algorithm implementation differences**.

## 2. Existing Parity Result
- **Status:** FAIL
- **MAE:** 2.3256
- **Max AE:** 7.5531
- **SRCC:** 0.9118
- **Requirement:** MAE < 0.1 for high parity, MAE < 2.0 for acceptable parity.

## 3. Mismatch Attribution
Analysis of 50 images:
- **Branch Matches:** 43
- **Branch Switches:** 7 (These images show the largest C6 diffs)
- **Primary Attribution:**
  - TOPIQ: 27 images
  - KonIQ: 16 images
  - Branch Switch: 7 images

## 4. Top Mismatch Metadata
| Filename | Orientation | Abs Diff C6 | Attribution |
| :--- | :---: | :---: | :--- |
| 20230201_181510.jpg | 6 | 7.55 | Branch Switch |
| 1675564423029-24.jpg | 6 | 7.43 | TOPIQ |
| 20250204_174237(0).jpg | 6 | 7.37 | Branch Switch |
| 20250205_000515.jpg | 6 | 6.41 | TOPIQ |
| 20250204_135520.jpg | 6 | 4.81 | TOPIQ |

**Finding:** 100% of top mismatches have EXIF Orientation 6.

## 5. WSL Preprocessing Audit
- **Decoding:** PIL `Image.open().convert("RGB")`. **Does NOT handle EXIF orientation.**
- **KonIQ:** `tf.image.resize(..., (224, 224))` (Stretch, Bilinear) + `/ 255.0`.
- **TOPIQ:** `tf.image.resize_with_pad(..., 384, 384)` (Aspect-ratio, Bilinear) + No normalization.

## 6. Android Preprocessing Evidence
- **Decoding:** Android native / Flutter ImagePicker typically auto-transposes images based on EXIF.
- **Resize:** TFLite `ImageProcessor` or Flutter `ui.decodeImageFromList`.
- **Normalization:** Matches WSL (verified via `docs/tflite_mobile_handoff.md`).

## 7. Likely Root Causes
1. **EXIF Orientation Mismatch (CRITICAL):** WSL sees a landscape image; Android sees a portrait image. Verified: Applying `ImageOps.exif_transpose` in WSL reduced Max AE from 7.55 to 5.23.
2. **C6 Branch Sensitivity (High):** Small differences in KonIQ/TOPIQ can flip the `min()` branch, especially near the `topiq - koniq = 11.43` transition point.
3. **Resize Algorithm Differences (Medium):** TensorFlow `bilinear` vs TFLite/Android `bilinear` implementation details (rounding, coordinate systems).
4. **HEIF Decode Path (Low):** `pillow-heif` vs Android native HEIF.

## 8. Recommended Fix Plan

### Option A: Make WSL reference mimic Flutter exactly (Recommended)
- **Purpose:** Fix the benchmarking tool to match reality.
- **Change:** Update `run_wsl_c6_parity.py` to use `ImageOps.exif_transpose(img)` during loading.
- **Risk:** Low.
- **Codex Implementation:** YES.

### Option B: Make Flutter preprocessing mimic WSL benchmark exactly
- **Purpose:** Force Android to ignore orientation (not recommended for UX).
- **Risk:** High (user sees rotated images).
- **Codex Implementation:** NO.

### Option C: Export debug tensors from Flutter
- **Purpose:** Absolute verification of intermediate states.
- **Change:** Add `log(image_tensor.sublist(0, 10))` in Flutter.
- **Risk:** Medium (requires Flutter repo access).
- **Codex Implementation:** NO.

## 9. No-Go / Go Decision
- **Android log-only C6:** OK (functional, but parity is low).
- **Production Replacement:** **NO-GO** until parity is resolved (MAE < 1.0, Max AE < 3.0).

## 10. Next Codex Prompt
"Update `outputs/c6_android_wsl_parity_20260527/run_wsl_c6_parity.py` to use `ImageOps.exif_transpose` after opening images with PIL. Also ensure `pillow-heif` is registered for all runs. Re-run the parity check and update the report."
