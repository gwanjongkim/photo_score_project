# Context Notes

- Scope is intentionally limited to the WSL parity diagnostic artifacts under `outputs/c6_android_wsl_parity_exif_fixed_20260528/`.
- The existing parity result under `outputs/c6_android_wsl_parity_20260527/` must remain intact for old-vs-new comparison.
- The C6 scoring formula, model files, model exports, training code, and Flutter repo are out of scope.
- The diagnosis report identified ignored EXIF Orientation 6 as the primary mismatch driver, with residual error likely from resize implementation differences and C6 branch sensitivity.
- The corrected script registers `pillow-heif` only if importable so JPEG parity runs do not depend on optional HEIF support at import time.
- `ImageOps.exif_transpose` is applied once immediately after PIL open; the transposed RGB tensor is shared by KonIQ and TOPIQ preprocessing.
- Re-run processed 50/50 Android CSV rows with 0 failures. Numeric score columns have 0 NaNs and finite values.
- New C6 metrics are MAE 1.3470, Max AE 5.2387, Median AE 0.8212, SRCC 0.9690. This improves the old C6 MAE 2.3256 and Max AE 7.5531 but misses the requested Max AE <= 5.0 log-only threshold.
- C6 top-k overlap is 3/5 and 9/10; ranking is broadly preserved at top 10 but not exact at top 5.
