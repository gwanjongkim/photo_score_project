# Checklist

- [x] Copy the existing WSL C6 parity script into the EXIF-fixed output directory.
- [x] Register `pillow-heif` only when available.
- [x] Apply PIL EXIF transpose before RGB conversion.
- [x] Use the same corrected image tensor for KonIQ and TOPIQ preprocessing.
- [x] Emit EXIF orientation and before/after image-size fields.
- [x] Re-run parity for `data/debug/c6_parity_20260527/android_c6_scores.csv` and `test_vila`.
- [x] Generate `wsl_c6_scores.csv`, `android_vs_wsl_parity.csv`, and `report.md`.
- [x] Verify no NaNs, no scale inversion, and roughly preserved top-k ranking.
