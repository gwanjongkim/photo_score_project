# Context Notes

## 2026-05-29
- Original parity failed with C6 MAE 2.3256, Max AE 7.5531, SRCC 0.9118.
- Mismatch diagnosis found the top C6 mismatches all had EXIF Orientation 6.
- The requested fix is limited to the copied diagnostic script under this output directory.
- Existing model paths, output scaling, and C6 formula must stay unchanged.
- Copied script now imports `ImageOps`, applies `ImageOps.exif_transpose(img)` immediately after `Image.open`, and writes outputs to this directory.
- Report generation was expanded to include the required before-vs-after table, top 10 C6 mismatches, EXIF orientation counts, mismatch attribution, NaN check, and score-scale inversion check.
- `.venv/bin/python` does not have `pillow_heif`; the copied script now preserves HEIF registration when available but does not require the package at import time.
- Required `.venv/bin/python` run completed. It matched 49 images and failed `IMG_20240519_215827_510.jpg` because the active environment could not decode HEIF.
- New C6 metrics: MAE 1.3690, median AE 0.8261, Max AE 5.2387, SRCC 0.9691. No NaN or score-scale inversion was found.
- Decision remains FAIL because Max AE is above 5.0. Remaining top-10 mismatches are still mostly EXIF Orientation 6, with TOPIQ as the main attribution count.
