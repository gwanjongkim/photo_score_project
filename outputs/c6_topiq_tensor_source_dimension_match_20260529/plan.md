# Plan

## Goal
Test whether resizing WSL's EXIF-transposed RGB source image to Android's logged decoded dimensions before PIL bilinear `384x384` resize-with-pad explains the remaining TOPIQ tensor fingerprint mismatch.

## Confirmed Inputs
- Android tensor fingerprints: `data/debug/c6_parity_20260527/android_topiq_tensor_fingerprint_rows.csv`
- Previous fingerprint comparison report and CSVs under `outputs/c6_topiq_tensor_fingerprint_compare_20260529/`
- Images: `test_vila`

## Assumptions
- The Android fingerprint CSV describes the actual tensor passed into TOPIQ mixed112.
- Android's logged `original_decoded_width` / `original_decoded_height` are the effective decoded source dimensions before `resizeWithPad`.
- The source-dimension-matching resize uses PIL bilinear as a diagnostic approximation of Android decode downsampling.

## Verification
1. Compile the diagnostic script.
2. Run the script with HEIF registration enabled.
3. Confirm all 50 Android rows are processed with zero failures.
4. Compare source-dimension-matched metrics against previous `pil_bilinear_resize_with_pad`.
