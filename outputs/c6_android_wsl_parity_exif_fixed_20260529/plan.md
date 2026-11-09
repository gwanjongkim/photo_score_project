# Plan

## Goal
Re-run the C6 Android-vs-WSL parity check with the WSL diagnostic script matching Android/user-visible EXIF orientation handling.

## Assumptions
- Android Flutter respects EXIF orientation during decode.
- WSL should apply `ImageOps.exif_transpose(img)` before RGB conversion, resize, or model preprocessing.
- KonIQ/TOPIQ model paths, score scaling, preprocessing contracts, and C6 formula remain unchanged.
- The existing `outputs/c6_android_wsl_parity_20260527/` outputs remain intact.

## Steps
1. Copy the original WSL parity script into this output directory.
2. Patch only image loading to apply EXIF orientation before `convert("RGB")`.
3. Run the copied script with `.venv/bin/python`.
4. Validate generated CSV/report artifacts and compare C6 metrics against the original result.
5. If parity still misses thresholds, identify whether remaining C6 mismatch is mainly KonIQ, TOPIQ, branch switching, or orientation-related.
