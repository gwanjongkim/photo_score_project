# Plan

## Goal
Compare Android TOPIQ tensor fingerprints against WSL-generated fingerprints for each candidate preprocessing backend.

## Confirmed Inputs
- Android tensor fingerprint CSV: `data/debug/c6_parity_20260527/android_topiq_tensor_fingerprint_rows.csv`
- Android score CSV: `data/debug/c6_parity_20260527/android_c6_scores.csv`
- Image directory: `test_vila`
- Prior backend ablation report and CSVs under `outputs/c6_topiq_preprocess_backend_ablation_20260529/`

## Assumptions To Keep Separate
- The Android CSV fingerprint rows represent the actual tensor passed into TOPIQ mixed112.
- Android decoded dimensions are after Android's orientation handling.
- WSL applies `ImageOps.exif_transpose` before RGB conversion and preprocessing.

## Verification
1. Generate WSL tensor fingerprints for all four backend candidates.
2. Compare checksum, aggregate stats, channel stats, patch means, first/last values, and geometry against Android rows.
3. Write `backend_fingerprint_summary.csv`, `per_image_fingerprint_comparison.csv`, and `report.md`.
