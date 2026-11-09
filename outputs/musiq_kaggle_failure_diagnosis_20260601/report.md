# MUSIQ Kaggle Failure Diagnosis Report

## 1. Executive Summary
The official Kaggle MUSIQ 100-image sanity benchmark failure was traced to a specific image in the FLIVE dataset (`AVA__325065.jpg`) that was in **grayscale (L) mode**. This caused a `ConcatOp` dimension mismatch in MUSIQ's internal TensorFlow preprocessing. A robust fallback mechanism—converting to **RGB** and **re-encoding as JPEG**—was tested and confirmed to fix the issue across all three MUSIQ variants (`koniq-10k`, `spaq`, `paq2piq`). The benchmark script has been patched, and a rerun achieved 100% completion (900/900 predictions).

## 2. Files Inspected
- `outputs/musiq_kaggle_sanity_20260601/report.md`: Initial failure report.
- `outputs/musiq_kaggle_sanity_20260601/failures.csv`: Failure trace identifying the specific image and error.
- `scripts/technical_iqa/run_musiq_kaggle_sanity_benchmark.py`: Original benchmark script.

## 3. Failed Image Inspection
- **Path**: `data/raw/paq2piq/voc_emotic_ava/AVA__325065.jpg`
- **Mode**: `L` (Grayscale)
- **Dimensions**: 552x600 (Aspect Ratio: 0.92)
- **Status**: Not corrupt, PIL opens successfully, but incompatible with MUSIQ's direct-byte signature when in grayscale.
- **Findings**: Full results in `outputs/musiq_kaggle_failure_diagnosis_20260601/failed_image_inspection.csv`.

## 4. Fallback Tests
All tested fallbacks that included **RGB conversion** were successful:
- **Fallback 2 (Re-encode RGB)**: Success. Score (koniq-10k): 70.64.
- **Fallback 3 (Resize 1024 + RGB)**: Success.
- **Fallback 4 (Resize 768 + RGB)**: Success.
- **Original Bytes**: Failed as expected with `ConcatOp` error.
- **Findings**: Full results in `outputs/musiq_kaggle_failure_diagnosis_20260601/fallback_test_results.csv`.

## 5. Patch Decision
Patched `scripts/technical_iqa/run_musiq_kaggle_sanity_benchmark_fallback.py` to:
1. Attempt inference with original image bytes first (preserving performance and authenticity for 99.7% of images).
2. On failure, catch the exception and attempt inference with **EXIF-transposed, RGB-converted JPEG re-encoded bytes**.
3. Record the `call_method` in the predictions to track fallback usage.

## 6. Optional Rerun Results
The 100-image sanity benchmark was rerun using the fallback script:
- **Predictions**: 900 / 900 (0 failures).
- **FLIVE (paq2piq)**: SRCC 0.4815 (Still below production 0.574).
- **KonIQ-10k (koniq-10k)**: SRCC 0.9275 (Slightly below production 0.934).
- **SPAQ (spaq)**: SRCC 0.9460 (Outperforming production 0.873).

## 7. Production Baseline Implication
MUSIQ is a clear winner for **SPAQ** (+0.073 SRCC). For **KonIQ-10k**, it is competitive but not a definitive upgrade. For **FLIVE**, the current official variants are notably weaker than the production baseline, suggesting MUSIQ may not be the optimal teacher for that specific distribution.

## 8. Final Decision
`MUSIQ_FAILURE_FIXED_AND_READY_FOR_FULL_BENCHMARK`

## 9. Next Step
Proceed with a full benchmark of the `spaq` and `koniq-10k` variants to confirm if they should replace the existing teachers for those datasets.
