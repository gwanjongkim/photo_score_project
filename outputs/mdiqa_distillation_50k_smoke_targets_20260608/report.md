# MDIQA 50k Distillation 5k Smoke Target Report

## 1. Executive Summary
The 5k MDIQA target generation smoke test for the recommended 50k Policy B distillation manifest has successfully completed. The inference pipeline generated valid targets for exactly 5,000 images sampled from the uncached (missing) rows. No NaNs, out-of-bounds scores, or crashes occurred. The generated score distributions align with expectations.

**Final Decision: MDIQA_50K_SMOKE_TARGETS_READY_FOR_FULL_INFERENCE**

## 2. Sources Inspected
- `outputs/mdiqa_distillation_50k_manifest_audit_20260608/candidate_50k_manifest_preview.csv`
- `outputs/official_protocol_benchmark_manifest_20260604/official_benchmark_test_manifest.csv`

## 3. Smoke Manifest Construction
A random 5,000-row subset of the uncached Policy B candidates was constructed, ensuring strict separation from the official test set.
- FLIVE: 3,500 images
- SPAQ: 1,000 images
- KonIQ-10k: 500 images
- Total: 5,000 images

## 4. MDIQA Inference Execution
Inference was run using the established variant policy:
- **KonIQ-10k**: `mdiqa_koniq` (seed1)
- **SPAQ**: `mdiqa_xd`
- **FLIVE**: `mdiqa_xd`
The inference completed successfully on the GPU utilizing PyTorch and the `CFANet_Shared` architecture. Execution for 5k images took approximately 2 minutes (roughly ~40-45 images per second).

## 5. Target Distribution
Dataset-wise target score analysis shows expected variation and bounds:
| Dataset | n | Min | Max | Mean | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FLIVE** | 3500 | 18.01 | 82.89 | 63.69 | 10.75 |
| **KonIQ** | 500 | 45.12 | 68.25 | 60.00 | 4.87 |
| **SPAQ** | 1000 | 11.97 | 79.49 | 46.05 | 18.70 |

*Note: KonIQ targets remain highly compressed around the mean (~60), which matches the known teacher distribution behavior on this dataset. SPAQ exhibits the widest variance, successfully reaching both low (~12) and high (~79) extremes.*

## 6. Failure and Leakage Audit
- **Failures**: 0 rows failed decoding or inference.
- **Leakage**: 0 rows overlap with the official benchmark test set.
- **Missing/NaNs**: 0 targets returned NaN or Infinity.

## 7. Scale Validation
- All generated scores correctly fall within the expected `0` to `100` range. The MDIQA raw outputs (which vary between `0.0~1.0` and `0~100` depending on the head) were properly standardized to the `0-100` scale.

## 8. Final Decision
**MDIQA_50K_SMOKE_TARGETS_READY_FOR_FULL_INFERENCE**

## 9. Next Actions
Proceed with executing the full missing MDIQA inference (approx 38k remaining rows) to finalize the 50k student distillation manifest.
