# MDIQA 50k Distillation Manifest Audit Report

## 1. Executive Summary
This audit evaluates candidate 50k training manifests for MDIQA student distillation. The primary goals are to increase training coverage to address the generalization gap identified in Phase 13, maintain dataset balance, and avoid evaluation leakage. 

We recommend **Policy B (Balanced)**, which yields exactly 50k image-only rows by utilizing all non-test KonIQ and SPAQ images and supplementing with FLIVE image rows. Since MDIQA inference is very fast (estimated ~16 minutes for the missing ~43k rows), we can safely proceed with this robust dataset composition without heavy reliance on biased cached predictions.

**Final Decision: MDIQA_50K_MANIFEST_READY_FOR_SMOKE**

## 2. Sources Inspected
- `outputs/student_distillation_manifest_14105_20260603/student_distillation_manifest.csv`
- `outputs/official_protocol_benchmark_manifest_20260604/official_benchmark_test_manifest.csv`
- `outputs/mdiqa_correct_protocol_full_benchmark_20260601/predictions.csv`
- `outputs/official_protocol_mdiqa_signal_benchmark_20260604/predictions.csv`

## 3. Existing MDIQA Cache Inventory
- MDIQA Cache 1 (Correct Protocol Benchmark): 23,103 rows
- MDIQA Cache 2 (Official Signal Benchmark): 4,968 rows
- Total unique cached paths identified: 28,064 rows

## 4. Candidate 50k Policies
We simulated four potential policies:
- **Policy A (Image-only conservative)**: Uses only images, avoiding FLIVE patches.
- **Policy B (Balanced)**: KonIQ (~7.1k), SPAQ (~8.9k), and FLIVE images (~33k) to reach 50k. Patches were not needed.
- **Policy C (Hard-absent emphasis)**: Only KonIQ and SPAQ, emphasizing rows absent from prior student training.
- **Policy D (Cache-first)**: Prioritizes reusing the 28k cached rows, supplementing with uncached rows to hit 50k. (Risk: Introduces ~30k FLIVE patches implicitly present in the cache).

## 5. Dataset Balance and Coverage
**Recommended Policy B Summary:**
- **KonIQ rows**: 7,193
- **SPAQ rows**: 8,916
- **FLIVE image rows**: 33,891
- **FLIVE patch rows**: 0
- **Total rows**: 50,000

*Note: All official test split images are strictly excluded.*

## 6. Missing Inference Estimate
For Policy B:
- **Cached rows**: 6,418
- **Missing rows**: 43,582

## 7. Runtime and Storage Estimate
- **Estimated Speed**: ~44.8 images/second (based on prior 23k batch taking 515s).
- **Estimated Runtime**: ~971 seconds (~16.2 minutes).
- **Storage**: Highly feasible; prediction CSV will be approximately 5-10MB.

## 8. Leakage Risk Audit
- **Official Test Overlap**: 0 rows. 
We verified strict separation. No images from the 4,967-row official test benchmark manifest are included in the 50k training pool.

## 9. Recommended 50k Policy
**Policy B (Balanced 50k)**. It strictly avoids FLIVE patches, providing a clean image-only distillation target, while safely omitting official test sets. The inference cost of 16 minutes is trivial.

## 10. Smoke Plan Before Long Run
Prior to kicking off the full 50k student distillation training:
- Execute MDIQA inference on a 5k sampled subset of the manifest.
- Verify target distribution (min/max/mean/std) on the newly inferred targets.
- Ensure no NaNs or unexpected scaling artifacts are present.

## 11. Final Decision
**MDIQA_50K_MANIFEST_READY_FOR_SMOKE**

## 12. Next Actions
- Generate the final MDIQA targets for the 43,582 missing rows using `mdiqa_xd` (for SPAQ/FLIVE) and `mdiqa_koniq` seed1 (for KonIQ).
- Merge into the final 50k distillation manifest.
- Kick off a 5k student distillation smoke run.
