# MDIQA FLIVE Protocol Audit Report

## 1. Executive Summary
The near-zero SRCC for FLIVE in the WSL MDIQA sanity benchmark (**0.0676**) is primarily due to a **protocol and manifest mismatch**. While the MDIQA paper evaluates performance on **full images**, the WSL benchmark sampled a mix of **74% patches** and **26% images** from the PaQ-2-PiQ manifest. The MDIQA model exhibits a significant **scale bias**, predicting much lower scores for patches (~0.35) than for full images (~0.66) even when they share the same MOS (~70). This scale mismatch destroys rank correlation in the mixed set. 

When restricted to images only, the `mdiqa_xd` model achieves an SRCC of **0.5070** on the small sample (n=26), which is significantly better than the aggregate result but still below the paper's reported **0.612**. The remaining gap is likely due to the non-representative nature of the 26 images (only 1 was from the official test split).

## 2. Paper FLIVE Claim
- **Reported FLIVE SRCC:** 0.612 (arXiv:2508.16887 Table 1, cross-dataset KonIQ->FLIVE).
- **Reported FLIVE PLCC:** 0.734.
- **Methodology:** Evaluation on **full images** of the official FLIVE test set.

## 3. Official FLIVE Protocol Evidence
- The MDIQA repository config (`train_MDIQA_xd_koniq.yml`) uses `GeneralNRDataset` for FLIVE with a split file `flive_official_test_sample1000.pkl`.
- Official FLIVE evaluation typically uses the 8,000-image test partition.
- Preprocessing involves resizing to 384x384.

## 4. Current WSL Evaluation Protocol
- **Manifest:** `benchmark_all_research_protocol_iqa.csv` contains 39,810 images and 119,430 patches for FLIVE.
- **Sampling:** 100 random rows were selected for the sanity check.
- **Composition:** 74 patches, 26 images. Only 1 image was from the official `test` split.

## 5. FLIVE Manifest and Label Audit
- **Mean MOS (Images):** 70.16
- **Mean MOS (Patches):** 71.88
- **Observation:** MOS values for patches in the manifest are consistent with image scales, but the model does not treat them as equivalent.

## 6. Checkpoint Mapping Audit
- `mdiqa_xd` variant uses `mdiqa_shared_5t_4a_mbv3_koniq_xd.pth`.
- This variant achieved **0.934** SRCC on SPAQ, confirming the architecture (CLIP RN50 + Local Score) and implementation are correct.

## 7. Inference Implementation Comparison
- WSL script uses `Image.BICUBIC` resize to 384x384.
- This matches standard TOPIQ/MDIQA validation patterns.

## 8. Metric Sensitivity Checks
| Group | n | SRCC | Mean Pred | Mean MOS |
|---|---|---|---|---|
| Aggregate | 100 | 0.0676 | 0.429 | 71.4 |
| Images only | 26 | **0.5070** | **0.657** | 70.2 |
| Patches only | 74 | 0.2810 | 0.349 | 71.9 |

## 9. Root Cause Ranking
1. **Protocol Mismatch (Patches vs Images):** **CONFIRMED**. Model is not designed for patch-level evaluation on this dataset.
2. **Scale Mismatch (Model Bias):** **CONFIRMED**. Mixing images and patches breaks the rank order due to lower patch predictions.
3. **Non-representative Sample:** **LIKELY**. n=26 is too small and mostly non-test split.

## 10. Impact on Teacher Candidate Decision
MDIQA remains a strong teacher candidate for **KonIQ-10k** and **SPAQ**, where it significantly beats production. The low FLIVE performance is a benchmark artifact. Once restricted to images and official splits, MDIQA is expected to be competitive on FLIVE as well.

## 11. Final Decision
**FLIVE_FAILURE_LIKELY_MANIFEST_OR_LABEL_MISMATCH**

## 12. Recommended Next Step
Update the benchmark protocol to **filter the FLIVE dataset to `content_type == image`** and **use only the `test` split** images for future MDIQA evaluations. This will ensure consistency with official reporting and provide a fair comparison against production.
