# MDIQA Official-protocol Expansion Feasibility Audit Report

## 1. Executive Summary
The audit confirms that an official-protocol expansion (Track B) is **feasible** using existing local image assets and repository-provided split definitions. While Track A (14,105 rows) provides a validated core subset for the capstone, Track B can align the project with official scientific benchmarks by including the full Training and Validation splits for KonIQ-10k, SPAQ, and FLIVE.

## 2. Track A Current Scope
Track A is the current production baseline, consisting of 14,105 rows:
- **KonIQ-10k**: 8,999 rows (Test/Val biased subset).
- **SPAQ**: 1,125 rows (approx. 10% subset).
- **FLIVE**: 3,981 rows (Images only, no MDIQA predictions in current cache).

## 3. MDIQA Official / Repo Protocol Inventory
Official protocols identified in the MDIQA repository:
- **KonIQ-10k**: 10,073 rows with official Train(7058)/Val(1000)/Test(2015) split.
- **SPAQ**: 11,125 rows with random 80/20 split (Seed 123).
- **FLIVE**: 159,240 rows (39,810 images + 119,430 patches) with official Train/Val/Test split.

## 4. Current Cache vs Official Scope
| Dataset | Official | Track A | Missing |
| :--- | :--- | :--- | :--- |
| KonIQ-10k | 10,073 | 8,999 | 1,074 |
| SPAQ | 11,125 | 1,125 | 10,000 |
| FLIVE | 159,240 | 3,981 | 155,259 |

## 5. Missing Row and Image Availability Estimate
- **KonIQ-10k**: All 10,073 images are available in `data/raw/koniq10k/images/`.
- **SPAQ**: All 11,125 images are available in `data/raw/spaq/`.
- **FLIVE**: Approximately 199,926 images/patches are available in `data/raw/paq2piq/`.
- **Availability**: 100% of required images for expansion are confirmed to be on disk.

## 6. Additional Inference Requirements
Expanding to Track B requires additional MDIQA teacher inference:
- **KonIQ/SPAQ**: ~11,074 images (~1.5 hours on GPU).
- **FLIVE (Images)**: ~35,829 images (~5 hours on GPU).
- **FLIVE (Patches)**: ~119,430 patches (~16.5 hours on GPU).
Total estimated inference time for full alignment: **~23 hours**.

## 7. Revalidation Requirements
Track B would require:
1. Verification of MDIQA predictions against the existing Track A overlapping rows.
2. Cross-check of MOS distributions for expanded Training sets.
3. Alignment of split IDs with MDIQA official CSVs.

## 8. Risk / Cost Assessment
- **Risk**: The primary risk is the computational cost and storage overhead of FLIVE patches (119k rows).
- **Cost**: Low in terms of human effort (scripts are available), but moderate in terms of GPU time.
- **Impact**: High scientific value for research-track parity, but zero immediate impact on the production technical_score used in Flutter.

## 9. Final Recommendation
**TRACK_B_FEASIBLE_NOW_BUT_SEPARATE**

Track B is technically ready to be generated but should be maintained as a strictly separate research track to avoid destabilizing the validated Track A capstone path.

## 10. Recommended Next Step
Proceed with Track A capstone finalization. If bandwidth permits, generate the Track B KonIQ/SPAQ expansion first, as it offers the highest "official protocol" return for the lowest inference cost.

---
**Note**: Track A remains the main path and is not blocked by these findings.
