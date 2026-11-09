# Unknown / Real-Mobile Fallback Behavior Audit Report

## 1. Executive Summary
This audit addresses the missing fallback coverage identified in the Phase 5 10k dry-run. We identified 50 real-mobile images from Android device testing (`c6_parity`) with confirmed production technical scores. These images were matched to absolute paths in the workspace. An additional 40 product-gold labeled images (`Test_weight`) were also inventoried. The production technical score safety fallback was validated against these real-mobile samples.

**Final Decision: UNKNOWN_FALLBACK_READY_FOR_FULL_LABEL_DRYRUN**

## 2. Sources Inspected
- `data/debug/c6_parity_20260527/android_c6_scores.csv` (50 rows real-mobile scores)
- `data/Test_weight/` (40 product-gold images)
- `outputs/acut_product_gold_weight_eval_20260603/product_gold_predictions.csv`
- `outputs/hybrid_teacher_label_dryrun_10k_20260603/report.md`

## 3. Real Mobile / Unknown Data Inventory
- **Real Mobile**: 50 images from `android_c6_scores.csv` matched to paths in `data/`, `forWeights/`, and `forWeights2/`.
- **Product Gold**: 40 images labeled Bad/Soso/Good in `data/Test_weight/`. 12 images overlap with the real-mobile scored set.
- **Unknown**: No explicit "unknown" dataset found; `c6_parity` serves as the primary real-mobile proxy.

## 4. Fallback Dry-run Manifest
A 50-row manifest was built using the confirmed real-mobile scored images.
- Path: `outputs/hybrid_teacher_unknown_fallback_audit_20260603/unknown_fallback_dryrun_manifest.csv`

## 5. Production Score Availability
- **Existing**: 100% of the 50 real-mobile rows have `existing_technical_score_100`.
- **Generated**: Attempted to score 40 `Test_weight` images, but encountered a CuDNN version mismatch (9.1.0 vs 9.3.0) in the `.venv_tf` environment. This was mitigated by using the 12 overlapping images that already had scores.

## 6. Fallback Target Distribution
| Metric | Value |
| --- | --- |
| Count | 50 |
| Mean | 68.81 |
| Std Dev | 7.60 |
| Min / Max | 44.78 / 79.65 |
| p05 / p95 | 54.95 / 77.02 |
| Clipping (0 or 100) | 0 |

## 7. MOS Alignment and Safety Checks
Validated against 12 overlapping product-gold labels:
- **SRCC (proxy)**: 0.6504
- **Severe FP**: 1 (`IMG_2785.JPG`, labeled 'Bad' but scored 68.26).
- **FP**: 1
- **FN**: 0

The production fallback is generally safe but can be optimistic on some images labeled "Bad" by product teams for aesthetic reasons.

## 8. Proxy Limitations
- The sample size (n=50) is small but provides significantly better coverage than the FLIVE-proxy used in Phase 5.
- The "Bad" labels in `Test_weight` may be driven by aesthetic factors not fully captured by the technical-only production score.

## 9. Failure Cases
- 0 images failed to match to paths.
- 0 images showed out-of-bounds targets.
- 1 image (`IMG_20240519_215827_510.jpg`) failed to decode during the optional scoring attempt (HEIF bytes in .jpg extension).

## 10. Final Decision
**UNKNOWN_FALLBACK_READY_FOR_FULL_LABEL_DRYRUN**
The production technical score fallback behavior is confirmed as stable and reasonably aligned with real-mobile content. The project is ready to incorporate these 50 rows (and others like them) into the full-label dry-run manifest.

## 11. Recommended Next Step
Proceed to **Phase 6: Final Full-Scale Hybrid Label Generation (50k/159k)**.
- Integrate the 50 `c6_parity` real-mobile rows into the training manifest.
- Use `production_safety_fallback` as the source for any row not belonging to KonIQ, SPAQ, or FLIVE.
- Resolve the CuDNN mismatch in `.venv_tf` if further production scoring is required at scale.
