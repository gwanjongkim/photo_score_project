# Phase 5 policy_f_shift_cap3 10k Hybrid Teacher-label Dry-run Report

## 1. Executive Summary
- Run mode: `smoke`.
- Final decision: `SMOKE_ONLY_NOT_A_FULL_10K_DECISION`.
- Selected policy: `policy_f_shift_cap3`.
- Rows: `50` total, `50` valid.
- Applied MDIQA shift: `3.0`.

## 2. Inputs
- `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_label_design_review_2k_20260603/hybrid_label_dryrun_2k_manifest.csv`
- `/home/omen_pc1/photo_score_project/outputs/mdiqa_correct_protocol_full_benchmark_20260601/predictions.csv`
- `/home/omen_pc1/photo_score_project/outputs/paired_production_mdiqa_significance_expanded_20260603/paired_predictions_expanded.csv`
- `/home/omen_pc1/photo_score_project/outputs/technical_iqa_combo_benchmark_20260524/fusion_predictions_flive.csv`
- `/home/omen_pc1/photo_score_project/outputs/research_protocol_iqa_manifests_20260530/benchmark_all_research_protocol_iqa.csv`
- `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_adjusted_calibration_2k_20260603/recommended_adjusted_policy.json`

## 3. Confirmed Policy
- KonIQ-10k: MDIQA seed1 plus cap3 source-step smoothing.
- SPAQ: MDIQA xd plus cap3 source-step smoothing.
- FLIVE: Production technical score.
- Fallback/Unknown: Production safety fallback.

## 4. Dataset and Source Counts
| dataset | selected_teacher_source | n |
| --- | --- | --- |
| fallback_proxy | production_safety_fallback | 5 |
| flive | production | 15 |
| koniq10k | mdiqa_seed1_policy_f_shift_cap3 | 20 |
| spaq | mdiqa_xd_policy_f_shift_cap3 | 10 |

## 5. Target Distribution Summary
- Distribution CSV: `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_label_dryrun_10k_20260603_smoke/hybrid_label_distribution_summary_10k.csv` rows: `12`.

## 6. MOS Alignment
- fallback_proxy: n=5, SRCC=0.6000, PLCC=0.3231, MAE=1.8960, aligned_MAE=1.0660
- flive: n=15, SRCC=0.5643, PLCC=0.6637, MAE=3.4540, aligned_MAE=2.8239
- koniq10k: n=20, SRCC=0.8797, PLCC=0.9673, MAE=5.5231, aligned_MAE=2.1268
- spaq: n=10, SRCC=0.8303, PLCC=0.9660, MAE=4.2773, aligned_MAE=4.4150

## 7. Safety Cases
- None.

## 8. Range Calibration Audit
- Source mean shift: `18.4709` vs 2k `19.6235`.
- Source std ratio: `6.3675`.
- Source p95/p05 gap mismatch: `38.0341` vs 2k `34.9940`.
- Range audit CSV: `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_label_dryrun_10k_20260603_smoke/range_calibration_audit_10k.csv`.

## 9. Fallback Proxy Limitation
- Real mobile/unknown rows were not available in the requested cached sources.
- Fallback proxy uses non-overlapping FLIVE image rows with existing production scores when available.
- This is not representative real mobile fallback data.

## 10. Failure Cases
- None.

## 11. Policy Summary
- Summary CSV: `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_label_dryrun_10k_20260603_smoke/policy_f_shift_cap3_summary_10k.csv`.

## 12. Final Decision
SMOKE_ONLY_NOT_A_FULL_10K_DECISION

## 13. Recommended Next Step
- Run the full 10k dry-run command only after reviewing smoke output.
