# Phase 5 policy_f_shift_cap3 10k Hybrid Teacher-label Dry-run Report

## 1. Executive Summary
- Run mode: `full_10k`.
- Final decision: `HYBRID_LABEL_10K_NEEDS_REAL_MOBILE_FALLBACK_DATA`.
- Selected policy: `policy_f_shift_cap3`.
- Rows: `7606` total, `7606` valid.
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
| flive | production | 3981 |
| koniq10k | mdiqa_seed1_policy_f_shift_cap3 | 2500 |
| spaq | mdiqa_xd_policy_f_shift_cap3 | 1125 |

## 5. Target Distribution Summary
- Distribution CSV: `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_label_dryrun_10k_20260603/hybrid_label_distribution_summary_10k.csv` rows: `9`.

## 6. MOS Alignment
- flive: n=3981, SRCC=0.6303, PLCC=0.7567, MAE=3.1184, aligned_MAE=3.0220
- koniq10k: n=2500, SRCC=0.9450, PLCC=0.9582, MAE=5.3720, aligned_MAE=2.4809
- spaq: n=1125, SRCC=0.9136, PLCC=0.9187, MAE=6.9406, aligned_MAE=6.6582

## 7. Safety Cases
- None.

## 8. Range Calibration Audit
- Source mean shift: `15.4364` vs 2k `19.6235`.
- Source std ratio: `2.9749`.
- Source p95/p05 gap mismatch: `32.3943` vs 2k `34.9940`.
- Range audit CSV: `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_label_dryrun_10k_20260603/range_calibration_audit_10k.csv`.

## 9. Fallback Proxy Limitation
- Real mobile/unknown rows were not available in the requested cached sources.
- Fallback proxy uses non-overlapping FLIVE image rows with existing production scores when available.
- This is not representative real mobile fallback data.

## 10. Failure Cases
- Failure rows: `3`. See `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_label_dryrun_10k_20260603/hybrid_label_failure_cases_10k.csv`.

## 11. Policy Summary
- Summary CSV: `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_label_dryrun_10k_20260603/policy_f_shift_cap3_summary_10k.csv`.

## 12. Final Decision
HYBRID_LABEL_10K_NEEDS_REAL_MOBILE_FALLBACK_DATA

## 13. Recommended Next Step
- Define or score real mobile/unknown fallback rows before judging the full-label dry-run gate.
