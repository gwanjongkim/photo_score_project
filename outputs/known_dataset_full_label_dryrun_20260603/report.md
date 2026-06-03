# Known-dataset-only Full-label Dry-run Report

## 1. Executive Summary
- Final decision: `KNOWN_DATASET_FULL_LABEL_NEEDS_COVERAGE_FIX`.
- Rows: `14105` total, `14105` valid.
- Selected policy: `policy_f_shift_cap3`.
- Applied MDIQA shift: `3.0000`.
- Coverage note: cached MDIQA availability is below scope estimate for KonIQ/SPAQ.

## 2. Sources Inspected
- `/home/omen_pc1/photo_score_project/outputs/mdiqa_correct_protocol_full_benchmark_20260601/predictions.csv`
- `/home/omen_pc1/photo_score_project/outputs/paired_production_mdiqa_significance_expanded_20260603/paired_predictions_expanded.csv`
- `/home/omen_pc1/photo_score_project/outputs/technical_iqa_combo_benchmark_20260524/fusion_predictions_flive.csv`
- `/home/omen_pc1/photo_score_project/outputs/research_protocol_iqa_manifests_20260530/benchmark_all_research_protocol_iqa.csv`
- `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_adjusted_calibration_2k_20260603/recommended_adjusted_policy.json`
- `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_final_label_scope_decision_20260603/known_dataset_label_scope.csv`
- `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_final_label_scope_decision_20260603/report.md`
- `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_label_dryrun_10k_20260603/report.md`

## 3. Confirmed Label Scope
- KonIQ-10k: include, MDIQA seed1 + policy_f_shift_cap3.
- SPAQ: include, MDIQA xd + policy_f_shift_cap3.
- FLIVE: include, production technical score, image rows only.
- Unknown/fallback: excluded from main labels.
- c6_parity real-mobile: holdout only.
- Test_weight/product-gold: aesthetic/product validation only.

## 4. Manifest Construction
- Base metadata deduped by `image_path`, preferring `test` split when repeated across research protocol runs.
- KonIQ/SPAQ rows require cached MDIQA prediction, image content_type, existing image path, finite target.
- FLIVE rows require image content_type and cached production `flive_score` keyed by basename.
- Duplicate image_path rows are not allowed.

## 5. Dataset Counts
| dataset | n |
| --- | --- |
| flive | 3981 |
| koniq10k | 8999 |
| spaq | 1125 |

Source counts:
| selected_teacher_source | n |
| --- | --- |
| mdiqa_seed1_policy_f_shift_cap3 | 8999 |
| mdiqa_xd_policy_f_shift_cap3 | 1125 |
| production | 3981 |

## 6. Target Distribution
- Distribution CSV: `/home/omen_pc1/photo_score_project/outputs/known_dataset_full_label_dryrun_20260603/known_dataset_distribution_summary.csv` rows: `9`.

## 7. MOS Alignment
- flive: n=3981, SRCC=0.6303, PLCC=0.7567, MAE=3.1184, aligned_MAE=3.0220, FP=0, severe_FP=0, FN=0.
- koniq10k: n=8999, SRCC=0.9443, PLCC=0.9555, MAE=5.2906, aligned_MAE=2.4661, FP=0, severe_FP=0, FN=0.
- spaq: n=1125, SRCC=0.9136, PLCC=0.9187, MAE=6.9406, aligned_MAE=6.6582, FP=0, severe_FP=0, FN=0.

## 8. Safety Cases
- None.

## 9. Range Audit
- MDIQA vs Production mean shift: `12.2377`; Phase 5 10k: `15.4364`.
- Std ratio: `2.0418`; Phase 5 10k: `2.9749`.
- p95/p05 gap mismatch: `14.9453`; Phase 5 10k: `32.3943`.
- Clipping count: `0`; clipping pct: `0.0000`.
- Range audit CSV: `/home/omen_pc1/photo_score_project/outputs/known_dataset_full_label_dryrun_20260603/known_dataset_range_audit.csv`.

## 10. Excluded Holdout Data
| dataset | status | expected_or_estimated_rows | excluded_reason |
| --- | --- | --- | --- |
| unknown_fallback | excluded_future_work | 0 | excluded by final scope decision; product-gold labels are not strict technical MOS |
| real_mobile_c6 | excluded_holdout | 50 | c6_parity real-mobile holdout only |
| test_weight_gold | excluded_validation_only | 40 | Test_weight/product-gold is aesthetic/product validation only |
| duplicate_image_path | ok | 0 | audit row |

## 11. Failure Cases
- Failure rows: `2`. See `/home/omen_pc1/photo_score_project/outputs/known_dataset_full_label_dryrun_20260603/known_dataset_failure_cases.csv`.
| stage | dataset | error |
| --- | --- | --- |
| coverage | koniq10k | expected_or_estimated_rows=10073; valid_cached_rows=8999 |
| coverage | spaq | expected_or_estimated_rows=11125; valid_cached_rows=1125 |

## 12. Final Decision
KNOWN_DATASET_FULL_LABEL_NEEDS_COVERAGE_FIX

## 13. Recommended Next Step
- Fill missing cached MDIQA coverage for KonIQ/SPAQ, then rerun this dry-run only.
