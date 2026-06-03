# Student Distillation Manifest 14105 Preparation Report

## 1. Executive Summary
- Final decision: `STUDENT_MANIFEST_14105_READY_FOR_SMOKE_TRAINING`.
- Rows: `14105` valid, `0` removed.
- Split seed: `20260603`.
- Scope: validated known-dataset teacher-cache subset, not MDIQA official full-dataset relabeling.

## 2. Sources Inspected
- `/home/omen_pc1/photo_score_project/outputs/known_dataset_full_label_dryrun_20260603/known_dataset_full_label_dryrun_manifest.csv`
- `/home/omen_pc1/photo_score_project/outputs/known_dataset_full_label_coverage_audit_20260603/report.md`
- `/home/omen_pc1/photo_score_project/outputs/known_dataset_full_label_dryrun_20260603/report.md`
- Existing scripts inspected: `make_student_training_manifest.py`, `make_student_training_manifest_50k.py`.

## 3. Scope Statement
- Option A: keep current validated cache-subset direction.
- KonIQ-10k: MDIQA seed1 + policy_f_shift_cap3.
- SPAQ: MDIQA xd + policy_f_shift_cap3.
- FLIVE: Production technical score.
- Unknown/fallback excluded; c6_parity holdout only; Test_weight/product-gold validation only.
- No extra MDIQA inference, training, TFLite export, Flutter, production scoring, or model asset changes.

## 4. Manifest Construction
- Kept `selected_teacher_target_0_100` as main target.
- Added `target_norm = selected_teacher_target_0_100 / 100.0`.
- Preserved original split as `split_original`; new student split stored in `split`.
- Split unit: `source_group` when present, falling back to `image_path`.

## 5. Train/Val/Test Split
| split | n |
| --- | --- |
| train | 11283 |
| val | 1409 |
| test | 1413 |

| split | row_count | row_pct | target_pct | pct_deviation | source_group_count | image_path_count |
| --- | --- | --- | --- | --- | --- | --- |
| train | 11283 | 0.7999291031549096 | 0.8 | -7.089684509042904e-05 | 11283 | 11283 |
| val | 1409 | 0.0998936547323644 | 0.1 | -0.00010634526763560193 | 1409 | 1409 |
| test | 1413 | 0.10017724211272598 | 0.1 | 0.00017724211272597545 | 1413 | 1413 |
- Source groups: `14105` total; multi-row groups: `0`.

## 6. Dataset Balance
| dataset | n |
| --- | --- |
| flive | 3981 |
| koniq10k | 8999 |
| spaq | 1125 |

| split | key | row_count | split_pct | overall_pct | count_delta_vs_expected |
| --- | --- | --- | --- | --- | --- |
| train | flive | 3184 | 0.282194451830187 | 0.28224034030485645 | -0.8000000000001819 |
| train | koniq10k | 7199 | 0.6380395284941948 | 0.638000708968451 | -0.2000000000007276 |
| train | spaq | 900 | 0.07976601967561818 | 0.07975895072669266 | 0.0 |
| val | flive | 398 | 0.28246983676366216 | 0.28224034030485645 | -0.10000000000002274 |
| val | koniq10k | 899 | 0.638041163946061 | 0.638000708968451 | -0.900000000000091 |
| val | spaq | 112 | 0.0794889992902768 | 0.07975895072669266 | -0.5 |
| test | flive | 399 | 0.2823779193205945 | 0.28224034030485645 | 0.8999999999999773 |
| test | koniq10k | 901 | 0.637650389242746 | 0.638000708968451 | 1.099999999999909 |
| test | spaq | 113 | 0.0799716914366596 | 0.07975895072669266 | 0.5 |

## 7. Target Distribution
| split | n | min | max | mean | std | p05 | p50 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 1413 | 17.63276296854019 | 81.68309020996094 | 64.16381880309847 | 10.407427761926089 | 47.23043489456177 | 65.72410750389099 | 76.32285919189454 |
| train | 11283 | 14.804594844579697 | 88.80587768554688 | 64.51609819456321 | 9.806648292098444 | 50.07804560661316 | 65.6771867275238 | 76.39292755126954 |
| val | 1409 | 18.707634389400482 | 84.98268127441406 | 64.48151303369835 | 10.00462831421296 | 49.44910454750061 | 65.94662356376648 | 76.63396606445312 |
- Full distribution CSV: `/home/omen_pc1/photo_score_project/outputs/student_distillation_manifest_14105_20260603/target_distribution_summary.csv`.

## 8. Leakage Audit
| check | value | status |
| --- | --- | --- |
| input_rows | 14105 | pass |
| valid_rows | 14105 | pass |
| removed_invalid_rows | 0 | pass |
| duplicate_image_path | 0 | pass |
| image_path_overlap_across_splits | 0 | pass |
| source_group_leakage_across_splits | 0 | pass |
| target_out_of_range_0_100 | 0 | pass |
| target_norm_out_of_range_0_1 | 0 | pass |
| missing_student_split | 0 | pass |
| dataset_missing_from_split |  | pass |
| train_rows | 11283 | pass |
| val_rows | 1409 | pass |
| test_rows | 1413 | pass |

## 9. Safety Checks
| split | mos_n | srcc | plcc | mae | linear_aligned_mae | fp | severe_fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 11283 | 0.8620820508847455 | 0.8829534442233433 | 4.8081005244441135 | 4.726762470519143 | 0 | 0 | 0 |
| val | 1409 | 0.8576883883444169 | 0.8794594507139386 | 4.9337847457692625 | 4.852302463943524 | 0 | 0 | 0 |
| test | 1413 | 0.8634086806315228 | 0.8953987619732503 | 4.693137719169277 | 4.627218834041934 | 0 | 0 | 0 |

## 10. Limitations
- This is a validated teacher-cache subset, not full official KonIQ/SPAQ/FLIVE relabeling.
- SPAQ coverage follows correct-protocol cached test split only.
- Real-mobile/unknown fallback labels remain excluded from main supervision.

## 11. Final Decision
STUDENT_MANIFEST_14105_READY_FOR_SMOKE_TRAINING

## 12. Recommended Next Step
- Run a bounded smoke training job only; do not start full distillation yet.
