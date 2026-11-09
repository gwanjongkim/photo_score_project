# Student Training Manifest Implementation Report

## 1. Summary
- guarded_labels_csv: outputs/topiq_teacher_label_dryrun_20260601/guarded_labels.csv
- validation_manifest_csv: outputs/topiq_teacher_validation_20260531/validation_manifest.csv
- holdout_manifest_csv: outputs/topiq_teacher_guard_holdout_20260601/holdout_manifest.csv
- output_csv: outputs/student_training_manifest_design_20260601/student_training_manifest.csv
- input rows: 9000
- final decision: READY_FOR_FIRST_STUDENT_TRAINING
- decision reason: Manifest passed duplicate, leakage, group, dataset, and ratio checks.

## 2. Input Validation
- required columns missing: []
- qcut method: qcut
- target decile bins: 10

## 3. Source Group Derivation
- source groups: 8903
- multi-row source groups: 95
- patch rule: paths containing `/patches/` use filename before `_patch_`; other paths use filename.

## 4. Split Policy
- target ratio: train 70%, val 15%, test 15%
- split unit: source_group
- stratification key: dataset + `__` + target_decile
- stratification buckets: 29
- assignment: deterministic greedy per stratification bucket toward row-count targets

## 5. Split Counts
| student_split | row_count | group_count | actual_pct | target_pct | pct_deviation |
| --- | --- | --- | --- | --- | --- |
| train | 6301 | 6204 | 0.700111 | 0.700000 | 0.000111 |
| val | 1354 | 1354 | 0.150444 | 0.150000 | 0.000444 |
| test | 1345 | 1345 | 0.149444 | 0.150000 | -0.000556 |

## 6. Dataset Distribution by Split
| student_split | dataset | row_count |
| --- | --- | --- |
| test | flive | 448 |
| test | koniq10k | 448 |
| test | spaq | 449 |
| train | flive | 2100 |
| train | koniq10k | 2101 |
| train | spaq | 2100 |
| val | flive | 452 |
| val | koniq10k | 451 |
| val | spaq | 451 |

## 7. Target Decile Distribution by Split
| student_split | target_decile | row_count |
| --- | --- | --- |
| test | d0 | 135 |
| test | d1 | 134 |
| test | d2 | 136 |
| test | d3 | 135 |
| test | d4 | 136 |
| test | d5 | 137 |
| test | d6 | 134 |
| test | d7 | 135 |
| test | d8 | 133 |
| test | d9 | 130 |
| train | d0 | 630 |
| train | d1 | 631 |
| train | d2 | 628 |
| train | d3 | 629 |
| train | d4 | 628 |
| train | d5 | 625 |
| train | d6 | 630 |
| train | d7 | 628 |
| train | d8 | 633 |
| train | d9 | 639 |
| val | d0 | 135 |
| val | d1 | 135 |
| val | d2 | 136 |
| val | d3 | 136 |
| val | d4 | 136 |
| val | d5 | 139 |
| val | d6 | 136 |
| val | d7 | 136 |
| val | d8 | 134 |
| val | d9 | 131 |

## 8. Leakage Checks
- image_path duplicate count: 0
- validation image_path overlap: 0
- holdout image_path overlap: 0
- source_group overlap across student train/val/test: 0
- all splits non-empty: True
- each dataset appears in each split: True
- datasets by split: {'train': ['flive', 'koniq10k', 'spaq'], 'val': ['flive', 'koniq10k', 'spaq'], 'test': ['flive', 'koniq10k', 'spaq']}

## 9. Deviations From Target Ratio
- pass threshold used here: max absolute split-ratio deviation <= 0.03
- max absolute deviation: 0.000556

## 10. Risks
- Greedy group stratification optimizes row-count balance inside each dataset/decile bucket but does not guarantee exact global ratios.
- FLIVE patch groups are protected by source_group, which can create small unavoidable stratification deviations.
- This manifest is based on the 9k guarded-label dry run only; it is not the full technical IQA training corpus.

## 11. Final Decision
READY_FOR_FIRST_STUDENT_TRAINING
