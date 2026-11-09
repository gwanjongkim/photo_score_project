# 50k Student Training Manifest Report

## 1. Summary
- guarded_labels_csv: outputs/topiq_teacher_label_expansion_50k_20260601/guarded_labels.csv
- validation_manifest_csv: outputs/topiq_teacher_validation_20260531/validation_manifest.csv
- holdout_manifest_csv: outputs/topiq_teacher_guard_holdout_20260601/holdout_manifest.csv
- output_csv: outputs/student_training_manifest_50k_20260601/student_training_manifest.csv
- seed: 20260601
- input rows: 50000
- final decision: READY_FOR_V3_STUDENT_TRAINING
- decision reason: Manifest passed duplicate, leakage, group, dataset, ratio, and target-decile checks.

## 2. Input Validation
- dataset distribution: {'flive': 32000, 'koniq10k': 9000, 'spaq': 9000}
- missing values: {'image_path': 0, 'dataset': 0, 'split': 0, 'quality_score_0_100': 0, 'topiq_raw': 0, 'topiq_raw_x100': 0, 'calibrated_topiq_0_100': 0, 'guarded_target_0_100': 0}
- duplicate image_path count: 0
- guarded target min/max/mean/std: 10.700000 / 87.983333 / 66.396579 / 12.619546
- score range issues outside 0..100: {'quality_score_0_100': 0, 'topiq_raw_x100': 0, 'calibrated_topiq_0_100': 0, 'guarded_target_0_100': 0}

## 3. Source Group Derivation
- total source_group count: 41549
- multi-row source_group count: 7258
- max rows per source_group: 4
- patch rule: paths containing `/patches/` use filename before `_patch_`; other paths use filename.

Dataset-wise source_group counts:
| dataset | source_group_count |
| --- | --- |
| flive | 23549 |
| koniq10k | 9000 |
| spaq | 9000 |

## 4. Target Decile Binning
- method: qcut
- bins created: 10

## 5. Split Policy
- target ratio: train 70%, val 15%, test 15%
- split unit: source_group
- stratification key: dataset + `__` + target_decile
- stratification buckets: 30
- bucket size min/max: 19 / 4548
- small buckets below 30 rows: {'flive__d0': 19}
- assignment: deterministic greedy per stratification bucket toward row-count targets

## 6. Split Counts
| student_split | row_count | group_count | actual_pct | target_pct | pct_deviation |
| --- | --- | --- | --- | --- | --- |
| train | 34998 | 26753 | 0.699960 | 0.700000 | -0.000040 |
| val | 7507 | 7403 | 0.150140 | 0.150000 | 0.000140 |
| test | 7495 | 7393 | 0.149900 | 0.150000 | -0.000100 |

## 7. Dataset Distribution by Split
| student_split | dataset | row_count |
| --- | --- | --- |
| test | flive | 4799 |
| test | koniq10k | 1348 |
| test | spaq | 1348 |
| train | flive | 22400 |
| train | koniq10k | 6299 |
| train | spaq | 6299 |
| val | flive | 4801 |
| val | koniq10k | 1353 |
| val | spaq | 1353 |

## 8. Target Decile Distribution by Split
| student_split | target_decile | row_count |
| --- | --- | --- |
| test | d0 | 750 |
| test | d1 | 790 |
| test | d2 | 848 |
| test | d3 | 845 |
| test | d4 | 855 |
| test | d5 | 824 |
| test | d6 | 747 |
| test | d7 | 677 |
| test | d8 | 600 |
| test | d9 | 559 |
| train | d0 | 3498 |
| train | d1 | 3415 |
| train | d2 | 3314 |
| train | d3 | 3310 |
| train | d4 | 3297 |
| train | d5 | 3358 |
| train | d6 | 3498 |
| train | d7 | 3639 |
| train | d8 | 3795 |
| train | d9 | 3874 |
| val | d0 | 752 |
| val | d1 | 795 |
| val | d2 | 838 |
| val | d3 | 845 |
| val | d4 | 848 |
| val | d5 | 818 |
| val | d6 | 755 |
| val | d7 | 684 |
| val | d8 | 605 |
| val | d9 | 567 |

## 9. Dataset-Decile Distribution by Split
| student_split | dataset | target_decile | row_count |
| --- | --- | --- | --- |
| test | flive | d0 | 3 |
| test | flive | d1 | 206 |
| test | flive | d2 | 494 |
| test | flive | d3 | 581 |
| test | flive | d4 | 675 |
| test | flive | d5 | 700 |
| test | flive | d6 | 647 |
| test | flive | d7 | 586 |
| test | flive | d8 | 517 |
| test | flive | d9 | 390 |
| test | koniq10k | d0 | 130 |
| test | koniq10k | d1 | 337 |
| test | koniq10k | d2 | 227 |
| test | koniq10k | d3 | 180 |
| test | koniq10k | d4 | 128 |
| test | koniq10k | d5 | 89 |
| test | koniq10k | d6 | 74 |
| test | koniq10k | d7 | 65 |
| test | koniq10k | d8 | 56 |
| test | koniq10k | d9 | 62 |
| test | spaq | d0 | 617 |
| test | spaq | d1 | 247 |
| test | spaq | d2 | 127 |
| test | spaq | d3 | 84 |
| test | spaq | d4 | 52 |
| test | spaq | d5 | 35 |
| test | spaq | d6 | 26 |
| test | spaq | d7 | 26 |
| test | spaq | d8 | 27 |
| test | spaq | d9 | 107 |
| train | flive | d0 | 9 |
| train | flive | d1 | 689 |
| train | flive | d2 | 1662 |
| train | flive | d3 | 2081 |
| train | flive | d4 | 2453 |
| train | flive | d5 | 2776 |
| train | flive | d6 | 3026 |
| train | flive | d7 | 3212 |
| train | flive | d8 | 3404 |
| train | flive | d9 | 3088 |
| train | koniq10k | d0 | 608 |
| train | koniq10k | d1 | 1575 |
| train | koniq10k | d2 | 1058 |
| train | koniq10k | d3 | 839 |
| train | koniq10k | d4 | 599 |
| train | koniq10k | d5 | 416 |
| train | koniq10k | d6 | 348 |
| train | koniq10k | d7 | 306 |
| train | koniq10k | d8 | 262 |
| train | koniq10k | d9 | 288 |
| train | spaq | d0 | 2881 |
| train | spaq | d1 | 1151 |
| train | spaq | d2 | 594 |
| train | spaq | d3 | 390 |
| train | spaq | d4 | 245 |
| train | spaq | d5 | 166 |
| train | spaq | d6 | 124 |
| train | spaq | d7 | 121 |
| train | spaq | d8 | 129 |
| train | spaq | d9 | 498 |
| val | flive | d0 | 3 |
| val | flive | d1 | 210 |
| val | flive | d2 | 484 |
| val | flive | d3 | 581 |
| val | flive | d4 | 666 |
| val | flive | d5 | 693 |
| val | flive | d6 | 653 |
| val | flive | d7 | 592 |
| val | flive | d8 | 521 |
| val | flive | d9 | 398 |
| val | koniq10k | d0 | 131 |
| val | koniq10k | d1 | 338 |
| val | koniq10k | d2 | 227 |
| val | koniq10k | d3 | 180 |
| val | koniq10k | d4 | 129 |
| val | koniq10k | d5 | 89 |
| val | koniq10k | d6 | 75 |
| val | koniq10k | d7 | 66 |
| val | koniq10k | d8 | 56 |
| val | koniq10k | d9 | 62 |
| ... | 10 more rows |

## 10. Leakage Checks
- image_path duplicate count: 0
- validation image_path overlap: 0
- holdout image_path overlap: 0
- source_group overlap across student train/val/test: 0
- all splits non-empty: True
- every dataset appears in train/val/test: True
- every split has reasonable rows: True
- datasets by split: {'train': ['flive', 'koniq10k', 'spaq'], 'val': ['flive', 'koniq10k', 'spaq'], 'test': ['flive', 'koniq10k', 'spaq']}

## 11. Deviations From Target Ratio
- pass threshold used here: max absolute split-ratio deviation <= 0.03
- max absolute deviation: 0.000140

## 12. Risks
- Exact stratification can be impossible when a source_group contains multiple patch rows.
- FLIVE has many multi-row source groups, so safety against leakage is prioritized over exact distribution matching.
- This manifest uses guarded teacher-label design targets; it is not a Flutter runtime formula.

## 13. Final Decision
READY_FOR_V3_STUDENT_TRAINING

## 14. Next Training Command Recommendation
Do not run until the v3 trainer exists and this manifest has been reviewed.

```bash
./.venv_gpu/bin/python scripts/technical_iqa/train_mobile_student_iqa_v3.py \
  --manifest_csv outputs/student_training_manifest_50k_20260601/student_training_manifest.csv \
  --out_dir outputs/mobile_student_iqa_v3_20260601 \
  --head_epochs 5 \
  --finetune_epochs 20 \
  --batch_size 32
```
