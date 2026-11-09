# Mobile Student IQA 50k Smoke Training Report

## 1. Executive Summary
- Final decision: `STUDENT_50K_SMOKE_NEEDS_PIPELINE_FIX`.
- Run type: bounded smoke training only; no full training, no TFLite export, no Flutter edits, no production `technical_score` edits.
- Runtime seconds: `45.95`.
- Usable train/val/test counts: `{'train': 96, 'val': 48, 'test': 48}`.
- Final train loss: `0.005933`.
- Final val loss: `0.006163`.

## 2. Sources Inspected
- `/home/omen_pc1/photo_score_project/outputs/mdiqa_distillation_50k_full_targets_20260608/final_50k_mdiqa_distillation_manifest.csv`
- `/home/omen_pc1/photo_score_project/outputs/mdiqa_distillation_50k_full_targets_20260608/report.md`
- `scripts/technical_iqa/train_mobile_student_iqa_14105_smoke.py`
- `scripts/technical_iqa/train_mobile_student_iqa_v4a.py`
- `scripts/technical_iqa/train_mobile_student_iqa_v4b.py`
- `scripts/technical_iqa/make_student_training_manifest_50k.py`

## 3. Manifest and Split Construction
- Input rows: `50000`.
- Official Track B rows excluded before split: `0`.
- Duplicate image paths in input: `0`.
- Full split counts before caps: `{'train': 40000, 'val': 5004, 'test': 4996}`.
- Capped counts before validation: `{'train': 96, 'val': 48, 'test': 48}`.
- Image-path overlap across splits: `0`.
- Source-group overlap across splits: `0`.
- Official Track B rows in train: `0`.

## 4. Environment / GPU Preflight
- TensorFlow: `2.21.0`.
- GPUs visible to TensorFlow: `[]`.
- GPU count: `0`.
- Execution mode: `CPU fallback smoke recorded; not acceptable for READY unless explicitly approved`.

## 5. Smoke Training Configuration
- Backbone: `efficientnetv2b0`.
- Image size: `256`.
- Batch size: `16`.
- Head epochs: `1`.
- Finetune epochs: `1`.
- Loss: `huber`.
- target_norm: `teacher_target_0_100 / 100`.
- ImageNet weights requested: `none`.
- ImageNet weights loaded: `False`.
- Backbone weights error: ``.
- Include preprocessing: `True`.
- Finetune trainable backbone layers: `13`.

## 6. Training History
| phase | phase_epoch | global_epoch | loss | mae | rmse | val_loss | val_mae | val_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| head | 1 | 1 | 0.006040 | 0.145101 | 0.161754 | 0.006169 | 0.147035 | 0.163948 |
| finetune | 1 | 2 | 0.005933 | 0.142988 | 0.159817 | 0.006163 | 0.146913 | 0.163840 |

## 7. Prediction and Metric Summary
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 96 | 96 | 14.297029 | 15.986950 | 0.061557 | 0.104712 | 50.430346 | 51.600659 | 50.648501 | 0.214088 | 15.662763 | 79.294342 | 58.257735 | 14.080753 | 0 |
| val | overall | 48 | 48 | 14.691329 | 16.383996 | 0.164242 | -0.225514 | 50.444871 | 51.596165 | 50.680102 | 0.264800 | 16.179299 | 75.623190 | 58.645259 | 14.255487 | 0 |
| test | overall | 48 | 48 | 13.886628 | 16.342699 | 0.101281 | 0.072629 | 50.448084 | 52.023697 | 50.673947 | 0.260754 | 13.986123 | 79.336214 | 59.026697 | 14.063430 | 0 |

## 8. Dataset-wise Metrics
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | flive | 65 | 65 | 13.814043 | 15.273691 | 0.079545 | 0.059053 | 50.430346 | 51.600659 | 50.652971 | 0.210031 | 29.685128 | 79.294342 | 62.469924 | 9.687180 | 0 |
| train | koniq10k | 14 | 14 | 10.192231 | 10.872764 | 0.164835 | 0.330618 | 50.462002 | 51.336628 | 50.704824 | 0.307558 | 52.891964 | 68.643862 | 60.897055 | 3.876758 | 0 |
| train | spaq | 17 | 17 | 19.524162 | 21.306339 | -0.098039 | -0.259202 | 50.482845 | 50.725418 | 50.585026 | 0.074440 | 15.662763 | 71.804798 | 39.978750 | 18.459390 | 0 |
| val | flive | 32 | 32 | 15.318684 | 16.371048 | 0.264296 | 0.018383 | 50.444871 | 51.239908 | 50.665301 | 0.209161 | 38.500124 | 75.623190 | 64.494137 | 8.763455 | 0 |
| val | koniq10k | 7 | 7 | 8.830235 | 9.779866 | 0.428571 | 0.381680 | 50.449729 | 50.670981 | 50.529450 | 0.084863 | 50.360352 | 65.563202 | 59.316439 | 4.325227 | 0 |
| val | spaq | 9 | 9 | 17.019364 | 20.108088 | 0.333333 | -0.126428 | 50.450039 | 51.596165 | 50.849905 | 0.406882 | 16.179299 | 66.461027 | 37.327223 | 14.825029 | 0 |
| test | flive | 32 | 32 | 14.270606 | 16.361758 | 0.087977 | -0.038628 | 50.448084 | 51.243079 | 50.634833 | 0.180191 | 38.623863 | 79.336214 | 63.446141 | 10.168750 | 0 |
| test | koniq10k | 7 | 7 | 7.812828 | 9.148092 | -0.178571 | -0.219449 | 50.572360 | 50.953114 | 50.773217 | 0.126623 | 51.087493 | 64.998138 | 58.586045 | 4.729529 | 0 |
| test | spaq | 9 | 9 | 17.245437 | 20.186852 | 0.633333 | 0.378983 | 50.469506 | 52.023697 | 50.735809 | 0.466670 | 13.986123 | 67.804164 | 43.655844 | 19.076505 | 0 |

## 9. Failure Cases
- Failure CSV: `/home/omen_pc1/photo_score_project/outputs/mobile_student_iqa_50k_smoke_20260608/failure_cases.csv`.
- Failure rows: `0`.

## 10. Collapse / Safety Checks
- Collapse check: `{'pred_std_min_overall': 0.21408845550986075, 'pred_min_overall': 50.43034553527832, 'pred_max_overall': 52.02369689941406, 'nan_inf_count': 0, 'pred_std_too_low': True, 'pred_outside_0_100': False}`.

## 11. Comparison Against 14k Student Baseline
- Previous 14k student official SRCC: `0.7436`.
- MDIQA teacher official SRCC: `0.8227`.
- Prior gap decision: `STUDENT_GAP_MAINLY_CAPACITY_AND_COVERAGE`.
- This smoke is not a final official benchmark; compare only for pipeline sanity.

## 12. Final Decision
STUDENT_50K_SMOKE_NEEDS_PIPELINE_FIX

## 13. Recommended Next Step
- Fix TensorFlow GPU visibility and rerun the same smoke command before controlled training.
