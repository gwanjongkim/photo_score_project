# Mobile Student IQA v2 Training Report

## 1. Summary
- Final decision: `NEEDS_MORE_TUNING`
- Input manifest: `outputs/student_training_manifest_design_20260601/student_training_manifest.csv`
- Output directory: `outputs/mobile_student_iqa_v2_20260601`
- Best model: `outputs/mobile_student_iqa_v2_20260601/best_model.keras`
- Final model: `outputs/mobile_student_iqa_v2_20260601/final_model.keras`

## 2. Dataset
- Split counts after image validation: `{'train': 6301, 'val': 1354, 'test': 1345}`
- Image load failures: `0`
- Target column: `guarded_target_0_100` normalized as `target_norm`
- Safety reference column: `quality_score_0_100`

## 3. Model
- Backbone: `tf.keras.applications.MobileNetV3Small`
- Input size: `224x224x3`
- Head: `GlobalAveragePooling2D -> Dense(128, relu) -> Dropout(0.2) -> Dense(1, sigmoid)`
- Include MobileNetV3 preprocessing in backbone: `True`
- Requested backbone weights: `imagenet`
- ImageNet weights loaded: `True`

## 4. Training Configuration
- Stage A head epochs: `5`
- Stage A learning rate: `0.0003`
- Stage B fine-tune epochs: `20`
- Stage B learning rate: `1e-05`
- Stage B requested unfreeze last N layers: `30`
- Stage B trainable backbone layers: `24`
- Stage B trainable BatchNorm layers: `0`
- Batch size: `32`
- Train steps per epoch: `197`
- Validation steps: `43`
- Loss: `Huber(delta=0.05) on normalized target`
- Completed history rows: `25`

## 5. Overall Metrics
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 6301 | 6301 | 5.410941 | 7.639426 | 0.797397 | 0.864156 | 9.980621 | 86.711067 | 62.417741 |
| val | overall | 1354 | 1354 | 7.277648 | 9.549469 | 0.640706 | 0.781769 | 12.497938 | 85.410315 | 62.945602 |
| test | overall | 1345 | 1345 | 7.494714 | 9.914824 | 0.646023 | 0.754783 | 15.657428 | 86.417550 | 62.724722 |

## 6. Dataset-wise Metrics
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | flive | 448 | 448 | 5.870127 | 7.820620 | 0.348981 | 0.389889 | 16.005577 | 83.204740 | 68.588928 |
| test | koniq10k | 448 | 448 | 6.332923 | 8.371381 | 0.572440 | 0.639486 | 31.139997 | 86.417550 | 65.371525 |
| test | spaq | 449 | 449 | 10.274887 | 12.787632 | 0.743032 | 0.738187 | 15.657428 | 79.490048 | 54.232670 |

## 7. Safety Metrics
| split | dataset | n | finite_n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- |
| val | overall | 1354 | 1354 | 28 | 8 | 1 |
| val | flive | 452 | 452 | 2 | 1 | 1 |
| val | koniq10k | 451 | 451 | 11 | 1 | 0 |
| val | spaq | 451 | 451 | 15 | 6 | 0 |
| test | overall | 1345 | 1345 | 23 | 5 | 0 |
| test | flive | 448 | 448 | 3 | 1 | 0 |
| test | koniq10k | 448 | 448 | 10 | 1 | 0 |
| test | spaq | 449 | 449 | 10 | 3 | 0 |

## 8. v1 Comparison
| model | test_srcc | test_mae | test_severe_fp |
| --- | --- | --- | --- |
| v1 | 0.394443 | 12.138425 | 10.000000 |
| v2 | 0.646023 | 7.494714 | 5.000000 |

## 9. Failure Cases
- Load failure CSV: `outputs/mobile_student_iqa_v2_20260601/load_errors.csv`
- Load failure count: `0`

## 10. Risks
- This is a tuned baseline training run and is not exported to mobile.
- The guarded target depends on MOS during label design and is not a Flutter runtime formula.
- If dataset-wise SRCC remains low, tune data balance or target design before distillation expansion.

## 11. Final Decision
- `NEEDS_MORE_TUNING`
- Test SRCC 0.6460 is below 0.85.
- Test severe FP count is 5, expected 0.
- At least one test dataset SRCC is below the no-collapse guard of 0.50.
