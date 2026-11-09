# Mobile Student IQA v2 Training Report

## 1. Summary
- Final decision: `NEEDS_MORE_TUNING`
- Input manifest: `outputs/student_training_manifest_design_20260601/student_training_manifest.csv`
- Output directory: `outputs/mobile_student_iqa_v2_training_loop_verify_20260601`
- Best model: `outputs/mobile_student_iqa_v2_training_loop_verify_20260601/best_model.keras`
- Final model: `outputs/mobile_student_iqa_v2_training_loop_verify_20260601/final_model.keras`

## 2. Dataset
- Split counts after image validation: `{'train': 512, 'val': 256, 'test': 256}`
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
- Stage A head epochs: `1`
- Stage A learning rate: `0.0003`
- Stage B fine-tune epochs: `4`
- Stage B learning rate: `1e-05`
- Stage B requested unfreeze last N layers: `30`
- Stage B trainable backbone layers: `24`
- Stage B trainable BatchNorm layers: `0`
- Batch size: `32`
- Train steps per epoch: `16`
- Validation steps: `8`
- Loss: `Huber(delta=0.05) on normalized target`
- Completed history rows: `5`

## 5. Overall Metrics
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 512 | 512 | 11.418406 | 14.939934 | 0.298032 | 0.352263 | 21.356483 | 90.183842 | 66.077651 |
| val | overall | 256 | 256 | 15.076555 | 19.731483 | 0.152481 | 0.133197 | 23.916893 | 92.063814 | 67.139359 |
| test | overall | 256 | 256 | 12.470493 | 16.329595 | 0.252699 | 0.260528 | 31.231707 | 91.743869 | 66.423755 |

## 6. Dataset-wise Metrics
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | flive | 82 | 82 | 8.449759 | 10.849608 | 0.059631 | 0.163983 | 33.201954 | 85.228646 | 68.263955 |
| test | koniq10k | 86 | 86 | 11.851492 | 14.966830 | 0.253512 | 0.215276 | 31.231707 | 91.743869 | 68.539234 |
| test | spaq | 88 | 88 | 16.822019 | 21.145274 | 0.207804 | 0.203268 | 35.921195 | 86.228275 | 62.641624 |

## 7. Safety Metrics
| split | dataset | n | finite_n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- |
| val | overall | 256 | 256 | 29 | 18 | 5 |
| val | flive | 87 | 87 | 0 | 0 | 1 |
| val | koniq10k | 81 | 81 | 7 | 2 | 2 |
| val | spaq | 88 | 88 | 22 | 16 | 2 |
| test | overall | 256 | 256 | 20 | 8 | 3 |
| test | flive | 82 | 82 | 0 | 0 | 0 |
| test | koniq10k | 86 | 86 | 7 | 1 | 0 |
| test | spaq | 88 | 88 | 13 | 7 | 3 |

## 8. v1 Comparison
| model | test_srcc | test_mae | test_severe_fp |
| --- | --- | --- | --- |
| v1 | 0.394443 | 12.138425 | 10.000000 |
| v2 | 0.252699 | 12.470493 | 8.000000 |

## 9. Failure Cases
- Load failure CSV: `outputs/mobile_student_iqa_v2_training_loop_verify_20260601/load_errors.csv`
- Load failure count: `0`

## 10. Risks
- This is a tuned baseline training run and is not exported to mobile.
- The guarded target depends on MOS during label design and is not a Flutter runtime formula.
- If dataset-wise SRCC remains low, tune data balance or target design before distillation expansion.

## 11. Final Decision
- `NEEDS_MORE_TUNING`
- Test SRCC 0.2527 is below 0.85.
- Test MAE 12.4705 is above 10.
- Test severe FP count is 8, expected 0.
- At least one test dataset SRCC is below the no-collapse guard of 0.50.
