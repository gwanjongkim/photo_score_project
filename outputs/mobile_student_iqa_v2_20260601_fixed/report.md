# Mobile Student IQA v2 Training Report

## 1. Summary
- Final decision: `NEEDS_MORE_TUNING`
- Input manifest: `outputs/student_training_manifest_design_20260601/student_training_manifest.csv`
- Output directory: `outputs/mobile_student_iqa_v2_20260601_fixed`
- Best model: `outputs/mobile_student_iqa_v2_20260601_fixed/best_model.keras`
- Final model: `outputs/mobile_student_iqa_v2_20260601_fixed/final_model.keras`

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
| train | overall | 6301 | 6301 | 3.956874 | 5.888472 | 0.880186 | 0.921919 | 8.094892 | 86.938858 | 62.097155 |
| val | overall | 1354 | 1354 | 7.040720 | 9.236782 | 0.652904 | 0.796844 | 12.821166 | 85.245210 | 62.611325 |
| test | overall | 1345 | 1345 | 7.234272 | 9.490268 | 0.666845 | 0.778319 | 13.378684 | 85.555083 | 62.408077 |

## 6. Dataset-wise Metrics
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | flive | 448 | 448 | 5.861572 | 7.772535 | 0.361239 | 0.396797 | 20.039552 | 83.605987 | 68.582505 |
| test | koniq10k | 448 | 448 | 6.155196 | 8.023422 | 0.602450 | 0.667942 | 32.414117 | 85.555083 | 65.190423 |
| test | spaq | 449 | 449 | 9.680588 | 12.053411 | 0.768217 | 0.766532 | 13.378684 | 80.832326 | 53.471251 |

## 7. Safety Metrics
| split | dataset | n | finite_n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- |
| val | overall | 1354 | 1354 | 23 | 5 | 1 |
| val | flive | 452 | 452 | 2 | 1 | 1 |
| val | koniq10k | 451 | 451 | 10 | 0 | 0 |
| val | spaq | 451 | 451 | 11 | 4 | 0 |
| test | overall | 1345 | 1345 | 18 | 5 | 0 |
| test | flive | 448 | 448 | 2 | 1 | 0 |
| test | koniq10k | 448 | 448 | 6 | 1 | 0 |
| test | spaq | 449 | 449 | 10 | 3 | 0 |

## 8. v1 Comparison
| model | test_srcc | test_mae | test_severe_fp |
| --- | --- | --- | --- |
| v1 | 0.394443 | 12.138425 | 10.000000 |
| v2 | 0.666845 | 7.234272 | 5.000000 |

## 9. Failure Cases
- Load failure CSV: `outputs/mobile_student_iqa_v2_20260601_fixed/load_errors.csv`
- Load failure count: `0`

## 10. Risks
- This is a tuned baseline training run and is not exported to mobile.
- The guarded target depends on MOS during label design and is not a Flutter runtime formula.
- If dataset-wise SRCC remains low, tune data balance or target design before distillation expansion.

## 11. Final Decision
- `NEEDS_MORE_TUNING`
- Test SRCC 0.6668 is below 0.85.
- Test severe FP count is 5, expected 0.
- At least one test dataset SRCC is below the no-collapse guard of 0.50.
