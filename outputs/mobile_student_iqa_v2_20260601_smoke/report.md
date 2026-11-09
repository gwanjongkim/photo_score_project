# Mobile Student IQA v2 Training Report

## 1. Summary
- Final decision: `NEEDS_MORE_TUNING`
- Input manifest: `outputs/student_training_manifest_design_20260601/student_training_manifest.csv`
- Output directory: `outputs/mobile_student_iqa_v2_20260601_smoke`
- Best model: `outputs/mobile_student_iqa_v2_20260601_smoke/best_model.keras`
- Final model: `outputs/mobile_student_iqa_v2_20260601_smoke/final_model.keras`

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
- Stage B fine-tune epochs: `1`
- Stage B learning rate: `1e-05`
- Stage B requested unfreeze last N layers: `30`
- Stage B trainable backbone layers: `24`
- Stage B trainable BatchNorm layers: `0`
- Batch size: `32`
- Train steps per epoch: `16`
- Validation steps: `8`
- Loss: `Huber(delta=0.05) on normalized target`
- Completed history rows: `2`

## 5. Overall Metrics
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 512 | 512 | 12.553152 | 15.976783 | 0.217751 | 0.274133 | 19.614449 | 89.362222 | 63.381733 |
| val | overall | 256 | 256 | 15.340287 | 19.849240 | 0.120311 | 0.104034 | 21.919778 | 91.644770 | 64.744552 |
| test | overall | 256 | 256 | 12.742419 | 16.502859 | 0.223984 | 0.229719 | 28.426999 | 91.910261 | 63.942354 |

## 6. Dataset-wise Metrics
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | flive | 82 | 82 | 9.506738 | 12.066002 | 0.030986 | 0.125072 | 32.960257 | 84.492081 | 65.980868 |
| test | koniq10k | 86 | 86 | 12.045771 | 15.010870 | 0.217944 | 0.183798 | 28.426999 | 91.910261 | 66.305587 |
| test | spaq | 88 | 88 | 16.438300 | 20.890366 | 0.150832 | 0.141614 | 32.184860 | 84.916216 | 59.733308 |

## 7. Safety Metrics
| split | dataset | n | finite_n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- |
| val | overall | 256 | 256 | 25 | 14 | 5 |
| val | flive | 87 | 87 | 0 | 0 | 1 |
| val | koniq10k | 81 | 81 | 7 | 2 | 2 |
| val | spaq | 88 | 88 | 18 | 12 | 2 |
| test | overall | 256 | 256 | 14 | 7 | 3 |
| test | flive | 82 | 82 | 0 | 0 | 0 |
| test | koniq10k | 86 | 86 | 6 | 1 | 0 |
| test | spaq | 88 | 88 | 8 | 6 | 3 |

## 8. v1 Comparison
| model | test_srcc | test_mae | test_severe_fp |
| --- | --- | --- | --- |
| v1 | 0.394443 | 12.138425 | 10.000000 |
| v2 | 0.223984 | 12.742419 | 7.000000 |

## 9. Failure Cases
- Load failure CSV: `outputs/mobile_student_iqa_v2_20260601_smoke/load_errors.csv`
- Load failure count: `0`

## 10. Risks
- This is a tuned baseline training run and is not exported to mobile.
- The guarded target depends on MOS during label design and is not a Flutter runtime formula.
- If dataset-wise SRCC remains low, tune data balance or target design before distillation expansion.

## 11. Final Decision
- `NEEDS_MORE_TUNING`
- Test SRCC 0.2240 is below 0.85.
- Test MAE 12.7424 is above 10.
- Test severe FP count is 7, expected 0.
- At least one test dataset SRCC is below the no-collapse guard of 0.50.
