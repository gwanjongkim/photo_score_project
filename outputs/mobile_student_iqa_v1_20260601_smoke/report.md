# Mobile Student IQA v1 Training Report

## 1. Summary
- Final decision: `NEEDS_STUDENT_TUNING`
- Input manifest: `outputs/student_training_manifest_design_20260601/student_training_manifest.csv`
- Output directory: `outputs/mobile_student_iqa_v1_20260601_smoke`
- Best model: `outputs/mobile_student_iqa_v1_20260601_smoke/best_model.keras`
- Final model: `outputs/mobile_student_iqa_v1_20260601_smoke/final_model.keras`

## 2. Dataset
- Split counts after image validation: `{'train': 6301, 'val': 1354, 'test': 1345}`
- Image load failures: `0`
- Target column: `guarded_target_0_100`
- Safety reference column: `quality_score_0_100`

## 3. Model
- Backbone: `tf.keras.applications.MobileNetV3Small`
- Input size: `224x224x3`
- Head: `GlobalAveragePooling2D -> Dense(128, relu) -> Dropout(0.2) -> Dense(1)`
- Include MobileNetV3 preprocessing in backbone: `True`
- Requested backbone weights: `imagenet`
- ImageNet weights loaded: `True`
- Backbone trainable: `False`

## 4. Training Configuration
- Epochs: `1`
- Batch size: `32`
- Train steps per epoch: `197`
- Validation steps: `43`
- Test prediction steps: `43`
- Learning rate: `0.0001`
- Loss: `Huber(delta=5.0)`
- Callbacks: EarlyStopping on `val_mae`, ReduceLROnPlateau on `val_mae`, ModelCheckpoint on `val_mae`.
- Completed epochs: `1`

## 5. Overall Metrics
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 6301 | 6301 | 25.290436 | 28.899541 | 0.137519 | 0.145806 | -1.499361 | 72.743835 | 38.742569 |
| val | overall | 1354 | 1354 | 24.964626 | 28.296174 | 0.195400 | 0.191412 | -0.751353 | 70.423325 | 39.026651 |
| test | overall | 1345 | 1345 | 24.683672 | 28.143147 | 0.194885 | 0.191729 | -0.044236 | 68.611092 | 39.321441 |

## 6. Dataset-wise Metrics
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | flive | 448 | 448 | 32.423850 | 34.315461 | 0.230039 | 0.210143 | -0.044236 | 66.696152 | 39.182038 |
| test | koniq10k | 448 | 448 | 23.206859 | 26.056694 | 0.173423 | 0.186511 | 10.161359 | 65.417679 | 40.785579 |
| test | spaq | 449 | 449 | 18.434258 | 22.808217 | 0.237009 | 0.242574 | 7.648971 | 68.611092 | 37.999656 |

## 7. Safety Metrics
| split | dataset | n | finite_n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- |
| val | overall | 1354 | 1354 | 0 | 0 | 162 |
| val | flive | 452 | 452 | 0 | 0 | 86 |
| val | koniq10k | 451 | 451 | 0 | 0 | 28 |
| val | spaq | 451 | 451 | 0 | 0 | 48 |
| test | overall | 1345 | 1345 | 0 | 0 | 163 |
| test | flive | 448 | 448 | 0 | 0 | 97 |
| test | koniq10k | 448 | 448 | 0 | 0 | 27 |
| test | spaq | 449 | 449 | 0 | 0 | 39 |

## 8. Failure Cases
- Load failure CSV: `outputs/mobile_student_iqa_v1_20260601_smoke/load_errors.csv`
- Load failure count: `0`

## 9. Risks
- This is a first baseline and is not exported to mobile.
- The guarded target depends on MOS during label design and is not a Flutter runtime formula.
- Dataset-wise FLIVE behavior must be reviewed before student training is expanded.

## 10. Final Decision
- `NEEDS_STUDENT_TUNING`
- Test SRCC 0.1949 is below 0.85.
- Test MAE 24.6837 is above 10.
- At least one test dataset SRCC is below the no-collapse guard of 0.50.
