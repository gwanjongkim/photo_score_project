# Mobile Student IQA v1 Training Report

## 1. Summary
- Final decision: `NEEDS_STUDENT_TUNING`
- Input manifest: `outputs/student_training_manifest_design_20260601/student_training_manifest.csv`
- Output directory: `outputs/mobile_student_iqa_v1_20260601`
- Best model: `outputs/mobile_student_iqa_v1_20260601/best_model.keras`
- Final model: `outputs/mobile_student_iqa_v1_20260601/final_model.keras`

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
- Epochs: `20`
- Batch size: `32`
- Train steps per epoch: `197`
- Validation steps: `43`
- Test prediction steps: `43`
- Learning rate: `0.0001`
- Loss: `Huber(delta=5.0)`
- Callbacks: EarlyStopping on `val_mae`, ReduceLROnPlateau on `val_mae`, ModelCheckpoint on `val_mae`.
- Completed epochs: `20`

## 5. Overall Metrics
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 6301 | 6301 | 11.867144 | 15.074259 | 0.400814 | 0.437412 | 2.999745 | 103.502586 | 60.218833 |
| val | overall | 1354 | 1354 | 11.875913 | 14.999640 | 0.416315 | 0.445503 | 13.377425 | 96.726357 | 60.200363 |
| test | overall | 1345 | 1345 | 12.138425 | 15.255040 | 0.394443 | 0.422607 | 14.680711 | 97.939384 | 60.582469 |

## 6. Dataset-wise Metrics
| split | dataset | n | finite_n | mae | rmse | srcc | plcc | pred_min | pred_max | pred_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | flive | 448 | 448 | 12.877265 | 16.513377 | 0.342096 | 0.306045 | 14.680711 | 97.853188 | 61.726977 |
| test | koniq10k | 448 | 448 | 9.877991 | 12.418866 | 0.329568 | 0.387361 | 28.451628 | 95.489174 | 63.696289 |
| test | spaq | 449 | 449 | 13.656631 | 16.466424 | 0.535123 | 0.539096 | 26.707403 | 97.939384 | 56.333624 |

## 7. Safety Metrics
| split | dataset | n | finite_n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- | --- |
| val | overall | 1354 | 1354 | 32 | 12 | 19 |
| val | flive | 452 | 452 | 1 | 1 | 11 |
| val | koniq10k | 451 | 451 | 12 | 1 | 1 |
| val | spaq | 451 | 451 | 19 | 10 | 7 |
| test | overall | 1345 | 1345 | 32 | 10 | 17 |
| test | flive | 448 | 448 | 3 | 2 | 13 |
| test | koniq10k | 448 | 448 | 13 | 1 | 2 |
| test | spaq | 449 | 449 | 16 | 7 | 2 |

## 8. Failure Cases
- Load failure CSV: `outputs/mobile_student_iqa_v1_20260601/load_errors.csv`
- Load failure count: `0`

## 9. Risks
- This is a first baseline and is not exported to mobile.
- The guarded target depends on MOS during label design and is not a Flutter runtime formula.
- Dataset-wise FLIVE behavior must be reviewed before student training is expanded.

## 10. Final Decision
- `NEEDS_STUDENT_TUNING`
- Test SRCC 0.3944 is below 0.85.
- Test MAE 12.1384 is above 10.
- Severe FP count is 44, expected 0.
- At least one test dataset SRCC is below the no-collapse guard of 0.50.
