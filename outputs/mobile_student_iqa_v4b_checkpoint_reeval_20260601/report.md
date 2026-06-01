# Mobile Student IQA v4b Checkpoint Re-evaluation Report

## 1. Executive Summary
- Final decision: `BEST_CHECKPOINT_STILL_NEEDS_TUNING`
- Manifest: `outputs/student_training_manifest_50k_20260601/student_training_manifest.csv`
- Checkpoint directory: `outputs/mobile_student_iqa_v4b_20260601`
- Output directory: `outputs/mobile_student_iqa_v4b_checkpoint_reeval_20260601`
- Validated split counts: `{'val': 7507, 'test': 7495}`
- Image validation failures: `0`
- Best raw checkpoint: `final_model`
- Best raw metrics: SRCC `0.649433`, MAE `5.026057`, severe FP `11`, min dataset SRCC `0.472453`
- Best calibrated checkpoint: `best_by_val_mae_model` (isotonic), SRCC `0.651000`, MAE `4.612393`, severe FP `20`

## 2. Checkpoints Evaluated
| checkpoint               | val_mae  | val_srcc | test_mae | test_srcc | test_severe_fp | raw_readiness_satisfied |
| ------------------------ | -------- | -------- | -------- | --------- | -------------- | ----------------------- |
| best_model               | 5.170461 | 0.647608 | 5.109911 | 0.647852  | 12             | False                   |
| best_by_val_mae_model    | 5.010316 | 0.651415 | 4.966383 | 0.651385  | 13             | False                   |
| best_by_val_srcc_model   | 5.010316 | 0.651415 | 4.966383 | 0.651385  | 13             | False                   |
| best_by_val_safety_model | 7.171849 | 0.568935 | 7.092405 | 0.575179  | 6              | False                   |
| final_model              | 5.065358 | 0.650444 | 5.026057 | 0.649433  | 11             | False                   |

## 3. Raw Test Metrics
| checkpoint               | test_mae | test_rmse | test_srcc | test_plcc | test_pred_min | test_pred_max | test_pred_mean | test_pred_std |
| ------------------------ | -------- | --------- | --------- | --------- | ------------- | ------------- | -------------- | ------------- |
| best_model               | 5.109911 | 6.897205  | 0.647852  | 0.839575  | 12.916234     | 81.723177     | 64.599378      | 10.984720     |
| best_by_val_mae_model    | 4.966383 | 6.765429  | 0.651385  | 0.842397  | 13.976920     | 82.538080     | 65.059250      | 10.968598     |
| best_by_val_srcc_model   | 4.966383 | 6.765429  | 0.651385  | 0.842397  | 13.976920     | 82.538080     | 65.059250      | 10.968598     |
| best_by_val_safety_model | 7.092405 | 8.930882  | 0.575179  | 0.737481  | 12.376033     | 83.475929     | 62.854999      | 9.174472      |
| final_model              | 5.026057 | 6.813489  | 0.649433  | 0.842374  | 14.114782     | 82.387769     | 64.748062      | 10.974870     |

## 4. Dataset-wise Raw Metrics
| checkpoint               | split | dataset  | n    | mae      | rmse      | srcc     | plcc     | pred_min  | pred_max  | pred_mean | pred_std  |
| ------------------------ | ----- | -------- | ---- | -------- | --------- | -------- | -------- | --------- | --------- | --------- | --------- |
| best_model               | test  | flive    | 4799 | 4.377028 | 5.794917  | 0.469685 | 0.504906 | 19.238821 | 81.723177 | 68.343656 | 5.595164  |
| best_model               | test  | koniq10k | 1348 | 5.502585 | 7.406388  | 0.654138 | 0.698445 | 27.199319 | 81.315696 | 64.722486 | 7.374522  |
| best_model               | test  | spaq     | 1348 | 7.326364 | 9.491854  | 0.853889 | 0.858898 | 12.916234 | 78.763974 | 51.146306 | 16.557668 |
| best_by_val_mae_model    | test  | flive    | 4799 | 4.157837 | 5.532546  | 0.473825 | 0.511475 | 21.996062 | 82.538080 | 68.799842 | 5.529631  |
| best_by_val_mae_model    | test  | koniq10k | 1348 | 5.535180 | 7.465663  | 0.655647 | 0.699622 | 26.413080 | 81.962454 | 65.044662 | 7.523518  |
| best_by_val_mae_model    | test  | spaq     | 1348 | 7.276079 | 9.475437  | 0.855991 | 0.861623 | 13.976920 | 79.759908 | 51.757000 | 16.634683 |
| best_by_val_srcc_model   | test  | flive    | 4799 | 4.157837 | 5.532546  | 0.473825 | 0.511475 | 21.996062 | 82.538080 | 68.799842 | 5.529631  |
| best_by_val_srcc_model   | test  | koniq10k | 1348 | 5.535180 | 7.465663  | 0.655647 | 0.699622 | 26.413080 | 81.962454 | 65.044662 | 7.523518  |
| best_by_val_srcc_model   | test  | spaq     | 1348 | 7.276079 | 9.475437  | 0.855991 | 0.861623 | 13.976920 | 79.759908 | 51.757000 | 16.634683 |
| best_by_val_safety_model | test  | flive    | 4799 | 6.506710 | 7.998543  | 0.396683 | 0.443042 | 16.214189 | 83.475929 | 65.611780 | 6.268935  |
| best_by_val_safety_model | test  | koniq10k | 1348 | 6.303469 | 7.948172  | 0.586754 | 0.627864 | 34.681290 | 83.073974 | 62.712900 | 6.394336  |
| best_by_val_safety_model | test  | spaq     | 1348 | 9.966466 | 12.350713 | 0.769393 | 0.759212 | 12.376033 | 77.193880 | 53.182714 | 12.906243 |
| final_model              | test  | flive    | 4799 | 4.280824 | 5.689904  | 0.472453 | 0.508103 | 22.240277 | 82.387769 | 68.466872 | 5.532689  |
| final_model              | test  | koniq10k | 1348 | 5.516186 | 7.428753  | 0.655849 | 0.700132 | 26.044524 | 81.326467 | 64.889349 | 7.543999  |
| final_model              | test  | spaq     | 1348 | 7.189026 | 9.363507  | 0.857271 | 0.863549 | 14.114782 | 79.963785 | 51.367484 | 16.598720 |

## 5. Safety Metrics
| checkpoint               | split | dataset  | n    | fp_count | severe_fp_count | fn_count |
| ------------------------ | ----- | -------- | ---- | -------- | --------------- | -------- |
| best_model               | val   | overall  | 7507 | 72       | 8               | 1        |
| best_model               | val   | flive    | 4801 | 30       | 1               | 1        |
| best_model               | val   | koniq10k | 1353 | 20       | 3               | 0        |
| best_model               | val   | spaq     | 1353 | 22       | 4               | 0        |
| best_model               | test  | overall  | 7495 | 74       | 12              | 7        |
| best_model               | test  | flive    | 4799 | 30       | 1               | 4        |
| best_model               | test  | koniq10k | 1348 | 26       | 3               | 0        |
| best_model               | test  | spaq     | 1348 | 18       | 8               | 3        |
| best_by_val_mae_model    | val   | overall  | 7507 | 82       | 11              | 2        |
| best_by_val_mae_model    | val   | flive    | 4801 | 32       | 1               | 2        |
| best_by_val_mae_model    | val   | koniq10k | 1353 | 21       | 4               | 0        |
| best_by_val_mae_model    | val   | spaq     | 1353 | 29       | 6               | 0        |
| best_by_val_mae_model    | test  | overall  | 7495 | 82       | 13              | 5        |
| best_by_val_mae_model    | test  | flive    | 4799 | 31       | 1               | 3        |
| best_by_val_mae_model    | test  | koniq10k | 1348 | 29       | 3               | 0        |
| best_by_val_mae_model    | test  | spaq     | 1348 | 22       | 9               | 2        |
| best_by_val_srcc_model   | val   | overall  | 7507 | 82       | 11              | 2        |
| best_by_val_srcc_model   | val   | flive    | 4801 | 32       | 1               | 2        |
| best_by_val_srcc_model   | val   | koniq10k | 1353 | 21       | 4               | 0        |
| best_by_val_srcc_model   | val   | spaq     | 1353 | 29       | 6               | 0        |
| best_by_val_srcc_model   | test  | overall  | 7495 | 82       | 13              | 5        |
| best_by_val_srcc_model   | test  | flive    | 4799 | 31       | 1               | 3        |
| best_by_val_srcc_model   | test  | koniq10k | 1348 | 29       | 3               | 0        |
| best_by_val_srcc_model   | test  | spaq     | 1348 | 22       | 9               | 2        |
| best_by_val_safety_model | val   | overall  | 7507 | 40       | 5               | 2        |
| best_by_val_safety_model | val   | flive    | 4801 | 19       | 0               | 2        |
| best_by_val_safety_model | val   | koniq10k | 1353 | 10       | 1               | 0        |
| best_by_val_safety_model | val   | spaq     | 1353 | 11       | 4               | 0        |
| best_by_val_safety_model | test  | overall  | 7495 | 43       | 6               | 4        |
| best_by_val_safety_model | test  | flive    | 4799 | 15       | 0               | 4        |
| best_by_val_safety_model | test  | koniq10k | 1348 | 14       | 3               | 0        |
| best_by_val_safety_model | test  | spaq     | 1348 | 14       | 3               | 0        |
| final_model              | val   | overall  | 7507 | 75       | 10              | 2        |
| final_model              | val   | flive    | 4801 | 31       | 1               | 2        |
| final_model              | val   | koniq10k | 1353 | 20       | 4               | 0        |
| final_model              | val   | spaq     | 1353 | 24       | 5               | 0        |
| final_model              | test  | overall  | 7495 | 75       | 11              | 7        |
| final_model              | test  | flive    | 4799 | 31       | 1               | 4        |
| final_model              | test  | koniq10k | 1348 | 27       | 3               | 0        |
| final_model              | test  | spaq     | 1348 | 17       | 7               | 3        |

## 6. Post-hoc Calibration Metrics
| checkpoint               | method   | dataset  | mae      | rmse      | srcc     | plcc     | fp_count | severe_fp_count | fn_count |
| ------------------------ | -------- | -------- | -------- | --------- | -------- | -------- | -------- | --------------- | -------- |
| best_model               | isotonic | overall  | 4.643669 | 6.624340  | 0.647127 | 0.845814 | 118      | 20              | 6        |
| best_model               | isotonic | flive    | 3.595877 | 5.003704  | 0.468316 | 0.491656 | 42       | 1               | 4        |
| best_model               | isotonic | koniq10k | 5.681126 | 7.846401  | 0.653072 | 0.694481 | 42       | 5               | 0        |
| best_model               | isotonic | spaq     | 7.336444 | 9.658522  | 0.852918 | 0.860249 | 34       | 14              | 2        |
| best_by_val_mae_model    | isotonic | overall  | 4.612393 | 6.583824  | 0.651000 | 0.847873 | 112      | 20              | 4        |
| best_by_val_mae_model    | isotonic | flive    | 3.573110 | 4.956647  | 0.472879 | 0.496458 | 39       | 1               | 3        |
| best_by_val_mae_model    | isotonic | koniq10k | 5.627970 | 7.759930  | 0.656195 | 0.697928 | 37       | 5               | 0        |
| best_by_val_mae_model    | isotonic | spaq     | 7.296753 | 9.660729  | 0.856077 | 0.861073 | 36       | 14              | 1        |
| best_by_val_srcc_model   | isotonic | overall  | 4.612393 | 6.583824  | 0.651000 | 0.847873 | 112      | 20              | 4        |
| best_by_val_srcc_model   | isotonic | flive    | 3.573110 | 4.956647  | 0.472879 | 0.496458 | 39       | 1               | 3        |
| best_by_val_srcc_model   | isotonic | koniq10k | 5.627970 | 7.759930  | 0.656195 | 0.697928 | 37       | 5               | 0        |
| best_by_val_srcc_model   | isotonic | spaq     | 7.296753 | 9.660729  | 0.856077 | 0.861073 | 36       | 14              | 1        |
| best_by_val_safety_model | isotonic | overall  | 5.517734 | 7.963003  | 0.575438 | 0.767469 | 148      | 40              | 7        |
| best_by_val_safety_model | isotonic | flive    | 4.182598 | 5.888822  | 0.397808 | 0.452612 | 36       | 0               | 5        |
| best_by_val_safety_model | isotonic | koniq10k | 6.094954 | 8.397938  | 0.584895 | 0.634716 | 47       | 8               | 0        |
| best_by_val_safety_model | isotonic | spaq     | 9.693721 | 12.592817 | 0.768142 | 0.777399 | 65       | 32              | 2        |
| final_model              | isotonic | overall  | 4.611615 | 6.577584  | 0.649180 | 0.848177 | 120      | 20              | 4        |
| final_model              | isotonic | flive    | 3.577570 | 4.962263  | 0.472169 | 0.493601 | 43       | 1               | 3        |
| final_model              | isotonic | koniq10k | 5.656802 | 7.813581  | 0.654827 | 0.697477 | 41       | 5               | 0        |
| final_model              | isotonic | spaq     | 7.247720 | 9.583277  | 0.857579 | 0.863178 | 36       | 14              | 1        |

## 7. Comparison with v3/v4a/v4b Report
| checkpoint               | test_srcc | test_mae | test_severe_fp | flive_test_srcc | koniq10k_test_srcc | spaq_test_srcc |
| ------------------------ | --------- | -------- | -------------- | --------------- | ------------------ | -------------- |
| v3_headline              | 0.658204  | 4.543416 | 28             | 0.514385        | 0.643786           | 0.840971       |
| v4a_headline             | 0.630522  | 5.410899 | 13             | 0.449674        | 0.639833           | 0.832751       |
| v4b_report_headline      | 0.647898  | 5.108165 | 12             | 0.469730        | 0.654136           | 0.853897       |
| best_model               | 0.647852  | 5.109911 | 12             | 0.469685        | 0.654138           | 0.853889       |
| best_by_val_mae_model    | 0.651385  | 4.966383 | 13             | 0.473825        | 0.655647           | 0.855991       |
| best_by_val_srcc_model   | 0.651385  | 4.966383 | 13             | 0.473825        | 0.655647           | 0.855991       |
| best_by_val_safety_model | 0.575179  | 7.092405 | 6              | 0.396683        | 0.586754           | 0.769393       |
| final_model              | 0.649433  | 5.026057 | 11             | 0.472453        | 0.655849           | 0.857271       |

## 8. Best Checkpoint Recommendation
- Recommend `final_model` as the best raw checkpoint for any further baseline comparison work, but only if the final decision permits it.
- Calibration analysis favors `best_by_val_mae_model`; this does not modify the raw model.

## 9. Readiness Decision
- `BEST_CHECKPOINT_STILL_NEEDS_TUNING`
- final_model is the best raw tradeoff but misses readiness.
- test_srcc=0.649433, test_mae=5.026057, severe_fp=11, dataset_srcc_min=0.472453.

## 10. Risks
- Post-hoc calibration is analysis-only and must not be treated as a saved model improvement.
- The v4b split is reused from the manifest after image validation; any future manifest change would invalidate direct comparison.
- Readiness is judged on raw checkpoint predictions only.

## 11. Next Step
- Keep v4b in tuning; use this report to decide whether the best raw checkpoint is worth a targeted follow-up run.
