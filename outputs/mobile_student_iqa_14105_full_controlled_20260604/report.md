# Mobile Student IQA 14105 Longer Controlled Training Report

## 1. Executive Summary
- Final decision: `STUDENT_14105_LONGER_READY_FOR_FULL_CONTROLLED_TRAINING`.
- Run type: longer controlled training only, not final full distillation.
- Runtime seconds: `2249.40`.
- Train/val/test samples used: `{'train': 11283, 'val': 1409, 'test': 1413}`.
- Final train loss: `0.000742`.
- Final val loss: `0.001458`.
- Best val loss: `0.001458`.

## 2. Sources Inspected
- `outputs/student_distillation_manifest_14105_20260603/student_distillation_manifest.csv`
- `/home/omen_pc1/photo_score_project/outputs/mobile_student_iqa_14105_short_train_20260604/report.md`
- `/home/omen_pc1/photo_score_project/outputs/mobile_student_iqa_14105_short_train_20260604/history.csv`
- `/home/omen_pc1/photo_score_project/outputs/mobile_student_iqa_14105_short_train_20260604/metrics.csv`
- `/home/omen_pc1/photo_score_project/outputs/student_distillation_manifest_14105_20260603/report.md`
- `/home/omen_pc1/photo_score_project/outputs/known_dataset_full_label_coverage_audit_20260603/report.md`
- `/home/omen_pc1/photo_score_project/outputs/known_dataset_full_label_dryrun_20260603/report.md`
- Reused smoke trainer functions from `scripts/technical_iqa/train_mobile_student_iqa_14105_smoke.py`.

## 3. Manifest and Scope
- Scope: validated known-dataset teacher-cache subset, not MDIQA official full-dataset relabeling.
- Unknown/fallback excluded; c6_parity holdout only; Test_weight/product-gold validation only.
- Target: `target_norm`; metrics use 0..100 scale.
- Image load failure rows: `0`.

## 4. Environment / GPU Preflight
- TensorFlow: `2.21.0`.
- Built with CUDA: `True`.
- GPUs visible: `["PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')"]`.
- GPU count: `1`.

## 5. Training Configuration
- Backbone: `efficientnetv2b0`.
- Image size: `256`.
- Batch size: `16`.
- Head epochs: `5`.
- Finetune epochs: `20`.
- Loss: `huber`.
- ImageNet weights requested: `imagenet`.
- ImageNet weights loaded: `True`.
- Include preprocessing: `True`.
- Requested unfreeze_last_n: `24`.
- Finetune trainable backbone layers: `19`.

## 6. Training History
| phase | phase_epoch | global_epoch | loss | mae | rmse | val_loss | val_mae | val_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| head | 1 | 1 | 0.001980 | 0.059582 | 0.084496 | 0.002387 | 0.067996 | 0.090738 |
| head | 2 | 2 | 0.001306 | 0.044714 | 0.062359 | 0.001952 | 0.058475 | 0.078845 |
| head | 3 | 3 | 0.001138 | 0.040787 | 0.057326 | 0.001883 | 0.057100 | 0.076089 |
| head | 4 | 4 | 0.001050 | 0.038681 | 0.054361 | 0.001876 | 0.056853 | 0.076662 |
| head | 5 | 5 | 0.000961 | 0.036459 | 0.051608 | 0.001747 | 0.053946 | 0.072825 |
| finetune | 1 | 6 | 0.001124 | 0.040328 | 0.055094 | 0.001727 | 0.053786 | 0.071660 |
| finetune | 2 | 7 | 0.001026 | 0.038085 | 0.052143 | 0.001684 | 0.052956 | 0.070344 |
| finetune | 3 | 8 | 0.000992 | 0.037287 | 0.051072 | 0.001657 | 0.052436 | 0.069517 |
| finetune | 4 | 9 | 0.000952 | 0.036380 | 0.050077 | 0.001661 | 0.052562 | 0.069658 |
| finetune | 5 | 10 | 0.000946 | 0.036205 | 0.049351 | 0.001602 | 0.051200 | 0.068001 |
| finetune | 6 | 11 | 0.000908 | 0.035224 | 0.048220 | 0.001588 | 0.050871 | 0.067794 |
| finetune | 7 | 12 | 0.000904 | 0.035227 | 0.048183 | 0.001548 | 0.049985 | 0.066646 |
| finetune | 8 | 13 | 0.000861 | 0.034156 | 0.046660 | 0.001526 | 0.049440 | 0.066066 |
| finetune | 9 | 14 | 0.000846 | 0.033784 | 0.046318 | 0.001565 | 0.050379 | 0.067016 |
| finetune | 10 | 15 | 0.000862 | 0.034213 | 0.046750 | 0.001545 | 0.049871 | 0.066532 |
| finetune | 11 | 16 | 0.000827 | 0.033356 | 0.045437 | 0.001516 | 0.049217 | 0.065746 |
| finetune | 12 | 17 | 0.000809 | 0.032903 | 0.045043 | 0.001500 | 0.048887 | 0.065246 |
| finetune | 13 | 18 | 0.000804 | 0.032762 | 0.044703 | 0.001497 | 0.048806 | 0.065191 |
| finetune | 14 | 19 | 0.000791 | 0.032357 | 0.044382 | 0.001493 | 0.048755 | 0.065097 |
| finetune | 15 | 20 | 0.000783 | 0.032286 | 0.044079 | 0.001463 | 0.048033 | 0.064336 |
| finetune | 16 | 21 | 0.000762 | 0.031561 | 0.043628 | 0.001471 | 0.048224 | 0.064512 |
| finetune | 17 | 22 | 0.000763 | 0.031607 | 0.043362 | 0.001472 | 0.048241 | 0.064628 |
| finetune | 18 | 23 | 0.000745 | 0.031278 | 0.042581 | 0.001475 | 0.048359 | 0.064694 |
| finetune | 19 | 24 | 0.000749 | 0.031367 | 0.042906 | 0.001508 | 0.049244 | 0.065448 |
| finetune | 20 | 25 | 0.000742 | 0.031110 | 0.042662 | 0.001458 | 0.048011 | 0.064158 |

## 7. Best Checkpoint Summary
| monitor | best_epoch | best_value | checkpoint |
| --- | --- | --- | --- |
| val_loss | 25 | 0.001458 | not_saved_separately |
| val_mae | 25 | 0.048011 | outputs/mobile_student_iqa_14105_full_controlled_20260604/best_by_val_mae_model.keras |
- `best_by_val_srcc_model.keras` not implemented; SRCC is computed after final prediction pass to keep training bounded.

## 8. Prediction and Metric Summary
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 11283 | 11283 | 3.881279 | 5.093766 | 0.867959 | 0.922091 | 12.194797 | 77.286083 | 61.325947 | 7.872038 | 14.804595 | 88.805878 | 64.516098 | 9.806214 | 0 |
| val | overall | 1409 | 1409 | 4.801111 | 6.415775 | 0.714702 | 0.829117 | 15.040964 | 72.021180 | 61.419491 | 7.568933 | 18.707634 | 84.982681 | 64.481513 | 10.001077 | 0 |
| test | overall | 1413 | 1413 | 4.796548 | 6.422966 | 0.694477 | 0.842257 | 11.613049 | 76.829225 | 61.107487 | 8.085109 | 17.632763 | 81.683090 | 64.163819 | 10.403744 | 0 |

## 9. Dataset-wise Metrics
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | flive | 398 | 398 | 9.021419 | 9.689580 | 0.507349 | 0.589966 | 42.753744 | 72.021180 | 64.488155 | 4.001939 |
| val | koniq10k | 899 | 899 | 2.639641 | 3.385779 | 0.764299 | 0.770159 | 28.115582 | 70.414871 | 61.685049 | 4.412657 |
| val | spaq | 112 | 112 | 7.153600 | 9.601153 | 0.863575 | 0.869730 | 15.040964 | 70.822990 | 48.383206 | 17.422200 |
| test | flive | 399 | 399 | 8.963079 | 9.719480 | 0.517562 | 0.516904 | 43.584919 | 71.672857 | 64.372674 | 3.849801 |
| test | koniq10k | 901 | 901 | 2.724276 | 3.497422 | 0.723024 | 0.757147 | 31.617707 | 71.463186 | 61.663904 | 4.555742 |
| test | spaq | 113 | 113 | 6.607798 | 9.206938 | 0.898355 | 0.878527 | 11.613049 | 76.829225 | 45.141637 | 17.451639 |

## 10. Comparison Against Short Controlled Training
| split | dataset | metric | short | longer | delta | improved |
| --- | --- | --- | --- | --- | --- | --- |
| val | overall | mae_0_100 | 5.099313 | 4.801111 | -0.298202 | True |
| val | overall | srcc | 0.643581 | 0.714702 | 0.071120 | True |
| val | overall | plcc | 0.730197 | 0.829117 | 0.098920 | True |
| val | overall | severe_fp_count | 1.000000 | 1.000000 | 0.000000 | False |
| val | overall | pred_std | 6.711669 | 7.568933 | 0.857264 | True |
| val | flive | mae_0_100 | 7.135755 | 9.021419 | 1.885665 | False |
| val | flive | srcc | 0.324147 | 0.507349 | 0.183202 | True |
| val | flive | plcc | 0.444457 | 0.589966 | 0.145509 | True |
| val | flive | severe_fp_count | 0.000000 | 0.000000 | 0.000000 | False |
| val | flive | pred_std | 4.706282 | 4.001939 | -0.704343 | False |
| val | koniq10k | mae_0_100 | 3.272865 | 2.639641 | -0.633224 | True |
| val | koniq10k | srcc | 0.598899 | 0.764299 | 0.165400 | True |
| val | koniq10k | plcc | 0.640944 | 0.770159 | 0.129215 | True |
| val | koniq10k | severe_fp_count | 0.000000 | 0.000000 | 0.000000 | False |
| val | koniq10k | pred_std | 4.811435 | 4.412657 | -0.398779 | False |
| val | spaq | mae_0_100 | 12.603411 | 7.153600 | -5.449811 | True |
| val | spaq | srcc | 0.775292 | 0.863575 | 0.088283 | True |
| val | spaq | plcc | 0.712797 | 0.869730 | 0.156932 | True |
| val | spaq | severe_fp_count | 1.000000 | 1.000000 | 0.000000 | False |
| val | spaq | pred_std | 13.084322 | 17.422200 | 4.337878 | True |
| test | overall | mae_0_100 | 5.008367 | 4.796548 | -0.211819 | True |
| test | overall | srcc | 0.610541 | 0.694477 | 0.083936 | True |
| test | overall | plcc | 0.751056 | 0.842257 | 0.091201 | True |
| test | overall | severe_fp_count | 1.000000 | 0.000000 | -1.000000 | True |
| test | overall | pred_std | 7.382373 | 8.085109 | 0.702737 | True |
| test | flive | mae_0_100 | 7.030894 | 8.963079 | 1.932184 | False |
| test | flive | srcc | 0.287497 | 0.517562 | 0.230064 | True |
| test | flive | plcc | 0.329384 | 0.516904 | 0.187520 | True |
| test | flive | severe_fp_count | 0.000000 | 0.000000 | 0.000000 | False |
| test | flive | pred_std | 4.635648 | 3.849801 | -0.785847 | False |
| test | koniq10k | mae_0_100 | 3.300183 | 2.724276 | -0.575907 | True |
| test | koniq10k | srcc | 0.564519 | 0.723024 | 0.158505 | True |
| test | koniq10k | plcc | 0.604393 | 0.757147 | 0.152754 | True |
| test | koniq10k | severe_fp_count | 0.000000 | 0.000000 | 0.000000 | False |
| test | koniq10k | pred_std | 4.655618 | 4.555742 | -0.099876 | False |
| test | spaq | mae_0_100 | 11.605539 | 6.607798 | -4.997741 | True |
| test | spaq | srcc | 0.782181 | 0.898355 | 0.116174 | True |
| test | spaq | plcc | 0.736443 | 0.878527 | 0.142083 | True |
| test | spaq | severe_fp_count | 1.000000 | 0.000000 | -1.000000 | True |
| test | spaq | pred_std | 15.334057 | 17.451639 | 2.117582 | True |

## 11. FLIVE Weakness Analysis
- val FLIVE: SRCC `0.5073` (`+0.1832` vs short), MAE `9.0214`.
- test FLIVE: SRCC `0.5176` (`+0.2301` vs short), MAE `8.9631`.

## 12. Failure Cases
- Failure CSV: `outputs/mobile_student_iqa_14105_full_controlled_20260604/failure_cases.csv`.
- Failure rows: `0`.

## 13. Collapse / Safety Checks
- Collapse check: `{'pred_std_min_overall': 7.56893298210777, 'pred_min_overall': 11.613048613071442, 'pred_max_overall': 77.28608250617981, 'nan_inf_count': 0, 'pred_std_too_low': False, 'pred_outside_0_100': False}`.

| split | dataset | n | fp_count | severe_fp_count | fn_count |
| --- | --- | --- | --- | --- | --- |
| train | overall | 11283 | 8 | 0 | 0 |
| train | flive | 3184 | 1 | 0 | 0 |
| train | koniq10k | 7199 | 3 | 0 | 0 |
| train | spaq | 900 | 4 | 0 | 0 |
| val | overall | 1409 | 5 | 1 | 0 |
| val | flive | 398 | 1 | 0 | 0 |
| val | koniq10k | 899 | 2 | 0 | 0 |
| val | spaq | 112 | 2 | 1 | 0 |
| test | overall | 1413 | 1 | 0 | 0 |
| test | flive | 399 | 0 | 0 | 0 |
| test | koniq10k | 901 | 0 | 0 | 0 |
| test | spaq | 113 | 1 | 0 | 0 |

## 14. Final Decision
STUDENT_14105_LONGER_READY_FOR_FULL_CONTROLLED_TRAINING

## 15. Recommended Next Step
- Prepare a full controlled training plan with explicit stop criteria; do not run final full distillation yet.
