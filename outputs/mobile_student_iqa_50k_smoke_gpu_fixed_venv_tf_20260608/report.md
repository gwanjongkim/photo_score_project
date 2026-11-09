# Mobile Student IQA 50k Smoke Training Report

## 1. Executive Summary
- Final decision: `STUDENT_50K_SMOKE_READY_FOR_CONTROLLED_TRAINING`.
- Run type: bounded smoke training only; no full training, no TFLite export, no Flutter edits, no production `technical_score` edits.
- Runtime seconds: `155.24`.
- Usable train/val/test counts: `{'train': 4000, 'val': 1000, 'test': 1000}`.
- Final train loss: `0.002758`.
- Final val loss: `0.002619`.

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
- Capped counts before validation: `{'train': 4000, 'val': 1000, 'test': 1000}`.
- Image-path overlap across splits: `0`.
- Source-group overlap across splits: `0`.
- Official Track B rows in train: `0`.

## 4. Environment / GPU Preflight
- TensorFlow: `2.21.0`.
- GPUs visible to TensorFlow: `["PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')"]`.
- GPU count: `1`.
- Execution mode: `GPU`.

## 5. Smoke Training Configuration
- Backbone: `efficientnetv2b0`.
- Image size: `256`.
- Batch size: `16`.
- Head epochs: `1`.
- Finetune epochs: `1`.
- Loss: `huber`.
- target_norm: `teacher_target_0_100 / 100`.
- ImageNet weights requested: `imagenet`.
- ImageNet weights loaded: `True`.
- Backbone weights error: ``.
- Include preprocessing: `True`.
- Finetune trainable backbone layers: `13`.

## 6. Training History
| phase | phase_epoch | global_epoch | loss | mae | rmse | val_loss | val_mae | val_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| head | 1 | 1 | 0.003686 | 0.095508 | 0.127795 | 0.002666 | 0.074274 | 0.098709 |
| finetune | 1 | 2 | 0.002758 | 0.076308 | 0.101662 | 0.002619 | 0.073375 | 0.096651 |

## 7. Prediction and Metric Summary
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 4000 | 4000 | 6.527414 | 8.795195 | 0.696896 | 0.775880 | 8.970194 | 79.262793 | 60.340567 | 9.462186 | 10.749967 | 87.818849 | 59.978032 | 13.790484 | 0 |
| val | overall | 1000 | 1000 | 7.337507 | 9.665148 | 0.554848 | 0.684287 | 18.681565 | 80.598521 | 60.784498 | 8.815579 | 14.501648 | 79.914570 | 60.067715 | 13.214069 | 0 |
| test | overall | 1000 | 1000 | 7.519026 | 9.909618 | 0.522288 | 0.663622 | 18.924768 | 77.414262 | 60.785109 | 7.993017 | 14.627834 | 83.644652 | 60.163288 | 13.182265 | 0 |

## 8. Dataset-wise Metrics
| split | dataset | n | finite_n | mae_0_100 | rmse_0_100 | srcc | plcc | pred_min | pred_max | pred_mean | pred_std | target_min | target_max | target_mean | target_std | nan_inf_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | flive | 2712 | 2712 | 6.245620 | 8.144098 | 0.624507 | 0.648305 | 23.749536 | 79.262793 | 62.921591 | 6.356701 | 15.932895 | 87.818849 | 63.685449 | 10.626349 | 0 |
| train | koniq10k | 575 | 575 | 3.838153 | 4.910283 | 0.665202 | 0.676765 | 25.733304 | 75.542587 | 60.078846 | 6.625743 | 44.442070 | 68.092257 | 59.699285 | 4.900317 | 0 |
| train | spaq | 713 | 713 | 9.768016 | 12.737565 | 0.813456 | 0.792060 | 8.970194 | 74.013072 | 50.734328 | 13.982667 | 10.749967 | 79.486680 | 46.101122 | 19.316609 | 0 |
| val | flive | 677 | 677 | 7.218950 | 9.314174 | 0.413697 | 0.436341 | 38.275450 | 80.598521 | 63.289084 | 5.725976 | 18.996963 | 79.914570 | 63.660596 | 10.249056 | 0 |
| val | koniq10k | 144 | 144 | 4.411978 | 5.986224 | 0.680854 | 0.697041 | 27.255186 | 73.870611 | 59.920439 | 8.328379 | 44.806111 | 67.099279 | 59.557322 | 5.600893 | 0 |
| val | spaq | 179 | 179 | 10.139402 | 12.842519 | 0.761907 | 0.766326 | 18.681565 | 77.595288 | 52.006953 | 12.368698 | 14.501648 | 78.360319 | 46.889595 | 18.173238 | 0 |
| test | flive | 678 | 678 | 7.343791 | 9.426788 | 0.436291 | 0.460874 | 39.802736 | 77.414262 | 62.745597 | 5.360166 | 23.296875 | 83.644652 | 63.373657 | 10.584621 | 0 |
| test | koniq10k | 144 | 144 | 4.179769 | 5.478290 | 0.514468 | 0.576618 | 33.423796 | 76.535177 | 60.974644 | 6.667704 | 46.197146 | 67.445320 | 60.551394 | 4.239804 | 0 |
| test | spaq | 178 | 178 | 10.887914 | 13.745035 | 0.707242 | 0.735159 | 18.924768 | 69.229281 | 53.164302 | 11.864813 | 14.627834 | 75.627726 | 47.621057 | 18.392440 | 0 |

## 9. Failure Cases
- Failure CSV: `outputs/mobile_student_iqa_50k_smoke_gpu_fixed_venv_tf_20260608/failure_cases.csv`.
- Failure rows: `0`.

## 10. Collapse / Safety Checks
- Collapse check: `{'pred_std_min_overall': 7.993017392217919, 'pred_min_overall': 8.970194309949875, 'pred_max_overall': 80.59852123260498, 'nan_inf_count': 0, 'pred_std_too_low': False, 'pred_outside_0_100': False}`.

## 11. Comparison Against 14k Student Baseline
- Previous 14k student official SRCC: `0.7436`.
- MDIQA teacher official SRCC: `0.8227`.
- Prior gap decision: `STUDENT_GAP_MAINLY_CAPACITY_AND_COVERAGE`.
- This smoke is not a final official benchmark; compare only for pipeline sanity.

## 12. Final Decision
STUDENT_50K_SMOKE_READY_FOR_CONTROLLED_TRAINING

## 13. Recommended Next Step
- Start a controlled, still bounded 50k training run using the saved clean split.
