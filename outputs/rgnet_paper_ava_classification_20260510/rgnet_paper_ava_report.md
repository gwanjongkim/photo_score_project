# RGNet Paper-Oriented AVA Classification Report

## 1. Environment
- pwd: `/home/omen_pc1/photo_score_project`
- Original date: `Sun May 10 21:54:00 KST 2026`
- Rerun GPU check: `2026-05-10T22:20:39.233356+09:00`
- Python: `Python 3.12.3`
- TensorFlow: `2.20.0`
- TensorFlow GPU visibility with `./.venv_gpu/bin/python`: `["PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')"]`
- nvidia-smi: not used for rerun; it is not on PATH in this WSL session.
- git status summary at initial setup: 90 changed/untracked status lines; pre-existing dirty worktree not cleaned.

## 2. Motivation
The AADB regression track has a strong paper-oriented v1 mean baseline: RGNet-paper-v1 `agg_mean_full` with AADB full test SRCC `0.6818650940218989`, PLCC `0.6877526576657776`, MAE `0.11975742876529694`, and RMSE `0.14860332788070868`.

This Track A extends that paper-oriented direction to AVA binary classification. The experiment label is **RGNet-paper-AVA-classification approximation**.

## 3. Dataset and Labels
- Label rule: `label = 1 if mean_score >= 5.0 else 0`.
- Required columns were present for train, val, and test.
- Initial path check resolved all CSV image paths, but training decode found one truncated JPEG in the 4096-row train subset: `data/raw/ava/images/277832.jpg`.
- The isolated AVA train/eval loaders now skip missing or undecodable image rows and record skipped counts.
- train: 204406 samples, positives 145219, negatives 59187, positive ratio 0.7104.
- val: 25551 samples, positives 18003, negatives 7548, positive ratio 0.7046.
- test: 25551 samples, positives 18228, negatives 7323, positive ratio 0.7134.

The class balance is positive-heavy at roughly 70 percent positive across all splits, so accuracy alone is not sufficient.

## 4. Model Architecture
The isolated model is implemented in `src/models/rgnet_paper_ava.py`.

- DenseNet121 fully convolutional backbone with ImageNet no-top weights for mid-run.
- ASPP approximation with dilation rates `[1, 3, 6, 12, 18]`.
- Spatial positions from the post-ASPP feature map are region nodes.
- Cosine similarity adjacency with row-wise softmax normalization.
- Residual graph convolution blocks, default `graph_blocks=3`.
- Region-level scalar logits.
- Aggregation candidates: `mean` and `lse` with `r=4`.
- Activation placement: `raw_before_aggregation_then_sigmoid`.

Paper-aligned pieces are DenseNet121, multi-scale context approximation, region graph, graph blocks, and region-level aggregation. Still approximate pieces are exact DenseASPP/RegionGraph details and lack of official RGNet paper weights.

## 5. Smoke Results
- `./.venv_gpu/bin/python -m py_compile` passed for all new scripts.
- Forward smoke passed: input `[2, 256, 256, 3]`, output `[2, 1]`, finite probabilities in `[0, 1]`.
- Smoke train completed: 64 train / 32 val samples, 1 epoch, batch size 4, backbone weights `none`.
- Smoke save/load check max abs diff: `0.0`.
- Smoke metrics remain execution-only evidence because the smoke model predicted all positives.

## 6. Mid-run Results
| Candidate | Samples | Accuracy | F1 | Precision | Recall | ROC-AUC | AP | Val Loss | Seconds/Image | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mean | 4095 train / 1024 val / 1024 test | 0.740234 | 0.828829 | 0.783455 | 0.879781 | 0.720011 | 0.847095 | 0.559156 | 0.012494 | skipped 1 train image; val F1 0.811383. |
| lse_r4 | 4095 train / 1024 val / 1024 test | 0.746094 | 0.836478 | 0.775058 | 0.908470 | 0.743862 | 0.867100 | 0.525041 | 0.012160 | skipped 1 train image; val F1 0.841640. |

Validation split comparison:

| Candidate | Val Accuracy | Val F1 | Val Precision | Val Recall | Val ROC-AUC | Val AP | Val BCE |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean | 0.721680 | 0.811383 | 0.783887 | 0.840878 | 0.718949 | 0.849597 | 0.559156 |
| lse_r4 | 0.754883 | 0.841640 | 0.779206 | 0.914952 | 0.735840 | 0.855293 | 0.525041 |

Both candidates trained on GPU at batch size 8. TensorFlow emitted non-fatal GPU memory fragmentation and large-allocation warnings for both candidates.

## 7. Full AVA Training: lse_r4
Full AVA training completed for `lse_r4`.

| Field | Value |
|---|---:|
| Train samples | 204402 |
| Val samples | 25551 |
| Train skipped images | 4 |
| Val skipped images | 0 |
| Epochs requested | 20 |
| Epochs completed | 13 |
| Best epoch | 10 |
| Best val loss | 0.48175305128097534 |
| Final train loss | 0.4252810776233673 |
| Aggregation | `lse` |
| LSE r | 4 |
| Score activation mode | `raw_before_aggregation_then_sigmoid` |
| GPU visible through `./.venv_gpu/bin/python` | yes |
| Save/load max abs diff | 0.0 |

The full run used the same paper-oriented approximation direction as the mid-run: DenseNet121, ASPP approximation, spatial region nodes, cosine adjacency, residual graph convolution, region logits, and LSE aggregation.

## 8. Full AVA Evaluation: lse_r4
Full AVA evaluation completed from `outputs/rgnet_paper_ava_classification_20260510/full_train/lse_r4/final_model.keras`.

| Split | Samples | Skipped | Accuracy | Precision | Recall | F1 | ROC-AUC | AP | BCE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test | 25549 | 2 | 0.769658 | 0.805728 | 0.892242 | 0.846781 | 0.797906 | 0.901876 | 0.481819 |
| val | 25551 | 0 | 0.771633 | 0.800890 | 0.899517 | 0.847343 | 0.801449 | 0.897984 | 0.481745 |

Confusion matrices at threshold `0.50`:

| Split | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| test | 3402 | 3921 | 1964 | 16262 |
| val | 3522 | 4026 | 1809 | 16194 |

Accuracy is not enough for this AVA setup because the split is positive-heavy. The `0.50` threshold reaches good F1 and ranking metrics, but the confusion matrices show a large false-positive count, so F1, ROC-AUC, AP, BCE, balanced accuracy, specificity, and confusion matrices should remain part of any decision.

## 9. Full-run Comparison
The full `lse_r4` run improved over the mid-run candidate on the main validation and test metrics.

| Metric | Mid-run lse_r4 test | Full-run lse_r4 test | Mid-run lse_r4 val | Full-run lse_r4 val |
|---|---:|---:|---:|---:|
| Accuracy | 0.746094 | 0.769658 | 0.754883 | 0.771633 |
| F1 | 0.836478 | 0.846781 | 0.841640 | 0.847343 |
| ROC-AUC | 0.743862 | 0.797906 | 0.735840 | 0.801449 |
| AP | 0.867100 | 0.901876 | 0.855293 | 0.897984 |
| BCE | 0.516347 | 0.481819 | 0.525041 | 0.481745 |

The most important gain is not raw accuracy. The full run improved ranking quality substantially: test ROC-AUC increased from `0.743862` to `0.797906`, and test AP increased from `0.867100` to `0.901876`.

## 10. Decision
The full AVA `lse_r4` run completed successfully and is the current RGNet paper-oriented AVA classification baseline for this project.

Threshold calibration on the saved full-run prediction CSVs gives two different operating points:

- Best validation F1: threshold `0.41`, validation F1 `0.850860`, test F1 `0.851918`.
- Best validation balanced accuracy and Youden J: threshold `0.70`, validation balanced accuracy `0.727189`, test balanced accuracy `0.723243`.
- Threshold `0.50` remains a strong baseline, with validation F1 `0.847343` and test F1 `0.846781`.

The threshold should be reported with the metric target. For paper-oriented comparison, `0.50` is the simplest fixed-threshold baseline. For operational false-positive reduction, `0.70` is more useful, but it lowers recall and F1.

## 11. Paper Comparability
This remains a paper-oriented approximation, not an official RGNet paper reproduction.

The experiment follows the RGNet paper direction by covering AVA binary classification with a region-graph-style model and LSE region evidence aggregation. However, exact DenseASPP details, exact RegionGraph details, original training hyperparameters, and official author weights are not locally available here. The binary label rule also simplifies AVA into `mean_score >= 5.0`.

The practical app RGNet path is untouched by this report update. Flutter, `forWeights/`, and `src/models/rgnet.py` were not modified.

## 12. Next Step Recommendation
Preserve this full AVA `lse_r4` result as the current RGNet paper-oriented classification baseline. Use the calibration outputs to document threshold tradeoffs, then move to the A-LAMP-paper-AVA track before spending more effort on RGNet architecture tuning.
