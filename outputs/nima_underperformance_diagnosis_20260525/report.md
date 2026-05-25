# NIMA Underperformance Diagnosis Report

## 1. Executive Summary
- **Main Problem**: A combination of a **critical input range bug (double rescaling)** and **model collapse** due to insufficient discrimination training.
- **Double Rescaling**: The project feeds `[0, 1]` images to `EfficientNetV2B0`, which has an internal `Rescaling(1/255.0)` layer. The model effectively sees images with brightness capped at 0.0039.
- **Model Collapse**: The model predicts "positive" for ~97% of samples, defaulting to the majority class distribution. Accuracy (71.5%) matches the dataset's positive ratio (70.2%) exactly, indicating near-zero discriminative power.
- **Incomplete Training**: Training was interrupted by crashes and the restart run only completed 4 epochs, which is insufficient for the EMD loss to converge on a 200k-image dataset.
- **Preprocessing Mismatch**: Training used random cropping (maintaining aspect ratio), whereas evaluation used direct resizing (warping), leading to a significant domain shift.
- **Checkpoint Inconsistency**: `final_model.keras` and `best.weights.h5` show a massive 0.30 mean score difference, indicating that `restore_best_weights` might not have been correctly applied during the fit/save cycle.

## 2. Current Performance Recap
| Metric | Result (Local) | Paper Target (AVA-MobileNet) | Gap |
| :--- | :--- | :--- | :--- |
| Accuracy | 71.5% | ~80.36% | -8.86% |
| SRCC | 0.3449 | ~0.510 | -0.165 |
| PLCC | 0.3527 | ~0.518 | -0.165 |
| MAE | 0.5781 | N/A | - |

## 3. Model and Checkpoint Inventory
- **Backbone**: `EfficientNetV2B0` (stronger than paper's MobileNetV1).
- **Checkpoints**:
  - `best.weights.h5`: Saved at Epoch 1 (val_loss 0.0959).
  - `final_model.keras`: Saved after Epoch 4 (restored or latest).
- **Consistency**: Massive difference confirmed. Keras best mean score 5.52 vs Final mean score 5.82 on the same image.
- **Export Source**: TFLite model matches `best.weights.h5` (diff 0.003).

## 4. Training Configuration Audit
- **Optimizer**: Adam, Learning Rate 1e-4.
- **Loss**: EMD (r=2), implementation verified correct.
- **Epochs**: Restart run only did 4 epochs before stopping via EarlyStopping.
- **Mixed Precision**: Enabled (`mixed_float16`).
- **Batch Size**: 64 (confirmed from log).

## 5. Dataset and Label Audit
- **Split Sizes**: Train (204k), Val (25k), Test (25k). Matches standard AVA splits.
- **Label Validity**: Confirmed correct mapping. `dist_1..dist_10` sum to 1. Mean score formula is correct ($\sum P_i \times i$).
- **Distribution Statistics**: GT Positive Ratio is 70.2%. Prediction Positive Ratio is 96.8%. The model is severely biased.

## 6. Preprocessing Audit
- **Train**: Resize to 256 then `random_crop(224)`. Matches aspect ratio.
- **Eval**: `tf.image.resize(224, 224)`. Warps image.
- **Normalization**: Double rescaling bug found. `[0, 1]` input scaled by `1/255` internally. Effective range is `[0, 0.0039]`.
- **Normalization (Stem)**: ImageNet mean subtraction applied to almost-zero pixels, resulting in constant values of approx -2.1 across the entire image.

## 7. Metric Audit
- Metric formulas in evaluation scripts are verified correct.
- High accuracy is a "false friend" caused by class imbalance.

## 8. Root Cause Table
| Cause | Status | Evidence | Impact | Fix Priority |
| :--- | :--- | :--- | :--- | :--- |
| **Double Rescaling** | **Confirmed** | EfficientNetV2 internal layer config + `_decode_image` code. | Extreme | Highest |
| **Model Collapse** | **Confirmed** | 97% positive predictions on 70% positive data. | High | High |
| **Under-training** | **Confirmed** | Only 4 epochs completed; training interrupted. | High | High |
| **Eval Warping** | **Confirmed** | `resize` vs `crop` in dataset scripts. | Medium | Medium |
| **Checkpoint Bug**| **Likely** | `final_model.keras` != `best.weights.h5`. | Low-Med | Medium |

## 9. Recommended Next Actions
1. **Immediate (No retraining)**: Use `best.weights.h5` for all evaluations, as it is the most valid checkpoint produced.
2. **Retraining Fix**: Modify `_decode_image` to return images in `[0, 255]` range or use a custom EfficientNet backbone without internal rescaling.
3. **Retraining Fix**: Change Evaluation preprocessing to `center_crop` to match Training domain.
4. **Retraining Fix**: Use a lower learning rate (e.g., 1e-5 or 1e-6) for the backbone and 1e-4 for the head.
5. **Retraining Fix**: Train for at least 20-30 epochs or until convergence.
6. **Retraining Fix**: Use class-balanced sampling or weighted loss to counter AVA's positive bias.

## 10. What Not To Claim Yet
- Do not claim the current NIMA model is reliable for production.
- Do not claim that SRCC will reach 0.5+ immediately after fixing the rescaling bug; other factors (LR, epochs) are also significant.
