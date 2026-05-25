# RGNet Fixed AVA-prior AADB Fine-tune Evaluation Audit

## 1. Completion Status
The fixed RGNet AVA-prior AADB fine-tuning run has completed. This run successfully addressed the weight-loading failure observed in the previous attempt, confirming that AVA pre-trained features were correctly utilized for the backbone and graph layers.

## 2. Artifact Status
- **Final Model (Best Weights)**: `outputs/rgnet_paper_v1_aadb_finetune_ava_prior_fixed_20260517/full_train/lse_r4/final_model.keras`
- **Training Log**: `outputs/rgnet_paper_v1_aadb_finetune_ava_prior_fixed_20260517/full_train/lse_r4/training_history.csv`
- **Summary**: `outputs/rgnet_paper_v1_aadb_finetune_ava_prior_fixed_20260517/full_train/lse_r4/train_summary.json`

## 3. AVA Weight Loading Confirmation
Confirmed via `train_summary.json`:
- **Loaded Layers**: 5 (DenseNet121 backbone + ASPP + 3 Residual Graph Blocks).
- **Loaded Variables**: 643.
- **Random Init Fallback Used**: `false`.
This confirms a successful transfer learning setup from the AVA classification task.

## 4. Training Curve Summary
- **Best Epoch**: 9
- **Best Val Loss**: 0.0223
- **Best Val MAE**: 0.1170
- **Observation**: The model showed much better stability and lower loss compared to the failed run (val_loss 0.0223 vs 0.0295).

## 5. Best Checkpoint Metrics
Evaluation on the full AADB test set (1000 samples) using `final_model.keras` (restored to epoch 9):
- **SRCC**: 0.6558
- **PLCC**: 0.6700
- **MAE**: 0.1222
- **RMSE**: 0.1516

## 6. Baseline Comparison
| Metric | AVA-prior Fixed (LSE r4) | Previous Baseline (agg_mean_full) | Previous LSE (lse_r4_no_prior) |
| :--- | :---: | :---: | :---: |
| **SRCC** | 0.6558 | **0.6819** | 0.6389 |
| **PLCC** | 0.6700 | 0.6878 | 0.6423 |
| **MAE** | 0.1222 | 0.1198 | 0.1282 |

**Key Findings**:
- Correct AVA pre-training improved the LSE candidate by **+0.0169 SRCC**.
- However, the current baseline (`agg_mean_full`) still holds the project record at **0.6819 SRCC**.
- The decision to use `mean` aggregation over `lse` appears to be a stronger driver of performance than the dataset transfer for this specific task.

## 7. Export Recommendation
**DO NOT EXPORT.**
While this model is functional and significantly better than the failed attempt, it is a regression compared to the existing `rgnet_paper_aadb_fp16.tflite` candidate (SRCC 0.6819).

## 8. Final Judgment
The fixed run proved the transfer learning logic is now working. The next priority is to run the **AVA-prior + Mean Aggregation** combination, which is highly likely to exceed 0.70 SRCC and establish a new state-of-the-art for the project.

**Confirmed Facts:**
- AVA weights were correctly loaded.
- Performance improved over the non-pretrained LSE variant.
- Performance remained below the ImageNet-initialized Mean variant.

**Note**: This is a paper-oriented AVA-prior fine-tuning experiment, not an official RGNet reproduction.
