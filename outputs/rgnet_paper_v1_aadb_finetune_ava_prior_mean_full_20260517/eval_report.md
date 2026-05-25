# RGNet AVA-prior Mean AADB Fine-tune Evaluation Audit

## 1. Completion Status
The RGNet AVA-prior + Mean aggregation fine-tuning run has completed. The training successfully utilized the AVA pre-trained weights, and evaluation was performed on the full AADB test set.

## 2. Artifact Status
- **Final Model (Best Weights)**: `outputs/rgnet_paper_v1_aadb_finetune_ava_prior_mean_full_20260517/final_model.keras`
- **Training Log**: `outputs/rgnet_paper_v1_aadb_finetune_ava_prior_mean_full_20260517/training_history.csv`
- **Summary**: `outputs/rgnet_paper_v1_aadb_finetune_ava_prior_mean_full_20260517/train_summary.json`

## 3. AVA Weight Loading Confirmation
Confirmed via `train_summary.json`:
- **Loaded Layers**: 5 components (Backbone + Context + 3 Graph Blocks).
- **Random Init Fallback Used**: `false`.
Weight loading from the AVA classification run was successful and identical to the previous "Fixed LSE" run.

## 4. Training Curve Summary
- **Best Epoch**: 8
- **Best Val Loss**: 0.0230
- **Best Val MAE**: 0.1181
- **Observation**: While training was stable, the validation loss was higher than both the previous LSE-prior run (0.0223) and the ImageNet-Mean baseline (0.0224).

## 5. Best Checkpoint Metrics
Evaluation on the full AADB test set (1000 samples) using `final_model.keras`:
- **SRCC**: 0.6493
- **PLCC**: 0.6624
- **MAE**: 0.1223
- **RMSE**: 0.1525

## 6. Baseline Comparison
| Metric | New AVA-prior + Mean | Existing Baseline (agg_mean_full) | AVA-prior + LSE Fixed |
| :--- | :---: | :---: | :---: |
| **SRCC** | 0.6493 | **0.6819** | 0.6558 |
| **PLCC** | 0.6624 | 0.6878 | 0.6700 |
| **MAE** | 0.1223 | 0.1198 | 0.1222 |

**Analysis**:
- Counter-intuitively, the AVA pre-trained model performed **worse** than the ImageNet-initialized model for the Mean aggregation architecture.
- It also slightly underperformed the AVA-prior + LSE model (SRCC 0.6558).
- This suggests that Mean aggregation is highly effective with general features (ImageNet) but may suffer from feature interference or suboptimal gradient flow when initialized from a specific binary classification task (AVA High/Low).

## 7. Export Recommendation
**DO NOT EXPORT.**
This model is a regression in performance. The project state-of-the-art for RGNet remains the `agg_mean_full` candidate (SRCC 0.6819), which is already exported as `models/aesthetic/rgnet_paper_aadb_fp16.tflite`.

## 8. Final Judgment
The experiment successfully combined AVA pre-training with Mean aggregation, but the combination did not yield a performance improvement. The existing ImageNet-initialized Mean model remains the superior candidate. Future efforts should focus on either larger-scale regression pre-training (instead of classification) or deeper graph architectures.

**Confirmed Facts:**
- AVA weights were correctly loaded.
- Ranking metrics (SRCC 0.6493) did not beat the 0.6819 baseline.
- The model is inferior to the existing Flutter candidate.
