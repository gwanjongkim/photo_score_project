# RGNet AVA-prior AADB Fine-tune Evaluation Audit

## 1. Completion Status
The RGNet AADB fine-tuning run using the AVA pre-trained weights has completed. However, the audit reveals a critical failure in the weight-loading process that resulted in the model training from random initialization rather than the intended AVA weights.

## 2. Artifacts
- **Final Model**: `outputs/rgnet_paper_v1_aadb_finetune_ava_prior_full/final_model.keras`
- **Best Weights**: `outputs/rgnet_paper_v1_aadb_finetune_ava_prior_full/best.weights.h5`
- **Training Log**: `outputs/rgnet_paper_v1_aadb_finetune_ava_prior_full/training_history.csv`
- **Summary**: `outputs/rgnet_paper_v1_aadb_finetune_ava_prior_full/train_summary.json`

## 3. Training Curve
The model completed 11 epochs before early stopping triggered (patience=3).
- **Best Epoch**: 8
- **Best Val Loss**: 0.0295
- **Best Val MAE**: 0.1374
- **Observation**: Validation loss remained significantly higher than the previous ImageNet-initialized baseline (`agg_mean_full`, val_loss=0.0224).

## 4. Best Checkpoint
Confirmed via `outputs/rgnet_paper_v1_aadb_finetune_ava_prior_smoke.log` (and inferred for the full run) that the backbone weight loading failed:
- **Error**: `DenseNet121 ... weights unavailable, falling back to random init: A total of 241 objects could not be loaded.`
- **Cause**: The script attempted to load full RGNet weights (Backbone+GCN) into a standalone DenseNet backbone using the Keras applications API, which expects only DenseNet weights.

## 5. Evaluation Metrics
Evaluation on the full AADB test set (1000 samples) using `final_model.keras`:
- **SRCC**: 0.5034
- **PLCC**: 0.5072
- **MAE**: 0.1486
- **RMSE**: 0.1828

## 6. Baseline Comparison
| Metric | New AVA-prior (Fail) | Previous Baseline (agg_mean_full) | Change |
| :--- | :---: | :---: | :---: |
| **SRCC** | 0.5034 | **0.6819** | -0.1785 |
| **PLCC** | 0.5072 | 0.6878 | -0.1806 |
| **MAE** | 0.1486 | 0.1198 | +0.0288 |

The new model is significantly worse than the existing ImageNet-initialized candidate.

## 7. Export Recommendation
**DO NOT EXPORT.**
The current model is underperforming and does not represent a valid transfer learning experiment due to the random initialization fallback. Exporting this would be a regression in app performance.

## 8. Final Judgment
The fine-tuning run was **unsuccessful** in its goal of leveraging AVA pre-training. The model should be discarded. To fix this, the training script must be modified to build the full RGNet model first and then call `model.load_weights()` before starting the fine-tuning process, ensuring compatibility between the AVA classification and AADB regression model structures.
