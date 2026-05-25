# Mobile A-LAMP v2 4096 Best Checkpoint Evaluation

## 1. Completion Status
The evaluation of the Mobile A-LAMP v2 4096 model's best checkpoint has been completed. The `best_val_auc_model.keras` (Epoch 5) was re-evaluated on the 4096-sample validation set using the same preprocessing and label conventions as the training script.

## 2. Training Curve Summary
Based on `training_log.csv`:
- **Best val_auc**: 0.7506 at Epoch 5.
- **Final val_auc**: 0.6919 at Epoch 20.
- **Overfitting Indicator**: Validation loss increased from 0.5565 (Epoch 1) to 1.2428 (Epoch 20), while training AUC reached 0.9944. This confirms severe overfitting in the later stages of training.

## 3. Best Checkpoint Metrics
The re-evaluation of `best_val_auc_model.keras` yielded the following metrics:
- **ROC-AUC**: 0.7506
- **Average Precision (AP)**: 0.8704
- **Accuracy (threshold 0.5)**: 0.7163
- **Confusion Matrix (0.5)**: TN=729, FP=467, FN=695, TP=2205
- **Sample Count**: 4096

## 4. Final Checkpoint Overfitting
The final model (`final_model.keras`) significantly underperforms compared to the best checkpoint:
- **ROC-AUC**: ~0.7029 (vs 0.7506)
- **Average Precision**: ~0.8354 (vs 0.8704)
- **Validation Loss**: 1.2428 (vs 0.5603 at Epoch 5)
The final model is effectively unusable due to the loss of generalization.

## 5. Comparison with MPNet-only and Graph Fusion
| Model | ROC-AUC | Average Precision |
| :--- | :--- | :--- |
| **Mobile A-LAMP v2 (Best)** | **0.7506** | **0.8704** |
| MPNet-only Baseline | ~0.7000 | 0.8445 |
| Mobile A-LAMP v2 (Final) | 0.7029 | 0.8354 |
| GraphLite/GraphGCN Fusion | < 0.7000 | - |

Mobile A-LAMP v2 (Best) shows a significant improvement (+0.05 ROC-AUC) over the MPNet-only baseline and previous graph-based attempts.

## 6. Prediction Distribution
- **Pred Min**: 0.0297
- **Pred Max**: 0.9912
- **Pred Mean**: 0.5951
- **Pred Std**: 0.2369
- **Positive Ratio (0.5)**: 0.6523
The predictions show healthy variance and span almost the entire [0, 1] range, unlike overfit models that often collapse to extremes.

## 7. Model Selection
The `best_val_auc_model.keras` (Epoch 5) should be kept as the primary artifact. The `final_model.keras` should be discarded or marked as overfit.

## 8. Final Judgment
Confirmed facts:
- The Best Checkpoint (Epoch 5) outperforms the MPNet-only baseline by a margin of 0.05 ROC-AUC.
- The training process successfully captured visual and layout features before overfitting.
- Preprocessing and label conventions (threshold=5.0) are consistent with the project's goals.

Assumptions:
- The baseline metrics for MPNet-only and Graph Fusion provided for comparison are correct.

Note: This evaluation is part of an internal audit and does not constitute an official A-LAMP reproduction.
