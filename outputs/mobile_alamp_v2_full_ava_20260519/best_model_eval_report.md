# Mobile A-LAMP v2 Full-AVA Best Checkpoint Evaluation

## 1. Completion Status
- **Training completed**: Yes (20 epochs)
- **Dataset**: Full AVA (204,406 training samples, 25,551 validation samples)
- **Output Directory**: `outputs/mobile_alamp_v2_full_ava_20260519/`
- **Evaluation Status**: Completed using `best_val_auc_model.keras` (Epoch 2).

## 2. Training Curve Summary
- **Peak Performance**: Epoch 2 reached the highest validation ROC-AUC of **0.8097**.
- **Early Convergence**: The model reached peak ranking performance very quickly (Epoch 2), after which validation metrics began to degrade.
- **Divergence**: After Epoch 2, validation loss started to rise (0.566 -> 1.120) while training loss continued to fall (0.533 -> 0.200).

## 3. Best Checkpoint Metrics (Epoch 2)
Re-evaluated on the full AVA validation set:
- **ROC-AUC**: 0.8098
- **Average Precision (AP)**: 0.9033
- **Accuracy (Threshold 0.5)**: 0.7046
- **Confusion Matrix (0.5)**:
  - TP: 12,005 / FN: 5,947
  - TN: 5,999 / FP: 1,600
- **Positive Prediction Ratio**: 0.5325
- **Prediction Range**: [0.00007, 0.99996] (Mean: 0.5226, Std: 0.2702)

## 4. Final Checkpoint Overfitting (Epoch 20)
- **ROC-AUC**: 0.7460 (Log) / 0.7518 (Final summary)
- **Validation Loss**: 1.120 (vs. 0.566 at best epoch)
- **Overfitting Gap**: Train AUC (0.9747) vs. Val AUC (0.7460).
- **Judgment**: The final model is severely overfit and exhibits significantly worse ranking performance compared to the best checkpoint.

## 5. Comparison with 4096 Baseline
| Metric | 4096 Baseline | Full-AVA (Best) | Difference |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | ~0.7506 | **0.8098** | **+0.0592** |
| **AP** | ~0.8704 | **0.9033** | **+0.0329** |

The Full-AVA training yielded a substantial improvement in both ranking (AUC) and precision-recall balance (AP).

## 6. Prediction Distribution
- The best model shows a well-balanced prediction distribution (mean 0.52).
- The prediction range is full [0, 1], indicating strong discriminative power.
- Compared to the final model (mean 0.649), the best model is less biased towards the positive class.

## 7. Export Recommendation
- **Target Model**: `outputs/mobile_alamp_v2_full_ava_20260519/best_val_auc_model.keras`
- **Action**: Export to FP32 and FP16 TFLite.
- **Reason**: The best checkpoint provides a ~6% absolute boost in ROC-AUC over the previous baseline and ~5.8% over the overfit final checkpoint.

## 8. Final Judgment
The Mobile A-LAMP v2 full-AVA training was successful. While the model overfitted quickly, the early checkpoint (Epoch 2) captures a high-quality ranking capability that significantly exceeds the previous 4096-sample baseline. 

**Recommendation: PROCEED WITH TFLITE EXPORT OF BEST CHECKPOINT.**
