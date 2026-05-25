# A-LAMP Multi-Patch Teacher Improvement Plan

## 1. Summary
The A-LAMP Dual-Branch GCN prototype failed to outperform the Multi-Patch-only baseline, exhibiting severe positive bias and lower overall accuracy. Consequently, GCN development is halted for the teacher role. This plan refocuses on strengthening the Multi-Patch-only teacher, which is currently the best performing candidate (Accuracy 0.7633). Empirical analysis reveals a significant positive bias (Negative Mean Score: 0.57) and poor class separation. We propose a systematic approach to bridge the ~6.17% accuracy gap to the paper target (0.825) by addressing class imbalance, unfreezing the backbone, and calibrating the decision boundary.

## 2. Current Full-Test Metrics
Based on the full available AVA test set (25,443 samples):
- **Accuracy**: 0.7633
- **F1 Score**: 0.8482
- **ROC-AUC**: 0.7877
- **Average Precision**: 0.8955
- **Precision**: 0.7795
- **Recall**: 0.9303 (Very high, indicates positive bias)
- **Specificity**: 0.3527 (Very low, indicates failure on negative samples)
- **Positive Ratio (Ground Truth)**: ~71.09%
- **Predicted Positive Ratio (at 0.5)**: ~84.85%
- **Confusion Matrix**: TN=2594, FP=4761, FN=1261, TP=16827

## 3. Threshold Sweep Results
Analysis of the existing Multi-Patch predictions shows:
- **Best Accuracy**: 0.7646 at threshold **0.52**.
- **Best Balanced Accuracy**: 0.7149 at threshold **0.72** (Recall: 0.7126, Specificity: 0.7172).
- **Best F1**: 0.8499 at threshold **0.47**.
- **Balanced Tradeoff**: At threshold **0.65**, the model reaches 0.60 Specificity while maintaining 0.80 Recall (Overall Accuracy: 0.7476).

The fact that the best balanced accuracy requires a threshold of 0.72 confirms that the model's scores are heavily skewed towards the positive class.

## 4. False Positive / False Negative Analysis
- **High Confidence False Positives**: Mean score of top 10 FP is **0.9947**. The model is extremely confident in some "bad" images. This suggests the frozen ImageNet VGG16 backbone may be ignoring aesthetic defects that define the negative class.
- **Low Confidence False Negatives**: Mean score of top 10 FN is **0.0323**.
- **Score Distribution**: 
  - Positive Mean: 0.7923 (Std: 0.172)
  - Negative Mean: 0.5717 (Std: 0.224)
- **Observation**: The distributions overlap significantly. The negative median (0.59) is actually higher than the positive decision threshold of 0.5.

## 5. Positive Bias and Class Imbalance
The training data (AVA) is naturally imbalanced towards positive samples (labels > 5.0). Using standard Binary Cross-Entropy (BCE) with a frozen backbone and default weights results in a model that "plays it safe" by predicting positive for almost everything. This is reflected in the 93% recall but 35% specificity.

## 6. Candidate Improvement Experiments

| ID | Experiment | Hypothesis | Risk |
| :--- | :--- | :--- | :--- |
| **A** | **Threshold Calibration** | Moving the threshold to ~0.52-0.55 will yield a slight accuracy gain. | Zero. Read-only. |
| **B** | **Class Weights / Balanced BCE** | Weighting negative samples higher (e.g., 2.5:1) will force the model to learn "bad" features and improve specificity. | Low. Standard training tweak. |
| **C** | **VGG16 Partial Unfreeze** | Unfreezing `block5` or `block4+5` of VGG16 with a low LR will allow learning aesthetics-specific textures. | Medium. Potential overfitting. |
| **D** | **Focal Loss** | Using Focal Loss will focus training on hard negative/positive samples near the boundary. | Low. Replaces BCE. |
| **E** | **Stronger Head** | Adding a 2-layer MLP after the Mean+Max aggregation to increase reasoning capacity. | Low. Incremental params. |

## 7. Recommended First Experiment
**Experiment B + C: Class-Weighted Fine-Tuning**
1.  Apply class weights based on the actual distribution in the training JSONL (approx 1.0 for positive, 2.45 for negative).
2.  Unfreeze `block5` of VGG16.
3.  Train with a reduced Learning Rate (e.g., 1e-5 or 2e-5) for 10-15 epochs.
4.  Monitor Balanced Accuracy on the validation set.

## 8. Success Criteria
- **Weak Improvement**: Accuracy > 0.7633 AND ROC-AUC > 0.79.
- **Strong Teacher**: Accuracy >= 0.80 AND Balanced Accuracy >= 0.73.
- **Paper-Near**: Accuracy >= 0.815.
- **Metric Protection**: ROC-AUC and Average Precision must not decrease.

## 9. Runtime and Risk
- **Runtime**: Training on full AVA (~230k train images) with 5 patches takes ~3-4 hours per 10 epochs on a modern GPU.
- **Risk**: Low. We are starting from a verified baseline. If unfreezing causes instability, we fall back to frozen weights with class weights only.

## 10. Distillation Readiness
- **Status**: **NOT READY.**
- **Reasoning**: A 0.76 accuracy teacher with 35% specificity provides a poor signal for distillation. Improving the teacher's balanced performance is critical to ensuring the mobile student is robust across both good and bad images.

## 11. Final Recommendation
1.  **GCN branch development is officially stopped** for the A-LAMP teacher role.
2.  The **Multi-Patch VGG16** architecture is the current state-of-the-art for this project.
3.  The next step is to run **Class-Weighted Fine-tuning** (Experiment B+C) to push accuracy towards 0.80 and correct the classification bias.
