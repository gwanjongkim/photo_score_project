# A-LAMP Dual-Branch GCN Failure Analysis

## 1. Summary
The A-LAMP Dual-Branch GCN teacher prototype failed to outperform the Multi-Patch-only baseline on the 4096-sample AVA subset. While it achieved a superficially high F1 score (0.828), this was driven by an extreme positive bias (96.3% predicted positive ratio) and near-perfect recall (97.3%), rather than actual discriminative power. The model's ROC-AUC (0.6635) and Balanced Accuracy (0.5186) were significantly lower than the Multi-Patch-only teacher (ROC-AUC 0.7914, Balanced Accuracy 0.6391).

## 2. Fair Comparison Results
Matching was performed on the exact same 4081-sample subset (71.28% positive ratio).

| Metric | Multi-Patch-only | Dual-Branch GCN |
|---|---|---|
| ROC-AUC | **0.791372** | 0.663500 |
| Average Precision | **0.900265** | 0.822219 |
| Balanced Accuracy | **0.639081** | 0.518590 |
| Specificity | **0.348976** | 0.063993 |
| Accuracy | **0.762558** | 0.712080 |
| F1 | **0.848000** | 0.828141 |
| Predicted Positive Ratio | 0.849302 | 0.962509 |

## 3. Positive Bias Diagnosis
The Dual-Branch GCN prototype is severely biased toward positive predictions.
- **Actual Positive Ratio**: 71.3%
- **Dual-Branch Predicted Positive**: 96.3%
- **Multi-Patch Predicted Positive**: 84.9%

The Dual-Branch model correctly identifies only **75 out of 1172 negatives** (6.4% specificity), effectively failing to learn the concept of "low aesthetic quality" in the presence of the GCN branch.

## 4. Multi-Patch-only vs Dual-Branch Differences
- **Input Features**: Both use VGG16 patches. Dual-Branch adds a 15-dimensional object graph node feature vector.
- **Fusion Method**: Dual-Branch concatenates 256-d patch features with 128-d GCN features.
- **Training Duration**: Only 3 epochs were used for the Dual-Branch training, which is insufficient for the model to overcome the 71% class imbalance and refine the newly initialized GCN/Fusion weights.
- **Complexity**: The Dual-Branch model adds significant complexity (Graph Convolutions, pooling, fusion layers) without a corresponding increase in data or training time.

## 5. Graph Feature / Adjacency Analysis
- **Weak Signal**: The node features (15 dims: boxes, centers, area, class, confidence, edges) may be too sparse or noisy if object detection was inconsistent.
- **Adjacency**: Purely distance-based adjacency ($1/(1+d)$) might not capture complex photographic composition rules without extensive training.
- **Masking**: While `object_mask` is used, the `MaskedMeanPoolingLayer` might produce zero-vectors for images with no detected objects, which the model may have mapped to a positive bias.

## 6. Fusion and Training Analysis
- **Scale Mismatch**: Multi-Patch features are from a frozen, pretrained VGG16. GCN features are from a raw initialization. Without Batch Normalization or Layer Normalization in the fusion head, the GCN branch likely overwhelmed the patch branch.
- **Lack of Regularization**: No Batch Normalization is present in the fusion head, and the 0.4 dropout rate was insufficient to prevent the model from defaulting to the majority class.
- **Optimization**: Binary Crossentropy without class weights on a 71% positive dataset naturally encourages positive bias.

## 7. Why F1 Was Misleading
The Dual-Branch F1 (0.828) is only 2% lower than the Multi-Patch F1 (0.848). However, this high F1 is a "false friend" caused by the high positive ratio of the dataset and the model's high recall (97%). If the model predicted "positive" for every sample, it would achieve an F1 of ~0.83 on this specific subset. Thus, the Dual-Branch model's F1 reflects the dataset's distribution more than the model's quality.

## 8. Root Cause Hypotheses
1. **Insufficient Training**: 3 epochs is not enough to train a GCN branch and fusion head from scratch on a small (4096) subset.
2. **Architecture Instability**: Lack of normalization in the fusion head allowed the uncalibrated GCN branch to destabilize the validated patch features.
3. **Class Imbalance Default**: The model "solved" the training objective by leaning into the 71% majority class, a tendency exacerbated by the noisy GCN signal.

## 9. Stop / Redesign Decision
**Decision: STOP.**
The current Dual-Branch GCN prototype should be archived. It significantly degrades performance relative to the Multi-Patch-only teacher and exhibits unacceptable positive bias. It fails the success criterion of beating the baseline in ROC-AUC or Balanced Accuracy on a fair subset comparison.

## 10. Recommended Next Step
1. **Strengthen Multi-Patch Teacher**: Focus on improving the backbone (e.g., VGG16 to ResNet/EfficientNet) or the patch selection logic rather than adding layout graphs.
2. **Redesign Layout Branch (Long-term)**: If GCN is revisited, it requires:
   - Much longer training (e.g., 20-50 epochs).
   - Normalization (LayerNorm/BatchNorm) in the fusion layers.
   - Class-weighted loss to counter the positive-heavy AVA distribution.
   - Evaluation on larger subsets to ensure the graph features generalize.
3. **Data Quality Audit**: Inspect the YOLO object detection quality in the 4096 subset to ensure the graph inputs are actually meaningful.
