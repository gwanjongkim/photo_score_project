# A-LAMP 1024 Threshold Calibration: MPNet vs Graph Fusion

## 1. Scope
This report compares the validation performance of three A-LAMP paper-oriented model variants after threshold calibration on the 1024-sample AVA subset:
1.  **MPNet-only**: Baseline multi-patch model (VGG16 shared backbone, V4 selector).
2.  **GraphLite**: MPNet fused with a flattened layout-aware attribute vector (MLP branch).
3.  **GraphGCN**: MPNet fused with a Graph Convolutional Network (GCN) branch using precomputed object-node features and adjacency.

## 2. Comparison Strategy
- **Direct Validation Comparison**: Fusion models (GraphLite/GraphGCN) were evaluated primarily on the validation split. MPNet-only results are also compared on validation for parity.
- **Label Rule**: All models use the strict positive rule: `positive if mean_score > 5.0`.
- **Metrics**: Comparison focuses on the **Threshold 0.50 Baseline** and the **Best Validation Balanced Accuracy** operating points.

> **Note:** For GraphLite and GraphGCN, `test_predictions` currently point to `val_predictions.csv`. Therefore, fusion test metrics are not treated as independent test results. This report uses validation metrics for direct comparison.

## 3. Validation Metrics (Threshold 0.50 Baseline)

| Metric | MPNet-only (Tuned) | GraphLite | GraphGCN |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.6768 | **0.6963** | **0.6963** |
| **Balanced Accuracy** | **0.5936** | 0.5930 | 0.5843 |
| **F1 Score** | 0.7765 | 0.7969 | **0.7992** |
| **Recall** | 0.7964 | 0.8449 | **0.8573** |
| **Specificity** | **0.3907** | 0.3411 | 0.3113 |
| **Precision** | **0.7576** | 0.7540 | 0.7485 |

*Note: Graph fusion variants show a slight increase in recall/F1 at the 0.5 threshold but suffer from lower specificity compared to the baseline.*

## 4. Best Validation Balanced Accuracy Comparison

| Metric | MPNet-only (Tuned) | GraphLite | GraphGCN |
| :--- | :---: | :---: | :---: |
| **Threshold** | 0.81 | 0.94 | 0.97 |
| **Balanced Accuracy** | **0.6407** | 0.6399 | 0.6402 |
| **F1 Score** | 0.6572 | **0.7078** | 0.6711 |
| **Recall** | 0.5429 | **0.6274** | 0.5651 |
| **Specificity** | **0.7384** | 0.6523 | 0.7152 |

## 5. Analysis
- **No Clear Lead**: Graph fusion (GraphLite/GraphGCN) did **not clearly outperform** the MPNet-only baseline on the 1024 validation subset. Peak balanced accuracy remains essentially plateaued at ~0.64 for all variants.
- **GraphGCN Specificity Signal**: GraphGCN exhibited a "small high-specificity signal" at extreme thresholds. For instance, at a threshold of 0.97, it maintained 0.64 balanced accuracy with 0.715 specificity, suggesting potential for more robust rejection of non-aesthetic images if tuned further.
- **Plateau Risk**: The similarity in performance suggests that the 1024-sample subset may be too small to differentiate the complex graph-based signals from the dominant VGG16 patch features.

## 6. Conclusion
The "Layout-Aware" graph branches (GCN and Lite) are correctly implemented and fused, but their empirical value is not yet evident at the 1024-sample scale. The project is currently seeing diminishing returns on architectural complexity for this data subset.

## 7. Recommendations
1.  **Primary Target**: Proceed with **MPNet-only 4096** expansion. This will provide a more stable and diverse feature base before re-evaluating fusion benefits.
2.  **Secondary Option**: Pursue **GraphGCN 4096** as an optional secondary experiment to see if the high-specificity signal matures on a larger dataset.
3.  **GraphLite**: Deprioritize GraphLite in favor of the more theoretically grounded GraphGCN for future layout-aware experiments.
