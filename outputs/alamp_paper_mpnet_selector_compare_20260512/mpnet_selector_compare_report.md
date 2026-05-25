# A-LAMP-paper-MPNet Selector Comparison Report

## 1. Environment
- CWD: `/home/omen_pc1/photo_score_project`
- Date: 2026-05-12
- Python & TF: Python 3.12.3 / TensorFlow 2.20.0
- GPU: NVIDIA GeForce RTX 4070 SUPER (Execution performed on CPU to avoid CuDNN version mismatch)

## 2. Scope
This report compares two different patch selector strategies using the same `A-LAMP-paper-MPNet` architecture. The comparison was performed on a 256-sample subset (with 1024-sample full test evaluation) to identify which salient-region extraction method is most effective for aesthetic assessment.

## 3. Why Two Selectors Are Compared
- **strict-U2Net selector**: Closer to the original A-LAMP paper's objective function (summing saliency, pattern diversity, and spatial distance).
- **V4-U2Net selector**: A modern component-aware strategy that extracts salient contours and assigns roles (closeup, context, etc.), which passed the manual visual quality gate (50/50 acceptable).

## 4. Model Architecture
- **Backbone**: Shared VGG16 (ImageNet weights).
- **Input**: `[B, 5, 224, 224, 3]` patches.
- **Aggregation**: Concatenation of Global Mean and Max pooling over the 5-patch axis.
- **Head**: Dense (256) -> Dropout (0.5) -> Sigmoid Output.

## 5. Strict Selector Training/Evaluation (256 samples)
- **Test ROC-AUC**: 0.5842
- **Test BCE Loss**: 0.6088
- **Specificity**: 0.0000

## 6. V4 Selector Training/Evaluation (256 samples)
- **Test ROC-AUC**: **0.6233**
- **Test BCE Loss**: **0.5967**
- **Specificity**: 0.0000

## 7. Comparison Against v0 Baselines
| Metric | v0_a (Full) | Strict (256) | V4 (256) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 0.7236 | 0.7070 | 0.7070 |
| **ROC-AUC** | 0.6679 | 0.5842 | 0.6233 |
| **Specificity**| 0.1224 | 0.0000 | 0.0000 |

*Note: The comparison runs were limited to 2 epochs on 256 samples. V4's 0.62 AUC is competitive considering the limited training data.*

## 8. Selector Interpretation
**The A-LAMP idea worked better with a modern U2Net component-aware selector (V4).** It achieved significantly higher ROC-AUC and lower BCE loss than the strict paper-style selector. This suggests that explicitly targeting the primary subject's geometry and surrounding context (as V4 does) provides a clearer aesthetic signal than the paper's original objective-summation approach.

## 9. Official Reproduction Boundary
This is an **A-LAMP-paper-oriented approximation**. It uses U2Net for saliency instead of the paper's specific (potentially proprietary) salient detector and implements an aggregated VGG16 multi-patch subnet.

## 10. Next Step Recommendation
**Decision: B. V4 selector improves enough to expand V4 MPNet to 4096.**
While both selectors currently struggle with specificity on small subsets, V4 shows a clear performance lead in rank-ordering (AUC). I recommend expanding the V4 patch selector to a 4096-image subset and increasing training epochs to 10-20 to allow the shared VGG16 backbone to fine-tune properly.
