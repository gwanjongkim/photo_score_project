# A-LAMP MPNet 4096 Expansion Experiment Report

## 1. Training Summary
- **Dataset Size**: 4096 images (AVA subset)
- **Architecture**: A-LAMP-paper-MPNet (Shared VGG16 backbone, multi-patch aggregation)
- **Patch Selector**: V4 Component-Aware Selector
- **Training Mode**: `two_phase_freeze_block5`
  - **Phase 1**: VGG16 backbone frozen (15 epochs)
  - **Phase 2**: VGG16 Block 5 unfrozen for fine-tuning (3 epochs completed)
- **Class Weighting**: Applied (`0: 2.4889`, `1: 1.0`) to balance the positive-skewed distribution (2922 positive / 1174 negative).

## 2. Best Checkpoint
- **Monitor Metric**: `val_auc`
- **Best Epoch**: 15 (end of Phase 1)
- **Checkpoint Location**: `outputs/alamp_paper_mpnet_4096_expansion/v4_4096/final_model.keras`
- **Weights Source**: `best_val_auc.weights.h5`

## 3. Validation Metrics (Best Epoch)
| Metric | Value |
| :--- | :--- |
| **ROC-AUC** | 0.7161 |
| **Average Precision** | 0.8466 |
| **Accuracy (0.50)** | 0.7175 |
| **BCE Loss** | 0.6205 |

## 4. Test Metrics (Independent 4096 Subset)
| Metric | Value |
| :--- | :--- |
| **ROC-AUC** | 0.7000 |
| **Average Precision** | 0.8445 |
| **Accuracy (0.50)** | 0.7014 |
| **F1 Score (0.50)** | 0.7920 |
| **Recall (0.50)** | 0.7975 |
| **Specificity (0.50)** | 0.4630 |

## 5. Threshold Calibration Results
Thresholds calibrated to optimize **Balanced Accuracy** (average of Recall and Specificity):

- **Optimal Threshold**: **0.73**
- **Test Balanced Accuracy**: **0.6488**
- **Test Recall**: 0.6009
- **Test Specificity**: 0.6967
- **Test Accuracy**: 0.6284

## 6. Comparison with 1024 MPNet V4
| Metric | 1024 Baseline (V4) | 4096 Expansion (V4) | Difference |
| :--- | :---: | :---: | :---: |
| **Test ROC-AUC** | 0.6233 | **0.7000** | **+0.0767** |
| **Val ROC-AUC** | 0.6138 | **0.7161** | **+0.1023** |
| **Specificity (0.50)**| 0.0000 | **0.4630** | **+0.4630** |

## 7. Comparison with A-LAMP-paper-AVA v0
| Metric | v0 Baseline (Full) | 4096 Expansion | Difference |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.7236 | 0.7014 | -0.0222 |
| **ROC-AUC** | 0.6679 | **0.7000** | **+0.0321** |
| **Specificity**| 0.1224 | **0.4630** | **+0.3406** |

## 8. Interpretation
- **Data Scaling Success**: The jump from 1024 to 4096 samples provided a massive boost to ROC-AUC (+0.10 val) and solved the specificity collapse issue.
- **Improved Separation**: The model is now significantly better at rank-ordering images than the original v0 baseline (+0.03 AUC) and provides much better rejection of non-aesthetic images (+0.34 specificity at 0.50 threshold).
- **Fine-Tuning Impact**: Phase 1 (frozen backbone) achieved the best AUC. Phase 2 (Block 5 fine-tuning) saw lower losses but slightly lower validation AUC, suggesting that VGG16 ImageNet features are already quite robust for this task at this scale.

## 9. Risks
- **Overfitting in Phase 2**: The increasing validation loss in the final epochs indicates the start of overfitting.
- **Plateauing at 0.70**: While 0.70 AUC is a milestone, it is still below SOTA for aesthetic scoring, suggesting architectural limits or further data needs.

## 10. Recommended Next Step
**Proceed with 4096 Graph Fusion Training.**
Since the MPNet-only baseline is now stable and high-performing at 4096 samples, we should re-run the `GraphGCN` and `GraphLite` fusion experiments using this improved backbone. The previous 1024-fusion run was inconclusive due to the weak baseline; the 4096 backbone should provide a clearer signal for layout-aware branches.

---
*Report generated on 2026-05-14 based on artifacts in `outputs/alamp_paper_mpnet_4096_expansion/v4_4096/`.*
