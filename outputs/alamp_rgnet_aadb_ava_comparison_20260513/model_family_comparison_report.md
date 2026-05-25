# A-LAMP and RGNet AADB/AVA Comparison Report

## 1. Scope
This report provides a consolidated comparison of the A-LAMP and RGNet model families across two distinct tasks:
1.  **AVA Binary Classification**: Distinguishing aesthetic vs. non-aesthetic photos (Paper-oriented).
2.  **AADB Regression**: Predicting continuous aesthetic scores and ranking (Practical app-oriented).

## 2. Evidence Sources
- **A-LAMP AVA Classification**: `outputs/alamp_paper_ava_classification_20260511/`
- **AADB A-LAMP Metrics**: `outputs/ava_retrain_rgnet_alamp_20260506/`
- **RGNet AVA Classification**: `outputs/rgnet_paper_ava_classification_20260510/`
- **AADB RGNet Metrics**: `outputs/rgnet_paper_consolidated_20260511/`

## 3. A-LAMP-paper-AVA v0 Metrics (Classification)
The A-LAMP paper reproduction track (v0) focuses on AVA binary classification (mean_score > 5.0).

| Variant | Accuracy | Recall | Specificity | F1 | ROC-AUC | AP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **v0_a** | 0.7236 | **0.9658** | 0.1224 | 0.8328 | **0.6679** | **0.8199** |
| **v0_b_fixed**| 0.7188 | 0.9521 | **0.1395** | 0.8284 | 0.6645 | 0.8180 |

*Note: v0 models show very high recall but very low specificity, suggesting an all-positive collapse tendency.*

## 4. Existing AADB A-LAMP Metrics (Regression)
The practical A-LAMP model (`alamp_aadb_gpu`) performs regression on AADB.

| Metric | Value |
| :--- | :--- |
| **SRCC (Spearman)** | **0.5843** |
| **PLCC (Pearson)** | 0.5859 |
| **MAE** | 0.1292 |
| **Pairwise Accuracy**| 0.7080 |

## 5. A-LAMP Comparison Decision
- **App-facing**: Keep **AADB Practical A-LAMP**. It is already integrated and provides ranking scores.
- **Paper-oriented**: Use **v0_a** as the AVA classification baseline, but acknowledge its specificity limitations.
- **Replacement**: Do not replace the app model with v0_a/v0_b. They are classification models and not directly suitable for ranking without further tuning.

## 6. RGNet-paper-AVA Metrics (Classification)
The RGNet track achieved significantly stronger results on AVA classification (mean_score >= 5.0).

| Variant | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | AP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **lse_r4 (Full)**| **0.7697** | **0.8057** | **0.8922** | **0.4646** | **0.8468** | **0.7979** | **0.9019** |

*Note: RGNet lse_r4 is the strongest classification result observed in the project.*

## 7. Existing AADB RGNet Metrics (Regression)
The AADB RGNet model (`agg_mean_full`) is the strongest regression model in the project.

| Metric | Value |
| :--- | :--- |
| **SRCC (Spearman)** | **0.6819** |
| **PLCC (Pearson)** | 0.6878 |
| **MAE** | 0.1198 |

## 8. RGNet Comparison Decision
- **AVA Classification**: **RGNet lse_r4** is much stronger than any A-LAMP variant.
- **AADB Regression**: **AADB RGNet** (SRCC 0.68) outperforms AADB A-LAMP (SRCC 0.58).
- **Capstone Report**: Emphasize RGNet as the project's high-performance backbone for both tasks.

## 9. Cross-Family Comparison Table

| Model | Task | Dataset | Accuracy / SRCC | ROC-AUC / MAE | Best Use |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AADB RGNet** | Regression | AADB | **0.6819 (SRCC)** | **0.1198 (MAE)**| **A-cut Ranking (App)** |
| **AADB A-LAMP** | Regression | AADB | 0.5843 (SRCC) | 0.1292 (MAE) | App Score (Deployed) |
| **RGNet AVA lse_r4**| Classification| AVA | **0.7697 (Acc)** | **0.7979 (AUC)**| **Paper Baseline** |
| **A-LAMP AVA v0_a** | Classification| AVA | 0.7236 (Acc) | 0.6679 (AUC) | Paper Baseline |
| **MPNet V4 1024** | Classification| AVA | 0.6719 (Acc) | 0.6629 (AUC) | Research Prototype |

## 10. Deployment Readiness
- **Deployed (TFLite)**: `rgnet_aadb_gpu.tflite`, `alamp_aadb_gpu.tflite`.
- **Not Deployed**: All AVA classification models (Keras weights only).

## 11. Paper-Oriented Value
RGNet provides the strongest empirical evidence for the project's ability to reproduce and improve upon aesthetic classification baselines.

## 12. Practical App Value
AADB RGNet and AADB A-LAMP are the workhorses of the photo scoring pipeline. RGNet is objectively stronger in ranking correlation (SRCC).

## 13. Final Recommendations
1.  **App-facing**: Continue using **AADB RGNet** as the primary high-quality scoring model.
2.  **AVA Paper**: Use **RGNet lse_r4** as the headline classification result.
3.  **A-LAMP Research**: Document A-LAMP results (v0, MPNet) as alternative architectures, noting that they currently trail RGNet in both tasks.
4.  **MPNet 4096**: Optional. Given RGNet's lead, MPNet 4096 should only be pursued if architectural diversity is desired for the report.

## 14. Missing Evidence / Caveats
- AVA classification and AADB regression are **not directly comparable**. A high accuracy on AVA does not guarantee high SRCC on AADB.
- The A-LAMP v0 models suffer from low specificity; they tend to over-predict the positive class.
