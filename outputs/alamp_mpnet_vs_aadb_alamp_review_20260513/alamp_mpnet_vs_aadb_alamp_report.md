# A-LAMP MPNet vs AADB A-LAMP Review

## 1. Scope
This report evaluates the status of two distinct A-LAMP (Adaptive Layout-Aware Multi-Patch Convolutional Neural Network) tracks within the Photo Score project:
1. **Existing Practical AADB A-LAMP**: A regression-based model trained on the AADB dataset, currently deployed in the application.
2. **A-LAMP-paper-MPNet 1024**: A research-oriented binary classification model trained on the AVA dataset, following the original A-LAMP paper architecture more closely.

## 2. Evidence Sources
- **AADB A-LAMP Metrics**: `outputs/ava_retrain_rgnet_alamp_20260506/baseline_eval/alamp_aadb_on_aadb_val512_metrics.json`
- **MPNet V4 1024 Metrics**: `outputs/alamp_paper_mpnet_selector_compare_20260512/eval/v4_1024_gpu_tuned/v4_1024_gpu_tuned/threshold_calibration_summary.json`
- **Deployment Status**: Checked against `models/aesthetic/` and `configs/`.

## 3. Existing AADB A-LAMP Metrics
The existing AADB A-LAMP model (`alamp_aadb_gpu`) performs regression on aesthetic scores (0.0 to 1.0).

| Metric | Value |
| :--- | :--- |
| **Dataset** | AADB (Validation Subset, 512 samples) |
| **SRCC (Spearman)** | **0.5843** |
| **PLCC (Pearson)** | 0.5859 |
| **MAE** | 0.1292 |
| **RMSE** | 0.1594 |
| **Pairwise Accuracy** | 0.7080 |

## 4. A-LAMP-paper-MPNet 1024 Metrics
The MPNet track (`v4_1024_gpu_tuned`) uses a V4-U2Net selector and performs binary classification on the AVA dataset (thresholded at score 5.0).

| Metric | Value (at Threshold 0.61) |
| :--- | :--- |
| **Dataset** | AVA (Test, 1024 samples) |
| **Accuracy** | **0.6719** |
| **Balanced Accuracy** | 0.6246 |
| **Precision** | 0.7897 |
| **Recall** | 0.7356 |
| **Specificity** | 0.5136 |
| **F1 Score** | 0.7617 |
| **ROC-AUC** | 0.6629 |
| **Average Precision** | 0.8181 |

## 5. Task Difference: AADB Regression vs AVA Classification
- **AADB A-LAMP**: Focuses on **ranking** and **scoring** individual photos based on a continuous scale. It is highly useful for the "A-cut" product feature where photos need to be compared and selected.
- **MPNet (AVA)**: Focuses on **distinguishing** "aesthetic" vs "non-aesthetic" photos. This is the standard task in aesthetic research papers but is less flexible for fine-grained ranking in its raw classification form.

## 6. Deployment Readiness
- **AADB A-LAMP**: **Fully Ready**. Already exported to TFLite (`alamp_aadb_gpu.tflite`) and integrated into the app scoring pipeline via `configs/stage5_reference.json`.
- **MPNet**: **Research Prototype**. Currently exists as Keras weights and evaluation logs. Not yet optimized or exported for mobile deployment.

## 7. Paper-Oriented Value
The MPNet track is essential for the academic/paper-reproduction side of the project. It validates that the multi-patch architecture with a V4 selector can achieve performance comparable to baseline references (v0_a) on the standard AVA benchmark.

## 8. Practical App Value
The AADB A-LAMP model remains superior for the application because:
1. It provides a continuous score.
2. It was trained on AADB, which contains more attribute-level variety relevant to consumer photography than the older AVA dataset.
3. It is already integrated into the production-ready TFLite weights.

## 9. Decision
**Recommendation: Keep AADB practical A-LAMP as the app-facing model; keep MPNet as the paper-oriented research baseline.**

The two tracks do not conflict; they serve different goals. The MPNet results are "honest" and show that the 1024-tuning reached a plateau similar to previous baselines, justifying a pause in this direction to focus on other high-priority areas.

## 10. Recommended Next Steps
1. **Document** these results in the Capstone final report, distinguishing between "Practical Deployment" (AADB A-LAMP) and "Academic Reproduction" (MPNet).
2. **Optional**: If 4096 expansion is pursued for MPNet, it should be treated as a pure research task with no immediate expectation of replacing the app-facing AADB A-LAMP model.
3. **Stabilize**: Ensure the `alamp_aadb_gpu.tflite` model remains the default in `configs/aesthetic_weight_lab.yaml`.
