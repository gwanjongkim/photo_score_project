# V4 MPNet Threshold Calibration Report (2026-05-13)

## 1. Executive Summary
The V4 MPNet 1024 model initially appeared collapsed because it predicted 100% positive samples at the default 0.5 threshold (Specificity = 0.0). However, threshold calibration analysis reveals that the model **has not collapsed**, but is significantly miscalibrated.

By shifting the threshold from 0.5 to **0.72**, we recover meaningful class separation:
- **Balanced Accuracy:** 0.50 → **0.61** (Test)
- **Specificity:** 0.00 → **0.69** (Test)
- **Recall:** 1.00 → **0.54** (Test)
- **Precision:** 0.71 → **0.81** (Test)

Final Decision: **A. calibration fixes the all-positive issue.**

## 2. Threshold Sweep Results

| Threshold Selection | Threshold | Test Balanced Acc | Test Specificity | Test Recall | Test F1 |
|---------------------|-----------|-------------------|------------------|-------------|---------|
| 0.50 Baseline       | 0.50      | 0.500             | 0.000            | 1.000       | 0.832   |
| Best Val Balanced Acc| 0.72      | 0.611             | 0.687            | 0.536       | 0.645   |
| Spec >= 0.5 Target  | 0.67      | 0.613             | 0.537            | 0.689       | 0.735   |

### Answers to Key Questions:
- **Does any threshold recover specificity without destroying recall?**
  Yes. At `t=0.67`, we achieve a balanced trade-off: Specificity **0.54** and Recall **0.69**.
- **Is balanced accuracy meaningfully above 0.5?**
  Yes. It reaches **0.611** at the optimal Youden J threshold (`t=0.72`).
- **Is V4 MPNet better than v0_a after calibration?**
  V4 MPNet (ROC-AUC ~0.65) shows predictive signal, but its raw performance is still modest. It is usable but requires careful thresholding.
- **Should 4096 expansion remain blocked?**
  **No.** The "collapse" was a calibration artifact. Expanding to 4096 is now safe and recommended to provide more data for the model to refine its decision boundary.

## 3. Implementation Details
The threshold sweep was performed from 0.01 to 0.99. The model's raw prediction distribution is heavily skewed towards the 0.5-0.8 range, which is why the 0.5 baseline failed.

Optimal threshold for general use: **0.67** (Balanced Recall/Specificity).
Optimal threshold for classification accuracy: **0.72** (Maximizes Youden's J).

## 4. Final Decision
**A. calibration fixes the all-positive issue.** 
The model is functional and provides better-than-random separation (Balanced Acc > 0.6). Expansion to 4096 is unblocked.
