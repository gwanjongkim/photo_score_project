# Stage 1 Benchmark Alignment Report

## 1. Summary
This report presents the Stage 1 benchmark rigor checks for paper-level aesthetic performance analysis. We evaluated the current best A-LAMP model on the full AVA test set and quantified the "Resize Gap" for RGNet on the AADB test set.

## 2. A-LAMP Full AVA Benchmark
- **Model:** `mobile_alamp_v2_fp16.tflite` (Mobile A-LAMP v2 full-AVA, A-LAMP-inspired)
- **Dataset:** Full Available AVA Test Split (25,549 usable samples after skipping 2 corrupted files).
- **Label Rule:** `mean_score > 5.0`

### Metrics Comparison
| Metric | Proxy (2,000 samples) | Full Test (25,549 samples) | Paper Target (CVPR 2017) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 0.6995 | **0.7049** | 0.8250 |
| **F1 / F-measure** | 0.7614 | **0.7647** | 0.9200 |
| **ROC-AUC** | N/A | **0.8055** | N/A |
| **Avg Precision** | N/A | **0.9061** | N/A |

### Findings
- Evaluation on the full AVA test set confirmed the reliability of the earlier proxy benchmark; accuracy improved by only **0.5%**.
- A massive **12.0% Accuracy gap** remains relative to the official paper target.
- The high Precision (0.883) but low Recall (0.674) suggests the model is conservative, likely due to architectural simplifications (MobileNetV3Small backbone vs. heavier backbones in papers).

## 3. RGNet Preprocessing Gap Analysis
- **Model:** `rgnet_paper_aadb_fp16.tflite` (paper-oriented RGNet)
- **Dataset:** AADB Test Split (1,000 samples)

### Metrics Comparison (Side-by-Side Preprocessing)
| Metric | TF-Native Preprocess | PIL / Mobile-like | Gap Size | Paper Target (WACV 2020) |
| :--- | :--- | :--- | :--- | :--- |
| **SRCC** | **0.6819** | 0.6139 | **0.0680** | 0.7104 |
| **PLCC** | **0.6878** | 0.6311 | **0.0567** | N/A |
| **MAE** | **0.1197** | 0.1481 | **0.0284** | N/A |
| **RMSE** | **0.1486** | 0.1828 | **0.0342** | N/A |

### Findings
- The **0.068 SRCC Resize Gap** was successfully reproduced on identical samples.
- The model is highly sensitive to interpolation differences between TensorFlow and PIL.
- Recovering this gap via mobile-resize retraining will bring the model to ~0.68 SRCC, which is within **0.03 SRCC** of the paper target.

## 4. Analytical Answers

### Does A-LAMP improve on full AVA?
Yes, but the improvement is marginal (+0.5% accuracy). The proxy result was an accurate representation of current model performance.

### How far is A-LAMP from paper targets?
It is **12.0% below** the Accuracy target (82.5%) and **0.155 below** the F-measure target (0.92).

### How large is the RGNet preprocessing gap?
The gap is **0.068 SRCC**, which represents a ~10% performance degradation when moving from training (TF) to deployment (PIL).

### Is resize retraining sufficient for paper-level targets?
- **For RGNet:** Likely **yes (near-target)**. Recovering the gap brings us to 0.6819, very close to 0.7104. Further gains would require architectural refinements (e.g., DenseASPP).
- **For A-LAMP:** **No**. The 12% gap is too large to be explained by resizing alone. It is primarily an **Architecture Gap** (Lightweight MobileNetV3 vs. Heavy VGG-based subnets).

## 5. Next Recommended Experiment
**Stage 2: Align and Distill**
1. **RGNet:** Retrain using PIL-Bilinear resize in the training loop to recover the 0.068 SRCC gap immediately.
2. **A-LAMP:** Implement a "Faithful Teacher" (heavy VGG-based dual-subnet) to establish the performance ceiling, then distill its knowledge into the mobile student to close the 12% gap.
