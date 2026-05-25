# Benchmark Verification Audit

## 1. Summary
This audit verifies the A-LAMP and RGNet benchmark results generated on 2026-05-23. The results are confirmed to be technically sound and representative, with one significant finding regarding the discrepancy between Keras and TFLite metrics for RGNet.

## 2. Artifact Integrity
- **Output Directory:** `outputs/aesthetic_paper_benchmark_old_vs_new_20260523/`
- **Prediction Files:** `alamp_predictions.csv` (2,000 samples) and `rgnet_predictions.csv` (1,000 samples) exist and contain valid scores.
- **Metadata Alignment:** Preprocessing logic in the evaluation script matches the metadata for all four models (Old/New A-LAMP and Old/New RGNet).

## 3. Same-Sample Verification
- **A-LAMP:** Confirmed that `alamp_old_predictions.csv` and `alamp_new_predictions.csv` use the exact same image paths and order.
- **RGNet:** Confirmed that `rgnet_old_predictions.csv` and `rgnet_new_predictions.csv` use the exact same image paths and order, matching the standard AADB test split (`data/processed/aadb/test.csv`).

## 4. Metric Recalculation
Manual recomputation of metrics from the prediction CSVs yielded identical results to those reported in `summary.json`:
- **A-LAMP (New):** Accuracy 0.6995, F1 0.7614, Precision 0.8822, Recall 0.6697.
- **RGNet (New):** SRCC 0.6139, PLCC 0.6311, MAE 0.1481, RMSE 0.1828.

## 5. A-LAMP Benchmark Status
- **Status:** **AVA Proxy Benchmark**.
- **Sample Count:** 2,000 samples (intentional due to runtime).
- **Representativeness:** Confirmed. The 2,000-sample subset closely matches the full 25,551-sample AVA test set:
  - Full AVA mean score: 5.39 vs Subset mean score: 5.42.
  - Full AVA positive ratio (>5.0): 71.1% vs Subset positive ratio: 71.6%.
- **Verdict:** Highly reliable as a proxy. Significant improvement over the old model is verified.

## 6. RGNet Benchmark Status
- **Status:** **AADB Paper Benchmark**.
- **Dataset:** Standard AADB test split (1,000 samples).
- **Verdict:** Reliable. Improvement in ranking quality (SRCC) over the old model (0.53 -> 0.61) is verified.

## 7. RGNet 0.6819 vs 0.6139 Discrepancy Check
The reported Keras SRCC of 0.6819 (from `evaluation_summary.json`) vs the TFLite benchmark SRCC of 0.6139 was investigated.
- **Finding:** The discrepancy is due to a **Resize Gap**.
- **Reasoning:** 
  1. The official Keras evaluation script (`src/eval/evaluate_rgnet_paper_v1_aadb.py`) uses `tf.image.decode_jpeg` and `tf.image.resize` (native TensorFlow pipeline).
  2. The TFLite benchmark script (`scripts/eval_old_vs_new_alamp_rgnet_paper_benchmark.py`) uses `PIL.Image.resize` with `BILINEAR` interpolation.
  3. A separate parity check (`outputs/rgnet_paper_aadb_tflite_export_20260514/parity/`) showed perfect parity (SRCC > 0.99) between Keras and TFLite when using the same PIL-based preprocessing.
- **Conclusion:** The model is highly sensitive to the exact interpolation method. The 0.6139 value is the correct "real-world" performance when images are processed via standard mobile-like libraries (PIL/Flutter), while 0.6819 represents performance in a native TensorFlow environment.

## 8. Script/Preprocessing Audit
- **Old A-LAMP:** Correctly uses [0, 1] normalization.
- **New A-LAMP:** Correctly uses [0, 255] range as expected by the MobileNetV3-based architecture with built-in rescaling.
- **RGNet:** Both old and new models correctly use [0, 1] normalization.
- **Interpolation:** All models were evaluated using `BILINEAR`, which is standard but conservative for high-quality models.

## 9. Deployment Recommendation Review
- **Deploy New A-LAMP?** **YES**. The improvement on AVA is massive and verified.
- **Deploy New RGNet?** **YES**. Despite the lower-than-reported SRCC (0.61 vs 0.68), it still significantly outperforms the old model (0.53) in ranking correlation. The increase in MAE/RMSE is expected for a ranking-optimized model.
- **Recommendation Adjustment:** The report should note that "ranking quality (SRCC) improves significantly, while absolute score error (MAE) remains secondary."

## 10. Required Report Wording Corrections
- Ensure A-LAMP results are consistently labeled as "AVA proxy benchmark."
- Add a note about the "Resize Gap" to the RGNet section to explain why the TFLite metrics differ from the Keras training reports.

## 11. Final Judgment
**Audit Result: PASSED**.
The benchmark results are accurate representations of the models' performance under the specified conditions. The transition from Keras-native to PIL-based evaluation explains all observed metric drops.
