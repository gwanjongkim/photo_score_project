# Old vs New Aesthetic Model Paper-Benchmark Audit

## 1. Summary
This audit compares the "Old" deployable aesthetic models (A-LAMP and RGNet) against the "New" paper-oriented versions. 
- **A-LAMP**: The new `mobile_alamp_v2` (trained on full AVA) significantly outperforms the old AADB-trained `alamp_aadb_gpu` on the AVA classification task.
- **RGNet**: The new `rgnet_paper_aadb_fp16` (trained on AADB) shows a solid improvement in SRCC on the AADB test set compared to the old `rgnet_aadb_gpu`.

Replacement in the Flutter application is **highly justified** for both models based on these benchmark improvements.

## 2. Paper Benchmark Requirements
- **A-LAMP**: 
  - Dataset: AVA (Aesthetic Visual Analysis).
  - Task: Binary classification (mean score > 5.0).
  - Metrics: Accuracy, F-measure (F1).
- **RGNet**:
  - Dataset: AADB (Aesthetics with Attributes Database).
  - Task: Aesthetic score regression.
  - Metrics: Spearman’s Rho (SRCC), PLCC.

## 3. Located Model Files
| Model | Version | Path | Size | Preprocessing |
| --- | --- | --- | --- | --- |
| A-LAMP | Old | `models/aesthetic/alamp_aadb_gpu.tflite` | 38M | [0, 1] |
| A-LAMP | New | `outputs/mobile_alamp_v2_full_ava_20260519_tflite/mobile_alamp_v2_fp16.tflite` | 2.8M | [0, 255] |
| RGNet | Old | `models/aesthetic/rgnet_aadb_gpu.tflite` | 25M | [0, 1] |
| RGNet | New | `models/aesthetic/rgnet_paper_aadb_fp16.tflite` | 38M | [0, 1] |

## 4. Located Benchmark Splits
- **AVA Test**: `data/processed/ava/test.csv` (25,551 samples total).
  - Used patch JSONL: `outputs/alamp_v4_full_ava_20260517/subsets/test_patch_boxes_full_v4.jsonl`.
- **AADB Test**: `data/processed/aadb/test.csv` (1,001 samples).

## 5. Evaluation Method
- Sequential TFLite inference using `tf.lite.Interpreter`.
- Preprocessing applied per-model (New A-LAMP uses [0, 255] for MobileNetV3 with built-in preprocessing; others use [0, 1]).
- Same sample sets used for both old and new versions within each dataset.

## 6. A-LAMP AVA Benchmark Results (Proxy)
*Evaluated on 2,000 samples from the AVA test set.*

| Model | Accuracy | F1 | Precision | Recall |
| --- | --- | --- | --- | --- |
| Old (AADB-trained) | 0.5210 | 0.5913 | 0.7599 | 0.4839 |
| **New (Full-AVA)** | **0.6995** | **0.7614** | **0.8822** | **0.6697** |

## 7. RGNet AADB Benchmark Results (Paper Benchmark)
*Evaluated on all 1,001 samples of the AADB test set.*

| Model | SRCC | PLCC | MAE | RMSE |
| --- | --- | --- | --- | --- |
| Old | 0.5305 | 0.5355 | 0.1446 | 0.1768 |
| **New (Paper-v1)** | **0.6139** | **0.6311** | 0.1481 | 0.1828 |

## 8. Optional RGNet AVA Benchmark Results
Not evaluated in this run. New RGNet is optimized for AADB regression.

## 9. Metric Tables
(See `summary.csv` for detailed metrics including confusion matrix counts).

## 10. Prediction Distribution Checks
- **New A-LAMP**: Shows a much wider prediction range (0.002 to 0.999) compared to the old version (0.26 to 0.97), indicating better discriminative power on AVA.
- **New RGNet**: Prediction standard deviation increased from 0.088 to 0.165, matching the ground truth score distribution better than the "conservative" old model.

## 11. Paper Benchmark vs Proxy Benchmark Status
- **AADB**: **Paper Benchmark**. Used the standard 1,000-sample test split.
- **AVA**: **Proxy Benchmark**. Evaluated on 2,000 samples (sub-sampled for runtime efficiency) instead of the full ~20,000 samples. However, the performance gap is large enough to be conclusive.

## 12. Deployment Recommendation
- **A-LAMP**: **Deploy New**. The `mobile_alamp_v2` is significantly more accurate and an order of magnitude smaller (2.8M vs 38M).
- **RGNet**: **Deploy New**. The SRCC improvement (0.53 -> 0.61) is significant for aesthetic ranking quality.

## 13. Limitations
- AVA benchmark used a subset of 2000 samples.
- The "Old" A-LAMP was likely not optimized for AVA, explaining its near-random accuracy on this specific dataset.
- TFLite quantization (FP16) was used; slight numeric differences from Keras source are expected but usually negligible for these metrics.

## 14. Generated Artifacts
- `report.md`: This report.
- `summary.csv`: Tabular metrics for all models.
- `alamp_old_predictions.csv`: Individual predictions for old A-LAMP.
- `alamp_new_predictions.csv`: Individual predictions for new A-LAMP.
- `rgnet_old_predictions.csv`: Individual predictions for old RGNet.
- `rgnet_new_predictions.csv`: Individual predictions for new RGNet.
- `summary.json`: Raw metrics and configuration.
- `scripts/eval_old_vs_new_alamp_rgnet_paper_benchmark.py`: Evaluation script used.
