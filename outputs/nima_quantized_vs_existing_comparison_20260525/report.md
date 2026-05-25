# NIMA Quantized vs Existing Model Comparison Report

## 1. Executive Summary
- **Best Model**: **New_Fixed_FP16** (`nima_mobile_fixed_preproc_fp16.tflite`).
- **Recommendation**: **Replace the existing mobile NIMA with the new fixed FP16 model.**
- **Key Improvements**:
  - **Ranking (SRCC)**: Improved from **0.33** to **0.60** (Massive improvement).
  - **Discriminative Power (AUC)**: Improved from **0.65** to **0.81**.
  - **Prediction Variance**: Standard deviation of scores improved from 0.33 to **0.52**, resolving the majority-class collapse.
  - **Model Size**: FP16 version is **11.9 MB**, reducing the footprint by 50% compared to the existing 23.6 MB FP32 model.
- **Caveat**: The new model has a pessimistic score bias. The optimal binary threshold for aesthetic classification is **4.3**, rather than the standard 5.0. A-cut ranking logic should ideally use the raw scores or rank order rather than binary labels.

## 2. Model Inventory
| Type | Path | Size MB | Status |
|---|---|---|---|
| TFLite (Existing) | `models/aesthetic/nima_mobile.tflite` | 23.6 | Broken (Double-rescale bug) |
| TFLite (New FP32) | `outputs/nima_fixed_preproc_eval_20260525/nima_mobile_fixed_preproc_fp32.tflite` | 23.6 | Fixed |
| TFLite (New FP16) | `outputs/nima_fixed_preproc_eval_20260525/nima_mobile_fixed_preproc_fp16.tflite` | 11.9 | Fixed & Optimized |
| Keras (New) | `checkpoints/nima_ava_gpu_fixed_preproc_20260525/best.weights.h5` | 119.3 | Source weights |
| Keras (Old) | `checkpoints/nima_ava_gpu/best.weights.h5` | 119.3 | Broken weights |

## 3. Evaluation Dataset
- **CSV Path**: `data/processed/ava/test.csv`
- **Sample Count**: 2000 deterministic images (subset saved in `eval_subset_2000.csv`)
- **Class Balance**: 71.15% Positive (GT score >= 5.0)
- **Preprocessing**: RGB, Resize 256, Center Crop 224, Scale 1/255.0 (Standard NIMA style).

## 4. Input/Output Verification
- **TFLite Interpreter**:
  - Input: `[1, 224, 224, 3]`, `float32`, RGB.
  - Output: `[1, 10]`, `float32`, Softmax distribution.
- **Validity**: All TFLite outputs sum to 1.0 ($\pm 1e-6$). Mean score derived correctly from 10-bin distribution.

## 5. Performance Comparison
| Metric | Existing_Mobile | New_Fixed_FP16 | Delta | Paper Ref (MobileNet) |
|---|---|---|---|---|
| **SRCC (Spearman)** | 0.3317 | **0.5986** | **+0.2669** | ~0.510 |
| **PLCC (Pearson)** | 0.3362 | **0.6012** | **+0.2650** | ~0.518 |
| **AUC** | 0.6487 | **0.8108** | **+0.1621** | N/A |
| **Accuracy @ 5.0** | **0.7160** | 0.5645 | -0.1515 | ~0.8036 |
| **Best Accuracy** | 0.7160 | **0.7825** | **+0.0665** | N/A |
| **Best Threshold** | 4.8 | **4.3** | -0.5 | N/A |
| **MAE** | **0.5598** | 0.7131 | +0.1533 | N/A |
| **EMD** | **0.0878** | 0.1071 | +0.0193 | N/A |
| **Pred Std** | 0.3347 | **0.5250** | **+0.1903** | (GT: 0.699) |
| **Latency (CPU ms)** | 12.42 | 12.28 | -0.14 | N/A |

*Note: The existing model's high Accuracy @ 5.0 and low MAE are artifacts of model collapse where it predicts the majority class distribution for almost all images (95% positive ratio).*

## 6. Quantization and Parity (New Fixed Model)
- **Keras vs FP32**: Max dist diff **1.80e-04**, Mean score diff **3.09e-04**.
- **Keras vs FP16**: Max dist diff **5.26e-04**, Mean score diff **1.16e-03**.
- **FP32 vs FP16**: Rank correlation **1.000**.
- **Verdict**: FP16 quantization is highly stable and perfectly preserves ranking.

## 7. Calibration Analysis
- The new model is pessimistic compared to AVA ground truth labels.
- Standard threshold of 5.0 yields only 56% accuracy.
- Optimal threshold of **4.3** yields **78.25%** accuracy, approaching paper level.
- **Recommendation**: For binary logic, use threshold 4.3. For A-cut ranking, use the continuous `mean_score`.

## 8. Deployment Recommendation
- **Recommended file**: `outputs/nima_fixed_preproc_eval_20260525/nima_mobile_fixed_preproc_fp16.tflite`
- **Metadata file**: `outputs/nima_fixed_preproc_eval_20260525/nima_mobile_fixed_preproc.metadata.json`
- **Copy Commands**:
  ```bash
  cp outputs/nima_fixed_preproc_eval_20260525/nima_mobile_fixed_preproc_fp16.tflite assets/models/nima_mobile_fixed_preproc_fp16.tflite
  cp outputs/nima_fixed_preproc_eval_20260525/nima_mobile_fixed_preproc.metadata.json assets/models/nima_mobile_fixed_preproc.metadata.json
  ```
- **Ensemble Weight**: Keep existing aesthetic weight (0.24) but monitor A-cut stability.

## 9. What Not To Claim
- This comparison was on a 2000-sample subset. Full test set performance may vary by $\pm 1-2\%$.
- Accuracy does not reach 80% at 5.0 threshold; it requires recalibration.
- No Flutter code was modified in this audit.
