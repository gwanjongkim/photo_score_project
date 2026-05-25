# RGNet Mobile Export Readiness Audit Report

## 1. Executive Summary
This audit compares the current **Practical Mobile RGNet** with the **Paper-Oriented AADB RGNet** (`agg_mean_full`) to determine its potential for future mobile deployment.

**Conclusion**: The Paper-Oriented AADB RGNet is significantly more accurate (SRCC 0.68 vs 0.50) and uses a similar architecture (DenseNet121 vs EfficientNetV2). It is a **strong FP16 TFLite export candidate**. Replacement of the current mobile model will require formal parity verification and preprocessing consistency checks.

## 2. Model Comparison Matrix

| Feature | Practical Mobile RGNet | Paper-Oriented AADB RGNet |
| :--- | :--- | :--- |
| **Model Label** | `rgnet_aadb_gpu` | `RGNet-paper-v1 agg_mean_full` |
| **Backbone** | `EfficientNetV2B0` | `DenseNet121` |
| **Input Shape** | `[1, 256, 256, 3]` | `[1, 256, 256, 3]` |
| **Preprocessing** | `x / 255.0` | `(x*255 - mean) / std` (Built-in) |
| **AADB SRCC** | ~0.50 (val512) | **0.6819** (full test) |
| **Output Type** | Scalar [0, 1] | Scalar [0, 1] |
| **TFLite Status** | Exported (26MB) | Not yet exported |

## 3. Preprocessing and Input Contract
- **Practical Model**: Accepts RGB normalized to `[0, 1]`. EfficientNetV2 usually handles scaling internally or expects `[0, 1]`.
- **Paper Model**: Also accepts RGB normalized to `[0, 1]`. It includes a `DenseNetV1UnitPreprocess` layer that internally scales to `[0, 255]` and applies ImageNet subtraction.
- **Decision**: Both models are drop-in compatible from the perspective of the application's image loading pipeline.

## 4. TFLite Conversion Analysis
The Paper-Oriented model (`src/models/rgnet_paper_v1.py`) uses the following custom layers:
- `DenseNetV1UnitPreprocess`: Simple arithmetic. (Safe)
- `ASPPContextModule`: Parallel `Conv2D` and `BatchNormalization`. (Safe)
- `V1RegionSimilarityAdjacency`: `l2_normalize`, `matmul`, `softmax`. (Safe)
- `V1ResidualGraphConvolution`: `matmul`, `Dense`, `LayerNormalization`. (Safe)
- `RegionScoreAggregation`: `reduce_mean`. (Safe)

**Blockers**: No significant architectural blockers identified. Standard Keras 3 `TFLiteConverter` should support all operations.

## 5. Export Plan
### 5.1 Export Commands (Recommended)
**FP32 (Baseline):**
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

**FP16 (Recommended for Mobile):**
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
```

### 5.2 Parity Testing
1. Load both `.keras` and `.tflite` models.
2. Generate 10-20 sample images (e.g. from `test_samples/`).
3. Run inference and compute `max_abs_diff`.
4. Target diff: `< 1e-4` for FP32/FP16.

## 6. Final Recommendation
**A. Paper AADB RGNet is a strong candidate for FP16 TFLite export.**
The model represents a massive jump in predictive power (SRCC +0.18) without significant architectural cost. It is recommended to proceed with export and parity testing to confirm its suitability for replacing `rgnet_aadb_gpu.tflite`. Full replacement status is pending these verification steps.
