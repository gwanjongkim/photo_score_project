# RGNet Practical vs Paper-Oriented FP16 TFLite Comparison Report

## 1. Executive Summary
This report evaluates the **Paper-Oriented AADB RGNet (FP16)** against the current **Practical Mobile RGNet** on a subset of the AADB validation dataset using CPU-only inference.

**Conclusion**: The Paper-Oriented model shows significantly higher correlation with human aesthetic judgements (Pearson **0.518** vs **0.414**), but exhibits roughly **8.1x higher CPU latency** (132ms vs 16ms) due to its larger backbone (DenseNet121 vs EfficientNetV2). The Paper-Oriented model is a superior quality candidate but poses a risk for real-time mobile performance if not accelerated by GPU/NPU.

## 2. Methodology
- **Dataset**: 200 images from `data/processed/aadb/val.csv`.
- **Environment**: CPU-only (CUDA_VISIBLE_DEVICES=""), TensorFlow Lite Interpreter.
- **Preprocessing**: 
  - Both: 256x256, RGB, [0, 1] normalization.
  - Paper model: Built-in DenseNet preprocessing (internal scaling and ImageNet mean/std).

## 3. Performance Metrics

| Metric | Practical Mobile RGNet | Paper-Oriented AADB RGNet (FP16) |
| :--- | :--- | :--- |
| **Pearson Correlation** | 0.4141 | **0.5185** |
| **Spearman Correlation** | 0.4038 | **0.4825** |
| **MAE** | **0.1312** | 0.1507 |
| **MSE** | **0.0282** | 0.0321 |
| **Avg Latency (CPU)** | **16.37 ms** | 132.74 ms |

## 4. Observations
- **Predictive Quality**: The Paper-Oriented model provides a ~10-25% improvement in correlation metrics. This aligns with earlier full-benchmark results.
- **Latency**: The Practical model is highly optimized for mobile CPU (EfficientNetV2). The Paper model (DenseNet121) is significantly heavier.
- **Ranking Agreement**: Spearman correlation between the two models is **0.505**, indicating they agree on overall ranking to a moderate degree but provide distinct perspectives.

## 5. Final Recommendations

### A. Performance vs. Latency Tradeoff
The Paper-Oriented model is clearly better at capturing aesthetic quality. However, the 132ms CPU latency might be too high for certain mobile "live" features without acceleration.

### B. Justification for On-Device Testing
Yes, the desktop evidence confirms the quality jump. On-device testing is mandatory to see if the **GPU/NPU acceleration** (OpenCL/NNAPI) can bring the Paper model's latency down to acceptable levels (< 50ms).

### C. Replacement Recommendation
**Postponed** pending on-device performance results.
- **If GPU acceleration works**: Replace.
- **If restricted to CPU**: Keep the Practical model for real-time tasks, or use the Paper model for background processing only.

## 6. Remaining Blockers
1. **On-Device GPU Parity**: Verification that DenseNet121 and the custom graph convolution layers are well-supported by mobile TFLite GPU delegates.
2. **Preprocessing Verification**: Ensure the [0, 1] scaling is perfectly consistent with the built-in preprocessing layers.
