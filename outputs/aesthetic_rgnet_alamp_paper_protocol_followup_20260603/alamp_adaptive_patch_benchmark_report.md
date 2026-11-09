# A-LAMP Adaptive Patch Benchmark Report

## Overview
This benchmark evaluates Mobile A-LAMP v2 using a **Paper-Inspired Adaptive Patch Selection** protocol. This replaces the previous fixed center/corner crop proxy, aligning closer to the original A-LAMP research methodology.

## Benchmark Configuration
- **Dataset**: AVA (Full local 25,548 test split)
- **Patch Selector**: Version v2 (Spectral Residual Saliency + Pattern Diversity + Overlap Constraint)
- **Patch Count**: 5 patches per image
- **Patch Size**: 224x224
- **Global Size**: 384x384
- **Model**: `mobile_alamp_v2_fp16.tflite`

## Full Results Summary (25,548 Images)
| Metric | Fixed Proxy (Full) | Adaptive Patch (Full) | Status |
| :--- | :--- | :--- | :--- |
| **SRCC** | 0.5067 | **0.5798** | +0.0731 Improvement |
| **PLCC** | 0.5052 | **0.5773** | +0.0721 Improvement |
| **MAE (Scaled 1-10)** | 2.6026 | **1.7299** | +0.8727 Improvement |
| **Binary Accuracy** | 0.7536 | 0.6333* | See Note |

*\*Note: The lower binary accuracy (63.3%) compared to the fixed proxy (75.4%) indicates a distribution shift. The adaptive protocol reveals a more conservative prediction distribution (42.6% predicted positive vs. 71.1% ground truth positive), whereas the fixed proxy happened to align better with the 5.0 threshold despite lower ranking fidelity.*

## Comparison & Analysis
- **Ranking & Regression Fidelity**: The transition to adaptive patch selection supports a significant improvement in both Spearman and Pearson correlations. This indicates that saliency-aware selection better captures the visual cues necessary for aesthetic regression.
- **Error Reduction**: The scaled MAE improved significantly, reducing absolute error by nearly 0.9 points on a 10-point scale.
- **Protocol Fidelity**: While the mobile TFLite model lacks the external layout graph/attribute graph input used in the original A-LAMP paper, the implementation of adaptive patches brings the evaluation significantly closer to the intended research protocol.

## Conclusion
The Mobile A-LAMP v2 evaluation is finalized as **PAPER-INSPIRED ADAPTIVE**. The full-scale benchmark results indicate that adaptive patch selection is a critical component for maximizing the performance of the multi-patch architecture, even in a resource-constrained mobile TFLite environment.
