# TOPIQ-lite Replacement Export Feasibility Report

## 1. Summary
**PASS**. TOPIQ-lite successfully converted to TFLite (FP32 and FP16) using **built-in operators only**. No Flex/Select TF Ops are required, ensuring maximum compatibility with Android GPU delegates and minimal APK bloat. Numerical parity with the Keras baseline is excellent, with maximum errors well below 0.1 on the 0-100 MOS scale. TOPIQ-lite is now officially cleared for the **Replacement Track** to succeed the existing KonIQ/FLIVE mobile models.

## 2. Export Results
| Dataset | FP32 size | FP16 size | Builtin-only? | Flex? | Erfc? | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SPAQ | 24.9 MB | 12.6 MB | **YES** | NO | NO | **OK** |
| KonIQ | 24.9 MB | 12.6 MB | **YES** | NO | NO | **OK** |
| FLIVE | 24.9 MB | 12.6 MB | **YES** | NO | NO | **OK** |

## 3. Tensor Signatures
| Dataset | Input shape | Input dtype | Output shape | Output dtype |
| :--- | :--- | :--- | :--- | :--- |
| All | [1, 384, 384, 3] | float32 | [1, 1] | float32 |

## 4. Op Analysis
The unique operators in the exported TFLite models include:
- `CONV_2D`, `DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`
- `ADD`, `MUL`, `CONCATENATION`
- `GLOBAL_AVERAGE_POOL_2D`
- `RESIZE_BILINEAR` (used in cross-scale attention)
- `SIGMOID` (output and attention gates)
- `HARD_SWISH` / `MEAN` / `PAD` (EfficientNetV2 internals)

**Android Support**: All detected ops are standard TFLite built-ins. Android 11+ and modern `tflite-flutter` versions should support these without any additional configuration or `select-tf-ops` dependencies.

## 5. Parity Results (N=8 per model)
| Dataset | FP32 max diff | FP16 max diff | FP16 mean diff | PASS? |
| :--- | :--- | :--- | :--- | :--- |
| **SPAQ** | 0.038 | 0.066 | 0.032 | **YES** |
| **KonIQ** | 0.022 | 0.060 | 0.025 | **YES** |
| **FLIVE** | 0.024 | 0.036 | 0.014 | **YES** |

## 6. Replacement Implication
TOPIQ-lite satisfies all technical requirements for replacement:
- **Accuracy**: Proved superior on SPAQ and competitive on KonIQ during 1024-probe.
- **Complexity**: Single-input, builtin-only.
- **Latency**: Estimated ~15-20ms on CPU, likely significantly lower on modern Android NPU/GPU.
- **Scale**: Reliable 0-100 mapping.

## 7. Full Training Priority
**C. Full SPAQ then KonIQ.**
SPAQ training should be prioritized first as it showed the highest performance gain over current baselines and is most representative of mobile photography quality. KonIQ training will follow to solidify general image quality robustness.

## 8. Replacement Roadmap
1. **Full-Train SPAQ**: Train TOPIQ-lite on the complete 11,125 SPAQ image set.
2. **Full-Train KonIQ**: Train TOPIQ-lite on the complete 10,073 KonIQ image set.
3. **Cross-Evaluation**: Benchmarking both TOPIQ-lite models against current `koniq_mobile` and `flive_image_mobile` on the full test sets.
4. **Final Model Selection**: Choose either the SPAQ model or a KonIQ-finetuned version as the **unified technical quality replacement**.
5. **Flutter Integration**: Replace the two old model calls with a single TOPIQ-lite inference call.

## 9. Final GO / NO-GO
**GO**. Export is feasible and numerically stable. Proceed to full-dataset training for SPAQ.

## 10. Files Created
- summary.csv
- parity_spaq.csv
- parity_koniq.csv
- parity_flive.csv
- topiq_lite_spaq_probe1024_fp16.tflite
- topiq_lite_koniq_probe1024_fp16.tflite
- topiq_lite_flive_probe1024_fp16.tflite
