# ICAA17K DAT TFLite Conversion Feasibility Report

## 1. ONNX Model Information
- **Latest ONNX Model Path:** `outputs/icaa_export_safe_onnx_20260514_195654/icaa_dat_export_safe.onnx`
- **Input Shape:** `image: [1, 3, 224, 224]`
- **Output Shape:** `scores: ['dynamic', 2]` (Batch dimension is dynamic, but values should be `[1, 2]`)

## 2. Operator Histogram Analysis
Total node count: ~7000+ (High complexity)

| Operator | Count | Risk Level | Notes |
| :--- | :--- | :--- | :--- |
| **GridSample** | 20 | **High** | Not natively supported by TFLite. Requires `Select TF Ops` (Flex) or custom implementation. |
| **Einsum** | 48 | **High** | Needs decomposition to MatMul/Transpose during conversion. |
| **Gather** | 605 | Medium | Generally supported but can be slow/problematic if indices are dynamic. |
| **LayerNormalization** | 63 | Medium | Native in newer TFLite, otherwise needs decomposition. |
| **Constant** | 3073 | Low | High count, likely from weights and grid indices. |
| **Conv** | 86 | Low | Standard support. |
| **MatMul / Gemm** | 79 | Low | Standard support. |

## 3. Environment Check
- **onnx:** 1.21.0 (Installed)
- **onnxruntime:** 1.26.0 (Installed)
- **tensorflow:** 2.20.0 (Installed)
- **onnx2tf:** **MISSING** (Crucial for high-fidelity conversion)
- **ai-edge-litert:** **MISSING**

## 4. Conversion Feasibility & Risks
- **Major Blocker:** `GridSample` (20 occurrences). Standard TFLite does not have a `GridSample` kernel. This will likely force the use of `SELECT_TF_OPS` (Flex delegate), which increases the binary size significantly and may impact performance on mobile.
- **Secondary Risk:** `Einsum` (48 occurrences). If not correctly decomposed, it will also fall back to Flex.
- **Model Size:** The high number of `Constant` and `Unsqueeze` nodes suggests a very large graph which might lead to long conversion times or memory issues during conversion.

## 5. Recommended Conversion Route
**ONNX -> onnx2tf -> TFLite**

This route is preferred over `tf-onnx` because `onnx2tf` provides better control over operator decomposition and can automatically handle many unsupported ONNX ops by converting them to equivalent TF operations that TFLite can then consume (with or without Flex).

## 6. Expected Next Steps
1. **Install onnx2tf:** `pip install onnx2tf`
2. **Attempt FP32 Conversion:**
   ```bash
   onnx2tf -i outputs/icaa_export_safe_onnx_20260514_195654/icaa_dat_export_safe.onnx \
           -o outputs/icaa_tflite_conversion_20260514/ \
           --non_verbose
   ```
3. **Handle GridSample:** If standard conversion fails, retry with `--enable_flex_ops`.
4. **Verification:** Compare ONNX Runtime output with TFLite Interpreter output on real images.

## 7. Conclusion
Conversion is **challenging but potentially feasible** using the Flex delegate. The presence of 20 `GridSample` ops makes a "pure" TFLite (without Flex) extremely unlikely without significant architectural changes to the DAT model.
