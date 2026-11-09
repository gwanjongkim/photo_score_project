# ICAA17K DAT TFLite / LiteRT Feasibility Result

## 1. Environment Versions
- **Python:** 3.12.3
- **TensorFlow:** 2.21.0
- **ONNX:** 1.20.1
- **ONNXRuntime:** 1.24.3
- **onnx2tf:** 2.4.0
- **ai-edge-litert:** 2.1.2
- **tf-keras:** 2.21.0

## 2. ONNX Model Path
- `outputs/icaa_export_safe_onnx_20260514_211156/icaa_dat_export_safe.onnx` (Opset 16)
- `outputs/icaa_export_safe_onnx_20260514_195654/icaa_dat_export_safe.onnx` (Opset 17)

## 3. Conversion Attempts

### A. onnx2tf (flatbuffer_direct)
- **Status:** **SUCCEEDED** (produced .tflite files)
- **Outputs:**
  - `icaa_dat_export_safe_float32.tflite` (336MB)
  - `icaa_dat_export_safe_float16.tflite` (169MB)
- **Runtime Test (Python TFLite / LiteRT Interpreter):**
  - **Status:** **FAILED**
  - **Error:** `Fill dimensions must be >= 0 got 2Node number 1469 (FILL) failed to invoke.`
  - **Investigation:** The TFLite file contains 20 CUSTOM operators named `ONNX_GRIDSAMPLE`. The interpreter passed `allocate_tensors` but crashed during `invoke` at an earlier `FILL` node.

### B. onnx2tf (tf_converter)
- **Status:** **FAILED**
- **Error:** `TypeError: Input 'y' of 'Sub' Op has type int64 that does not match type int32 of argument 'x'.`
- **Context:** Occurred during conversion of `wa/model/stages.0/attns.0/Slice`. This is a known issue where ONNX uses `int64` for indices/shapes and TensorFlow expects `int32`.

### C. Opset 11/12/13 Export
- **Opset 11:** FAILED (aten::einsum not supported).
- **Opset 12:** FAILED (repeat_interleave unknown input size).
- **Opset 13:** FAILED (aten::grid_sampler requires opset 16+).
- **Conclusion:** Lowering the opset to force decomposition of `GridSample` by PyTorch is blocked by other operators requiring higher opsets.

## 4. LiteRT Feasibility
- The generated models can be loaded by `ai_edge_litert`, but they exhibit the same runtime failures as the standard TFLite interpreter.
- LiteRT does not provide a native `GridSample` implementation, confirming the need for either decomposition or a custom delegate.

## 5. Summary and Recommendations

| Metric | Status |
| :--- | :--- |
| **FP32 Conversion** | Succeeded (via onnx2tf flatbuffer_direct) |
| **TFLite Runtime (Interpreter)** | FAILED (runtime error in FILL/GridSample) |
| **Flex Delegate Requirement** | LIKELY (for GridSample) |
| **Binary Size Risk** | HIGH (336MB FP32 model) |

**Final Recommendation:**
The ICAA17K DAT model is **NOT currently deployable** to standard TFLite/LiteRT without significant effort.

**Potential Next Steps:**
1. **Architectural Change:** Replace `DAttention` (which uses `GridSample`) with a more TFLite-friendly attention mechanism (e.g., standard `LocalAttention` or `GlobalAttention`).
2. **Decomposition:** Manually implement `GridSample` using standard PyTorch ops (`Gather`, `Bilinear Interpolation` logic) in the model code to avoid the `aten::grid_sampler` operator during ONNX export.
3. **Flex / SELECT_TF_OPS:** Attempt a pure TensorFlow-based conversion if the `int64` issues can be resolved, then use the Flex delegate on Android/iOS. However, the model size (336MB) remains a major blocker for mobile deployment.
4. **Teacher-Student Distillation:** Use the ICAA DAT model as a teacher to train a much smaller, TFLite-friendly student model (e.g., MobileNet-based). This is likely the most realistic path for a production mobile app.
