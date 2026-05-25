# ICAA export-safe v2 TFLite constant-output debug report

## Scope

Debug target:

- `outputs/icaa_tflite_conversion_v2_no_flex_20260514_220549/icaa_dat_export_safe_v2_float32.tflite`

Reference ONNX:

- `outputs/icaa_export_safe_v2_onnx_20260514_215455/icaa_dat_export_safe_v2.onnx`

No training, Flutter edits, official repo edits, or model implementation edits were performed.

## TFLite Graph Inspection

Command family used:

```bash
python - <<'PY'
# parsed the TFLite flatbuffer with outputs/icaa_tflite_conversion_v2_no_flex_20260514_220549/schema_generated.py
# counted operators and computed forward/backward tensor reachability
PY
```

Saved inspection JSON:

- `outputs/icaa_export_safe_v2_tflite_debug_20260514_222350/tflite_graph_inspection.json`

Result:

- tensor count: `5986`
- operator count: `3879`
- input tensor: `[0]`
- output tensor: `[5848]`
- input connected to output by forward reachability: `true`
- output depends on input by backward reachability: `true`
- reachable ops from input: `3879 / 3879`
- backward ops from output: `3879 / 3879`

Conclusion: the flatbuffer input is graph-connected. The failure is not a trivially disconnected input tensor.

## ONNX vs TFLite Operator Histogram

Saved comparison JSON:

- `outputs/icaa_export_safe_v2_tflite_debug_20260514_222350/onnx_vs_tflite_op_histogram.json`

ONNX total ops: `12634`

TFLite total ops: `3879`

ONNX histogram:

```json
{
  "Abs": 1,
  "Add": 528,
  "And": 200,
  "Cast": 382,
  "Clip": 80,
  "Concat": 541,
  "Constant": 4255,
  "ConstantOfShape": 160,
  "Conv": 86,
  "Div": 189,
  "Einsum": 48,
  "Equal": 20,
  "Erf": 34,
  "Expand": 40,
  "Flatten": 1,
  "Floor": 40,
  "Gather": 845,
  "Gemm": 3,
  "GlobalAveragePool": 1,
  "GreaterOrEqual": 80,
  "LayerNormalization": 63,
  "LessOrEqual": 110,
  "MatMul": 76,
  "Mul": 974,
  "Range": 40,
  "Reciprocal": 20,
  "Relu": 1,
  "Reshape": 588,
  "Shape": 925,
  "Sigmoid": 12,
  "Slice": 178,
  "Softmax": 24,
  "Squeeze": 1,
  "Sub": 170,
  "Tanh": 20,
  "Tile": 20,
  "Transpose": 302,
  "Unsqueeze": 1446,
  "Where": 130
}
```

TFLite histogram:

```json
{
  "ABS": 1,
  "ADD": 626,
  "BATCH_MATMUL": 124,
  "CAST": 80,
  "CONCATENATION": 39,
  "CONV_2D": 74,
  "DEPTHWISE_CONV_2D": 12,
  "DIV": 63,
  "FLOOR": 40,
  "FULLY_CONNECTED": 3,
  "GATHER": 120,
  "GELU": 34,
  "GREATER_EQUAL": 80,
  "LESS": 80,
  "LESS_EQUAL": 110,
  "LOGICAL_AND": 200,
  "LOGISTIC": 12,
  "MEAN": 127,
  "MINIMUM": 76,
  "MUL": 523,
  "RELU": 77,
  "RELU6": 4,
  "RESHAPE": 590,
  "SELECT": 80,
  "SELECT_V2": 110,
  "SLICE": 16,
  "SOFTMAX": 24,
  "SPLIT": 54,
  "SQRT": 63,
  "SUB": 153,
  "TANH": 20,
  "TILE": 20,
  "TRANSPOSE": 244
}
```

## Input Sensitivity Checks

Saved summary JSON:

- `outputs/icaa_export_safe_v2_tflite_debug_20260514_222350/tflite_attempt_sensitivity_summary.json`

Probe inputs:

- zero input
- one input
- seeded random normal input
- one real ICAA17K image with ImageNet normalization

Reference ONNX sensitivity:

- zero: `[[0.25746325, 0.46316642]]`
- one: `[[0.26993352, 0.4735747]]`
- random: `[[0.20815423, 0.41366604]]`
- real image: `[[0.671119, 0.597687]]`

Original TFLite output:

- zero: `[[0.5152620077, 0.5379891396]]`
- one: `[[0.5152620077, 0.5379891396]]`
- random: `[[0.5152620077, 0.5379891992]]`
- real image: `[[0.5152620077, 0.5379891396]]`
- max non-zero-input diff vs zero: `5.960464477539063e-08`
- rejected as constant.

Reference builtin kernels, without XNNPACK, produced the same constant result.

## Intermediate Tensor Diagnosis

Saved JSON:

- `outputs/icaa_export_safe_v2_tflite_debug_20260514_222350/tflite_intermediate_zero_vs_one_diff.json`
- `outputs/icaa_export_safe_v2_tflite_debug_20260514_222350/tflite_intermediate_zero_vs_real_diff.json`

Findings:

- TFLite input changes do perturb many intermediate tensors.
- For zero vs real image, `cls_norm` output tensor `5803` has large differences:
  - max diff: `8.123847961425781`
  - mean diff: `0.49871764`
- Immediately after TFLite `GlobalAveragePool` / `MEAN` tensor `5816`, the difference collapses:
  - max diff: `6.4074993e-07`
  - mean diff: `4.1414026e-08`
- PyTorch at the same semantic stage does not collapse:
  - `cls_norm` zero vs real max/mean: `1.1398508549 / 0.0530280061`
  - `gap` zero vs real max/mean: `0.3921433985 / 0.0455789641`

Conclusion: the converted TFLite graph is semantically wrong around the final layout/pooling path. It is connected, but the signal collapses at `GlobalAveragePool`/`MEAN`.

## Alternative Conversion Attempts

### Attempt 1: avoid flatbuffer direct

Command:

```bash
onnx2tf -i outputs/icaa_export_safe_v2_onnx_20260514_215455/icaa_dat_export_safe_v2.onnx -o outputs/icaa_export_safe_v2_tflite_debug_20260514_222350/attempt_tf_converter -tb tf_converter -n
```

Result: failed. Error at `wa/model/stages.2/attns.12/Add_1`.

Key error:

```text
Dimensions must be equal, but are 4 and 49 ... input shapes: [16,4,49,49], [1,49,49,16]
```

No TFLite/SavedModel emitted.

### Attempt 2: disable ONNX simplification

Command:

```bash
onnx2tf -i outputs/icaa_export_safe_v2_onnx_20260514_215455/icaa_dat_export_safe_v2.onnx -o outputs/icaa_export_safe_v2_tflite_debug_20260514_222350/attempt_flatbuffer_no_onnxsim -tb flatbuffer_direct -nuo -n
```

Result: TFLite emitted, but rejected as constant.

- input shape: `[1, 224, 224, 3]`
- max non-zero-input diff vs zero: `5.960464477539063e-08`

### Attempt 3: preserve NCHW input

Command:

```bash
onnx2tf -i outputs/icaa_export_safe_v2_onnx_20260514_215455/icaa_dat_export_safe_v2.onnx -o outputs/icaa_export_safe_v2_tflite_debug_20260514_222350/attempt_flatbuffer_keep_nchw -tb flatbuffer_direct -k image -n
```

Result: TFLite emitted, but rejected as constant.

- input shape: `[1, 3, 224, 224]`
- max non-zero-input diff vs zero: `0.0`

### Attempt 4: preserve NHWC input

Command:

```bash
onnx2tf -i outputs/icaa_export_safe_v2_onnx_20260514_215455/icaa_dat_export_safe_v2.onnx -o outputs/icaa_export_safe_v2_tflite_debug_20260514_222350/attempt_flatbuffer_keep_nhwc -tb flatbuffer_direct -kt image -n
```

Result: TFLite emitted, but rejected as constant.

- input shape: `[1, 3, 224, 224]`
- max non-zero-input diff vs zero: `0.0`

### Attempt 5: flatbuffer-direct SavedModel output

Command:

```bash
onnx2tf -i outputs/icaa_export_safe_v2_onnx_20260514_215455/icaa_dat_export_safe_v2.onnx -o outputs/icaa_export_safe_v2_tflite_debug_20260514_222350/attempt_flatbuffer_saved_model -tb flatbuffer_direct -fdosm -n
```

Result: failed to emit SavedModel.

Key error:

```text
ModelIR->SavedModel exporter does not support some op types in this model. unsupported_op_types=['GELU']
```

A partial float32 TFLite was emitted before failure, but it was rejected as constant.

### Attempt 6: flatbuffer-direct SavedModel output with pseudo GELU

Command:

```bash
onnx2tf -i outputs/icaa_export_safe_v2_onnx_20260514_215455/icaa_dat_export_safe_v2.onnx -o outputs/icaa_export_safe_v2_tflite_debug_20260514_222350/attempt_flatbuffer_saved_model_pseudo_gelu -tb flatbuffer_direct -fdosm -rtpo gelu -n
```

Result: failed to emit SavedModel.

Key error:

```text
ValueError: `input.shape.rank` must be at least 5. Received: input.shape=(4, 14, 14, 128) with rank 4
```

A partial float32 TFLite was emitted before failure, but it was rejected as constant.

### Attempt 7: tf_converter with NCHW preservation

Command:

```bash
onnx2tf -i outputs/icaa_export_safe_v2_onnx_20260514_215455/icaa_dat_export_safe_v2.onnx -o outputs/icaa_export_safe_v2_tflite_debug_20260514_222350/attempt_tf_converter_keep_nchw -tb tf_converter -k image -n
```

Result: failed with the same Add layout mismatch as Attempt 1.

## Decision

All generated FP32 TFLite variants were rejected because their outputs are effectively constant with respect to input.

The current evidence points to a layout semantic error in the ONNX-to-TFLite conversion, especially at or immediately before the final `GlobalAveragePool`/`MEAN` path, rather than a disconnected input tensor or a runtime delegate issue.

Do not use the generated FP32 TFLite for deployment or Flutter smoke testing.
