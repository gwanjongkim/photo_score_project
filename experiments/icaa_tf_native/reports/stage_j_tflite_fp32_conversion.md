# Stage J: FP32 TFLite Conversion and Verification

## Summary
The verified TensorFlow SavedModel from Stage I has been converted to FP32 TFLite and rerun from the current working tree. The latest rerun passed builtin-op conversion, interpreter load, `allocate_tensors()`, `invoke()`, input sensitivity, and SavedModel-vs-TFLite parity.

## Execution Details
- **Exact command run**: `python experiments/icaa_tf_native/scripts/stage_j_convert_tflite_fp32.py`
- **SavedModel path**: `outputs/icaa_tf_native_savedmodel_20260516_134439/saved_model`
- **Latest output directory**: `outputs/icaa_tf_native_tflite_fp32_20260516_140027/`
- **TFLite model path**: `outputs/icaa_tf_native_tflite_fp32_20260516_140027/icaa_dat_tf_native_fp32.tflite`
- **TFLite file size**: 334.66 MiB (350,915,988 bytes)
- **SHA-256**: `7a29a2de7463a27187c2d205e324c4ed0197f0009d49d1c14471e21087dcb7e7`

## Conversion Results
- **Conversion source**: Stage I SavedModel wrapped as a fixed-shape concrete function.
- **Fixed input signature**: `[1, 224, 224, 3]`, `float32`.
- **FP32 only**: No FP16, INT8, representative dataset, optimizer, or quantization setting is enabled.
- **Builtin ops conversion**: Succeeded.
- **SELECT_TF_OPS required**: No.
- **Custom/Flex ops found in FlatBuffer**: None.

## Interpreter Details
- **Input**: `serving_default_image:0`, shape `[1, 224, 224, 3]`, shape signature `[1, 224, 224, 3]`, dtype `float32`, quantization `(0.0, 0)`.
- **Output**: `StatefulPartitionedCall:0`, shape `[1, 2]`, shape signature `[1, 2]`, dtype `float32`, quantization `(0.0, 0)`.
- **Fixed-shape limitation**: The exported model has fixed batch size `1`; multi-image inference must call the interpreter one image at a time or use a separately exported batch-capable model.

## TFLite Op Summary
- **Total builtin interpreter ops**: 3,598.
- **Builtin operator types**: `ABS`, `ADD`, `BATCH_MATMUL`, `CAST`, `CONCATENATION`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `DIV`, `EQUAL`, `EXP`, `FLOOR`, `FULLY_CONNECTED`, `GATHER`, `GELU`, `GREATER_EQUAL`, `LESS_EQUAL`, `LOGICAL_AND`, `LOGISTIC`, `MEAN`, `MINIMUM`, `MUL`, `NEG`, `REDUCE_MAX`, `RELU`, `RESHAPE`, `RSQRT`, `SELECT_V2`, `SOFTMAX`, `SPLIT`, `SQUARED_DIFFERENCE`, `STRIDED_SLICE`, `SUB`, `SUM`, `TANH`, `TILE`, `TRANSPOSE`, `ZEROS_LIKE`.
- **FlatBuffer custom operator codes**: None.
- **SELECT_TF_OPS/Flex operator codes**: None.

## Verification Results

### Input Sensitivity Probe
The model was tested on zero, one, and random inputs to verify outputs are not constant.

- **Zero vs One max diff**: 0.0111754239.
- **Zero vs Random max diff**: 0.0507589877.
- **Result**: Passed.

### SavedModel vs TFLite Parity
Real-image validation used 16 ICAA17K test images.

| Test Case | Status | Max Abs Diff | Mean Abs Diff |
| :--- | :--- | ---: | ---: |
| Random normalized input | `pass_preferred` | 1.0430813e-07 | 9.6857548e-08 |
| 16 real ICAA17K images | `pass_preferred` | 6.6399574e-05 | 2.5602058e-06 |

## Previous vs Latest Rerun
- **Previous output**: `outputs/icaa_tf_native_tflite_fp32_20260516_134634/`.
- **Latest output**: `outputs/icaa_tf_native_tflite_fp32_20260516_140027/`.
- **TFLite byte comparison**: Same size and same SHA-256.
- **Prediction CSV**: Same 16 real-image rows; max prediction-field delta vs previous run was `6.6459179e-05` after applying the CPU/no-oneDNN environment settings before TensorFlow import.
- **Input sensitivity JSON**: Identical.
- **Report/log differences**: Timestamped output paths and SavedModel-side parity values changed; all latest values remain inside preferred FP32 thresholds.

## Conclusion
Stage J is clean after review. The FP32 TFLite model is reproducible from the current working tree, uses builtin TFLite ops only, and matches the Stage I SavedModel within the configured FP32 parity thresholds.

- **Safe to proceed to Stage K FP16 conversion**: Yes.
- **Android/Flutter smoke testing**: Can be planned later after Stage K and any later quantization/export gates; no Android or Flutter testing was started in Stage J.
