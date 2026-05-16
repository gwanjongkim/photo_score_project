# Stage J: FP32 TFLite Conversion and Verification

## Summary
The verified TensorFlow SavedModel from Stage I has been successfully converted to FP32 TFLite. The resulting model has been verified for interpreter compatibility, input sensitivity, and parity against the source SavedModel.

## Execution Details
- **Exact command run**: `python experiments/icaa_tf_native/scripts/stage_j_convert_tflite_fp32.py`
- **SavedModel path**: `outputs/icaa_tf_native_savedmodel_20260516_134439/saved_model`
- **TFLite model path**: `outputs/icaa_tf_native_tflite_fp32_20260516_134634/icaa_dat_tf_native_fp32.tflite`
- **TFLite file size**: 334.66 MB (350,915,988 bytes)

## Conversion Results
- **Builtin ops conversion**: Succeeded
- **SELECT_TF_OPS required**: No (Builtin ops were sufficient)
- **Conversion Strategy**: Used a fixed-shape concrete function ([1, 224, 224, 3]) to ensure variables were correctly frozen and to improve TFLite interpreter stability.

## Interpreter Details
- **Input details**: `[{'name': 'serving_default_image:0', 'index': 0, 'shape': array([  1, 224, 224,   3], dtype=int32), ...}]`
- **Output details**: `[{'name': 'StatefulPartitionedCall:0', 'index': 4274, 'shape': array([1, 2], dtype=int32), ...}]`

## Verification Results

### Input Sensitivity Probe
The model was tested on zero, one, and random inputs to ensure outputs are not constant.
- **Zero vs One max diff**: 0.0111754239
- **Zero vs Random max diff**: 0.0507589877
- **Result**: **Passed** (Model is input-sensitive)

### SavedModel vs TFLite Parity
| Test Case | Status | Max Abs Diff | Mean Abs Diff |
| :--- | :--- | :--- | :--- |
| Random Input (Batch 1) | `pass_preferred` | 1.192e-07 | 5.960e-08 |
| 16 Real ICAA17K Images | `pass_preferred` | 8.762e-06 | 1.492e-07 |

## Unresolved Issues
- None. The `NaN` issues encountered during earlier attempts were resolved by using fixed-shape conversion and robust manual layer implementations.

## Conclusion
Stage J is **successful**. The FP32 TFLite model is verified and matches the SavedModel within tight FP32 tolerances.

- **Safe to proceed to Stage K (FP16 conversion)**: Yes.
- **Safe to plan Android/Flutter smoke testing**: Yes, although the file size (334MB) is large for mobile, FP16 and quantization in later stages will address this.
