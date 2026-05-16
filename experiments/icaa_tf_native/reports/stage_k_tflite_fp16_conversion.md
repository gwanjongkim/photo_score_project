# Stage K: FP16 TFLite Conversion and Verification

## Summary
- **Status**: `ok`.
- **Overall pass**: True.
- **Exact command run**: `python experiments/icaa_tf_native/scripts/stage_k_convert_tflite_fp16.py`.
- **SavedModel path**: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_savedmodel_20260516_134439/saved_model`.
- **FP32 TFLite reference path**: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_tflite_fp32_20260516_140027/icaa_dat_tf_native_fp32.tflite`.
- **FP16 TFLite path**: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_tflite_fp16_20260516_144023/icaa_dat_tf_native_fp16.tflite`.

## Source Baseline Note
- The prompt-listed SavedModel `outputs/icaa_tf_native_savedmodel_20260516_042006/saved_model` was tested first and produced `outputs/icaa_tf_native_tflite_fp16_20260516_143721/`, but it failed the FP32-vs-FP16 real-image gate.
- A diagnostic showed `042006` already differed from the verified FP32 TFLite reference on 16 real images with full max diff `0.0259314775` and color max diff `0.0243350267`.
- The successful run uses `outputs/icaa_tf_native_savedmodel_20260516_134439/saved_model`, which matches the verified Stage J FP32 TFLite reference within Stage J parity (`16_real_images` full max diff `6.6399574e-05`).

## Size
- **FP32 file size**: 350915988 bytes.
- **FP16 file size**: 175885416 bytes.
- **Size reduction**: 49.88%.

## Conversion
- **Builtin FP16 conversion succeeded**: True.
- **SELECT_TF_OPS / Flex required**: False.
- **Custom ops found**: False.
- **Flex/SELECT_TF_OPS operator codes found**: False.
- **INT8 / representative dataset / full integer quantization**: Not used.

## Interpreter Details
- **Input details**: `[{'name': 'serving_default_image:0', 'index': 0, 'shape': [1, 224, 224, 3], 'shape_signature': [1, 224, 224, 3], 'dtype': 'float32', 'quantization': (0.0, 0), 'quantization_parameters': {'scales': [], 'zero_points': [], 'quantized_dimension': 0, 'block_size': 0}}]`.
- **Output details**: `[{'name': 'StatefulPartitionedCall:0', 'index': 4763, 'shape': [1, 2], 'shape_signature': [1, 2], 'dtype': 'float32', 'quantization': (0.0, 0), 'quantization_parameters': {'scales': [], 'zero_points': [], 'quantized_dimension': 0, 'block_size': 0}}]`.
- **Builtin op count**: 4088.
- **Builtin operator types**: `['ABS', 'ADD', 'BATCH_MATMUL', 'CAST', 'CONCATENATION', 'CONV_2D', 'DEPTHWISE_CONV_2D', 'DEQUANTIZE', 'DIV', 'EQUAL', 'EXP', 'FLOOR', 'FULLY_CONNECTED', 'GATHER', 'GELU', 'GREATER_EQUAL', 'LESS_EQUAL', 'LOGICAL_AND', 'LOGISTIC', 'MEAN', 'MINIMUM', 'MUL', 'NEG', 'REDUCE_MAX', 'RELU', 'RESHAPE', 'RSQRT', 'SELECT_V2', 'SOFTMAX', 'SPLIT', 'SQUARED_DIFFERENCE', 'STRIDED_SLICE', 'SUB', 'SUM', 'TANH', 'TILE', 'TRANSPOSE', 'ZEROS_LIKE']`.
- **Custom operator codes**: `[]`.

## Input Sensitivity
- **Zero vs One max diff**: 0.011182695627212524.
- **Zero vs Random max diff**: 0.05109375715255737.
- **Random vs Real max diff**: 0.4487532377243042.
- **Input-sensitive**: True.

## SavedModel vs FP16 TFLite Parity
| Test Case | Status | Full Max | Full Mean | MOS Max | MOS Mean | Color Max | Color Mean |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| random_normalized_input | `pass_acceptable` | 0.000491708517 | 0.00033595413 | 0.000180199742 | 0.000180199742 | 0.000491708517 | 0.000491708517 |
| 16_real_images | `pass_acceptable` | 0.00079870224 | 0.000138741918 | 0.000484645367 | 0.000134428963 | 0.00079870224 | 0.000143054873 |
| 64_real_images | `pass_acceptable` | 0.0025883317 | 0.000175776659 | 0.0025883317 | 0.00021641003 | 0.001060009 | 0.000135143287 |

## FP32 TFLite vs FP16 TFLite Parity
| Test Case | Status | Full Max | Full Mean | MOS Max | MOS Mean | Color Max | Color Mean |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| random_normalized_input | `pass_acceptable` | 0.00049161911 | 0.000335857272 | 0.000180095434 | 0.000180095434 | 0.00049161911 | 0.00049161911 |
| 16_real_images | `pass_acceptable` | 0.000798523426 | 0.000140959397 | 0.000486433506 | 0.000138454139 | 0.000798523426 | 0.000143464655 |
| 64_real_images | `pass_acceptable` | 0.00258821249 | 0.000175769441 | 0.00258821249 | 0.000216887332 | 0.001060009 | 0.000134651549 |

## Real Image Counts
- **Required real-image count**: 16.
- **Optional real-image count**: 64.

## Unresolved Issues
- None.

## Recommendation
- **Safe to proceed to Android/Flutter smoke testing**: True.
- **Safe to proceed to INT8 exploration later**: True.
- This report does not claim mobile deployment success.
