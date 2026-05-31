# Full INT8 Failure Analysis

## Confirmed Facts
- **Model path**: `/home/omen_pc1/photo_score_project/outputs/icaa_ptq_candidates_20260601/icaa_dat_tf_native_full_integer_int8_float_io.tflite`.
- **Input details**: `[{'name': 'serving_default_image:0', 'index': 0, 'shape': [1, 224, 224, 3], 'shape_signature': [1, 224, 224, 3], 'dtype': 'float32', 'quantization': (0.0, 0), 'quantization_parameters': {'scales': [], 'zero_points': [], 'quantized_dimension': 0, 'block_size': 0}}]`.
- **Output details**: `[{'name': 'StatefulPartitionedCall:0', 'index': 4721, 'shape': [1, 2], 'shape_signature': [1, 2], 'dtype': 'float32', 'quantization': (0.0, 0), 'quantization_parameters': {'scales': [], 'zero_points': [], 'quantized_dimension': 0, 'block_size': 0}}]`.
- **Total ops**: 6471.
- **DIV op count**: 10.
- **Custom/Flex ops found**: [].
- **Failed invoke cases**: 4 of 4.

## Minimal Invoke Failure
- **zeros**: `failed` - RuntimeError: tensorflow/lite/kernels/div.cc:242 data[i] != 0 was not true.Node number 593 (DIV) failed to invoke.
- **ones**: `failed` - RuntimeError: tensorflow/lite/kernels/div.cc:242 data[i] != 0 was not true.Node number 593 (DIV) failed to invoke.
- **random_normal**: `failed` - RuntimeError: tensorflow/lite/kernels/div.cc:242 data[i] != 0 was not true.Node number 593 (DIV) failed to invoke.
- **first_icaa_image**: `failed` - RuntimeError: tensorflow/lite/kernels/div.cc:242 data[i] != 0 was not true.Node number 593 (DIV) failed to invoke.

## Failed Node
- **Failed node from exception**: 593.
- **Failed op inventory row**: `{'op_index': 593, 'op_name': 'DIV', 'is_div': True, 'is_quantized_by_tensor_dtype': True, 'input_indices': '[1336, 1338]', 'output_indices': '[1339]', 'input_dtypes': '["int8", "int8"]', 'output_dtypes': '["int8"]', 'input_names': '["tf_icaa17k_dat_1/stage2_1/attn_1_1/Exp", "tf_icaa17k_dat_1/stage2_1/attn_1_1/add_34"]', 'output_names': '["tf_icaa17k_dat_1/stage2_1/attn_1_1/truediv_4"]', 'input_quant': '[{"quantization": [0.003921568859368563, -128], "scale_count": 1, "zero_point_count": 1, "scale_min": 0.003921568859368563, "scale_max": 0.003921568859368563, "zero_scale_count": 0, "nan_or_inf_scale_count": 0, "zero_point_min": -128, "zero_point_max": -128, "quantized_dimension": 0}, {"quantization": [0.7384021282196045, -128], "scale_count": 1, "zero_point_count": 1, "scale_min": 0.7384021282196045, "scale_max": 0.7384021282196045, "zero_scale_count": 0, "nan_or_inf_scale_count": 0, "zero_point_min": -128, "zero_point_max": -128, "quantized_dimension": 0}]', 'output_quant': '[{"quantization": [0.003921549301594496, -128], "scale_count": 1, "zero_point_count": 1, "scale_min": 0.003921549301594496, "scale_max": 0.003921549301594496, "zero_scale_count": 0, "nan_or_inf_scale_count": 0, "zero_point_min": -128, "zero_point_max": -128, "quantized_dimension": 0}]', 'scale_issues': '[]'}`.
- **Failed op tensor runtime details**: `{'available': True, 'op_index': 593, 'op_name': 'DIV', 'inputs': [{'tensor_index': 1336, 'name': 'tf_icaa17k_dat_1/stage2_1/attn_1_1/Exp', 'dtype': 'int8', 'shape': [16, 196, 196], 'scale_count': 1, 'scale_min': 0.003921568859368563, 'scale_max': 0.003921568859368563, 'zero_scale_count': 0, 'nan_or_inf_scale_count': 0, 'zero_point_min': -128, 'zero_point_max': -128, 'runtime_stats': {'shape': [16, 196, 196], 'dtype': 'int8', 'size': 614656, 'nan_count': 0, 'finite_count': 614656, 'zero_count': 0, 'min': -128.0, 'max': 127.0, 'mean': -8.340725544044147}}, {'tensor_index': 1338, 'name': 'tf_icaa17k_dat_1/stage2_1/attn_1_1/add_34', 'dtype': 'int8', 'shape': [16, 196, 1], 'scale_count': 1, 'scale_min': 0.7384021282196045, 'scale_max': 0.7384021282196045, 'zero_scale_count': 0, 'nan_or_inf_scale_count': 0, 'zero_point_min': -128, 'zero_point_max': -128, 'runtime_stats': {'shape': [16, 196, 1], 'dtype': 'int8', 'size': 3136, 'nan_count': 0, 'finite_count': 3136, 'zero_count': 20, 'min': -127.0, 'max': 127.0, 'mean': -3.65625}}], 'outputs': [{'tensor_index': 1339, 'name': 'tf_icaa17k_dat_1/stage2_1/attn_1_1/truediv_4', 'dtype': 'int8', 'shape': [16, 196, 196], 'scale_count': 1, 'scale_min': 0.003921549301594496, 'scale_max': 0.003921549301594496, 'zero_scale_count': 0, 'nan_or_inf_scale_count': 0, 'zero_point_min': -128, 'zero_point_max': -128, 'runtime_stats': {'shape': [16, 196, 196], 'dtype': 'int8', 'size': 614656, 'nan_count': 0, 'finite_count': 614656, 'zero_count': 614656, 'min': 0.0, 'max': 0.0, 'mean': 0.0}}]}`.

## DIV Quantization
- **DIV ops quantized by tensor dtype**: 10.
- **DIV op indices**: `[593, 954, 1315, 1676, 2037, 2398, 2759, 3120, 3481, 3849]`.
- **DIV ops with scale issues**: `[]`.

## Quantization Parameter Scan
- **Tensor dtype counts**: `{'bool': 240, 'float32': 389, 'int32': 611, 'int8': 3482}`.
- **Quantized tensor count**: 3482.
- **Quantized tensors missing scales**: 0.
- **Quantized tensors with zero scales**: 0.
- **Quantized tensors with NaN/Inf scales**: 0.

## Conservative Interpretation
- The TFLite flatbuffer is valid enough to allocate and expose float32 input/output, but it is not runtime-safe in this local interpreter test.
- The preprocessing normalization is external to this TFLite model; the failing DIV is inside the model graph, not the Python preprocessing step.
- The failed DIV node uses int8 input and output tensors; the denominator tensor has valid quantization parameters but contains literal zero values at runtime.
- This is more consistent with a quantized graph numeric failure around attention normalization than with invalid external input dtype.
- Representative calibration remains a plausible contributor if the calibrated dynamic ranges allow a denominator tensor to collapse to zero, but this script does not prove calibration is the sole cause.
