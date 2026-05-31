# ICAA Stage L PTQ Conversion Report

## Summary
- **Status**: `ok`.
- **Exact command run**: `python experiments/icaa_tf_native/scripts/stage_l_convert_tflite_ptq.py`.
- **SavedModel path**: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_savedmodel_20260516_134439/saved_model`.
- **Output directory**: `/home/omen_pc1/photo_score_project/outputs/icaa_ptq_candidates_20260601`.
- **Representative image count requested**: 64.
- **Representative image count used**: 64.
- **Preprocessing contract**: RGB resize to `224x224`, `pixel / 255.0`, then ImageNet mean/std.
- **Speed claim**: none; no benchmark was run.

## Conversion Outcomes
- **Dynamic range PTQ succeeded**: True.
- **Full integer INT8 float-IO PTQ succeeded**: True.
- **Full integer INT8 input/output stayed float32**: True.
- **INT8 int-IO fallback produced**: False.

## Validation Inventory
| Model | Status | Size Bytes | Input | Output | Shape OK | Flex Ops | Builtins Only | Internal Float32 Tensors |
| :--- | :--- | ---: | :--- | :--- | :--- | :--- | :--- | ---: |
| dynamic_range_ptq | `succeeded` | 92150672 | `float32 [1, 224, 224, 3]` | `float32 [1, 2]` | yes | no | yes | 3808 |
| full_integer_int8_float_io | `succeeded` | 91598496 | `float32 [1, 224, 224, 3]` | `float32 [1, 2]` | yes | no | yes | 387 |

## Ops Summary

### dynamic_range_ptq

- **TFLITE_BUILTINS only**: True.
- **Flex ops exist**: False.
- **Builtin operator codes**: `['ABS', 'ADD', 'BATCH_MATMUL', 'CAST', 'CONCATENATION', 'CONV_2D', 'DEPTHWISE_CONV_2D', 'DEQUANTIZE', 'DIV', 'EQUAL', 'EXP', 'FLOOR', 'FULLY_CONNECTED', 'GATHER', 'GELU', 'GREATER_EQUAL', 'LESS_EQUAL', 'LOGICAL_AND', 'LOGISTIC', 'MEAN', 'MINIMUM', 'MUL', 'NEG', 'REDUCE_MAX', 'RELU', 'RESHAPE', 'RSQRT', 'SELECT_V2', 'SOFTMAX', 'SPLIT', 'SQUARED_DIFFERENCE', 'STRIDED_SLICE', 'SUB', 'SUM', 'TANH', 'TILE', 'TRANSPOSE', 'ZEROS_LIKE']`.
- **Custom operator codes**: `[]`.
- **Tensor dtype counts**: `{'bool': 240, 'float32': 3810, 'int32': 1158, 'int8': 298}`.
- **Ops summary**: `RESHAPE=661, ADD=533, MUL=522, SUB=202, TRANSPOSE=162, FULLY_CONNECTED=149, RELU=132, MEAN=127, SELECT_V2=100, CAST=80, GATHER=80, MINIMUM=80, SPLIT=74, LESS_EQUAL=70, RSQRT=63, SQUARED_DIFFERENCE=63, LOGICAL_AND=60, EQUAL=50, BATCH_MATMUL=48, FLOOR=40, GREATER_EQUAL=40, ZEROS_LIKE=40, CONCATENATION=39, GELU=34, ... (+14 more)`.

### full_integer_int8_float_io

- **TFLITE_BUILTINS only**: True.
- **Flex ops exist**: False.
- **Builtin operator codes**: `['ABS', 'ADD', 'BATCH_MATMUL', 'CAST', 'CONCATENATION', 'CONV_2D', 'DEPTHWISE_CONV_2D', 'DEQUANTIZE', 'DIV', 'EQUAL', 'EXP', 'FLOOR', 'FULLY_CONNECTED', 'GATHER', 'GELU', 'GREATER_EQUAL', 'LESS_EQUAL', 'LOGICAL_AND', 'LOGISTIC', 'MEAN', 'MINIMUM', 'MUL', 'NEG', 'QUANTIZE', 'REDUCE_MAX', 'RELU', 'RESHAPE', 'RSQRT', 'SELECT_V2', 'SOFTMAX', 'SPLIT', 'SQUARED_DIFFERENCE', 'STRIDED_SLICE', 'SUB', 'SUM', 'TANH', 'TILE', 'TRANSPOSE', 'ZEROS_LIKE']`.
- **Custom operator codes**: `[]`.
- **Tensor dtype counts**: `{'bool': 240, 'float32': 389, 'int32': 611, 'int8': 3482}`.
- **Ops summary**: `RESHAPE=661, ADD=533, MUL=522, DELEGATE=267, SUB=202, DEQUANTIZE=172, TRANSPOSE=162, FULLY_CONNECTED=149, QUANTIZE=134, RELU=132, MEAN=127, SELECT_V2=100, CAST=80, GATHER=80, MINIMUM=80, SPLIT=74, LESS_EQUAL=70, RSQRT=63, SQUARED_DIFFERENCE=63, LOGICAL_AND=60, EQUAL=50, BATCH_MATMUL=48, FLOOR=40, GREATER_EQUAL=40, ... (+16 more)`.

## Conversion Errors

- None.

## Generated Smoke Script
- `/home/omen_pc1/photo_score_project/outputs/icaa_ptq_candidates_20260601/smoke_compare_ptq.py`.
- It compares candidates against `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_tflite_fp16_20260516_144023/icaa_dat_tf_native_fp16.tflite` when that FP16 reference exists.

## Next Step
- Run a device-side or representative desktop latency benchmark before making any speed or deployment claim.
