# ICAA PTQ Smoke Comparison

- **Reference path**: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_tflite_fp16_20260516_144023/icaa_dat_tf_native_fp16.tflite`.
- **Image count**: 16.
- **Preprocessing**: RGB resize to `224x224`, `pixel / 255.0`, then ImageNet mean/std.

| Candidate | Status | MAE | MaxAE | NaN Count | Range Errors | Input DType | Output DType | Error |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| icaa_dat_tf_native_dynamic_range_ptq | `ok` | 0.0030072899535298347 | 0.008145809173583984 | 0 | 0 | `float32` | `float32` |  |
| icaa_dat_tf_native_full_integer_int8_float_io | `failed` |  |  |  |  | `` | `` | tensorflow/lite/kernels/div.cc:242 data[i] != 0 was not true.Node number 593 (DIV) failed to invoke. |

No speed or latency claim is made by this smoke comparison.
