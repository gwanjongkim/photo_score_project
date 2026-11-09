# Next Action

**Selected Path:** B. No INT8/QAT found; create dynamic-range/PTQ INT8 candidate first.

## Next Steps to Execute:
1. Copy `experiments/icaa_tf_native/scripts/stage_k_convert_tflite_fp16.py` to a new script `stage_l_convert_tflite_ptq.py`.
2. Modify the TFLiteConverter configuration in `stage_l_convert_tflite_ptq.py` to build two new models:
   - Dynamic Range PTQ (`tf.lite.Optimize.DEFAULT`)
   - Full Integer INT8 PTQ using the `representative_dataset` generator and `tf.lite.OpsSet.TFLITE_BUILTINS_INT8`.
3. Ensure both new models expose `float32` inputs and outputs to preserve the current Flutter contract.
4. Run the conversion script and check the internal parity reports.
5. If conversion succeeds, proceed to offline WSL ranking benchmarks using the newly generated models against `icaa_dat_tf_native_fp16.tflite`.