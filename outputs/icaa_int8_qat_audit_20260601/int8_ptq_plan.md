# ICAA INT8 PTQ Plan

## Level 1: Dynamic Range Quantization
1. **Source Model:** `outputs/icaa_tf_native_savedmodel_20260516_134439/saved_model`
2. **Script:** Create `stage_l_convert_tflite_ptq_dynamic.py` based on `stage_k`.
3. **Conversion:** Apply `tf.lite.Optimize.DEFAULT` without a representative dataset.
4. **Validation:** Run parity checks against `icaa_dat_tf_native_fp16.tflite` to ensure scores are identical or very close.

## Level 2: Full Integer Quantization (INT8 PTQ)
1. **Source Model:** Same TF SavedModel.
2. **Representative Dataset:** Utilize `data/raw/icaa17k` with preprocessing `(pixel/255.0 - mean) / std`. Sample ~100-500 images randomly.
3. **Script:** Create `stage_m_convert_tflite_ptq_int8.py`.
4. **Conversion Params:**
   - `optimizations = [tf.lite.Optimize.DEFAULT]`
   - `representative_dataset = representative_data_gen`
   - `target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]`
   - `inference_input_type = tf.float32` (Important for Flutter Contract)
   - `inference_output_type = tf.float32`
5. **Validation:** Calculate MAE, MaxAE, SRCC against FP16 TFLite outputs on the 50-image A-cut test set. Ensure no NaN or range errors.