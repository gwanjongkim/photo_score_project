# ICAA PTQ candidate context notes

- Assumption: this is diagnostic-only experimental work. No Flutter project files, production metadata, scoring formulas, or existing FP16 TFLite artifacts should be modified.
- Source SavedModel is fixed to `outputs/icaa_tf_native_savedmodel_20260516_134439/saved_model`, matching the task prompt and the successful Stage K report.
- FP16 smoke reference is `outputs/icaa_tf_native_tflite_fp16_20260516_144023/icaa_dat_tf_native_fp16.tflite` when available.
- Preprocessing contract remains RGB resize to `224x224`, `pixel / 255.0`, then ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- TFLite external input/output should remain float32 unless the full-integer converter cannot produce a usable candidate with that contract.
- Stage L attempts dynamic-range PTQ first, then full-integer INT8 with `TFLITE_BUILTINS_INT8` and explicit float32 input/output. An int8 input/output fallback is only attempted if the float32 IO full-integer path fails.
- Stage L conversion succeeded for both dynamic-range PTQ and full-integer INT8 float32 IO. Smoke comparison succeeded only for dynamic-range PTQ; full-integer INT8 failed at invoke with TFLite `DIV` node 593 reporting a zero denominator.
