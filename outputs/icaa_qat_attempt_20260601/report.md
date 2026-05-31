# ICAA QAT Attempt Report

## Final Decision
- **Decision**: E. QAT cannot be applied safely; stop ICAA acceleration attempts for now.
- **QAT feasible**: False.
- **QAT training ran**: False.
- **QAT TFLite export succeeded**: False.
- **QAT TFLite invokes successfully**: Not applicable; no QAT candidate was produced.
- **Production readiness**: Not proven.

## Confirmed Facts
- All files from this attempt were created under `outputs/icaa_qat_attempt_20260601/`.
- No Flutter files were modified.
- Existing FP16/PTQ artifacts were not overwritten.
- The SavedModel exists and can be loaded for inference with `tf.saved_model.load`.
- The SavedModel cannot be loaded as a trainable Keras model with `tf.keras.models.load_model` under this Keras 3 environment.
- The source TensorFlow model can be rebuilt from code plus the PyTorch checkpoint in `.venv_gpu`.
- The rebuilt source model is a subclassed custom-layer `TFICAA17KDAT`, not a Functional/Sequential model.
- `tf.keras.models.clone_model` failed on `TFICAA17KDAT`.
- `tensorflow_model_optimization` is not installed in the checked interpreters.

## Why Training Did Not Run
- Standard TensorFlow Model Optimization Toolkit QAT requires TFMOT APIs such as `quantize_model` or `quantize_apply`; those APIs are unavailable.
- Installing TFMOT would modify the environment outside the allowed output directory.
- Even with TFMOT installed, the current source model would still need custom serialization/quantization support because it is subclassed, custom-layer-heavy, and not cloneable.
- Manual fake-quant insertion would be a new custom quantization implementation and is not safe for this final conservative attempt.

## Prior Quantization Context
- Dynamic PTQ was faster in WSL but failed model-level parity: MAE `0.003328498899936676`, MaxAE `0.009905099868774414`, P90 AE `0.006955149769783022`, and 81 of 100 values over `0.001`.
- Full INT8 float-IO converted but failed at runtime at quantized DIV node 593.
- Since direct QAT is blocked, no improved QAT parity or latency metrics are available.

## Generated Outputs
- `qat_feasibility_audit.py`
- `qat_feasibility_audit.json`
- `qat_feasibility_report.md`
- `report.md`
- `next_action.md`
- `plan.md`
- `checklist.md`
- `context-notes.md`

## Not Created
- `train_icaa_qat.py`: not created because direct QAT is not feasible.
- `export_icaa_qat_tflite.py`: not created because no QAT model exists to export.
- `evaluate_icaa_qat.py`: not created because no QAT candidate exists to evaluate.
- `qat_eval_report.md`, `qat_eval_comparison.csv`, `qat_latency_report.md`: not created because evaluation did not run.

## Conservative Conclusion
ICAA quantization/acceleration should stop for now. A future path would require a separate student-distillation project or a deliberate engineering pass to make `TFICAA17KDAT` cloneable/serializable and compatible with layer-level quantization configs.
