# ICAA QAT attempt context notes

- Scope is limited to `outputs/icaa_qat_attempt_20260601/`.
- Production Flutter artifacts, production metadata, existing FP16/PTQ artifacts, preprocessing contract, and A-cut scoring formulas must not be modified.
- Prior Dynamic PTQ review failed model-level parity with MAE `0.003328498899936676`, MaxAE `0.009905099868774414`, P90 AE `0.006955149769783022`, and 81 of 100 output values over `0.001`.
- Prior Full INT8 float-IO review found repeated runtime failure at quantized DIV node 593.
- Default `python` lacks `tensorflow_model_optimization` and `torch`; `.venv_gpu` has TensorFlow `2.21.0`, `tf_keras 2.21.0`, and `torch 2.5.1+cu121`, but still lacks `tensorflow_model_optimization`.
- Feasibility audit with `.venv_gpu` confirmed `tf.saved_model.load` works, but `tf.keras.models.load_model` cannot load the SavedModel under Keras 3.
- The source model can be rebuilt as `TFICAA17KDAT` and loaded from the PyTorch checkpoint, but it is a subclassed custom-layer model with no Functional inputs/outputs.
- `tf.keras.models.clone_model` failed on `TFICAA17KDAT` because deserialization passes unsupported constructor fields such as `trainable`.
- Direct QAT is not safe here because standard TFMOT tooling is absent and the recoverable model is not a cloneable Functional/Sequential model.
