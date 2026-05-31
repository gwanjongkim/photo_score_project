# Next Action

## Selected Decision
E. QAT cannot be applied safely; stop ICAA acceleration attempts for now.

## Action
- Do not deploy Dynamic PTQ.
- Do not deploy Full INT8 PTQ.
- Do not create or copy a QAT model into Flutter.
- Stop ICAA quantization/acceleration work for now.

## Reason
- Direct QAT is blocked by current artifacts and tooling.
- The SavedModel is inference-loadable but not trainable via Keras `load_model`.
- The source model is recoverable only as a subclassed custom-layer model and is not cloneable with `tf.keras.models.clone_model`.
- TFMOT is not installed, and adding it would modify the environment outside this task's allowed output directory.

## Future Work If Reopened Later
- Treat any follow-up as a separate student-distillation or model-rearchitecture task, not as a continuation of this QAT attempt.
- A new trainable student should be designed to preserve the same ICAA input/output contract and pass the `<= 0.001` model-level parity gate before mobile benchmarking.
