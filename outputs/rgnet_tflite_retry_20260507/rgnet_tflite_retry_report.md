# RGNet TFLite Retry Report

## 1. Environment
- Working directory: `/home/omen_pc1/photo_score_project`.
- Date command at retry start: `Thu May  7 22:20:28 KST 2026`.
- Python: `Python 3.12.3`.
- Input model: `outputs/ava_pretrain_aadb_finetune_20260507/rgnet_ava_pretrain_aadb_finetune/final_model.keras`.
- Output directory: `outputs/rgnet_tflite_retry_20260507`.
- Git status was dirty before this retry; unrelated modified/untracked files were left untouched.

## 2. Reference Keras Output
- Reference preprocessing: RGB, resize 256x256, float32 / 255.0.
- Reference image: `data/raw/ava/images/85837.jpg`.
- Direct `final_model.keras` score: `0.46629470586776733`.

## 3. Existing Failure Diagnosis
- Previous SavedModel export conversion mode: `select_tf_ops`.
- Previous SavedModel required Select TF ops: `True`.
- Previous requested TFLite allocation error: `Select TensorFlow op(s), included in the given model, is(are) not supported by this interpreter. Make sure you apply/link the Flex delegate before inference. For the Android, it can be resolved by adding "org.tensorflow:tensorflow-lite-select-tf-ops" dependency. See instructions: https://www.tensorflow.org/lite/guide/ops_selectNode number 1 (FlexMul) failed to prepare.`.
- SavedModel vs checkpoint max diff before TFLite allocation: `5.841255187988281e-06`.
- Previous rebuild weights vs checkpoint max diff: `0.17772206664085388`.
- Previous rebuild TFLite vs checkpoint max diff: `0.17774948477745056`.
- Diagnosis: SavedModel export fails the builtin requirement because mixed-float16 TensorFlow ops require Flex. The builtin rebuild path fails earlier than conversion because the float32 rebuild is not numerically equivalent to the mixed-policy Keras checkpoint.

## 4. Direct Keras Built-in TFLite Attempt
- Output target: `outputs/rgnet_tflite_retry_20260507/tflite/rgnet_ava_aadb_finetune_direct_keras_builtin.tflite`.
- Conversion ok: `False`.
- Verification passed: `False`.
- Error type: `ConverterError`.
- Error summary: `Could not translate MLIR to FlatBuffer./home/omen_pc1/photo_score_project/.venv_gpu/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py:696:1: error: 'tf.Mul' op is neither a custom op nor a flex op self._concrete_variable_creation_fn = tracing_compilation.trace_function( ^ /home/omen_pc1/photo_score_project/.venv_gpu/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py:178:1: note: called from concrete_function = _maybe_define_function( ^ /home/omen_pc1/photo_score_project/.venv_gpu/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py:283:1: note: called from concrete_function = _create_concrete_function( ^ /home/omen_pc1/photo_score_project/.venv_gpu/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py:310:1: n...`.

## 5. Rebuild Built-in TFLite Attempt
- Strict float32 rebuild v2 weight coverage: `True` with `373` weights.
- Max weight abs diff after copy: `0.0`.
- Float32 rebuild Keras vs original Keras smoke diff: `0.17772197723388672`.
- Float32 rebuild Keras vs original Keras 21-image max diff: `0.33626319468021393`.
- Float32 rebuild result: failed before TFLite export; no v2 TFLite was written.
- Mixed-policy rebuild Keras parity before conversion: `True` with max diff `0.0`.
- Mixed-policy builtin conversion result: failed; converter required Flex for float16 TensorFlow ops.

## 6. Select TF Ops Diagnostic
- Diagnostic artifact: `outputs/rgnet_tflite_retry_20260507/tflite/rgnet_ava_aadb_finetune_select_tf_ops_diagnostic.tflite`.
- Requires Select TF ops: `True`.
- Preferred for deployment: `False`.
- Allocation ok with local standard interpreter: `False`.
- Allocation error: `Select TensorFlow op(s), included in the given model, is(are) not supported by this interpreter. Make sure you apply/link the Flex delegate before inference. For the Android, it can be resolved by adding "org.tensorflow:tensorflow-lite-select-tf-ops" dependency. See instructions: https://www.tensorflow.org/lite/guide/ops_selectNode number 1 (FlexMul) failed to prepare.`.
- Decision: diagnostic only; not a deployment artifact.

## 7. Numeric Parity Results
| Candidate | Built-in only | Keras parity before conversion | TFLite allocation | Keras vs TFLite parity | Result |
|---|---:|---:|---:|---:|---|
| Direct Keras builtin | yes | n/a | n/a | n/a | conversion failed |
| Strict float32 rebuild builtin | yes | failed | n/a | n/a | not exported |
| Mixed-policy rebuild builtin | yes | passed | n/a | n/a | conversion failed |
| Select TF ops diagnostic | no | n/a | failed | n/a | non-preferred diagnostic |

## 8. Fixed-Subset Evaluation
- Status: skipped.
- Reason: no builtin TFLite candidate passed numeric parity.
- Previous fine-tuned Keras reference remains AADB val512 SRCC/PLCC/MAE/RMSE `0.6749 / 0.6835 / 0.1104 / 0.1381`.

## 9. Deployment Decision
- Classification: `not deployment-ready, teacher-only, needs source/export fix`.
- Deployment-ready: `False`.
- Reason: No standard builtin TFLite artifact both loads and matches the fine-tuned Keras checkpoint within 1e-4.
- No Flutter copy was performed.
- Recommendation: keep RGNet fine-tuned Keras as teacher-only until a source/export fix produces a builtin TFLite with max diff <= 1e-4.
