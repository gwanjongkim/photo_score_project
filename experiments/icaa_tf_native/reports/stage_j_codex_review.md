# Stage J Codex Review

## Scope
- Review target: Stage J FP32 TFLite conversion for the native TensorFlow/Keras ICAA17K DAT model.
- Explicitly not started: Stage K FP16, INT8, Flutter, Android smoke testing, training, official repo edits, A-cut source edits, and model architecture changes.
- Allowed write area used: `experiments/icaa_tf_native/` and new `outputs/icaa_tf_native_tflite_fp32_*` artifacts only.

## Git Status Summary
- Initial `git status --short` was already dirty before this review, including tracked modifications outside Stage J such as `.gitignore`, `README.md`, docs, earlier Stage G/H reports, TensorFlow-native model files, requirements, and RGNet files, plus many untracked files.
- Initial `git diff --stat` showed 15 tracked files changed before this review: 314 insertions and 700 deletions.
- Review-scoped tracked changes made here:
  - `experiments/icaa_tf_native/scripts/stage_j_convert_tflite_fp32.py`
  - `experiments/icaa_tf_native/reports/stage_j_tflite_fp32_conversion.md`
  - `experiments/icaa_tf_native/reports/stage_j_codex_review.md`
  - `experiments/icaa_tf_native/reports/checklist.md`
  - `experiments/icaa_tf_native/reports/context-notes.md`
- No unrelated dirty files were reverted or modified.

## Files Reviewed
- `experiments/icaa_tf_native/scripts/stage_j_convert_tflite_fp32.py`
- `experiments/icaa_tf_native/reports/stage_j_tflite_fp32_conversion.md`
- `outputs/icaa_tf_native_tflite_fp32_20260516_134634/stage_j_tflite_fp32_report.json`
- `outputs/icaa_tf_native_tflite_fp32_20260516_134634/stage_j_tflite_fp32_predictions.csv`
- `outputs/icaa_tf_native_tflite_fp32_20260516_134634/stage_j_tflite_fp32_log.txt`
- `outputs/icaa_tf_native_tflite_fp32_20260516_134634/tflite_input_sensitivity_report.json`
- `outputs/icaa_tf_native_tflite_fp32_20260516_134634/icaa_dat_tf_native_fp32.tflite`
- Latest rerun artifacts under `outputs/icaa_tf_native_tflite_fp32_20260516_140027/`

## Script Review
- The script selected the latest Stage I SavedModel, resolving to `outputs/icaa_tf_native_savedmodel_20260516_134439/saved_model`.
- Conversion is derived from that SavedModel through a fixed-shape concrete function with input signature `[1, 224, 224, 3]`, dtype `float32`.
- FP32-only behavior is clean: no FP16 setting, no INT8 path, no representative dataset, no converter optimization, and no quantization setting.
- Builtin TFLite conversion is attempted first. `SELECT_TF_OPS` is only configured in the exception fallback path if builtin conversion fails.
- Required artifacts are written: `.tflite`, JSON report, log, input-sensitivity JSON, and real-image prediction CSV.
- Interpreter load, `allocate_tensors()`, `invoke()`, input sensitivity, and SavedModel-vs-TFLite parity are exercised.
- TFLite inference is explicitly performed one image at a time because the exported TFLite input shape is fixed batch `1`.

## Fixes Applied
- Moved `import tensorflow as tf` below the existing environment setup in `stage_j_convert_tflite_fp32.py` so `CUDA_VISIBLE_DEVICES=-1`, `TF_ENABLE_ONEDNN_OPTS=0`, `TF_CPP_MIN_LOG_LEVEL=3`, and `MPLCONFIGDIR` apply before TensorFlow initializes.
- Updated `stage_j_tflite_fp32_conversion.md` to correct the stale real-image mean-diff value and add explicit command, paths, dtype/shape, fixed batch-1 limitation, op/custom-op summary, previous-vs-latest comparison, and Stage K readiness.
- No model architecture, conversion mode, FP16/INT8 path, official repo file, A-cut source, Flutter file, or training code was changed.

## Latest Stage J Rerun
- Command: `python experiments/icaa_tf_native/scripts/stage_j_convert_tflite_fp32.py`
- Latest output directory: `outputs/icaa_tf_native_tflite_fp32_20260516_140027/`
- TFLite path: `outputs/icaa_tf_native_tflite_fp32_20260516_140027/icaa_dat_tf_native_fp32.tflite`
- Status: passed.
- Builtin conversion: passed.
- `SELECT_TF_OPS` required: no.
- Interpreter load: passed.
- `allocate_tensors()`: passed.
- `invoke()`: passed.
- Input sensitivity: passed.
- SavedModel-vs-TFLite parity: passed preferred thresholds.

## Previous vs Latest
| Field | Previous `20260516_134634` | Latest `20260516_140027` |
| :--- | :--- | :--- |
| TFLite size | 350,915,988 bytes | 350,915,988 bytes |
| SHA-256 | `7a29a2de7463a27187c2d205e324c4ed0197f0009d49d1c14471e21087dcb7e7` | `7a29a2de7463a27187c2d205e324c4ed0197f0009d49d1c14471e21087dcb7e7` |
| Builtin conversion | true | true |
| SELECT_TF_OPS required | false | false |
| Zero vs one diff | 0.0111754239 | 0.0111754239 |
| Zero vs random diff | 0.0507589877 | 0.0507589877 |
| Random parity max/mean | 1.1920929e-07 / 1.0430813e-07 | 1.0430813e-07 / 9.6857548e-08 |
| 16-real-image parity max/mean | 8.7618828e-06 / 6.1001629e-07 | 6.6399574e-05 / 2.5602058e-06 |
| Parity status | `pass_preferred` | `pass_preferred` |

The TFLite FlatBuffer itself is byte-identical. The parity deltas changed after making the script's CPU/no-oneDNN environment settings effective before TensorFlow import; the latest values still pass the preferred FP32 thresholds.

## TFLite Model Inspection
- Model exists: yes.
- File size: 350,915,988 bytes.
- Input: shape `[1, 224, 224, 3]`, shape signature `[1, 224, 224, 3]`, dtype `float32`, quantization `(0.0, 0)`.
- Output: shape `[1, 2]`, shape signature `[1, 2]`, dtype `float32`, quantization `(0.0, 0)`.
- Fixed-shape limitation: batch size is fixed to `1`; batch inference must loop one image at a time.
- Builtin interpreter op count with `BUILTIN_REF`: 3,598.
- Builtin operator types: `ABS`, `ADD`, `BATCH_MATMUL`, `CAST`, `CONCATENATION`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `DIV`, `EQUAL`, `EXP`, `FLOOR`, `FULLY_CONNECTED`, `GATHER`, `GELU`, `GREATER_EQUAL`, `LESS_EQUAL`, `LOGICAL_AND`, `LOGISTIC`, `MEAN`, `MINIMUM`, `MUL`, `NEG`, `REDUCE_MAX`, `RELU`, `RESHAPE`, `RSQRT`, `SELECT_V2`, `SOFTMAX`, `SPLIT`, `SQUARED_DIFFERENCE`, `STRIDED_SLICE`, `SUB`, `SUM`, `TANH`, `TILE`, `TRANSPOSE`, `ZEROS_LIKE`.
- FlatBuffer custom operator codes: none.
- SELECT_TF_OPS/Flex operator codes: none.
- Runtime delegate note: the default interpreter may create the XNNPACK CPU delegate, but the FlatBuffer does not require delegate, custom, or Flex ops.

## Issues Found
- Documentation issue: the previous Markdown report had a stale/mismatched real-image mean-diff value and did not clearly spell out dtype/shape, fixed batch-1 inference, or op/custom-op summary.
- Script cleanup issue: TensorFlow was imported before environment variables intended to control CPU/no-oneDNN/log behavior.
- No conversion correctness blocker was found.

## Recommendation
Stage J is safe to build on. Proceed to Stage K FP16 conversion when ready, using `outputs/icaa_tf_native_tflite_fp32_20260516_140027/icaa_dat_tf_native_fp32.tflite` and the Stage I SavedModel path above as the reviewed FP32 baseline. Android/Flutter smoke testing can be planned later after the next export gates; it was not started here.
