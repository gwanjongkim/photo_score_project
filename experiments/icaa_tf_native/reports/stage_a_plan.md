# Stage A Plan

Goal: inspect the official PyTorch DAT checkpoint and export-safe v3 model without training, conversion, or source-file edits, then produce mapping evidence for a future TensorFlow/Keras native implementation.

Success criteria:
1. Create only Stage A artifacts under `experiments/icaa_tf_native/` and `outputs/icaa_tf_native_state_dict_inspection/`.
2. Load the official checkpoint on CPU and confirm the state_dict structure.
3. Produce a complete CSV inventory with shape, dtype, module grouping, likely TensorFlow layer type, mapping rule, and notes.
4. Instantiate the export-safe v3 DAT model with the official config, strict-load the checkpoint, and record dummy output shape.
5. Save the JSON summary and Markdown report with totals, breakdowns, uncertainty, command, warnings, and errors.

