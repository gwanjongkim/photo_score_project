# Stage C Plan

Goal: implement only the TensorFlow/Keras patch projection block and verify it against the export-safe PyTorch DAT `patch_proj` block.

Success criteria:
1. Add an NHWC TensorFlow block containing only Conv2D and channel LayerNormalization.
2. Load only the four official patch projection checkpoint tensors into the TensorFlow block.
3. Strict-load the official checkpoint into the export-safe PyTorch DAT and use `model.patch_proj` as the reference.
4. Run deterministic random-input parity for `[1, 3, 224, 224]` vs `[1, 224, 224, 3]`.
5. Save timestamped JSON and log artifacts under `outputs/icaa_tf_native_stage_c_*`.
6. Write a Stage C Markdown report with output shapes, diff metrics, unresolved issues, and the Stage D go/no-go decision.

