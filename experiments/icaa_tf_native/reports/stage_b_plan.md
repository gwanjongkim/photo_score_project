# Stage B Plan

Goal: validate basic PyTorch-to-TensorFlow/Keras weight mapping rules with real ICAA17K DAT checkpoint tensors before implementing any TensorFlow DAT model.

Success criteria:
1. Use actual checkpoint tensors for Conv2D, Dense, LayerNorm, GELU, and an isolated MLP mapping check.
2. Compare PyTorch and TensorFlow outputs with deterministic seeds and record max/mean absolute differences.
3. Mark a test inconclusive or failed when the mapping cannot be determined safely.
4. Save timestamped JSON and log artifacts under `outputs/icaa_tf_native_stage_b_*`.
5. Write a concise Stage B Markdown report with verified rules, unresolved issues, and the Stage C go/no-go decision.

