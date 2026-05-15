# Stage G-3 TransformerStage 3 Plan

## Scope
- Implement and verify only DAT `TransformerStage` stage `3`.
- Keep Stage G-2 marked unresolved for full strict parity.
- Work only under `experiments/icaa_tf_native/` and `outputs/icaa_tf_native_*`.
- Do not implement full DAT, train, convert to TFLite, or touch official/A-cut/Flutter files.

## Steps
1. Inspect PyTorch stage `3` and checkpoint keys directly.
   - Verify: report records class names, shapes, `stage_spec`, residual order, drop-path eval behavior, and `stages.3.*` tensor names.
2. Add `TFTransformerStage3`.
   - Verify: it reuses `TFLocalAttention`, `TFDAttentionBaseline`, and `TFTransformerMLP`; NHWC internal layout; stage spec `["L", "D"]`.
3. Add `stage_g3_transformer_stage3_parity.py`.
   - Verify: script strict-loads the checkpoint, captures PyTorch stage `3` inputs/outputs, maps only `stages.3.*`, runs random/captured/real-image cases, and writes JSON/log/debug artifacts.
4. Run the Stage G-3 script.
   - Verify: outputs are compared against preferred `1e-5` and acceptable `1e-4` thresholds and markdown report states whether full TensorFlow DAT assembly parity is safe to attempt.
