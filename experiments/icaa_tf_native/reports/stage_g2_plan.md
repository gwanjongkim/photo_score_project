# Stage G-2 TransformerStage 2 Plan

## Scope
- Implement TensorFlow/Keras parity for `TransformerStage` stage `2` only.
- Use only files under `experiments/icaa_tf_native/` and `outputs/icaa_tf_native_*`.
- Do not implement full DAT, stage `3`, training, or TFLite conversion.

## Steps
1. Inspect PyTorch stage `2` and `stages.2.*` checkpoint tensors.
   - Verify: script report includes exact class names, 18-block structure, residual order, drop path behavior, and tensor names.
2. Extend NHWC TensorFlow stage support for `TFTransformerStage2`.
   - Verify: it composes stage-2 local and deformable attention blocks plus MLP and LayerNorm mapping only.
3. Add Stage G-2 parity script.
   - Verify: script strict-loads the checkpoint, captures PyTorch stage-2 inputs/outputs, maps TensorFlow weights, and compares outputs.
4. Run Stage G-2 parity.
   - Verify: JSON report, log, optional debug summary with per-block residual diffs, and Markdown report are saved in the requested paths.
