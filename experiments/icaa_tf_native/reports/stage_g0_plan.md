# Stage G-0 TransformerStage 0 Plan

## Scope
- Implement TensorFlow/Keras parity for `TransformerStage` stage `0` only.
- Use only files under `experiments/icaa_tf_native/` and `outputs/icaa_tf_native_*`.
- Do not implement full DAT, later stages, training, or TFLite conversion.

## Steps
1. Inspect PyTorch stage `0` and `stages.0.*` checkpoint tensors.
   - Verify: script report includes exact class names, stage structure, residual order, drop path behavior, and tensor names.
2. Implement NHWC TensorFlow `TFTransformerStage0`.
   - Verify: it contains only stage-0 local and shifted-window blocks plus MLP and LayerNorm mapping.
3. Add Stage G-0 parity script.
   - Verify: script strict-loads the checkpoint, captures PyTorch stage-0 inputs/outputs, maps TensorFlow weights, and compares outputs.
4. Run Stage G-0 parity.
   - Verify: JSON report, log, optional debug summary, and Markdown report are saved in the requested paths.
