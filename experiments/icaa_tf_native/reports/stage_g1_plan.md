# Stage G-1 TransformerStage 1 Plan

## Scope
- Implement TensorFlow/Keras parity for `TransformerStage` stage `1` only.
- Use only files under `experiments/icaa_tf_native/` and `outputs/icaa_tf_native_*`.
- Do not implement full DAT, stages `2`/`3`, training, or TFLite conversion.

## Steps
1. Inspect PyTorch stage `1` and `stages.1.*` checkpoint tensors.
   - Verify: script report includes exact class names, stage structure, residual order, drop path behavior, and tensor names.
2. Extend NHWC TensorFlow stage support for `TFTransformerStage1`.
   - Verify: it contains only stage-1 local and shifted-window blocks plus MLP and LayerNorm mapping.
3. Add Stage G-1 parity script.
   - Verify: script strict-loads the checkpoint, captures PyTorch stage-1 inputs/outputs, maps TensorFlow weights, and compares outputs.
4. Run Stage G-1 parity.
   - Verify: JSON report, log, optional debug summary, and Markdown report are saved in the requested paths.
