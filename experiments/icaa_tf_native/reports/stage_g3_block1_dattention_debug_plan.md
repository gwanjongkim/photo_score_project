# Stage G-3 Block1 DAttention Surgical Debug Plan

## Scope
- Diagnose only Stage 3 block `1` DAttention parity.
- Use PyTorch-captured block `1` norm-attention output as the direct input to both PyTorch and TensorFlow DAttention.
- Keep Stage G-2 and Stage G-3 full strict parity unresolved unless this diagnostic proves otherwise.
- Do not modify model code, train, convert, implement full DAT, or proceed to full DAT assembly.

## Steps
1. Load the official checkpoint into PyTorch export-safe v3 DAT and TensorFlow `TFTransformerStage3`.
   - Verify: strict PyTorch load succeeds and only `stages.3.attns.1.*` keys are reported for the target DAttention block.
2. Capture PyTorch Stage 3 block `1` norm-attention tensors for the three requested inputs.
   - Verify: the direct TensorFlow DAttention input is exactly the PyTorch tensor transposed from NCHW to NHWC.
3. Capture DAttention subcomponents on the same input.
   - Verify: report includes q, offset, pos, reference, sampled feature, k, v, logits before/after RPE, RPE bias, softmax, weighted output, projection output, and final output.
4. Write artifacts and Markdown report.
   - Verify: JSON, log, optional debug tensor summary, and report are saved, with a numeric-drift versus implementation-bug decision.
