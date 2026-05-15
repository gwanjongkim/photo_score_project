# Stage G-2 MLP Surgical Debug Plan

## Scope
- Diagnose only Stage 2 block `3` and block `4` MLP parity.
- Use only `experiments/icaa_tf_native/` and `outputs/icaa_tf_native_*`.
- Do not modify model code, train, convert, implement full DAT, or proceed to Stage G-3.

## Steps
1. Load the official checkpoint into PyTorch export-safe v3 DAT and TensorFlow `TFTransformerStage2`.
   - Verify: strict PyTorch load succeeds and only `stages.2.mlps.3.*` / `stages.2.mlps.4.*` are reported for the target MLP keys.
2. Capture full-flow block `3` and `4` subcomponents for the three requested inputs.
   - Verify: report includes block input, norm1, attention output, attention residual, norm2, MLP linear1, GELU, linear2, and MLP residual.
3. Run direct same-input MLP parity from PyTorch-captured `norm2`.
   - Verify: direct MLP comparisons decide implementation bug versus accumulated numeric drift.
4. Write artifacts and Markdown report.
   - Verify: JSON, log, optional debug tensor summary, and report are saved.
