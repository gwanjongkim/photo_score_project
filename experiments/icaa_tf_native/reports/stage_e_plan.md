# Stage E Plan

Goal: implement only the first DAT local-attention / early block in TensorFlow and verify it against the export-safe PyTorch `stages.0` depth-0 block.

Success criteria:
1. Document the exact PyTorch class names, residual order, eval-mode drop path behavior, tensor layouts, and checkpoint tensors.
2. Add an NHWC TensorFlow implementation for the stage-0 depth-0 block only.
3. Map only the stage-0 depth-0 LayerNorm, LocalAttention, relative-position, and MLP tensors.
4. Compare parity on random block input, patch projection output from a deterministic random image, and patch projection output from three real ICAA17K images.
5. Save JSON, log, and debug tensor summaries under `outputs/icaa_tf_native_stage_e_*`.
6. Write a Stage E Markdown report with unresolved issues and a Stage F go/no-go decision.

