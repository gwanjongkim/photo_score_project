# Stage F Plan

Goal: implement only the first DAT deformable/delegate attention module in TensorFlow and verify it against PyTorch `DAttentionBaseline`.

Success criteria:
1. Inspect and document the first `DAttentionBaseline` instance, its stage/block index, checkpoint tensors, shape, offset path, sampler, and relative-position path.
2. Add an NHWC TensorFlow implementation for `stages.2.attns.1` only.
3. Map only that module's checkpoint tensors.
4. Compare parity on deterministic random module input, a captured deterministic random-image input, and captured real-image inputs.
5. Save JSON, log, and debug summaries under `outputs/icaa_tf_native_stage_f_*`.
6. Write a Stage F Markdown report with unresolved issues and a Stage G go/no-go decision.

