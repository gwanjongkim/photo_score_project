# Stage D Plan

Goal: implement only the TensorFlow/Keras SoftHistogram module and verify it against the export-safe PyTorch SoftHistogram used by DAT.

Success criteria:
1. Inspect and document the PyTorch `SoftHistogram` constructor, state entries, grouped Conv1d math, input shape, and output shape.
2. Add a TensorFlow custom layer that reproduces only SoftHistogram math.
3. Copy the real checkpoint SoftHistogram state into the TensorFlow layer.
4. Determine the real pre-SoftHistogram tensor shape through the export-safe PyTorch DAT path.
5. Run parity on deterministic random, zeros, ones, and one captured real pre-SoftHistogram tensor.
6. Save timestamped JSON, log, and captured tensor summary artifacts under `outputs/icaa_tf_native_stage_d_*`.
7. Write a Stage D Markdown report with a clear Stage E go/no-go decision.

