# A-LAMP Paper AVA v0 Checklist

- [ ] Inspect practical A-LAMP, native-size dataset, and RGNet paper AVA patterns without modifying them.
- [x] Create isolated A-LAMP-paper-AVA-v0 model, train script, eval script, and config.
- [x] Run py_compile with ./.venv_gpu/bin/python.
- [x] Check TensorFlow GPU visibility with ./.venv_gpu/bin/python.
- [x] Run forward smoke for v0_a and v0_b.
- [x] Run tiny smoke train/eval for v0_a.
- [x] Optionally smoke v0_b only if memory is safe.
- [x] Run bounded v0_a mid-run if smoke passes.
- [x] Compare mid-run results carefully against RGNet references with sample-size caveats.
- [x] Write report, summary JSON, and command log.
- [x] Report changed files and suggested git add list without committing.

## 2026-05-11 v0_b_fixed XLA Failure Fix

- [x] Inspect the current v0_b model, training compile path, and failed output evidence.
- [x] Replace v0_b dynamic/global-layout fusion with fixed-shape Keras tensor operations.
- [x] Add explicit `jit_compile=False` where the model is compiled.
- [x] Run requested py_compile validation.
- [x] Run v0_b forward smoke with batch size 2 and fixed input shapes.
- [x] Run v0_b tiny smoke train with 64 train / 32 val samples.
- [x] Run v0_b_fixed mid-train only if smoke passes.
- [x] Run v0_b_fixed mid-eval only if mid-train succeeds.
- [x] Update report, summary JSON, command log, and context notes with exact outcomes.
