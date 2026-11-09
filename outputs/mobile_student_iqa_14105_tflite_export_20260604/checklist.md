# Phase 12 checklist

- [x] Inspect existing TFLite export/parity scripts under `scripts/technical_iqa` and `outputs`.
- [x] Inspect Phase 11 checkpoint evaluation report and selected checkpoint.
- [x] Create Phase 12 export/parity script only because existing parity script is old hard-coded path.
- [x] Py-compile Phase 12 export script.
- [x] Run GPU-safe `.venv_tf` export/parity command.
- [x] Export FP32 and FP16 TFLite.
- [x] Verify shapes, op compatibility, analyzer output, and model sizes.
- [x] Run bounded stratified parity sample.
- [x] Write hybrid runtime design and final report.
- [x] Confirm no Flutter edit, production scoring edit, training, full distillation, model-asset replacement, or commit.
