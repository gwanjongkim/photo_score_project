# Technical IQA signal comparison checklist

- [x] Inspect existing model files under `exports/tflite/`.
- [x] Inspect student checkpoint/TFLite outputs.
- [x] Inspect previous hybrid benchmark outputs.
- [x] Inspect production technical eval predictions/formula.
- [x] Inspect TOPIQ/MUSIQ caches and protocol scripts.
- [x] Create phase-scoped benchmark script.
- [x] Py-compile benchmark script.
- [x] Run benchmark once in `.venv_tf`.
- [x] Confirm coverage recovery targets.
- [x] Write required outputs.
- [x] Create handoff note.
- [x] Confirm no Flutter edit, production scoring edit, training, full distillation, TFLite export, or commit.
