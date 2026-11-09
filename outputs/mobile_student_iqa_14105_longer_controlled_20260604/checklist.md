# Phase 9 checklist

- [x] Inspect Phase 8 short report, metrics, history, and trainer wrappers.
- [x] Confirm train/val/test manifest row counts: 11283 / 1409 / 1413 data rows.
- [x] Create Phase 9 longer controlled wrapper without overwriting smoke/short scripts.
- [x] Py-compile Phase 9 wrapper.
- [x] Run GPU preflight with `.venv_tf` and ImageNet weights.
- [x] Run bounded longer controlled training only: 11283/1409/1413 samples, 5 head + 10 finetune epochs.
- [x] Write required outputs and comparison against short controlled training.
- [x] Save `best_by_val_mae_model.keras` and `final_model.keras`.
- [x] Confirm no final full distillation, TFLite export, Flutter edit, production scoring edit, production model asset edit, or commit.
