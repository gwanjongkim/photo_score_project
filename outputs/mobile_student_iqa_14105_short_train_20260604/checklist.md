# Phase 8 checklist

- [x] Inspect prior smoke report and smoke trainer.
- [x] Reuse smoke trainer functions through short-run wrapper.
- [x] Py-compile short trainer.
- [x] Run GPU preflight with `.venv_tf` and ImageNet weights.
- [x] Run bounded short training only: 4000/1000/1000 samples, 3 head + 3 finetune epochs.
- [x] Write required metrics, predictions, failure cases, sample manifest, report, and smoke comparison.
- [x] Save smoke-only checkpoint artifact: `short_model.keras`.
- [x] Confirm no TFLite export, Flutter edit, production scoring edit, or production model asset creation.
