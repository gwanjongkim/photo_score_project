# DistortionGuard-IQA v1 Synthetic Smoke Checklist

- [x] Confirm source manifest columns and local design direction.
- [x] Keep the implementation isolated from model/training/export/runtime code.
- [x] Add `src/data/generate_distortionguard_synthetic.py`.
- [x] Support requested CLI arguments and defaults.
- [x] Generate all requested distortion types and severities.
- [x] Skip WebP with a warning if Pillow lacks WebP support.
- [x] Record unreadable images as failed rows instead of crashing.
- [x] Write manifest, pair CSV, and summary JSON.
- [x] Run `python -m py_compile`.
- [x] Run the requested 20-image smoke command.
- [x] Inspect manifest, pairs, and summary outputs.
