# TechIQA-Guard v1 Dataset Builder Checklist

- [x] Inspect local design report and available SPAQ/KonIQ/FLIVE/TOPIQ inputs.
- [x] Keep this work additive and avoid stable training, export, and Flutter files.
- [x] Add isolated `src/data/build_techiqa_guard_dataset.py`.
- [x] Include manual false positives even when images or MOS are missing.
- [x] Mine hard false positives with `delta_mixed112_existing >= 5` and strong hard false positives with `>= 8`.
- [x] Preserve original MOS labels and write guard target columns instead of forcing hard-FP labels.
- [x] Write train/val/test/hard-fp/smoke manifests plus summary and README under this directory.
- [x] Run `py_compile`, `--help`, the default builder command, and requested output inspections.
