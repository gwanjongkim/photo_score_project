# TechIQA-Guard v1 Dataset

Generated manifests for direct single-output technical IQA training.

This is dataset preparation only. It is not teacher-student distillation, multi-head training, model training, or TFLite export.

## Hard False-Positive Rules

- `hard_false_positive`: `delta_mixed112_existing >= 5` or manual user flag.
- `strong_hard_false_positive`: `delta_mixed112_existing >= 8`.
- Manual user-flagged files are always included even when MOS or images are missing.
- Original MOS is preserved when available. Hard-FP guard columns carry cap metadata.

## Files

- `train.csv` and `val.csv`: direct single-head training manifests from mixed_112 inputs when available.
- `test_flive.csv`, `test_koniq.csv`, `test_spaq.csv`: separate held-out test manifests.
- `hard_false_positive.csv`: manual and mined hard false-positive rows.
- `smoke_v1.csv`: balanced smoke manifest plus all hard false positives.
- `summary.json`: counts, inputs, manual file discovery status, and warnings.

## Row Counts

- `train`: 16384
- `val`: 2048
- `test_flive`: 3981
- `test_koniq`: 1008
- `test_spaq`: 1125
- `hard_false_positive`: 3
- `smoke_v1`: 450
