# DistortionGuard-IQA v1 Synthetic Dataset

This directory holds smoke artifacts for the DistortionGuard-IQA v1 synthetic distortion dataset generator.

Default smoke outputs:
- `synthetic_smoke/`: distorted image files grouped by distortion type and severity.
- `synthetic_smoke_manifest.csv`: per-distorted-image metadata.
- `synthetic_smoke_pairs.csv`: weak ranking pairs where lower distortion severity is treated as better quality.
- `synthetic_smoke_summary.json`: run-level counts and generation status.

The initial generator is intentionally dataset-only. It does not train models, change trainers, export TFLite, or touch Flutter runtime assets.
