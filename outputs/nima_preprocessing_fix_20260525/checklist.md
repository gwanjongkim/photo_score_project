# NIMA preprocessing fix checklist

- [x] Confirm existing NIMA EfficientNetV2B0 preprocessing bug.
- [x] Set NIMA EfficientNetV2B0 to `include_preprocessing=False`.
- [x] Keep NIMA outputs, EMD loss, and expected mean score unchanged.
- [x] Align NIMA validation/test preprocessing to resize-256 plus center-crop-224.
- [x] Protect fixed retraining from overwriting existing NIMA checkpoints.
- [x] Run smoke checks only.
- [x] Write final report with recommended retraining command.
