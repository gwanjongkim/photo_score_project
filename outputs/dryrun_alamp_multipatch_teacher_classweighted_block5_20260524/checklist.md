# Checklist

- [x] Inspect current Multi-Patch teacher model, dataset, trainer, evaluator, and config.
- [x] Add block5-only VGG16 unfreeze support while preserving frozen/full-trainable defaults.
- [x] Add optional class-weight sample weights from active training labels.
- [x] Add validation balanced metrics and train summary metadata.
- [x] Run `py_compile` validation.
- [x] Run the requested smoke training.
- [x] Run the requested small validation experiment if smoke succeeds.
- [x] Inspect output artifacts and summarize readiness for full training.
