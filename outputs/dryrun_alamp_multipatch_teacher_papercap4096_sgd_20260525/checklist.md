# Checklist

- [x] Inspect current optimizer implementation and Adam recipe baseline result.
- [x] Add `momentum`, `nesterov`, and `weight_decay` flags if missing.
- [x] Preserve default optimizer behavior.
- [x] Record optimizer settings in train summary.
- [x] Run `py_compile` validation.
- [x] Run default compatibility smoke.
- [x] Run SGD paper-capacity smoke.
- [x] Run SGD `lr=1e-3` 8192/2048 midrun.
- [x] Run SGD `lr=1e-4` 8192/2048 midrun only if needed.
- [x] Compare against Adam recipe midrun `val_auc=0.6696473360061646`.
