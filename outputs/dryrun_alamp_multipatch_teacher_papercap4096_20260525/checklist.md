# Checklist

- [x] Inspect current Multi-Patch teacher model/trainer/evaluator/config.
- [x] Add `gap` and `flatten_dense` patch projection modes.
- [x] Add configurable `patch_feature_dim`, `head_layers`, and `head_dropout`.
- [x] Preserve existing default model behavior unless new flags are passed.
- [x] Record capacity settings and parameter counts in `train_summary.json`.
- [x] Add OOM-aware training failure message.
- [x] Run `py_compile` validation.
- [x] Run default compatibility smoke.
- [x] Run paper-capacity 4096 smoke at batch size 1.
- [x] Inspect artifacts and report feasibility.
