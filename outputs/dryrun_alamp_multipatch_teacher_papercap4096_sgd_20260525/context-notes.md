# Context Notes

- Existing trainer already has `--optimizer adam/sgd`, but SGD currently hard-codes `momentum=0.9` and has no `nesterov` or `weight_decay` flags.
- This remains the A-LAMP Multi-Patch teacher paper-capacity variant, not full A-LAMP and not a paper reproduction.
- First SGD comparison keeps layer-level `dense_l2=1e-5` and `bias_l2=1e-5`, with optimizer-level `weight_decay=0.0`, to avoid double-decay.
- Adam recipe comparison target is the 8192/2048 midrun best/final `val_auc=0.6696473360061646`.
- The optimizer flag already existed. This patch adds explicit `momentum`, `nesterov`, and optimizer-level `weight_decay` controls; default config remains `optimizer=adam`, `momentum=0.0`, `nesterov=false`, `weight_decay=0.0`.
- `py_compile` passed for trainer, model, dataset, and evaluator after the SGD optimizer patch.
- Default compatibility smoke completed with `optimizer=adam`, `momentum=0.0`, `nesterov=false`, `weight_decay=0.0`, total params `14977345`, trainable params `262657`.
- SGD paper-capacity smoke completed with `optimizer=sgd`, `learning_rate=1e-3`, `momentum=0.9`, `nesterov=false`, `weight_decay=0.0`; no OOM. Smoke loss was very high (`val_loss=494.3457`), so `lr=1e-3` may be unstable but still proceeds to bounded midrun per request.
- SGD `lr=1e-3` 8192/2048 midrun completed but diverged: loss was `nan` from epoch 1 and `val_auc` stayed at `0.5000`.
- SGD `lr=1e-4` 8192/2048 midrun completed stably. Epoch 3 train AUC `0.7429`, val AUC `0.6577`, below Adam recipe `0.6696473360061646`.
- Comparison: SGD `lr=1e-4` val AUC is `0.011997` below Adam; train-val AUC gap is `0.08527`, similar to Adam gap `0.08081`.
