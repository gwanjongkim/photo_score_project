# Context Notes

- This experiment is still an A-LAMP Multi-Patch teacher baseline, not a full A-LAMP reproduction.
- The GCN branch remains stopped for the teacher role; no Dual-Branch GCN files should be edited.
- Class weights should use the active training records after sample limits so dry runs and full runs are internally consistent.
- Baseline commands without new flags must preserve existing behavior: frozen VGG16 backbone and no sample weights.
- `--unfreeze_from_layer block5` is implemented as a VGG16 layer-name prefix match, so `block5_conv1`, `block5_conv2`, `block5_conv3`, and `block5_pool` become trainable when `--backbone_trainable true`.
- Class weighting is implemented as tf.data sample weights instead of Keras `class_weight`, which keeps the generator dataset format explicit and avoids compatibility ambiguity.
- Validation balanced metrics are computed after training from a non-repeated validation dataset at threshold 0.5.
- `py_compile` passed for the edited model, dataset, trainer, and evaluator files.
- The first smoke attempt failed only on sandboxed VGG16 ImageNet weight download; the escalated rerun used GPU and completed.
- Smoke used 32 train / 16 val records, computed class weights `{0: 1.6, 1: 0.7272727272727273}`, and trained only VGG16 `block5_*` layers.
- Small run used 1024 train / 256 val records, computed class weights `{0: 1.3950953678474114, 1: 0.7792998477929984}`, and completed two epochs without an input-exhaustion warning.
- Small run validation diagnostics at threshold 0.5: balanced accuracy `0.5470629865534324`, specificity `0.4444444444444444`, predicted positive ratio `0.61328125`.
