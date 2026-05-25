# Context Notes

- This remains the A-LAMP Multi-Patch teacher paper-capacity variant, not full A-LAMP and not a paper reproduction because the Layout-Aware subnet is still absent.
- The audit identified missing L2 regularization, overly high learning rate, missing horizontal flip augmentation, and a shallower shared projection as likely overfitting causes.
- Defaults must preserve the current GAP and paper-capacity behavior unless the new flags are passed.
- Horizontal flip is applied consistently across the five cropped patch tensors for a sample; spatial flip is equivalent before/after VGG16 per-pixel preprocessing, and validation/test remain unaugmented.
- L2 is applied to Dense kernels and biases only when `dense_l2`/`bias_l2` are greater than zero, so the default path has no regularizer losses.
- `patch_projection_layers=1` keeps the existing `patch_projection_dense` layer name; extra shared projection layers use numbered names.
- The trainer accepts `optimizer=adam` or `optimizer=sgd`; Adam remains the config/default behavior.
- `py_compile` passed for model, dataset, trainer, and evaluator after the recipe patch.
- Default compatibility smoke completed with GAP mode, `dense_l2=0`, `bias_l2=0`, `random_horizontal_flip=false`, total params `14977345`, and trainable params `262657`.
- Recipe smoke completed with `flatten_dense`, `patch_projection_layers=2`, `dense_l2=1e-5`, `bias_l2=1e-5`, `random_horizontal_flip=true`, total params `184604481`, and trainable params `169889793`.
- Recipe 8192/2048 midrun completed 3 epochs. Epoch 3 train AUC `0.7505`, val AUC `0.6696`; best val AUC `0.66965`.
- Compared to previous midrun final val AUC `0.6458`, the recipe run improved by `0.02385`. Train-val AUC gap reduced from about `0.2879` to about `0.0808`.
