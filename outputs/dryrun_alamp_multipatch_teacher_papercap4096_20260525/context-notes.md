# Context Notes

- This is the "A-LAMP Multi-Patch teacher paper-capacity variant", not full A-LAMP and not a paper reproduction because Layout-Aware A-LAMP is still missing.
- External adaptive patch boxes remain patch selections, not ground-truth labels.
- VGG16 preprocessing should stay in the existing dataset path using `tf.keras.applications.vgg16.preprocess_input` on RGB float pixels in `[0,255]`.
- Because the 4096 Flatten+Dense path is memory-heavy, validation stops at smoke runs unless the user explicitly asks for longer training.
- The default GAP path keeps the existing head layer names `teacher_dense` and `teacher_dropout` when `head_layers=1`.
- The paper-capacity path applies shared VGG16 to merged patch batches, then shared `Flatten -> Dense(patch_feature_dim)` before restoring `[batch, 5, feature_dim]`.
- Weights-only evaluation now reads the same capacity fields from config so future paper-capacity checkpoints can be rebuilt without changing evaluator architecture code.
- `py_compile` passed for model, trainer, and evaluator after the capacity patch.
- Default compatibility smoke completed with `patch_projection_mode=gap`, `patch_feature_dim=512`, `head_layers=1`, total params `14977345`, and trainable params `262657`.
- Paper-capacity smoke completed at batch size 1 with total params `167823169`, trainable params `153108481`, frozen VGG16 params `14714688`, and no OOM.
- Paper-capacity TensorFlow GPU memory summary reported current/peak bytes after model build `674500096` / `1546840576`, and after training `2016218624` / `4060562176`.
- Paper-capacity artifacts are large: `best.weights.h5` and `final_model.keras` are each about `1.8G`.
