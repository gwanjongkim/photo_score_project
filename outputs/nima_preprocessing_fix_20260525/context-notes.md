# NIMA preprocessing fix context notes

- 2026-05-25: Scope is limited to NIMA model/dataset/export/inference code plus this output directory.
- 2026-05-25: Existing NIMA model omits `include_preprocessing`, so Keras defaults are active for EfficientNetV2B0.
- 2026-05-25: Existing AVA distribution dataset uses `tf.image.convert_image_dtype(..., tf.float32)`, which returns image tensors in `[0,1]`.
- 2026-05-25: Fixed design keeps project images in `[0,1]` and disables EfficientNetV2 internal preprocessing in the model.
- 2026-05-25: Export weights-only rebuild now calls the same NIMA model builder with `backbone_weights=None`, so export architecture follows the fixed training architecture.
- 2026-05-25: NIMA validation/test preprocessing now uses deterministic resize-to-256 plus center-crop-224. Training remains resize-to-256 plus random-crop-224 plus random horizontal flip.
- 2026-05-25: `train_nima.py` now refuses to write into an output directory that already has `best.weights.h5`, `final_model.keras`, or `saved_model`.
- 2026-05-25: Keras accepted `include_preprocessing=False` with `backbone_weights=None` and with explicit local ImageNet weights at `/home/omen_pc1/.keras/models/efficientnetv2-b0_notop.h5`; the `"imagenet"` alias attempted a network download in the sandbox.
- 2026-05-25: Added optional `--backbone_weights` to NIMA training so retraining can use the explicit local ImageNet weights path without changing existing CLI defaults.
