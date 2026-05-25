# Context Notes - A-LAMP Multi-Patch Teacher Baseline

## Decisions
- Scope is a Multi-Patch teacher baseline only, not a full A-LAMP reproduction.
- External JSONL `patch_boxes` are treated as adaptive patch selections and never as ground-truth labels.
- Labels are binary AVA high/low targets derived from `mean_score > 5.0`.
- The dataset loader will use PIL for deterministic crop/resize, clip boxes to image dimensions before crop, and apply `tf.keras.applications.vgg16.preprocess_input` to RGB float pixels in `[0,255]`.
- The model will use one shared VGG16 ImageNet backbone across all 5 patches, global average pooling, orderless mean+max aggregation, and a sigmoid binary classification head.
- The trainer will default to frozen VGG16 backbone and will not start full training.

## Validation Log
- Source implementation added in new isolated files only:
  `src/datasets/alamp_external_patch_dataset.py`,
  `src/models/alamp_multipatch_teacher.py`,
  `src/train/train_alamp_multipatch_teacher.py`, and
  `configs/paper_benchmarks/alamp_multipatch_teacher_ava.yaml`.
- Existing practical A-LAMP, Mobile A-LAMP v2, Flutter, TFLite, and `forWeights/` files were not modified.
- `python -m py_compile src/models/alamp_multipatch_teacher.py` passed.
- `python -m py_compile src/datasets/alamp_external_patch_dataset.py` passed.
- `python -m py_compile src/train/train_alamp_multipatch_teacher.py` passed.
- The first sandboxed smoke attempt failed because Keras could not fetch the VGG16 ImageNet weights under restricted network access.
- The escalated rerun of the requested smoke command completed with GPU visible and frozen VGG16 backbone.
- Smoke metrics from `training_history.csv`: train accuracy `0.5`, train AUC `0.40909090638160706`, val accuracy `0.5`, val AUC `0.6666666269302368`, val precision `1.0`, val recall `0.3333333432674408`.
- Smoke artifacts were written to `outputs/dryrun_alamp_multipatch_teacher_20260524/best.weights.h5`, `final_model.keras`, `training_history.csv`, and `train_summary.json`.
