# Conservative Mobile Aesthetic Student Handoff

## Model

- Experiment: `student_conservative_aesthetic_full_aadb_20260423`
- Final mobile artifact: `outputs/student_conservative_aesthetic_full_aadb_20260423/export/student_conservative_aesthetic_full_aadb_20260423_fp16_builtin.tflite`
- Metadata: `outputs/student_conservative_aesthetic_full_aadb_20260423/export/student_conservative_aesthetic_full_aadb_20260423.metadata.json`
- Verification: `outputs/student_conservative_aesthetic_full_aadb_20260423/export/student_conservative_aesthetic_full_aadb_20260423.verify_test_vila.json`
- Comparison summary: `outputs/student_conservative_aesthetic_full_aadb_20260423/comparison/summary.json`

## Teacher Formula

`conservative_teacher_score = 0.35 * rgnet_aadb + 0.20 * alamp_aadb + 0.30 * aadb_composition + 0.15 * nima_ava_unit`

Where:

- `rgnet_aadb` uses the repo's existing practical `[0, 1]` score
- `alamp_aadb` uses the repo's existing practical `[0, 1]` score
- `aadb_composition` uses the repo's existing practical `[0, 1]` score
- `nima_ava_unit = clip((nima_mean_score - 1.0) / 9.0, 0.0, 1.0)`

## Training Data

- Source dataset: AADB train/val path already wired in the repo
- Conservative teacher manifest:
  - `outputs/distill_conservative_aesthetic_full_aadb_20260423/teacher_labels_train.csv` (`7612` rows)
  - `outputs/distill_conservative_aesthetic_full_aadb_20260423/teacher_labels_val.csv` (`846` rows)
  - combined manifest: `outputs/distill_conservative_aesthetic_full_aadb_20260423/teacher_labels_trainval.csv`
- Training config: `configs/student_distill_conservative_full_aadb_20260423.json`
- Teacher config: `configs/student_distill_teacher_conservative_aadb.json`

## Student Contract

- Architecture: MobileNetV2 `alpha=0.5` with scalar sigmoid head
- Input tensor: `float32`, shape `[1, 224, 224, 3]`, NHWC
- Preprocessing:
  - decode to RGB
  - direct resize to `224x224` with bilinear interpolation
  - normalize to `[0, 1]`
- Output tensor: `float32`, shape `[1, 1]`
- Output meaning: single scalar aesthetic score in `[0, 1]`, higher is better
- No explanation text is produced by this model

## Key Results

- AADB val vs conservative teacher:
  - MAE `0.028400`
  - RMSE `0.035289`
  - Pearson `0.873829`
  - Spearman `0.858600`
  - Pairwise accuracy `0.90568`
  - Top-k overlap@10 `0.70`
- AADB val vs old `stage5_student_full_aadb` baseline:
  - lower MAE/RMSE
  - higher Pearson/Spearman
  - higher pairwise accuracy
- `test_vila` Keras vs conservative teacher:
  - MAE `0.049526`
  - RMSE `0.063052`
  - Pearson `0.619106`
  - Spearman `0.669796`
  - Pairwise accuracy `0.7917`
  - Top-k overlap@10 `0.30`
- `test_vila` old baseline vs conservative teacher:
  - MAE `0.052225`
  - RMSE `0.065980`
  - Pearson `0.686500`
  - Spearman `0.679082`
  - Pairwise accuracy `0.8015`
  - Top-k overlap@10 `0.60`
- `test_vila` score spread:
  - new student range `0.237463`, std `0.050375`
  - old baseline range `0.206454`, std `0.044533`
  - teacher range `0.257308`, std `0.065644`

## TFLite Verification

- Export mode: builtin ops only
- Quantization: float16
- TFLite size: `1,738,468` bytes
- `test_vila` parity vs Keras:
  - max abs diff `0.002140`
  - mean abs diff `0.000616`
  - p95 abs diff `0.001639`
- `test_vila` TFLite vs conservative teacher:
  - MAE `0.049329`
  - RMSE `0.062967`
  - Pearson `0.620777`
  - Spearman `0.675204`
  - Pairwise accuracy `0.7957`

## Known Weaknesses

- The new student is less compressed than the old mobile student, but `test_vila` ranking quality still does not consistently beat the old `stage5_student_full_aadb`.
- One `test_vila` file is unreadable and was skipped:
  - `test_vila/IMG_20240519_215827_510.jpg`

## Recommendation

- Treat this as the next candidate mobile aesthetic scorer for Flutter integration experiments because it is a single-score, builtin-op TFLite model with better conservative-teacher fidelity on AADB val and a wider score spread on real images.
- Do not treat it as a production replacement for the old mobile scorer yet; real-image ranking generalization still needs another iteration.
