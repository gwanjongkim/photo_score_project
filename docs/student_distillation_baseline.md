# Student Distillation Baseline

This document covers the first-pass student distillation pipeline added for the capstone `photo_score_project` repo.

## Scope

This baseline distills only the stage5 aesthetic teacher aggregate.

It does not try to reproduce:

- technical-quality models
- pairwise reranking
- diversity reranking
- Gemini or VILA explanation layers

The stage5 bundle still produces these teacher-side per-model signals:

- `composition_aadb`
- `nima_ava`
- `alamp_aadb`
- `musiq_aadb`
- `rgnet_aadb`

Second-pass note:

- the first-pass distill code used its own equal-weight five-model fusion
- the current default teacher config now reuses the production `aesthetic_component` aggregation path
- this is intentional because the production A-cut path is the target that the student should match most closely

Current production caveat:

- the production `aesthetic_component` path is defined by selector weights, not by a separate dedicated five-model teacher-fusion file
- that means the current live production aesthetic target can differ from the original equal-weight five-model interpretation
- the new consistency checker makes that difference visible on a real image sample

## Files

Main entrypoints:

- [src/distill/generate_teacher_labels.py](/home/omen_pc1/photo_score_project/src/distill/generate_teacher_labels.py)
- [src/distill/train_student.py](/home/omen_pc1/photo_score_project/src/distill/train_student.py)
- [src/distill/evaluate_student.py](/home/omen_pc1/photo_score_project/src/distill/evaluate_student.py)
- [src/distill/export_student_tflite.py](/home/omen_pc1/photo_score_project/src/distill/export_student_tflite.py)

Default configs:

- [configs/student_distill_teacher.json](/home/omen_pc1/photo_score_project/configs/student_distill_teacher.json)
- [configs/student_distill_baseline.json](/home/omen_pc1/photo_score_project/configs/student_distill_baseline.json)

## 1. Generate Teacher Labels

Before generating a full manifest, verify production-vs-distill agreement on a small sample:

```bash
MPLCONFIGDIR=/tmp/photo_score_project_mplconfig PYTHONPATH=. ./.venv_gpu/bin/python -m src.distill.check_teacher_consistency \
  --input_csv data/processed/aadb/val.csv \
  --output_dir outputs/distill_consistency_smoke \
  --limit 16
```

This writes:

- `teacher_consistency_comparison.csv`
- `teacher_consistency_summary.json`

Use that first to confirm the current teacher config matches the production aggregation path you intend to distill.

Example: build a first baseline manifest from the existing AADB split files.

```bash
PYTHONPATH=. ./.venv_gpu/bin/python -m src.distill.generate_teacher_labels \
  --input_csv data/processed/aadb/train.csv \
  --input_csv data/processed/aadb/val.csv \
  --input_csv data/processed/aadb/test.csv \
  --output_csv data/distill/stage5_aadb_teacher_labels.csv
```

What it writes per labeled image:

- `image_path`
- `split`
- `teacher_aesthetic_score`
- optional teacher model scores: `aadb_score`, `nima_mean_score`, `alamp_score`, `musiq_score`, `rgnet_score`, `pairwise_recovered_score`
- teacher bookkeeping: models used, missing models, stage failures, source manifest

Useful options:

- `--resume`: append only missing images to an existing output CSV
- `--limit N`: quick smoke run on the first `N` rows
- `--allow_partial_teacher_scores`: keep rows with incomplete teacher scores

For the first serious baseline, keep partial scores off so the target stays consistent.

## 2. Train The Student

The current student backbone is MobileNetV2.

The default baseline is a MobileNetV2 student with `alpha=0.5`, a single scalar sigmoid head, and pointwise regression to the teacher aggregate.

```bash
PYTHONPATH=. ./.venv_gpu/bin/python -m src.distill.train_student \
  --config configs/student_distill_baseline.json
```

The default config expects:

- teacher manifest: `data/distill/stage5_aadb_teacher_labels.csv`
- train split: `train`
- val split: `val`
- test split: `test`

Artifacts are written under the configured checkpoint directory, including:

- `best.weights.h5`
- `final_model.keras`
- `saved_model/`
- `resolved_config.json`
- `history.json`
- `evaluation/val_metrics.json`
- `evaluation/val_predictions.csv`

The optional pairwise ranking scaffold exists in the config, but it is off by default.

Training now computes ranking-oriented validation metrics each epoch and checkpoints by default on `val_spearman`, while learning-rate reduction still follows `val_loss`.

## 3. Evaluate

Run evaluation again later, or on different splits, without retraining:

```bash
PYTHONPATH=. ./.venv_gpu/bin/python -m src.distill.evaluate_student \
  --checkpoint_dir checkpoints/stage5_student_aadb_baseline \
  --splits val,test
```

Evaluation outputs include:

- regression metrics: `mae`, `rmse`, `mse`, `r2`, `pearson`, `spearman`
- ranking-oriented metrics: sampled pairwise accuracy and top-k overlap
- per-image prediction CSVs for direct teacher-vs-student comparison

## 4. Export Path

A concrete export path is included:

```bash
PYTHONPATH=. ./.venv_gpu/bin/python -m src.distill.export_student_tflite \
  --checkpoint_dir checkpoints/stage5_student_aadb_baseline \
  --quantization float16
```

This exports a TFLite file from the rebuilt MobileNetV2 student architecture plus `best.weights.h5`.

Current export assumptions:

- input size: `224 x 224`
- color: RGB
- layout: NHWC
- dtype: `float32`
- normalization: `pixel_value / 255.0`
- output: scalar teacher-style aesthetic score in `[0, 1]`

Supported export modes:

- `--quantization no_quant`
- `--quantization float16`

The current practical quantized path is post-training float16 quantization. Full QAT is not part of this patch.

## What Remains Before Product TFLite Deployment

- verify numeric parity between Keras and TFLite on a held-out image set
- benchmark latency and memory on the target Flutter device tier
- decide whether `no_quant` or `float16` is the product default
- choose the final training corpus beyond the first baseline AADB manifest
- if production aesthetic aggregation changes later, rerun the consistency checker and regenerate teacher labels
