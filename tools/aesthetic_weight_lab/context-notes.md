# Aesthetic Weight Lab Context Notes

- 2026-05-08: The working tree already had unrelated modified and untracked files, including `src/` changes. This task will not revert or edit those files.
- 2026-05-08: The new tool is intentionally placed under `tools/aesthetic_weight_lab/` and avoids runtime imports from `src/` so training/model code is not touched.
- 2026-05-08: Baseline TFLite artifacts exist under `exports/tflite/`: `nima_mobile.tflite`, `rgnet_aadb_gpu.tflite`, and `alamp_aadb_gpu.tflite`, with metadata and verification JSON files.
- 2026-05-08: The fine-tuned A-LAMP candidate exists at `outputs/ava_pretrain_aadb_finetune_20260507/tflite/alamp_ava_aadb_finetune.tflite` with metadata and verification JSON files.
- 2026-05-08: Fine-tuned RGNet artifacts also exist locally, but they are excluded because prior parity/deployment checks failed and the task explicitly forbids bundling them.
- 2026-05-08: The A-LAMP preprocessing will mirror the repo's deterministic approach: global resize-with-pad, saliency proposal from edge/luminance variance/color variance, non-max overlap reduction, and fixed-anchor fallback.
- 2026-05-08: Initial smoke run showed this environment does not expose `Interpreter` via `from tensorflow.lite import Interpreter`; the fallback now imports TensorFlow and returns `tf.lite.Interpreter`.
- 2026-05-08: `requirements.txt` now includes `PyYAML>=6.0` because the team-facing config files are YAML and a fresh install should not rely on incidental transitive packages.
