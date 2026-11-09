# Aesthetic Weight Lab

TFLite-only local tool for comparing NIMA, RGNet, and A-LAMP aesthetic scores on private team images. It writes raw score files and a portable HTML report where weights can be adjusted in the browser without rerunning models.

## Setup

Install Git LFS before clone or pull so `models/**/*.tflite` files are downloaded correctly.

```bash
git lfs install
git lfs pull
```

Install the repo Python dependencies in the environment you use for local scoring. This repo already uses TensorFlow Lite through TensorFlow in `requirements.txt`.

## Add Images

Put private test images in:

```text
test_images/
```

Supported extensions by default are `jpg`, `jpeg`, `png`, and `webp`.

Do not commit private images. The repo ignores `test_images/*` except `test_images/.gitkeep` and `test_images/README.md`.

## Run Baseline Config

```bash
PYTHONPATH=. ./.venv_gpu/bin/python tools/aesthetic_weight_lab/run_aesthetic_weight_lab.py \
  --input_dir test_images \
  --config configs/aesthetic_weight_lab.yaml \
  --output_dir outputs/aesthetic_weight_lab_demo
```

Open:

```text
outputs/aesthetic_weight_lab_demo/report.html
```

The output folder also contains:

- `raw_scores.csv`
- `raw_scores.json`
- `weight_presets.json`
- `command_log.txt`
- `thumbs/`
- `copied_images/` when enabled by config.

## Run Fine-tuned A-LAMP Candidate

```bash
PYTHONPATH=. ./.venv_gpu/bin/python tools/aesthetic_weight_lab/run_aesthetic_weight_lab.py \
  --input_dir test_images \
  --config configs/aesthetic_weight_lab_alamp_finetuned.yaml \
  --output_dir outputs/aesthetic_weight_lab_alamp_finetuned_demo
```

This config keeps the baseline NIMA and RGNet models and swaps only A-LAMP to `models/aesthetic/alamp_ava_aadb_finetune.tflite`.

## Browser Weight Lab

Open `report.html` locally. Use the NIMA, RGNet, and A-LAMP weight inputs or preset buttons to rerank results in the browser. The report auto-normalizes weights if their sum is not exactly 1.

## Model Swaps

To swap model files, edit the YAML config. For a new model type, add the model type metadata to `tools/aesthetic_weight_lab/model_registry.py` and add the corresponding runner behavior only if the model has a new input/output contract.

## Important Limits

This tool is for team qualitative scoring and weight comparison. Small private folders are not scientific benchmark results.

The RGNet AVA-to-AADB fine-tuned TFLite model is intentionally excluded because it failed TFLite deployment/parity checks. Do not add it to this bundle until parity is fixed and verified.

This tool does not modify training code, Flutter code, or `src/`.
