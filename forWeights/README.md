# Aesthetic Weight Lab

Standalone TFLite-based Aesthetic Weight Lab for team testing. Add local images, run one command, then open an offline HTML report to compare NIMA, RGNet, and A-LAMP scores with adjustable browser-side weights.

This folder is self-contained for the team tool. It does not include training code, Flutter code, personal test images, experiment outputs, `.keras` files, or `.h5` files.

## 1. Install Git LFS

Install Git LFS before clone or pull so the TFLite model files download correctly.

```bash
git lfs install
git lfs pull
```

## 2. Install Python Dependencies

From inside `forWeights/`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The runner tries `tflite-runtime` first if it is installed. TensorFlow is kept as the safe default dependency because the runner falls back to `tf.lite.Interpreter`.

## 3. Add Images

Put private test images into:

```text
test_images/
```

Supported defaults are `.jpg`, `.jpeg`, `.png`, and `.webp`.

Do not commit private images. This folder ignores `test_images/*` except `.gitkeep` and `README.md`.

## 4. Run Baseline Config

```bash
python tools/aesthetic_weight_lab/run_aesthetic_weight_lab.py \
  --input_dir test_images \
  --config configs/aesthetic_weight_lab.yaml \
  --output_dir outputs/aesthetic_weight_lab_demo
```

Open:

```text
outputs/aesthetic_weight_lab_demo/report.html
```

## 5. Run Fine-tuned A-LAMP Config

```bash
python tools/aesthetic_weight_lab/run_aesthetic_weight_lab.py \
  --input_dir test_images \
  --config configs/aesthetic_weight_lab_alamp_finetuned.yaml \
  --output_dir outputs/aesthetic_weight_lab_alamp_finetuned_demo
```

Open:

```text
outputs/aesthetic_weight_lab_alamp_finetuned_demo/report.html
```

## 6. Use The Report

The HTML report is portable and works offline as a local file. It shows image names, thumbnails, NIMA score, RGNet score, A-LAMP score, final weighted score, and rank.

Use the NIMA, RGNet, and A-LAMP weight inputs or preset buttons to rerank results in the browser. The report auto-normalizes weights if their sum is not exactly 1, so reranking does not require rerunning models.

## 7. Limits

This tool is for team qualitative testing and weight comparison. Do not treat small private image tests as scientific benchmark results.

The RGNet fine-tuned model is intentionally excluded because it failed TFLite deployment/parity checks. Do not add it until deployment parity is fixed and verified.
