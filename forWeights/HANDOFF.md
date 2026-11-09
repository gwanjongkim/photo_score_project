# Aesthetic Weight Lab Standalone Handoff

## 1. Files Copied

Copied into `forWeights/`.

- `README.md`
- `requirements.txt`
- `.gitignore`
- `.gitattributes`
- `configs/aesthetic_weight_lab.yaml`
- `configs/aesthetic_weight_lab_alamp_finetuned.yaml`
- `tools/aesthetic_weight_lab/run_aesthetic_weight_lab.py`
- `tools/aesthetic_weight_lab/html_report.py`
- `tools/aesthetic_weight_lab/tflite_model_runner.py`
- `tools/aesthetic_weight_lab/model_registry.py`
- `tools/aesthetic_weight_lab/README.md`
- `models/aesthetic/MODEL_CARD.md`
- `models/aesthetic/SHA256SUMS.txt`
- `models/aesthetic/nima_mobile.tflite`
- `models/aesthetic/nima_mobile.metadata.json`
- `models/aesthetic/nima_mobile.verify.json`
- `models/aesthetic/rgnet_aadb_gpu.tflite`
- `models/aesthetic/rgnet_aadb_gpu.metadata.json`
- `models/aesthetic/rgnet_aadb_gpu.verify.json`
- `models/aesthetic/alamp_aadb_gpu.tflite`
- `models/aesthetic/alamp_aadb_gpu.metadata.json`
- `models/aesthetic/alamp_aadb_gpu.verify.json`
- `models/aesthetic/alamp_ava_aadb_finetune.tflite`
- `models/aesthetic/alamp_ava_aadb_finetune.metadata.json`
- `models/aesthetic/alamp_ava_aadb_finetune.verify.json`
- `test_images/.gitkeep`
- `test_images/README.md`

## 2. Files Intentionally Excluded

- `outputs/` from the main repo.
- `test_vila/`.
- Private image files from `test_images/`.
- `tools/aesthetic_weight_lab/__pycache__/`.
- `tools/aesthetic_weight_lab/plan.md`.
- `tools/aesthetic_weight_lab/checklist.md`.
- `tools/aesthetic_weight_lab/context-notes.md`.
- `outputs/ava_pretrain_aadb_finetune_20260507/tflite/rgnet_ava_aadb_finetune.tflite`.
- `outputs/rgnet_tflite_retry_20260507/tflite/*`.
- Any `.keras` files.
- Any `.h5` files.

Validation created local smoke outputs under `forWeights/outputs/`. After validation, those generated outputs and the generated Python `__pycache__/` directory were removed so the standalone folder matches the requested clean upload structure. Future local outputs are ignored by `forWeights/.gitignore` and should not be committed.

## 3. Model Bundle Contents And Sizes

- `models/aesthetic/nima_mobile.tflite` - 24M.
- `models/aesthetic/rgnet_aadb_gpu.tflite` - 25M.
- `models/aesthetic/alamp_aadb_gpu.tflite` - 38M.
- `models/aesthetic/alamp_ava_aadb_finetune.tflite` - 38M.

Metadata and verification JSON sidecars are included for each model. `models/aesthetic/SHA256SUMS.txt` covers all bundled TFLite and sidecar JSON files.

The fine-tuned RGNet TFLite artifacts are excluded because they failed TFLite deployment/parity checks.

## 4. Validation Results

Working directory for validation: `/home/omen_pc1/photo_score_project`.

- Requested command `python -m py_compile ...` could not run because `python` is not installed on PATH in this environment.
- Equivalent compile check with the repo interpreter passed.

```bash
./.venv_gpu/bin/python -m py_compile \
  forWeights/tools/aesthetic_weight_lab/run_aesthetic_weight_lab.py \
  forWeights/tools/aesthetic_weight_lab/html_report.py \
  forWeights/tools/aesthetic_weight_lab/tflite_model_runner.py \
  forWeights/tools/aesthetic_weight_lab/model_registry.py
```

- Checksum validation passed from inside `forWeights/`.

```bash
sha256sum -c models/aesthetic/SHA256SUMS.txt
```

- `forWeights/test_images/` contains only `.gitkeep` and `README.md`.
- No `.keras`, `.h5`, failed fine-tuned RGNet TFLite, `test_vila/`, or copied main-repo `outputs/` files were found in the standalone bundle.
- Created temporary smoke image outside the repo at `/tmp/forWeights_smoke/synthetic_smoke.jpg`.
- Baseline smoke run passed.

```bash
../.venv_gpu/bin/python tools/aesthetic_weight_lab/run_aesthetic_weight_lab.py \
  --input_dir /tmp/forWeights_smoke \
  --config configs/aesthetic_weight_lab.yaml \
  --output_dir outputs/baseline_smoke
```

Baseline smoke outputs exist:

- `outputs/baseline_smoke/report.html`.
- `outputs/baseline_smoke/raw_scores.csv`.
- `outputs/baseline_smoke/raw_scores.json`.

Baseline smoke scores on the synthetic image:

- NIMA: `0.5365433163`.
- RGNet: `0.5130105019`.
- A-LAMP: `0.4784126282`.
- Final: `0.5095943605`.

- Fine-tuned A-LAMP smoke run passed.

```bash
../.venv_gpu/bin/python tools/aesthetic_weight_lab/run_aesthetic_weight_lab.py \
  --input_dir /tmp/forWeights_smoke \
  --config configs/aesthetic_weight_lab_alamp_finetuned.yaml \
  --output_dir outputs/alamp_finetuned_smoke
```

Fine-tuned smoke outputs exist:

- `outputs/alamp_finetuned_smoke/report.html`.
- `outputs/alamp_finetuned_smoke/raw_scores.csv`.
- `outputs/alamp_finetuned_smoke/raw_scores.json`.

Fine-tuned smoke scores on the synthetic image:

- NIMA: `0.5365433163`.
- RGNet: `0.5130105019`.
- A-LAMP: `0.4767650664`.
- Final: `0.5090506651`.

HTML validation found:

- `weight-nima`.
- `weight-rgnet`.
- `weight-alamp`.
- Preset buttons.
- Browser-side `updateScores()` reranking.

Both `raw_scores.json` files validate with `python -m json.tool` using the repo interpreter.

## 5. How To Upload This Folder To GitHub

Upload `forWeights/` as its own repository. Do not upload or commit generated `forWeights/outputs/`; it is ignored and should contain only local validation or scoring output when team members run the tool.

Install Git LFS before committing the model files so `.tflite` files are stored through LFS.

## 6. Exact Git Commands

Run these commands from `/home/omen_pc1/photo_score_project`.

```bash
cd forWeights
git init
git lfs install
git add .gitattributes .gitignore README.md requirements.txt HANDOFF.md
git add configs/
git add tools/
git add models/aesthetic/
git add test_images/.gitkeep test_images/README.md
git status
git commit -m "feat: add aesthetic weight lab"
```

Then add the target GitHub remote and push with your team's normal branch policy.

## 7. Git LFS Reminder

`.gitattributes` contains:

```text
models/**/*.tflite filter=lfs diff=lfs merge=lfs -text
```

Team members should run:

```bash
git lfs install
git lfs pull
```

before using the model bundle after clone or pull.

## 8. Limitations

- This is a qualitative team tool, not an official benchmark package.
- Private tests are not scientific benchmark results.
- The browser report reranks from already-computed raw scores; changing model files still requires rerunning the Python scorer.
- The runner supports the current NIMA distribution, scalar TFLite, and A-LAMP signature contracts.
- TensorFlow is the safe default dependency. `tflite-runtime` can be used if installed and compatible, because the runner tries it before falling back to `tf.lite.Interpreter`.
- RGNet fine-tuned artifacts remain excluded until a deployment/parity pass is locally verified.
