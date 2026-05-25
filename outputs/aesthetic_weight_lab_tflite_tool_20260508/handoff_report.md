# Aesthetic Weight Lab TFLite Tool Handoff

## 1. Files Created

- `.gitattributes`
- `configs/aesthetic_weight_lab.yaml`
- `configs/aesthetic_weight_lab_alamp_finetuned.yaml`
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
- `tools/aesthetic_weight_lab/run_aesthetic_weight_lab.py`
- `tools/aesthetic_weight_lab/html_report.py`
- `tools/aesthetic_weight_lab/tflite_model_runner.py`
- `tools/aesthetic_weight_lab/model_registry.py`
- `tools/aesthetic_weight_lab/README.md`
- `tools/aesthetic_weight_lab/plan.md`
- `tools/aesthetic_weight_lab/checklist.md`
- `tools/aesthetic_weight_lab/context-notes.md`
- `outputs/aesthetic_weight_lab_tflite_tool_20260508/baseline_smoke/*`
- `outputs/aesthetic_weight_lab_tflite_tool_20260508/alamp_finetuned_smoke/*`

## 2. Files Modified

- `.gitignore`
  - Keeps `outputs/` ignored.
  - Unignores only the team-shareable `models/aesthetic/` bundle.
  - Ignores private `test_images/*` while keeping `.gitkeep` and `README.md`.
- `requirements.txt`
  - Adds `PyYAML>=6.0` for YAML config loading.

No training code, Flutter code, or `src/` files were modified.

## 3. Model Bundle

Included under `models/aesthetic/`.

- `nima_mobile.tflite` from `exports/tflite/nima_mobile.tflite`, about 24 MB.
- `rgnet_aadb_gpu.tflite` from `exports/tflite/rgnet_aadb_gpu.tflite`, about 25 MB.
- `alamp_aadb_gpu.tflite` from `exports/tflite/alamp_aadb_gpu.tflite`, about 38 MB.
- `alamp_ava_aadb_finetune.tflite` from `outputs/ava_pretrain_aadb_finetune_20260507/tflite/alamp_ava_aadb_finetune.tflite`, about 38 MB.

Sidecar metadata and verify JSON files were copied where available. `models/aesthetic/SHA256SUMS.txt` verifies all bundled TFLite and sidecar JSON files.

Excluded intentionally.

- `outputs/ava_pretrain_aadb_finetune_20260507/tflite/rgnet_ava_aadb_finetune.tflite`
- `outputs/rgnet_tflite_retry_20260507/tflite/*`
- Any `.keras` or `.h5` files.

## 4. Configs

- `configs/aesthetic_weight_lab.yaml`
  - NIMA: `models/aesthetic/nima_mobile.tflite`
  - RGNet: `models/aesthetic/rgnet_aadb_gpu.tflite`
  - A-LAMP: `models/aesthetic/alamp_aadb_gpu.tflite`
  - Weights: NIMA `0.34`, RGNet `0.33`, A-LAMP `0.33`
- `configs/aesthetic_weight_lab_alamp_finetuned.yaml`
  - Same NIMA and RGNet baseline.
  - A-LAMP swapped to `models/aesthetic/alamp_ava_aadb_finetune.tflite`.

## 5. How Team Members Use It

1. Install Git LFS before clone or pull.
2. Put private images into `test_images/`.
3. Run the baseline command.

```bash
PYTHONPATH=. ./.venv_gpu/bin/python tools/aesthetic_weight_lab/run_aesthetic_weight_lab.py \
  --input_dir test_images \
  --config configs/aesthetic_weight_lab.yaml \
  --output_dir outputs/aesthetic_weight_lab_demo
```

4. Open `outputs/aesthetic_weight_lab_demo/report.html`.
5. Adjust NIMA, RGNet, and A-LAMP weights in the browser. The report reranks without rerunning models.

Fine-tuned A-LAMP comparison.

```bash
PYTHONPATH=. ./.venv_gpu/bin/python tools/aesthetic_weight_lab/run_aesthetic_weight_lab.py \
  --input_dir test_images \
  --config configs/aesthetic_weight_lab_alamp_finetuned.yaml \
  --output_dir outputs/aesthetic_weight_lab_alamp_finetuned_demo
```

## 6. Validation Results

- `./.venv_gpu/bin/python -m py_compile tools/aesthetic_weight_lab/*.py` passed.
- Model files exist under `models/aesthetic/`.
- `sha256sum -c models/aesthetic/SHA256SUMS.txt` passed.
- `test_images/` had no private image samples, only `.gitkeep` and `README.md`.
- Created a synthetic smoke image at `/tmp/aesthetic_weight_lab_smoke/synthetic_smoke.jpg`.
- Baseline smoke run passed.
  - Output: `outputs/aesthetic_weight_lab_tflite_tool_20260508/baseline_smoke/report.html`
  - `raw_scores.csv` and `raw_scores.json` exist.
  - Scores on synthetic sample: NIMA `0.5375546349`, RGNet `0.5119963884`, A-LAMP `0.4750898778`, final `0.5085070438`.
- Fine-tuned A-LAMP smoke run passed.
  - Output: `outputs/aesthetic_weight_lab_tflite_tool_20260508/alamp_finetuned_smoke/report.html`
  - `raw_scores.csv` and `raw_scores.json` exist.
  - Scores on synthetic sample: NIMA `0.5375546349`, RGNet `0.5119963884`, A-LAMP `0.4771580994`, final `0.5091895569`.
- Both HTML reports contain `weight-nima`, `weight-rgnet`, `weight-alamp`, preset buttons, and browser-side `updateScores()`.
- `raw_scores.json` validates with `python -m json.tool`.
- No personal images were copied into tracked repo paths. Smoke copied only the synthetic `/tmp` image into ignored `outputs/` folders.

## 7. Git LFS / GitHub Notes

- `.gitattributes` tracks `models/**/*.tflite` through Git LFS.
- `outputs/` remains ignored and should not be committed.
- Private `test_images/*` remains ignored.
- Suggested files to add are the new tool/config/model/test-image scaffold files plus `.gitignore`, `.gitattributes`, and `requirements.txt`.
- Do not add `outputs/aesthetic_weight_lab_tflite_tool_20260508/`; it is a handoff and smoke-output folder.

## 8. Limitations

- The tool is for qualitative team review and weight comparison, not a scientific benchmark claim.
- Small private image folders should not be treated as product-quality evidence.
- A-LAMP preprocessing is a deterministic NumPy/Pillow implementation matching the repo's documented approach: resize-with-pad global view, adaptive saliency-like patches, non-max overlap reduction, and fixed-anchor fallback.
- The CLI refuses to write into a non-empty output directory to avoid overwriting experiment outputs.
- Current runner supports `nima_distribution`, `scalar_tflite`, and `alamp_signature` model contracts.

## 9. Next Steps

- Team installs Git LFS and confirms the TFLite pointers resolve after clone.
- Team runs the baseline and fine-tuned A-LAMP configs on private image folders.
- Reviewers compare rank shifts in `report.html` and inspect `raw_scores.json` for patch boxes and NIMA distributions.
- Add future model types by extending `tools/aesthetic_weight_lab/model_registry.py` and the runner only when the input/output contract changes.
