# RGNet/A-LAMP AVA Retraining Experiment Report

## 1. Environment
- GPU: `nvidia-smi` showed NVIDIA GeForce RTX 4070 family, driver `591.55`, CUDA `13.1`. TensorFlow GPU was available only in escalated training commands; non-escalated evaluation/export ran on CPU and logged `cuInit UNKNOWN ERROR (100)`.
- Python: `./.venv_gpu/bin/python --version` -> `Python 3.12.3`.
- Project path: `/home/omen_pc1/photo_score_project`.
- Git status: dirty before/after this experiment with pre-existing modified/untracked files. No source code changes were made for this task.
- Dataset counts: AVA train `204402`, val `25551`, test `25551`; AADB fixed validation subset `512`.
- Output directory: `outputs/ava_retrain_rgnet_alamp_20260506/`.

## 2. Normalized AVA Manifest
Created:
- `data/processed/ava/train_unit.csv`
- `data/processed/ava/val_unit.csv`
- `data/processed/ava/test_unit.csv`

Formula: `aesthetic_unit_score = (mean_score - 1.0) / 9.0`.

Validation:
- `train_unit.csv`: `204402` rows, range `0.08988764..0.84126978`, missing `image_path`: `0`, first sample path exists: `True`.
- `val_unit.csv`: `25551` rows, range `0.12677108..0.84444444`, missing `image_path`: `0`, first sample path exists: `True`.
- `test_unit.csv`: `25551` rows, range `0.10986267..0.78784222`, missing `image_path`: `0`, first sample path exists: `True`.

Sample subsets created under `outputs/ava_retrain_rgnet_alamp_20260506/subsets/` for controlled baseline/post-train comparisons.

## 3. Baseline Results
Baseline fixed-subset results:

| Model | Dataset | Split | SRCC | PLCC | MAE | RMSE | Notes |
|---|---|---|---:|---:|---:|---:|---|
| RGNet AADB | AADB | AVA val512 | 0.1788 | 0.1930 | 0.1023 | 0.1260 | current AADB checkpoint |
| A-LAMP AADB | AADB | AVA val512 | 0.2894 | 0.2998 | 0.0973 | 0.1236 | current AADB checkpoint |
| RGNet AADB | AADB | AADB val512 | 0.4987 | 0.5094 | 0.1304 | 0.1619 | current AADB checkpoint |
| A-LAMP AADB | AADB | AADB val512 | 0.5843 | 0.5859 | 0.1292 | 0.1594 | current AADB checkpoint |

## 4. RGNet AVA Training
Command:

```bash
PYTHONPATH=. ./.venv_gpu/bin/python -u -m src.train.train_rgnet \
  --train_csv data/processed/ava/train_unit.csv \
  --val_csv data/processed/ava/val_unit.csv \
  --target_col aesthetic_unit_score \
  --image_size 256 \
  --batch_size 16 \
  --epochs 10 \
  --out_dir outputs/ava_retrain_rgnet_alamp_20260506/rgnet_ava_unit
```

Result:
- Output path: `outputs/ava_retrain_rgnet_alamp_20260506/rgnet_ava_unit/`.
- Saved `best.weights.h5`, `final_model.keras`, and `saved_model/`.
- Training stopped after epoch 5 because `src/train/train_rgnet.py:71-75` configures `EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)`.
- Best logged full-validation loss was epoch 2: `val_loss=0.0061`, `val_mae=0.0610`.

Fixed-subset validation:
- AVA val512: SRCC `0.5613`, PLCC `0.5697`, MAE `0.0599`, RMSE `0.0772`.
- AADB val512: SRCC `0.2887`, PLCC `0.3267`, MAE `0.1630`, RMSE `0.1990`.

## 5. A-LAMP AVA Training
Command:

```bash
PYTHONPATH=. ./.venv_gpu/bin/python -u -m src.train.train_alamp \
  --train_csv data/processed/ava/train_unit.csv \
  --val_csv data/processed/ava/val_unit.csv \
  --target_col aesthetic_unit_score \
  --batch_size 16 \
  --epochs 10 \
  --global_size 384 \
  --patch_size 224 \
  --num_patches 5 \
  --out_dir outputs/ava_retrain_rgnet_alamp_20260506/alamp_ava_unit
```

Result:
- Output path: `outputs/ava_retrain_rgnet_alamp_20260506/alamp_ava_unit/`.
- Saved `best.weights.h5`, `final_model.keras`, and `saved_model/`.
- Training stopped after epoch 4 because `src/train/train_alamp.py:84` configures `EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)`.
- Best logged full-validation loss was epoch 1: `val_loss=0.0047`, `val_mae=0.0536`.
- Training log included TensorFlow warning `Corrupt JPEG data: 451 extraneous bytes before marker 0xd9`; training completed.

Fixed-subset validation:
- AVA val512: SRCC `0.5548`, PLCC `0.5566`, MAE `0.0530`, RMSE `0.0686`.
- AADB val512: SRCC `0.3438`, PLCC `0.3628`, MAE `0.1468`, RMSE `0.1808`.

## 6. TFLite Export And Parity
Exported files:
- `outputs/ava_retrain_rgnet_alamp_20260506/tflite/rgnet_ava_unit.tflite` (`26050568` bytes)
- `outputs/ava_retrain_rgnet_alamp_20260506/tflite/rgnet_ava_unit.metadata.json`
- `outputs/ava_retrain_rgnet_alamp_20260506/tflite/rgnet_ava_unit.verify.json`
- `outputs/ava_retrain_rgnet_alamp_20260506/tflite/alamp_ava_unit.tflite` (`39098664` bytes)
- `outputs/ava_retrain_rgnet_alamp_20260506/tflite/alamp_ava_unit.metadata.json`
- `outputs/ava_retrain_rgnet_alamp_20260506/tflite/alamp_ava_unit.verify.json`

RGNet TFLite:
- Input: `[1, 256, 256, 3]`, `float32`, no quantization.
- Output: `[1, 1]`, `float32`, no quantization.
- Verifier result: failed. Smoke image TFLite score `0.4052223265`, Keras score `0.3305336833`, max absolute diff `0.0746886432`.
- AVA val512 TFLite metrics: SRCC `0.4992`, PLCC `0.4946`, MAE `0.0805`.

A-LAMP TFLite:
- Inputs: global `[1, 384, 384, 3]` and patches `[1, 5, 224, 224, 3]`, `float32`, no quantization.
- Output: `[1, 1]`, `float32`, no quantization.
- Verifier result: passed. TFLite vs float32 rebuild max diff `0.0000031590`; rebuild vs mixed-precision checkpoint max diff `0.0052133203`.
- AVA val512 TFLite metrics: SRCC `0.5136`, PLCC `0.5257`, MAE `0.0562`.

## 7. Comparison
| Model | Dataset | Split | SRCC | PLCC | MAE | RMSE | TFLite parity | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| RGNet AADB | AADB | AVA val512 | 0.1788 | 0.1930 | 0.1023 | 0.1260 | not applicable | baseline current model |
| RGNet AVA | AVA unit | AVA val512 | 0.5613 | 0.5697 | 0.0599 | 0.0772 | not applicable | Keras; improves AVA subset |
| RGNet AVA TFLite | AVA unit | AVA val512 | 0.4992 | 0.4946 | 0.0805 | 0.1019 | fail | TFLite parity failed on verifier |
| RGNet AADB | AADB | AADB val512 | 0.4987 | 0.5094 | 0.1304 | 0.1619 | not applicable | baseline current model |
| RGNet AVA | AVA unit | AADB val512 | 0.2887 | 0.3267 | 0.1630 | 0.1990 | not applicable | worse on AADB subset |
| A-LAMP AADB | AADB | AVA val512 | 0.2894 | 0.2998 | 0.0973 | 0.1236 | not applicable | baseline current model |
| A-LAMP AVA | AVA unit | AVA val512 | 0.5548 | 0.5566 | 0.0530 | 0.0686 | not applicable | Keras; improves AVA subset |
| A-LAMP AVA TFLite | AVA unit | AVA val512 | 0.5136 | 0.5257 | 0.0562 | 0.0724 | pass | TFLite verifier passed |
| A-LAMP AADB | AADB | AADB val512 | 0.5843 | 0.5859 | 0.1292 | 0.1594 | not applicable | baseline current model |
| A-LAMP AVA | AVA unit | AADB val512 | 0.3438 | 0.3628 | 0.1468 | 0.1808 | not applicable | worse on AADB subset |

Interpretation from local evidence only:
- On the fixed AVA val512 subset, both AVA-trained Keras models improved over their current AADB-trained baselines.
- On the fixed AADB val512 subset, both AVA-trained Keras models were worse than their current AADB-trained baselines.
- The RGNet AVA TFLite artifact is not deployment-ready because its verifier failed numeric parity.
- The A-LAMP AVA TFLite artifact passed verifier parity but still regressed on the AADB subset at the Keras level.

Current ensemble and Stage 5 student:
- `tools/run_aesthetic_ensemble_experiment.py:49-51` describes a folder-scoring comparison tool, not a direct command for evaluating fixed AVA/AADB label CSVs. Current ensemble was not run in this experiment.
- `tools/run_aesthetic_ensemble_experiment.py:116-126` and `:173-174` show the local weighted ensemble formula and normalization notes, but those were not recomputed with the new AVA checkpoints here.
- `docs/pozy_app_stage5_aesthetic_student_handoff.md:42-52` reports separate Stage 5 student candidate metrics against `teacher_aesthetic_score`, not against AVA scalar labels. Stage 5 student was not run on the fixed AVA/AADB subsets here.

## 8. Decision
- RGNet AVA Keras: better than current model on AVA val512; worse than current model on AADB val512; direct Flutter deployment failed due RGNet TFLite parity failure.
- RGNet AVA TFLite: failed for deployment parity in this experiment.
- A-LAMP AVA Keras: better than current model on AVA val512; worse than current model on AADB val512.
- A-LAMP AVA TFLite: export and parity passed, but replacement is inconclusive because AADB subset performance regressed.

Final classification:
- RGNet AVA: `inconclusive` for replacing current model; `failed` for immediate TFLite deployment until parity is fixed.
- A-LAMP AVA: `inconclusive`; useful AVA-domain candidate, not proven better for current app domain.

## 9. Recommendation
Do not copy these new models into Flutter yet.

Recommended next step: try AVA pretrain plus AADB fine-tune for RGNet and A-LAMP, then repeat the same fixed-subset comparison and TFLite parity checks. Direct AVA-only training is locally proven to improve AVA validation subset metrics, but it is also locally proven to regress on the AADB validation subset used here.

For mobile deployment, the safest path from this experiment is:
1. Keep current AADB models or the current app aesthetic path until replacement is validated on the app target domain.
2. Fix RGNet TFLite parity before considering `rgnet_ava_unit.tflite`.
3. Treat `alamp_ava_unit.tflite` as an AVA-domain candidate only, not as a direct app replacement.
4. Evaluate Stage 5 student separately on the same photo-selection target if it is being considered for Flutter, because local Stage 5 evidence is against a teacher target, not AVA normalized mean labels.
