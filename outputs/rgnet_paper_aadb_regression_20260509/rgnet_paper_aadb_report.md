# RGNet Paper-Oriented AADB Regression Report

## 1. Environment
- Project path: `/home/omen_pc1/photo_score_project`.
- Host/user/date: `DESKTOP-3T40JVV`, `omen_pc1`, `Sat May 9 18:08:01 KST 2026`.
- Python: `./.venv_gpu/bin/python --version` -> `Python 3.12.3`.
- GPU inventory: `nvidia-smi` sees an NVIDIA GeForce RTX 4070 SUPER with 12282 MiB total memory.
- TensorFlow smoke runtime: no GPU was visible inside the sandbox and TensorFlow logged `cuInit UNKNOWN ERROR (100)`, so the smoke run completed on CPU.
- Worktree note: unrelated modified/untracked files existed before this task. This track added isolated paper experiment files only.

## 2. Dataset
- Train CSV: `data/processed/aadb/train.csv`, 7612 rows.
- Validation CSV: `data/processed/aadb/val.csv`, 846 rows.
- Test CSV: `data/processed/aadb/test.csv`, 1000 rows.
- Image directory: `data/raw/aadb/images`.
- Image column: `image_path`.
- Target column: `score`.
- Target scale: already normalized to `[0, 1]`.

## 3. Model Architecture
Implemented file: `src/models/rgnet_paper.py`.

This is `RGNet-paper-v0 approximation`, not an official paper reproduction.

Implemented:
- DenseNet121 fully convolutional backbone, with ImageNet weights attempted when requested.
- DenseNet feature-map spatial positions treated as region nodes.
- 1x1 region feature projection.
- Cosine-similarity dense adjacency over region nodes.
- Row-wise softmax-normalized adjacency.
- Residual graph convolution blocks.
- Sigmoid scalar regression head for AADB aesthetic score in `[0, 1]`.

Approximated:
- Exact paper RegionGraph construction was not locally available with enough implementation detail.
- This v0 approximation uses spatial feature positions as nodes, not author-provided region definitions or official code.
- No official RGNet paper weights are used or claimed.

## 4. Training Setup
Implemented file: `src/train/train_rgnet_paper_aadb.py`.

Default/configured setup:
- Loss: MSE.
- Metric: MAE.
- Input size: 256.
- Default batch size: 8.
- Early stopping: `val_loss`, patience 3, `restore_best_weights=True`.
- Saves: `final_model.keras`, `best.weights.h5`, `training_history.csv`, `train_summary.json`.
- Config: `configs/paper_benchmarks/rgnet_paper_aadb_regression.yaml`.

Smoke command used random DenseNet weights to avoid depending on an external ImageNet download:
```bash
./.venv_gpu/bin/python -u -m src.train.train_rgnet_paper_aadb \
  --config configs/paper_benchmarks/rgnet_paper_aadb_regression.yaml \
  --out_dir outputs/rgnet_paper_aadb_regression_20260509/smoke_train_v2 \
  --backbone_weights none \
  --epochs 1 \
  --batch_size 2 \
  --max_train_samples 8 \
  --max_val_samples 4
```

## 5. Smoke Test Result
Passed.

- Model built at input shape `[2, 256, 256, 3]`.
- Forward output shape: `[2, 1]`.
- Smoke train completed 1 epoch on 8 training samples and 4 validation samples.
- Saved model path: `outputs/rgnet_paper_aadb_regression_20260509/smoke_train_v2/final_model.keras`.
- Best weights path: `outputs/rgnet_paper_aadb_regression_20260509/smoke_train_v2/best.weights.h5`.
- Training history: `outputs/rgnet_paper_aadb_regression_20260509/smoke_train_v2/training_history.csv`.
- Train summary: `outputs/rgnet_paper_aadb_regression_20260509/smoke_train_v2/train_summary.json`.
- Save/load verification max abs diff: `0.0`.
- Smoke validation loss/MAE: `0.03703468665480614` / `0.17081350088119507`.

## 6. Training Result
Only smoke training was run in this pass. Full DenseNet121 AADB training was not launched because it is materially more expensive and the requested smoke gate came first.

Controlled full-training command:
```bash
./.venv_gpu/bin/python -u -m src.train.train_rgnet_paper_aadb \
  --config configs/paper_benchmarks/rgnet_paper_aadb_regression.yaml \
  --out_dir outputs/rgnet_paper_aadb_regression_20260509/full_train \
  --epochs 20 \
  --batch_size 8 \
  --backbone_weights imagenet
```

If ImageNet weights are unavailable locally, the model builder catches that and falls back to random initialization while documenting the request in `train_summary.json`.

## 7. Evaluation Result
Implemented file: `src/eval/evaluate_rgnet_paper_aadb.py`.

Smoke evaluation command:
```bash
./.venv_gpu/bin/python -u -m src.eval.evaluate_rgnet_paper_aadb \
  --config configs/paper_benchmarks/rgnet_paper_aadb_regression.yaml \
  --model_path outputs/rgnet_paper_aadb_regression_20260509/smoke_train_v2/final_model.keras \
  --output_dir outputs/rgnet_paper_aadb_regression_20260509/smoke_eval_v2 \
  --batch_size 2 \
  --max_test_samples 8 \
  --max_val_samples 8
```

Smoke metrics are only a script/functionality check, not a performance claim.

| Split | Samples | SRCC | PLCC | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Test smoke subset | 8 | -0.2036 | -0.3124 | 0.1755 | 0.2052 |
| Val smoke subset | 8 | 0.1437 | 0.1589 | 0.1750 | 0.1972 |

Artifacts:
- `outputs/rgnet_paper_aadb_regression_20260509/smoke_eval_v2/evaluation_summary.json`.
- `outputs/rgnet_paper_aadb_regression_20260509/smoke_eval_v2/test_predictions.csv`.
- `outputs/rgnet_paper_aadb_regression_20260509/smoke_eval_v2/val_predictions.csv`.

## 8. Comparison Against Existing Practical RGNet
The new paper-oriented smoke run is not comparable to the existing practical RGNet metrics because it used tiny subsets and random backbone weights.

Existing local audit numbers:

| Model | Eval Split | SRCC | PLCC | MAE | RMSE | Source |
|---|---|---:|---:|---:|---:|---|
| Practical RGNet AADB baseline | AADB val512 | 0.4987 | 0.5094 | 0.1304 | 0.1619 | `outputs/ava_retrain_rgnet_alamp_20260506/ava_retrain_report.md` |
| Practical RGNet AVA->AADB fine-tuned | AADB val512 | 0.6749 | 0.6835 | 0.1104 | 0.1381 | `outputs/ava_pretrain_aadb_finetune_20260507/finetune_report.md` |

No Flutter code was modified, and nothing was copied into `forWeights`.

## 9. Paper Comparability
Matches the requested paper-oriented direction:
- DenseNet121 backbone instead of the practical EfficientNetV2B0 approximation.
- Fully convolutional feature extraction.
- Region/node extraction from a spatial feature map.
- Region similarity graph construction.
- Graph convolution over region nodes.
- AADB regression with MSE and MAE.

Does not establish official paper reproduction:
- RegionGraph exact details are approximated.
- No author code or official weights are used.
- The full AADB training/evaluation protocol has not yet been run.
- The smoke metrics are not paper-comparable results.

## 10. Next Steps
1. Run the controlled full AADB command above with GPU access.
2. Evaluate `full_train/final_model.keras` on the full AADB test split and optional validation split.
3. Compare full-run metrics against the practical RGNet baselines using the same split sizes.
4. Keep this track separate from Flutter and practical model deployment unless a later task explicitly asks for integration.
