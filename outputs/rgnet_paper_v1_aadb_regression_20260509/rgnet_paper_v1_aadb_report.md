# RGNet Paper-Oriented v1 AADB Regression Report

## 1. Environment
- `pwd`: `/home/omen_pc1/photo_score_project`.
- Date: `Sat May 9 19:43:00 KST 2026`.
- Python: `./.venv_gpu/bin/python --version` -> `Python 3.12.3`.
- TensorFlow: `2.20.0`.
- GPU inventory: `nvidia-smi` saw NVIDIA GeForce RTX 4070 SUPER with 12282 MiB total memory.
- GPU visibility: non-escalated TensorFlow runs saw no GPU and logged `cuInit UNKNOWN ERROR (100)`; escalated mid/full train/eval saw `PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')`.
- Git status summary: v1 source/config files were new. Existing untracked `src/models/rgnet.py` and `forWeights/` were present before this task and were not touched.

## 2. Baseline Context
Existing RGNet-paper-v0 full AADB test result:
- SRCC `0.6433`.
- PLCC `0.6499`.
- MAE `0.1255`.
- RMSE `0.1557`.

Existing practical RGNet references:
- Practical RGNet AADB baseline on AADB val512: SRCC `0.4987`, PLCC `0.5094`, MAE `0.1304`, RMSE `0.1619`.
- Practical RGNet AVA->AADB fine-tuned on AADB val512: SRCC `0.6749`, PLCC `0.6835`, MAE `0.1104`, RMSE `0.1381`.

v1 was tested because v0 had only a DenseNet121 feature map, cosine adjacency, residual graph convolution, and a scalar image-level head. v1 adds more paper-aligned context and region-score behavior while remaining an approximation.

## 3. Dataset
- Train: `data/processed/aadb/train.csv`, 7612 rows, score range `0.0..1.0`.
- Validation: `data/processed/aadb/val.csv`, 846 rows, score range `0.0..0.95`.
- Test: `data/processed/aadb/test.csv`, 1000 rows, score range `0.05..1.0`.
- Image directory: `data/raw/aadb/images`.
- Image column: `image_path`.
- Target column: `score`.

## 4. Architecture
Implemented v1 files:
- `src/models/rgnet_paper_v1.py`.
- `src/train/train_rgnet_paper_v1_aadb.py`.
- `src/eval/evaluate_rgnet_paper_v1_aadb.py`.
- `configs/paper_benchmarks/rgnet_paper_v1_aadb_regression.yaml`.

Architecture:
- DenseNet121 fully convolutional backbone.
- ASPP approximation after the DenseNet feature map.
- Dilation rates: `[1, 3, 6, 12, 18]`.
- Spatial positions from the post-ASPP feature map are region nodes.
- Node feature dimension: `256`.
- Cosine-similarity adjacency with temperature `0.25`.
- Row-wise softmax adjacency.
- Residual graph convolution blocks: `3`.
- Region-level sigmoid scalar score head.
- Aggregation: Log-Sum-Exp with `r=4`.

Implemented LSE formula:
`(1 / r) * log(mean(exp(r * region_scores)))`.

Sigmoid is applied before aggregation. This keeps each region score in `[0,1]`; mean, max, and LSE aggregation then remain bounded in the AADB target range. This is an implementation choice, not a confirmed official paper detail.

Paper-aligned changes:
- More context modeling than v0 through ASPP.
- Region-level score prediction instead of only a scalar fused image head.
- Three graph blocks by default.
- LSE-style score aggregation.

Still approximated:
- ASPP is not exact DenseASPP.
- RegionGraph uses spatial feature-map positions, not official region proposals or author code.
- No official paper weights are used.

## 5. Smoke Test
Pure forward smoke:
- Input: `[2, 256, 256, 3]`.
- Output: `[2, 1]`.
- Prediction range: `0.35727399587631226..0.3581896424293518`.
- All finite: `true`.
- In `[0,1]`: `true`.

Smoke training:
- Output: `outputs/rgnet_paper_v1_aadb_regression_20260509/smoke_train_v2/`.
- Samples: 16 train, 8 validation.
- Epochs: 1.
- Batch size: 2.
- Backbone weights: `none`.
- Best val_loss/val_mae: `0.17236243188381195` / `0.3674493432044983`.
- Save/load max abs diff: `0.0`.

Smoke eval:

| Split | Samples | SRCC | PLCC | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Test smoke | 8 | -0.1557 | -0.1882 | 0.4322 | 0.4571 |
| Val smoke | 8 | -0.8264 | -0.6750 | 0.3674 | 0.4152 |

Smoke metrics are functionality checks only.

## 6. Mid Run
Mid run completed.

- Output: `outputs/rgnet_paper_v1_aadb_regression_20260509/mid_train/`.
- Samples: 512 train, 128 validation.
- Epochs: 3.
- Batch size: 8.
- Backbone weights: ImageNet.
- GPU: visible in escalated run.
- Save/load max abs diff: `0.0`.
- Memory notes: TensorFlow logged allocator memory-pressure warnings at batch size 8, but no fatal OOM occurred.

Loss trend:
- Train loss: `0.0350702591`, `0.0184073560`, `0.0111448662`.
- Val loss: `0.0361035652`, `0.0359326415`, `0.0328521505`.
- Val MAE: `0.1554121077`, `0.1518580168`, `0.1411833614`.

## 7. Full Training
Full AADB training completed.

- Output: `outputs/rgnet_paper_v1_aadb_regression_20260509/full_train/`.
- Train samples: 7612.
- Val samples: 846.
- Epochs requested: 20.
- Epochs completed: 5.
- Best epoch: 2.
- Best val_loss: `0.022888537496328354`.
- Best val_mae: `0.11879914253950119`.
- Save/load max abs diff: `0.0`.

Training history:
- Train loss: `0.0276880916`, `0.0227470491`, `0.0200721789`, `0.0176165681`, `0.0159476269`.
- Val loss: `0.0333393589`, `0.0228885375`, `0.0237882137`, `0.0230343565`, `0.0239335839`.

Overfitting signs:
- After epoch 2, training loss kept improving while validation loss stayed worse than the epoch-2 best.
- Early stopping restored the best weights.

## 8. Full Evaluation

| Split | Samples | SRCC | PLCC | MAE | RMSE | seconds_per_image |
|---|---:|---:|---:|---:|---:|---:|
| AADB test | 1000 | 0.6389 | 0.6423 | 0.1282 | 0.1602 | 0.02391 |
| AADB val | 846 | 0.5832 | 0.5874 | 0.1188 | 0.1513 | 0.01329 |

Artifacts:
- `outputs/rgnet_paper_v1_aadb_regression_20260509/eval/evaluation_summary.json`.
- `outputs/rgnet_paper_v1_aadb_regression_20260509/eval/test_predictions.csv`.
- `outputs/rgnet_paper_v1_aadb_regression_20260509/eval/val_predictions.csv`.

## 9. Comparison

| Model | Split | SRCC | PLCC | MAE | RMSE | Notes |
|---|---|---:|---:|---:|---:|---|
| RGNet-paper-v0 | AADB test full | 0.6433 | 0.6499 | 0.1255 | 0.1557 | Same full test split; v0 baseline |
| RGNet-paper-v1 | AADB test full | 0.6389 | 0.6423 | 0.1282 | 0.1602 | Same full test split; slightly worse |
| RGNet-paper-v1 | AADB val full | 0.5832 | 0.5874 | 0.1188 | 0.1513 | Full validation split |
| Practical RGNet AADB baseline | AADB val512 | 0.4987 | 0.5094 | 0.1304 | 0.1619 | Different split size; not directly comparable |
| Practical RGNet AVA->AADB fine-tuned | AADB val512 | 0.6749 | 0.6835 | 0.1104 | 0.1381 | Different split size; not directly comparable |

Important split note:
- v0 and v1 test rows are directly comparable because both use full `data/processed/aadb/test.csv`.
- Practical baselines are AADB val512, so they are useful context but not direct full-test comparisons.

## 10. Paper Comparability
This is not an official reproduction.

v1 is more paper-aligned than the practical RGNet and v0 in these ways:
- DenseNet121 FCN backbone is retained.
- ASPP adds multi-scale context.
- Region-level scores are predicted.
- LSE aggregation is implemented.
- Graph reasoning uses three residual graph blocks by default.

Still approximate:
- ASPP is not exact DenseASPP.
- Exact RegionGraph construction is still approximated with spatial nodes and cosine adjacency.
- No author code, paper hyperparameter recipe, or official weights were used.

## 11. Decision
Classification: `worse than v0 on full AADB test; worth continuing only as an ablation path with tuning`.

Rationale:
- v1 test SRCC `0.6389` is lower than v0 test SRCC `0.6433`.
- v1 test PLCC, MAE, and RMSE are also slightly worse than v0.
- v1 did show a reasonable full-validation SRCC `0.5832`, but the test split is the stronger comparison target requested here.
- The added ASPP/LSE components may need tuning rather than replacing v0 as the default paper-oriented baseline.

## 12. Next Steps
1. Tune learning rate, starting with `5e-5` and/or freezing early DenseNet blocks for the first epochs.
2. Add stronger regularization or lower `head_dropout`/`graph_dropout` ablations, because full training overfit after epoch 2.
3. Run aggregation ablations: `mean`, `max`, and `lse` with `r` in `{2, 4, 8}`.
4. Try a lighter ASPP branch set, for example `[1, 3, 6, 12]`, to reduce memory pressure.
5. Add true DenseASPP only after an ASPP ablation beats v0.
6. Move to RGNet-paper-AVA-classification if the v1 AADB path stabilizes.
