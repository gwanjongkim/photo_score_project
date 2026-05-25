# AVA Pretrain + AADB Fine-tune Report

## 1. Environment
- GPU: NVIDIA GeForce RTX 4070 SUPER, 12282 MiB visible in `nvidia-smi`.
- Python: `./.venv_gpu/bin/python --version` -> Python 3.12.3.
- Git status: dirty before this experiment; unrelated modified/untracked files were left untouched.
- Output directory: `outputs/ava_pretrain_aadb_finetune_20260507`.

## 2. AADB Target Scale Check
- `data/processed/aadb/train.csv` score min/max: 0.000000 / 1.000000.
- `data/processed/aadb/val.csv` score min/max: 0.000000 / 0.950000.
- Normalization was not needed; existing `score` was used directly.

## 3. Fine-tune Setup
- RGNet initial model: `outputs/ava_retrain_rgnet_alamp_20260506/rgnet_ava_unit/final_model.keras`.
- A-LAMP initial model: `outputs/ava_retrain_rgnet_alamp_20260506/alamp_ava_unit/final_model.keras`.
- Learning rate: 1e-5.
- Batch size: RGNet 16, A-LAMP 16.
- Epochs: 10 max.
- Early stopping: `val_loss`, patience 3, `restore_best_weights=True`.
- Scripts used: experiment-only scripts under `outputs/ava_pretrain_aadb_finetune_20260507/scripts/`; no `src/` changes were made.

## 4. RGNet Fine-tune Result
- Epochs completed: 10 / 10.
- Best epoch: 10.
- Best val loss/mae: 0.019849520176649094 / 0.11031755805015564.
- Output files: `best.weights.h5`, `final_model.keras`, `saved_model/`, `training_history.csv`, `train_summary.json`.
- Warning: first sandbox attempt could not initialize CUDA and was stopped; partial CPU history was preserved as `training_history_cpu_aborted.csv`.

## 5. A-LAMP Fine-tune Result
- Epochs completed: 10 / 10.
- Best epoch: 10.
- Best val loss/mae: 0.020329773426055908 / 0.11302073299884796.
- Output files: `best.weights.h5`, `final_model.keras`, `saved_model/`, `training_history.csv`, `train_summary.json`.
- Warnings: none blocking in the fine-tune run.

## 6. Evaluation Comparison

| Model | Training path | Eval split | SRCC | PLCC | MAE | RMSE | top_k_overlap | pairwise_accuracy | seconds_per_image |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| RGNet | AADB baseline | AVA val512 | 0.1790 | 0.1930 | 0.1024 | 0.1261 | 0.1800 | 0.5539 | 0.02030 |
| RGNet | AADB baseline | AADB val512 | 0.4983 | 0.5094 | 0.1304 | 0.1619 | 0.4600 | 0.6810 | 0.02108 |
| RGNet | AVA-only | AVA val512 | 0.5613 | 0.5697 | 0.0599 | 0.0773 | 0.3600 | 0.7035 | 0.02059 |
| RGNet | AVA-only | AADB val512 | 0.2892 | 0.3271 | 0.1630 | 0.1990 | 0.3400 | 0.6013 | 0.02070 |
| RGNet | AVA->AADB fine-tuned | AVA val512 | 0.4865 | 0.5169 | 0.1511 | 0.1808 | 0.3800 | 0.6673 | 0.02055 |
| RGNet | AVA->AADB fine-tuned | AADB val512 | 0.6749 | 0.6835 | 0.1104 | 0.1381 | 0.5200 | 0.7513 | 0.02083 |
| A-LAMP | AADB baseline | AVA val512 | 0.2880 | 0.2996 | 0.0972 | 0.1231 | 0.2000 | 0.5987 | 0.05684 |
| A-LAMP | AADB baseline | AADB val512 | 0.5835 | 0.5855 | 0.1292 | 0.1595 | 0.4800 | 0.7076 | 0.05815 |
| A-LAMP | AVA-only | AVA val512 | 0.5548 | 0.5567 | 0.0530 | 0.0686 | 0.3600 | 0.7031 | 0.05844 |
| A-LAMP | AVA-only | AADB val512 | 0.3438 | 0.3631 | 0.1468 | 0.1808 | 0.2000 | 0.6271 | 0.05895 |
| A-LAMP | AVA->AADB fine-tuned | AVA val512 | 0.4515 | 0.4616 | 0.1256 | 0.1531 | 0.3200 | 0.6541 | 0.05729 |
| A-LAMP | AVA->AADB fine-tuned | AADB val512 | 0.6722 | 0.6824 | 0.1091 | 0.1375 | 0.5200 | 0.7513 | 0.06010 |

## 7. TFLite Export and Parity
- RGNet status: failed. Requested artifact was exported from SavedModel with Select TF ops; local TFLite allocation failed at `FlexMul`.
- RGNet input/output shapes: input `[{'name': 'serving_default_input_layer_1:0', 'shape': [1, 256, 256, 3], 'shape_signature': [-1, 256, 256, 3], 'dtype': 'float32', 'index': 0}]`, output `[{'name': 'StatefulPartitionedCall_1:0', 'shape': [1, 1], 'shape_signature': [-1, 1], 'dtype': 'float32', 'index': 608}]`.
- RGNet Keras vs requested TFLite max diff: unavailable because allocation failed. SavedModel vs checkpoint max diff: 5.841255187988281e-06. Rebuild TFLite vs checkpoint max diff: 0.17774948477745056.
- RGNet pass/fail: fail.
- A-LAMP status: passed. Export used float32 rebuild from weights with builtin TFLite ops.
- A-LAMP input/output shapes: input `[{'name': 'serving_default_global_view:0', 'shape': [1, 384, 384, 3], 'shape_signature': [-1, 384, 384, 3], 'dtype': 'float32', 'index': 0}, {'name': 'serving_default_patches:0', 'shape': [1, 5, 224, 224, 3], 'shape_signature': [-1, 5, 224, 224, 3], 'dtype': 'float32', 'index': 1}]`, output `[{'name': 'StatefulPartitionedCall_1:0', 'shape': [1, 1], 'shape_signature': [-1, 1], 'dtype': 'float32', 'index': 1044}]`.
- A-LAMP Keras vs TFLite max diff: 2.384185791015625e-07. Rebuild vs checkpoint smoke diff: 0.034421563148498535.
- A-LAMP pass/fail: pass.

## 8. Decision
- RGNet: better than current model; not deployment-ready; useful only as teacher until export/parity is fixed.
- A-LAMP: better than current model; deployment-ready by this TFLite smoke/parity gate; also useful as teacher/student distillation input.
- No Flutter deployment was performed.

## 9. Recommendation
- Keep current RGNet model path until RGNet export/parity is fixed.
- Use A-LAMP AVA->AADB fine-tuned model only as an explicit deployment candidate after review; do not copy it to Flutter automatically.
- Continue with the Stage 5 student path and consider the fine-tuned checkpoints as teacher/student distillation inputs.
- If RGNet is still required for mobile, fix the export path first; the current TFLite artifact is not deployable.
