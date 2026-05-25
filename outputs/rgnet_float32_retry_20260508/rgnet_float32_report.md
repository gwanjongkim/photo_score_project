# RGNet Float32-Only Training and TFLite Export Report

## 1. Environment

- TensorFlow version: `2.20.0`
- GPU: NVIDIA GeForce RTX 4070 family, 12282 MiB reported by `nvidia-smi`
- Mixed precision policy: `<DTypePolicy "float32">`
- Git status: dirty before this run; full `git status --short` output is in `command_log.txt`
- Output directory: `/home/omen_pc1/photo_score_project/outputs/rgnet_float32_retry_20260508`
- Source changes: no `src/` files were modified
- Flutter copy: not performed

## 2. Why This Experiment Was Needed

The previous RGNet AVA->AADB fine-tune had strong Keras metrics but was not deployable. SavedModel export needed Select TF ops, local TFLite allocation failed at `FlexMul`, strict float32 weight copying did not preserve Keras parity, and the mixed-policy checkpoint could not convert to builtin-only TFLite. This run retrained from the start under float32 policy and kept builtin-only TFLite plus Keras-vs-TFLite parity as hard gates.

## 3. Float32 Smoke Test

- Command: `./.venv_gpu/bin/python outputs/rgnet_float32_retry_20260508/scripts/train_rgnet_float32.py --train_csv outputs/ava_retrain_rgnet_alamp_20260506/subsets/ava_train_unit_smoke_128.csv --val_csv outputs/ava_retrain_rgnet_alamp_20260506/subsets/ava_val_unit_smoke_64.csv --target_col aesthetic_unit_score --image_size 256 --batch_size 8 --epochs 1 --out_dir outputs/rgnet_float32_retry_20260508/rgnet_float32_ava_smoke`
- Result: completed `1` epoch
- Best val loss/mae: `0.01752258464694023` / `0.11260972917079926`
- Smoke export/parity: pass
- Smoke 20-image max diff: `1.4901161193847656e-07`

## 4. Float32 AVA Pretrain

- Command: `./.venv_gpu/bin/python outputs/rgnet_float32_retry_20260508/scripts/train_rgnet_float32.py --train_csv data/processed/ava/train_unit.csv --val_csv data/processed/ava/val_unit.csv --target_col aesthetic_unit_score --image_size 256 --batch_size 16 --epochs 10 --out_dir outputs/rgnet_float32_retry_20260508/rgnet_float32_ava_pretrain`
- Epochs completed: `6`
- Best val loss/mae: `0.005768290255218744` / `0.05948878079652786`
- Output files: `best.weights.h5`, `final_model.keras`, `saved_model/`, `training_history.csv`, `train_summary.json`
- Dtype check: no mixed-float16 layers and no non-float32 weights

## 5. Float32 AADB Fine-tune

- Command: `./.venv_gpu/bin/python outputs/rgnet_float32_retry_20260508/scripts/finetune_rgnet_float32.py --source_model outputs/rgnet_float32_retry_20260508/rgnet_float32_ava_pretrain/final_model.keras --train_csv data/processed/aadb/train.csv --val_csv data/processed/aadb/val.csv --target_col score --image_size 256 --batch_size 16 --epochs 10 --learning_rate 1e-5 --out_dir outputs/rgnet_float32_retry_20260508/rgnet_float32_ava_pretrain_aadb_finetune`
- Epochs completed: `10`
- Best val loss/mae: `0.020968850702047348` / `0.11363933235406876`
- Output files: `best.weights.h5`, `final_model.keras`, `saved_model/`, `training_history.csv`, `train_summary.json`
- Dtype check: no mixed-float16 layers and no non-float32 weights

## 6. Keras Evaluation

| Model | Training path | Eval split | SRCC | PLCC | MAE | RMSE | top_k_overlap | pairwise_accuracy | seconds_per_image |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original RGNet AADB baseline | `checkpoints/rgnet_aadb_gpu/final_model.keras` | AVA val512 | 0.1408 | 0.1572 | 0.0971 | 0.1203 | 0.1800 | 0.5407 | 0.019635 |
| Original RGNet AADB baseline | `checkpoints/rgnet_aadb_gpu/final_model.keras` | AADB val512 | 0.4711 | 0.4670 | 0.1350 | 0.1678 | 0.4200 | 0.6696 | 0.022208 |
| Previous mixed RGNet AVA->AADB Keras | `outputs/ava_pretrain_aadb_finetune_20260507/rgnet_ava_pretrain_aadb_finetune/final_model.keras` | AVA val512 | 0.4521 | 0.4656 | 0.1323 | 0.1623 | 0.3600 | 0.6515 | 0.019414 |
| Previous mixed RGNet AVA->AADB Keras | `outputs/ava_pretrain_aadb_finetune_20260507/rgnet_ava_pretrain_aadb_finetune/final_model.keras` | AADB val512 | 0.5804 | 0.5844 | 0.2320 | 0.2703 | 0.5000 | 0.7175 | 0.023475 |
| New float32 RGNet AVA pretrain | `outputs/rgnet_float32_retry_20260508/rgnet_float32_ava_pretrain/final_model.keras` | AVA val512 | 0.4940 | 0.5058 | 0.0558 | 0.0727 | 0.2600 | 0.6843 | 0.015418 |
| New float32 RGNet AVA pretrain | `outputs/rgnet_float32_retry_20260508/rgnet_float32_ava_pretrain/final_model.keras` | AADB val512 | 0.2816 | 0.3057 | 0.1635 | 0.1979 | 0.2600 | 0.5963 | 0.017898 |
| New float32 RGNet AVA->AADB Keras | `outputs/rgnet_float32_retry_20260508/rgnet_float32_ava_pretrain_aadb_finetune/final_model.keras` | AVA val512 | 0.4568 | 0.4625 | 0.1196 | 0.1478 | 0.3400 | 0.6573 | 0.015953 |
| New float32 RGNet AVA->AADB Keras | `outputs/rgnet_float32_retry_20260508/rgnet_float32_ava_pretrain_aadb_finetune/final_model.keras` | AADB val512 | 0.5705 | 0.5728 | 0.2238 | 0.2622 | 0.4600 | 0.7186 | 0.017884 |

Notes:
- The original baseline and previous mixed comparison checkpoints contain `mixed_float16` layer policies when loaded; they were evaluated only as historical comparisons.
- The new float32 Keras artifacts loaded with no mixed-float16 layers and no non-float32 weights.

## 7. Builtin TFLite Export

- Conversion status: `success`
- Select TF ops required: `False`
- Flex ops: `[]`
- Input shape: `[1, 256, 256, 3]`
- Output shape: `[1, 1]`
- TFLite size: `24.84 MiB` (`26050560` bytes)
- Op count: `410`
- Unique ops: `ADD, BATCH_MATMUL, CONCATENATION, CONV_2D, DELEGATE, DEPTHWISE_CONV_2D, DIV, FILL, FULLY_CONNECTED, L2_NORMALIZATION, LOGISTIC, MATRIX_DIAG, MAXIMUM, MEAN, MUL, NEG, PACK, RESHAPE, RSQRT, SHAPE, SOFTMAX, SQUARED_DIFFERENCE, STRIDED_SLICE, SUB, SUM, TANH, TRANSPOSE`
- Keras model dtype summary: `{'global_policy': '<DTypePolicy "float32">', 'layer_count': 16, 'weight_count': 373, 'bad_mixed_float16_layers': [], 'bad_non_float32_weights': []}`

## 8. Keras vs TFLite Parity

- Reference image max abs diff: `7.927417755126953e-06`
- 20-image max abs diff: `1.9490718841552734e-05`
- Threshold: `0.0001`
- Pass/fail: `pass`

TFLite fixed-subset metrics:

| Model | Eval split | SRCC | PLCC | MAE | RMSE | top_k_overlap | pairwise_accuracy | seconds_per_image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| New float32 RGNet AVA->AADB TFLite | AVA val512 | 0.4569 | 0.4622 | 0.1195 | 0.1477 | 0.3400 | 0.6583 | 0.021506 |
| New float32 RGNet AVA->AADB TFLite | AADB val512 | 0.5709 | 0.5732 | 0.2234 | 0.2618 | 0.4600 | 0.7190 | 0.023869 |

## 9. Deployment Decision

- Classification: `deployment-ready`
- Rationale: builtin-only conversion passed, no Select TF ops were required, no Flex ops were reported, standard `tf.lite.Interpreter` loaded the model, and 20-image parity passed under `1e-4`.
- Metric caveat: AADB val512 TFLite SRCC `0.5709` beats the provided original RGNet AADB baseline target `0.4983`, but remains below the provided previous mixed fine-tuned Keras reference `0.6749` and slightly below the local prior mixed comparison in this run `0.5804`.
- Flutter status: no model was copied into Flutter.

## 10. Recommendation

Use the new float32 RGNet TFLite as a deployment candidate. It is the first RGNet artifact in this retry path that satisfies the builtin-only conversion and parity gates, but it should still be compared against the current app candidate stack before any Flutter integration because its AADB metric quality is below the strongest previous mixed Keras teacher.
