# RGNet Float32 Retry Context Notes

## 2026-05-08 21:16 KST

- Started in `/home/omen_pc1/photo_score_project`.
- Existing git status is dirty with many unrelated modified and untracked files; this run will not revert or touch them.
- No existing `outputs/rgnet_float32_retry_*` directory was found before creating `outputs/rgnet_float32_retry_20260508/`.
- TensorFlow import reported version `2.20.0`.
- Current global mixed precision policy before experiment scripts was `<DTypePolicy "float32">`.
- GPU is visible via `nvidia-smi`.
- Existing `src/train/train_rgnet.py` sets `tf.keras.mixed_precision.set_global_policy("mixed_float16")` whenever GPUs are visible, so it cannot guarantee float32-only training for this task.
- Decision: create experiment-only scripts in this output directory and set `mixed_precision.set_global_policy("float32")` at startup before every model creation or load.
- Prior memory and local artifacts show RGNet TFLite can have valid float32 IO metadata but still fail numeric parity; strict parity remains a hard deployment gate.

## 2026-05-08 21:18 KST

- Added experiment-only scripts:
  - `outputs/rgnet_float32_retry_20260508/scripts/train_rgnet_float32.py`
  - `outputs/rgnet_float32_retry_20260508/scripts/finetune_rgnet_float32.py`
  - `outputs/rgnet_float32_retry_20260508/scripts/export_rgnet_float32_tflite.py`
  - `outputs/rgnet_float32_retry_20260508/scripts/evaluate_rgnet_float32.py`
- The scripts compile with `./.venv_gpu/bin/python -m py_compile`.
- Original RGNet AADB baseline checkpoint path: `checkpoints/rgnet_aadb_gpu/final_model.keras`.
- Previous mixed AVA->AADB Keras checkpoint path: `outputs/ava_pretrain_aadb_finetune_20260507/rgnet_ava_pretrain_aadb_finetune/final_model.keras`.
- Previous retry summary confirms direct mixed Keras builtin conversion failed, SavedModel conversion needed Select TF ops, and Select TF ops allocation failed at `FlexMul`.

## 2026-05-08 21:25 KST

- First smoke command failed before training with `ModuleNotFoundError: No module named 'src'` because script execution from `outputs/.../scripts` did not include the repo root on `sys.path`.
- Fixed only the experiment scripts by adding a repo-root path bootstrap based on `Path(__file__).resolve().parents[3]`.

## 2026-05-08 21:27 KST

- Smoke train completed for 1 epoch and wrote `outputs/rgnet_float32_retry_20260508/rgnet_float32_ava_smoke/`.
- Smoke model dtype checks passed: global policy float32, no mixed-float16 layers, no non-float32 weights.
- Sandboxed TensorFlow could not initialize CUDA (`cuInit UNKNOWN ERROR (100)`), so smoke train ran with `visible_gpus: []`.
- ImageNet weights were not available in the sandbox and the download failed due name resolution, so smoke used the model builder's random-init fallback. This is acceptable only for export-compatibility smoke gating.
- Smoke builtin-only TFLite export passed. Verification JSON: `outputs/rgnet_float32_retry_20260508/tflite/rgnet_float32_ava_smoke.verify.json`.
- Smoke TFLite details: input `[1, 256, 256, 3]`, output `[1, 1]`, no Flex ops, 20-image max abs diff `1.4901161193847656e-07`.

## 2026-05-08 22:08 KST

- Full AVA pretrain completed under escalated GPU access with `visible_gpus: ["PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')"]`.
- Output directory: `outputs/rgnet_float32_retry_20260508/rgnet_float32_ava_pretrain/`.
- Early stopping completed after 6 epochs; best epoch was epoch 3.
- Best validation metrics: `val_loss=0.005768290255218744`, `val_mae=0.05948878079652786`.
- Dtype checks passed before and after training: no mixed-float16 layers and no non-float32 weights.
- The training log included a known TensorFlow JPEG warning: `Corrupt JPEG data: 451 extraneous bytes before marker 0xd9`.
- Adjusted experiment train/fine-tune scripts to default `verbose=2` for future runs so logs are epoch-level instead of progress-bar streams.

## 2026-05-08 22:17 KST

- Full AADB fine-tune completed under escalated GPU access with `visible_gpus: ["PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')"]`.
- Source model: `outputs/rgnet_float32_retry_20260508/rgnet_float32_ava_pretrain/final_model.keras`.
- Output directory: `outputs/rgnet_float32_retry_20260508/rgnet_float32_ava_pretrain_aadb_finetune/`.
- Fine-tune ran all 10 requested epochs; best epoch was epoch 7.
- Best validation metrics: `val_loss=0.020968850702047348`, `val_mae=0.11363933235406876`.
- Dtype checks passed after load and after training: no mixed-float16 layers and no non-float32 weights.

## 2026-05-08 22:29 KST

- Fixed-subset Keras evaluation completed for original baseline, previous mixed AVA->AADB Keras, new float32 AVA pretrain, and new float32 AVA->AADB fine-tune.
- The original baseline and previous mixed comparison checkpoints contain `mixed_float16` layer policies when loaded, so they were evaluated only with `--allow_mixed_loaded_model` as historical comparisons.
- New float32 Keras artifacts loaded with global policy float32, no mixed-float16 layers, and no non-float32 weights.
- New float32 Keras AADB val512 metrics: `SRCC=0.570531093650961`, `PLCC=0.5728452675471378`, `MAE=0.22376874089241028`, `RMSE=0.26218480987441833`.
- Builtin-only TFLite export passed for `outputs/rgnet_float32_retry_20260508/tflite/rgnet_float32_ava_aadb_finetune.tflite`.
- TFLite details: Select TF ops required `false`, Flex ops `[]`, input shape `[1, 256, 256, 3]`, output shape `[1, 1]`, size `26050560` bytes.
- Keras-vs-TFLite parity passed: reference max diff `7.927417755126953e-06`, 20-image max diff `1.9490718841552734e-05`, threshold `1e-4`.
- TFLite AADB val512 metrics: `SRCC=0.5708929646581344`, `PLCC=0.5731698530014868`, `MAE=0.22336196899414062`, `RMSE=0.26178353489942735`.
- Final artifacts written: `rgnet_float32_report.md` and `rgnet_float32_summary.json`.
- No Flutter copy was performed.
