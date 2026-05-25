# NIMA EfficientNetV2 preprocessing fix report

Date: 2026-05-25

## Files changed

- `src/models/nima_distribution.py`
- `src/datasets/ava_distribution_dataset.py`
- `src/train/train_nima.py`
- `src/infer/predict_nima.py`
- `src/export/tflite_presets.py`
- `src/export/verify_tflite.py`
- `outputs/nima_preprocessing_fix_20260525/checklist.md`
- `outputs/nima_preprocessing_fix_20260525/context-notes.md`
- `outputs/nima_preprocessing_fix_20260525/report.md`

## Exact bug fixed

The NIMA AVA dataset pipeline returns image tensors in `[0,1]` via `tf.image.convert_image_dtype(..., tf.float32)`.
The NIMA EfficientNetV2B0 backbone was constructed without `include_preprocessing=False`, so Keras used its default EfficientNetV2 preprocessing path, including an internal `Rescaling(1/255.0)`.

That made the backbone receive effective input values around `[0,0.0039]`, causing feature collapse.

## Why `include_preprocessing=False`

The project design for NIMA is now consistently `[0,1]` across train, validation/test, inference, and export verification.
Therefore the correct fix is model-side: construct EfficientNetV2B0 with `include_preprocessing=False`.

The dataset was not switched to `[0,255]`.
ICAA, A-LAMP, RGNet, FLIVE, KonIQ, MUSIQ, Flutter, and other non-NIMA paths were not modified.

## Verification

Smoke environment:

- TensorFlow: `2.21.0`
- Keras: `3.13.2`
- GPU present by `nvidia-smi`: RTX 4070-class GPU, 481 MiB used at check time

Smoke checks run:

- `./.venv_gpu/bin/python -m py_compile src/models/nima_distribution.py src/datasets/ava_distribution_dataset.py src/train/train_nima.py src/infer/predict_nima.py src/export/tflite_presets.py src/export/verify_tflite.py`
- Built fixed NIMA with `backbone_weights=None`.
- Compiled the model with existing `emd_loss`, `mean_score_mae`, and KL metric.
- Ran dummy `[2,224,224,3]` `[0,1]` input through the model.
- Ran 5 real AVA validation images through the fixed untrained model.
- Built the NIMA export rebuild model.
- Built the model with explicit local ImageNet weights: `/home/omen_pc1/.keras/models/efficientnetv2-b0_notop.h5`.
- Checked the NIMA train CLI exposes the optional `--backbone_weights` flag.

Smoke results:

- Model input shape: `(None, 224, 224, 3)`
- Model output shape: `(None, 10)`
- Dummy output shape: `[2, 10]`
- Dummy output sums: `[1.0, 0.9999999403953552]`
- Real AVA batch shape: `[5, 224, 224, 3]`
- Real AVA image range: min `0.0`, max `1.0`
- Real target shape: `[5, 10]`
- Real target sums: approximately `1.0`
- Real model output shape: `[5, 10]`
- Real model output sums: all `1.0`
- Rescaling layers in fixed training model: `[]`
- Rescaling layers in fixed export model: `[]`
- NIMA PIL eval/export verification array shape: `[224, 224, 3]`
- NIMA PIL eval/export verification range: min `0.05098039284348488`, max `1.0`

The Keras `"imagenet"` alias attempted a network download in the sandbox and failed DNS resolution.
The explicit local ImageNet weights path above loaded successfully with `include_preprocessing=False`, so the recommended retraining command uses that path.

## Preprocessing alignment

NIMA training remains:

- decode RGB
- convert to float `[0,1]`
- resize to at least `256x256`
- random crop to `224x224`
- random horizontal flip

NIMA validation/test/inference/export verification now use:

- decode/load RGB
- convert to float `[0,1]`
- resize to `256x256`
- deterministic center crop to `224x224`

Shared non-NIMA export/image loading behavior was left unchanged.

## Checkpoint protection

`train_nima.py` now refuses to write into an output directory that already contains any of:

- `best.weights.h5`
- `final_model.keras`
- `saved_model`

Old checkpoints were not deleted or overwritten.
Existing double-rescaled checkpoints should not be treated as fixed.

## Full retrain requirement

Full retraining is required.
The old `checkpoints/nima_ava_gpu` artifacts were trained with the double-rescaling bug, so they cannot become valid fixed-preprocessing checkpoints by code changes alone.

## Recommended retraining command

Do not run this until ready for a full training job:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib ./.venv_gpu/bin/python src/train/train_nima.py \
  --train_csv data/processed/ava/train_cleaned.csv \
  --val_csv data/processed/ava/val_cleaned.csv \
  --image_size 224 \
  --batch_size 64 \
  --epochs 30 \
  --learning_rate 1e-5 \
  --backbone_weights /home/omen_pc1/.keras/models/efficientnetv2-b0_notop.h5 \
  --out_dir checkpoints/nima_ava_gpu_fixed_preproc_20260525
```

Notes:

- This starts from ImageNet EfficientNetV2B0 weights, not an old broken NIMA checkpoint.
- `train_nima.py` does not currently support differential backbone/head learning rates; the command uses a conservative single LR of `1e-5`.
- After retraining, evaluate SRCC, PLCC, accuracy, MAE, and EMD before export/mobile use.
- Export to a new TFLite filename after retraining; do not overwrite existing `exports/tflite/nima_mobile.tflite`.
