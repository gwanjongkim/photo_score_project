# Aesthetic TFLite Model Bundle

## Included Models

- `nima_mobile.tflite`
  - Source copy: `exports/tflite/nima_mobile.tflite`
  - Input: RGB float32 `[1,224,224,3]`, pixel values divided by 255.
  - Output: `[1,10]` distribution over scores 1 through 10. The lab computes `sum(prob[i] * (i + 1))`, then converts to unit score with `(mean - 1) / 9`.
- `rgnet_aadb_gpu.tflite`
  - Source copy: `exports/tflite/rgnet_aadb_gpu.tflite`
  - Input: RGB float32 `[1,256,256,3]`, pixel values divided by 255.
  - Output: scalar unit aesthetic score `[1,1]`.
- `alamp_aadb_gpu.tflite`
  - Source copy: `exports/tflite/alamp_aadb_gpu.tflite`
  - Inputs: `global_view` RGB float32 `[1,384,384,3]` and `patches` RGB float32 `[1,5,224,224,3]`, pixel values divided by 255.
  - Output: scalar unit aesthetic score `[1,1]`.
- `alamp_ava_aadb_finetune.tflite`
  - Source copy: `outputs/ava_pretrain_aadb_finetune_20260507/tflite/alamp_ava_aadb_finetune.tflite`
  - Same A-LAMP input/output contract as `alamp_aadb_gpu.tflite`.
  - Provided as a candidate for qualitative comparison, not as a production replacement claim.

Metadata and verification JSON sidecars are included where available.

## Excluded Models

- `outputs/ava_pretrain_aadb_finetune_20260507/tflite/rgnet_ava_aadb_finetune.tflite`
- `outputs/rgnet_tflite_retry_20260507/tflite/*`
- Any `.keras` or `.h5` checkpoint files.

The fine-tuned RGNet TFLite artifacts are intentionally excluded because local TFLite deployment/parity checks failed. They should not be used in the team lab until parity is fixed and reverified.

## Intended Use

This bundle is for the Aesthetic Weight Lab: local qualitative scoring, side-by-side score inspection, and interactive weight comparison on team-provided images.

## Not Intended For

This bundle is not an official benchmark claim, not a replacement-readiness decision by itself, and not evidence that small private image tests generalize to product performance.
