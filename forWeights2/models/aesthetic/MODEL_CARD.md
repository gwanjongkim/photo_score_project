# forWeights2 Aesthetic TFLite Model Bundle

## Configured Models

- `nima_mobile.tflite`
  - Score key: `nima_score`.
  - Type: `nima_distribution`.
  - Input: RGB float32 `[1,224,224,3]`, pixel values divided by 255.
  - Output: `[1,10]` distribution over scores 1 through 10. The lab computes the expected score and maps it to 0-1 with `(mean - 1) / 9`.
- `rgnet_paper_aadb_fp16.tflite`
  - Score key: `rgnet_score`.
  - Type: `scalar_tflite`.
  - Configured input: RGB float32 `[1,256,256,3]`, pixel values divided by 255.
  - Output: scalar unit aesthetic score.
- `mobile_alamp_v2_fp16.tflite`
  - Score key: `alamp_score`.
  - Type: `alamp_signature`.
  - Inputs: full image `[1,384,384,3]` and patches `[1,5,224,224,3]`.
  - Local export metadata says this model uses float pixels 0-255 with MobileNetV3 preprocessing inside the model.
  - Output: scalar unit aesthetic score.
- `icaa_dat_tf_native_fp16.tflite`
  - Score key: `icaa_score`.
  - Type: `vector_tflite`.
  - Configured input: RGB float32 `[1,224,224,3]`, pixel values divided by 255.
  - Output: vector `[MOS, Color]` in the verified local export. `score_index: 0` selects MOS.

## Checksums

`SHA256SUMS.txt` covers the 4 configured `.tflite` files used by `configs/aesthetic_weight_lab.yaml`.

## Intended Use

This bundle is for local qualitative scoring, side-by-side score inspection, and interactive weight comparison on team-provided images.

## Not Intended For

This bundle is not an official benchmark claim, not a replacement-readiness decision by itself, and not evidence that small private image tests generalize to product performance.
