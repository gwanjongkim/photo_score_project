# TFLite Mobile Handoff

This document covers the priority mobile export targets that were exported and verified in this repo.

## Verified export source

All verified `.tflite` files in `exports/tflite/` were exported from the model architecture rebuilt in Python and loaded from `best.weights.h5`.

Reason:
- this produces builtin-op TFLite models that run in a normal TensorFlow Lite interpreter
- direct SavedModel conversion in this environment fell back to `SELECT_TF_OPS`, which is a worse Flutter deployment path

## Common preprocessing

Use the same preprocessing for all four exported models:

- color format: RGB
- resize: bilinear resize to `224 x 224`
- tensor layout: NHWC
- dtype: `float32`
- normalization: `pixel_value / 255.0`
- batch shape: `[1, 224, 224, 3]`

Do not apply ImageNet mean/std normalization.

## Model contracts

### `koniq_mobile.tflite`

- file: `exports/tflite/koniq_mobile.tflite`
- source weights: `checkpoints/technical_koniq_gpu/best.weights.h5`
- input: one RGB image tensor `[1, 224, 224, 3]`, `float32`
- output tensor: `[1, 1]`
- output meaning: raw KonIQ technical-quality score
- mobile postprocess: `normalized = clip(raw_score / 100.0, 0.0, 1.0)`
- notes: smallest practical first-pass mobile model in this repo

### `flive_image_mobile.tflite`

- file: `exports/tflite/flive_image_mobile.tflite`
- source weights: `checkpoints/technical_flive_image_gpu/best.weights.h5`
- input: one RGB image tensor `[1, 224, 224, 3]`, `float32`
- output tensor: `[1, 1]`
- output meaning: raw FLIVE full-image technical-quality score
- mobile postprocess: `normalized = clip(raw_score / 100.0, 0.0, 1.0)`
- notes: same deployment shape and preprocessing as KonIQ

### `aadb_mobile.tflite`

- file: `exports/tflite/aadb_mobile.tflite`
- source weights: `checkpoints/composition_aadb_gpu/best.weights.h5`
- input: one RGB image tensor `[1, 224, 224, 3]`, `float32`
- output tensor: `[1, 1]`
- output meaning: aesthetic score already in `[0, 1]`
- mobile postprocess: squeeze scalar and optionally clip to `[0, 1]`
- notes: heavier than the MobileNetV2 models because it uses EfficientNetV2B0

### `nima_mobile.tflite`

- file: `exports/tflite/nima_mobile.tflite`
- source weights: `checkpoints/nima_ava_gpu/best.weights.h5`
- input: one RGB image tensor `[1, 224, 224, 3]`, `float32`
- output tensor: `[1, 10]`
- output meaning: softmax distribution over aesthetic scores `1..10`
- mobile postprocess:
  - keep the full distribution if you want richer UI/explanations
  - or compute mean score with `sum(distribution[i] * (i + 1))`
  - optional unit score: `(mean_score - 1.0) / 9.0`
- notes: also EfficientNetV2B0, so expect a heavier latency/memory profile than KonIQ or FLIVE-image

## Verified artifact set

These files were created and checked with real TFLite inference:

- `exports/tflite/koniq_mobile.tflite`
- `exports/tflite/koniq_mobile.metadata.json`
- `exports/tflite/koniq_mobile.verify.json`
- `exports/tflite/flive_image_mobile.tflite`
- `exports/tflite/flive_image_mobile.metadata.json`
- `exports/tflite/flive_image_mobile.verify.json`
- `exports/tflite/aadb_mobile.tflite`
- `exports/tflite/aadb_mobile.metadata.json`
- `exports/tflite/aadb_mobile.verify.json`
- `exports/tflite/nima_mobile.tflite`
- `exports/tflite/nima_mobile.metadata.json`
- `exports/tflite/nima_mobile.verify.json`

## Flutter-side recommendation

Recommended integration order:

1. `koniq_mobile.tflite`
2. `flive_image_mobile.tflite`
3. `aadb_mobile.tflite`
4. `nima_mobile.tflite`

Practical first integration:

- start with KonIQ only, or KonIQ + FLIVE-image
- normalize both technical outputs with `/100.0` and clipping
- add AADB only if you want an aesthetic/composition signal on device
- add NIMA only if the larger output and heavier backbone are worth the extra latency

## Caveats

- These models are `float32` exports. Quantization was not part of this pass.
- The Python verifier used the standard TFLite interpreter with XNNPACK CPU delegate.
- TensorFlow warns that `tf.lite.Interpreter` is being replaced by LiteRT in future versions. That does not block current export artifacts, but a future mobile runtime update should prefer LiteRT-compatible integration paths.
