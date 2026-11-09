# Pozy App Stage5 Aesthetic Student Handoff

## Deployable artifact

- checkpoint: `checkpoints/student_full_aadb`
- export source used: rebuilt MobileNetV2 student loaded from `best.weights.h5`
- reason:
  - builtin-only TFLite export succeeded from rebuilt weights
  - direct builtin export from `saved_model` failed in this TensorFlow 2.20 environment
  - direct builtin export from `final_model.keras` also failed in this environment
- final artifact: `exports/tflite/stage5_student_full_aadb_aesthetic_mobile_fp32.tflite`
- artifact size: `3,411,060` bytes
- runtime requirement: standard builtin-op TFLite interpreter, no `SELECT_TF_OPS`

## App contract

- input tensor: `[1, 224, 224, 3]`
- input dtype: `float32`
- input layout: `NHWC`
- color format: `RGB`
- resize: direct bilinear resize to `224 x 224`
- normalization: `pixel_value / 255.0`
- do not apply ImageNet mean/std normalization
- output tensor: `[1, 1]`
- output dtype: `float32`
- output meaning: stage5 aesthetic component score distilled from `teacher_aesthetic_score / stage5_production_aesthetic_target_v2`
- score semantics: scalar already intended to live in `[0, 1]`; higher is better
- recommended app postprocess: squeeze to scalar float and optionally clip to `[0, 1]`
- observed validation operating range: about `0.310` to `0.736` on AADB val

## Verification

- Keras vs SavedModel on the 15-image real sample set: exact numeric match
- TFLite vs Keras max abs diff: `0.0028714239597320557`
- TFLite vs Keras mean abs diff: `0.0007897258037701249`
- ranking sanity on the 15-image real sample set: exact same full order, top-1, and top-5

## Candidate comparison

Comparison target: `teacher_aesthetic_score` on the full AADB validation split (`846` images).

| Candidate | TFLite size | Spearman | MAE | Pairwise acc. | Top-10 overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stage5_student_full_aadb` | `3.41 MB` | `0.8449` | `0.0260` | `0.9084` | `0.70` |
| `aadb_mobile` | `24.10 MB` | `0.9092` | `0.0462` | `0.9489` | `0.60` |
| `rgnet_aadb_gpu` | `26.09 MB` | `0.7979` | `0.0394` | `0.8780` | `0.40` |
| `nima_mobile` | `24.77 MB` | `0.3238` | `0.0468` | `0.6436` | `0.10` |

Practical interpretation:

- `aadb_mobile` keeps the strongest raw rank correlation on this validation slice, but it is a much larger single-model proxy for the aesthetic target.
- `stage5_student_full_aadb` is the only candidate explicitly trained against the current stage5 aesthetic target, has much better score calibration to that target than the older aesthetic exports, improves top-10 overlap over `aadb_mobile`, and is about `7x` smaller than the older aesthetic TFLite files.
- `rgnet_aadb_gpu` and `nima_mobile` are weaker choices for the default app path on both size and target fit.

Recommendation:

- make `stage5_student_full_aadb_aesthetic_mobile_fp32.tflite` the primary app aesthetic model candidate
- keep `aadb_mobile.tflite` as the main benchmark/fallback reference during app smoke testing

## App architecture note

- technical quality should continue coming from the existing technical models such as `koniq_mobile.tflite` and `flive_image_mobile.tflite`
- aesthetic quality should now come from `stage5_student_full_aadb_aesthetic_mobile_fp32.tflite`
- the default app path should treat this student output as the single on-device aesthetic signal, rather than fusing old `aadb_mobile`, `nima_mobile`, or `rgnet_aadb_gpu` outputs

## Exact next step

1. Add `exports/tflite/stage5_student_full_aadb_aesthetic_mobile_fp32.tflite` to the Flutter asset bundle.
2. Point the app’s aesthetic scorer to this model with the preprocessing contract above.
3. Keep the existing technical scorers unchanged.
4. Run an on-device smoke test on a small photo set and compare the app ordering against `exports/tflite/stage5_student_full_aadb_aesthetic_mobile_fp32.verify.json`.
