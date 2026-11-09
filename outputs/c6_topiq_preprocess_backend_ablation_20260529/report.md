# TOPIQ Preprocess Backend Ablation for Android Parity

## 1. Summary

Evaluated four WSL TOPIQ preprocessing backends against Android TOPIQ and C6 scores, with both FP16 and FP32 TOPIQ TFLite artifacts where available. KonIQ preprocessing and model loading were kept fixed to the latest EXIF-fixed WSL parity behavior.

Best backend by primary C6 sort `C6 MAE`, then `C6 Max AE`, then `TOPIQ MAE`: `pil_lanczos_resize_with_pad` with `fp16` TOPIQ.

Best backend by raw TOPIQ MAE: `tf_resize_with_pad_current` with `fp16` TOPIQ.

- Matched images for best backend: 50
- Failed source images: 0
- HEIF opener registered: True
- Best C6 MAE: 1.1397
- Best C6 Max AE: 4.6256
- Best C6 SRCC: 0.9744
- Best TOPIQ MAE: 1.3893
- Lowest raw TOPIQ MAE: 1.1965
- Score inversion: False

## 2. Inputs

- Android scores: `data/debug/c6_parity_20260527/android_c6_scores.csv`
- Images: `test_vila`
- KonIQ model: `exports/tflite/koniq_mobile.tflite`
- TOPIQ FP16: `outputs/final_topiq_lite_mixed112_export_20260517/topiq_lite_mixed112_frozen_fp16.tflite`
- TOPIQ FP32: `outputs/final_topiq_lite_mixed112_export_20260517/topiq_lite_mixed112_frozen_fp32.tflite`
- C6 formula: `c6 = min(0.7 * topiq + 0.3 * koniq, koniq + 8)`
- Decode: `pillow_heif.register_heif_opener()` when available, `ImageOps.exif_transpose(img)`, then RGB.
- Failed image details: None

## 3. Backend Variants

| Backend | Resize implementation | Pad value | Output range |
| --- | --- | ---: | --- |
| `tf_resize_with_pad_current` | `tf.image.resize_with_pad(384, 384)` | 0 | float32 `0..255` |
| `pil_bilinear_resize_with_pad` | PIL bilinear, `round(width * scale)` / `round(height * scale)` | 0 | float32 `0..255` |
| `pil_bicubic_resize_with_pad` | PIL bicubic, `round(width * scale)` / `round(height * scale)` | 0 | float32 `0..255` |
| `pil_lanczos_resize_with_pad` | PIL Lanczos, `round(width * scale)` / `round(height * scale)` | 0 | float32 `0..255` |

## 4. TOPIQ Parity Results

| backend | topiq_model | matched_count | topiq_mae | topiq_median_ae | topiq_max_ae | topiq_srcc | score_inversion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pil_lanczos_resize_with_pad | fp16 | 50 | 1.3893 | 1.1563 | 5.2691 | 0.9812 | False |
| pil_lanczos_resize_with_pad | fp32 | 50 | 1.3914 | 1.1620 | 5.2792 | 0.9812 | False |
| tf_resize_with_pad_current | fp16 | 50 | 1.1965 | 1.0157 | 3.5928 | 0.9753 | False |
| tf_resize_with_pad_current | fp32 | 50 | 1.1971 | 1.0297 | 3.5984 | 0.9753 | False |
| pil_bicubic_resize_with_pad | fp16 | 50 | 2.8165 | 2.7783 | 7.7322 | 0.9784 | False |
| pil_bicubic_resize_with_pad | fp32 | 50 | 2.8174 | 2.7857 | 7.7412 | 0.9789 | False |
| pil_bilinear_resize_with_pad | fp16 | 50 | 6.6458 | 6.5803 | 15.5376 | 0.9718 | False |
| pil_bilinear_resize_with_pad | fp32 | 50 | 6.6465 | 6.5828 | 15.5402 | 0.9724 | False |

## 5. C6 Parity Results

| backend | topiq_model | matched_count | c6_mae | c6_median_ae | c6_max_ae | c6_srcc | c6_abs_diff_gt_3 | c6_abs_diff_gt_5 | branch_switch_count | score_inversion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pil_lanczos_resize_with_pad | fp16 | 50 | 1.1397 | 0.8664 | 4.6256 | 0.9744 | 3 | 0 | 3 | False |
| pil_lanczos_resize_with_pad | fp32 | 50 | 1.1400 | 0.8670 | 4.6251 | 0.9744 | 3 | 0 | 3 | False |
| tf_resize_with_pad_current | fp16 | 50 | 1.3470 | 0.8212 | 5.2387 | 0.9690 | 6 | 1 | 2 | False |
| tf_resize_with_pad_current | fp32 | 50 | 1.3479 | 0.8275 | 5.2387 | 0.9690 | 6 | 1 | 2 | False |
| pil_bicubic_resize_with_pad | fp16 | 50 | 2.0170 | 1.7968 | 5.0467 | 0.9708 | 9 | 1 | 4 | False |
| pil_bicubic_resize_with_pad | fp32 | 50 | 2.0171 | 1.7937 | 5.0444 | 0.9708 | 9 | 1 | 4 | False |
| pil_bilinear_resize_with_pad | fp16 | 50 | 4.4995 | 4.4032 | 10.1542 | 0.9377 | 38 | 21 | 6 | False |
| pil_bilinear_resize_with_pad | fp32 | 50 | 4.4999 | 4.3981 | 10.1560 | 0.9377 | 38 | 21 | 6 | False |

## 6. Best Backend

- Selected backend: `pil_lanczos_resize_with_pad`
- TOPIQ model: `fp16`
- C6 MAE: 1.1397
- C6 Median AE: 0.8664
- C6 Max AE: 4.6256
- C6 SRCC: 0.9744
- TOPIQ MAE: 1.3893
- TOPIQ Max AE: 5.2691
- TOPIQ-only best backend: `tf_resize_with_pad_current` / `fp16`
- TOPIQ-only MAE: 1.1965
- TOPIQ-only Max AE: 3.5928
- C6 abs diff > 3: 3
- C6 abs diff > 5: 0
- Decision: D. Need Flutter tensor export

## 7. Remaining Top Mismatches

| filename | exif_orientation | android_c6_score | wsl_c6_score | abs_diff_c6 | diff_topiq | diff_koniq | android_c6_branch | wsl_c6_branch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20230201_181300.jpg | 6 | 61.2100 | 65.8356 | 4.6256 | -0.9086 | 5.2387 | koniq_cap | blend |
| 20250205_000515.jpg | 6 | 36.1400 | 39.9544 | 3.8144 | 5.1712 | 0.6686 | blend | blend |
| IMG_2729.JPG | 6 | 54.0100 | 50.6498 | 3.3602 | -3.9081 | -2.0682 | blend | blend |
| 20250205_072644.jpg | 6 | 66.2300 | 68.8990 | 2.6690 | 1.7187 | 4.6499 | koniq_cap | blend |
| 1675564423029-9.jpg | 6 | 71.0800 | 68.5475 | 2.5325 | 1.5023 | -11.9568 | blend | blend |
| 20250129_102520.jpg | 6 | 34.3300 | 36.7643 | 2.4343 | 5.2691 | -4.1868 | blend | blend |
| 20230201_181510.jpg | 6 | 59.7800 | 61.8424 | 2.0624 | -0.3682 | 2.0624 | koniq_cap | koniq_cap |
| IMG_2739.JPG | 8 | 50.7600 | 48.7033 | 2.0567 | -2.3780 | -1.3071 | blend | blend |
| IMG_20240519_215827_510.jpg | none | 73.4200 | 71.4526 | 1.9674 | -1.7535 | -2.4532 | blend | blend |
| 20240101_143158.jpg | 1 | 73.7100 | 71.8414 | 1.8686 | -2.3946 | -0.6214 | blend | blend |

## 8. Recommendation

Do not treat the C6-best backend as the confirmed Android TOPIQ preprocessing match. The current TensorFlow backend has the lowest raw TOPIQ MAE, while PIL Lanczos has the best C6 MAE/Max AE. That conflict points to C6 cap branch sensitivity, KonIQ interaction, or an unobserved tensor-level difference. Export Flutter TOPIQ input tensors for one top mismatch and one low-mismatch control before changing the WSL reference backend.
