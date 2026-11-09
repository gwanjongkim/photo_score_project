# TOPIQ Android vs WSL Tensor Fingerprint Comparison

## 1. Summary

Confirmed from Android CSV: 50 Android TOPIQ tensor fingerprint rows were available, each targeting `[1,384,384,3]`, `resizeWithPad`, `rawZeroTo255`, and pad value `0`.

Closest tested WSL backend by tensor fingerprint distance: `pil_bilinear_resize_with_pad`.

Selection basis: lowest mean relative checksum-sum difference, then checksum-square difference, tensor mean difference, patch mean difference, and channel mean difference.

Best backend metrics:
- Mean abs checksum_sum diff: 329906.660000
- Mean relative checksum_sum diff: 0.01298755
- Mean abs checksum_sum_sq diff: 43408821.420000
- Mean abs tensor mean diff: 0.745724
- Mean abs tensor std diff: 1.079805
- Mean abs channel mean diff: 0.994759
- Mean abs patch mean diff: 0.366151
- Content/pad dimension matches: 50/50
- Original decoded dimension matches: 0/50

Confirmed fact: this comparison uses Android-provided tensor fingerprints, not TOPIQ score fit, to choose the preprocessing backend.

Confirmed fact: all tested backends match Android's final resized content and pad dimensions for all 50 rows, but none match Android's logged decoded source dimensions. Android logs downsampled source dimensions such as `960x1705`, while WSL PIL loads full EXIF-transposed images such as `2252x4000` for the same files.

Assumption: the Android fingerprint CSV values are produced from the exact tensor passed into TOPIQ mixed112.

## 2. Inputs

- Android tensor fingerprints: `data/debug/c6_parity_20260527/android_topiq_tensor_fingerprint_rows.csv`
- Android scores: `data/debug/c6_parity_20260527/android_c6_scores.csv`
- Images: `test_vila`
- Prior backend ablation report: `outputs/c6_topiq_preprocess_backend_ablation_20260529/report.md`
- Prior backend summary: `outputs/c6_topiq_preprocess_backend_ablation_20260529/backend_summary.csv`
- Prior best per-image comparison: `outputs/c6_topiq_preprocess_backend_ablation_20260529/per_image_comparison_best.csv`

## 3. Backend Candidates

| Backend | WSL implementation |
| --- | --- |
| `tf_resize_with_pad_current` | `tf.image.resize_with_pad(384, 384)` on EXIF-transposed RGB float32 `0..255` |
| `pil_bilinear_resize_with_pad` | PIL bilinear resize, rounded content size, zero pad |
| `pil_bicubic_resize_with_pad` | PIL bicubic resize, rounded content size, zero pad |
| `pil_lanczos_resize_with_pad` | PIL Lanczos resize, rounded content size, zero pad |

## 4. Fingerprint Comparison Summary

| backend | matched_images | mean_abs_diff_checksum_sum | mean_rel_diff_checksum_sum | mean_abs_diff_checksum_sum_sq | mean_rel_diff_checksum_sum_sq | mean_abs_diff_tensor_mean | mean_abs_diff_tensor_std | mean_abs_diff_channel_mean | mean_abs_diff_patch_mean | first_12_near_total | last_12_near_total | content_pad_dimension_match_count | original_dimension_match_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pil_bilinear_resize_with_pad | 50 | 329907 | 0.0129875 | 4.34088e+07 | 0.0144057 | 0.745724 | 1.07981 | 0.994759 | 0.366151 | 600 | 600 | 50 | 0 |
| pil_bicubic_resize_with_pad | 50 | 330057 | 0.0130943 | 3.84207e+07 | 0.0105038 | 0.746065 | 0.805939 | 0.99555 | 0.371297 | 600 | 600 | 50 | 0 |
| pil_lanczos_resize_with_pad | 50 | 330359 | 0.0132197 | 3.78661e+07 | 0.00942571 | 0.746746 | 0.699825 | 0.996994 | 0.371964 | 600 | 600 | 50 | 0 |
| tf_resize_with_pad_current | 50 | 320435 | 0.0132395 | 8.43593e+07 | 0.0232657 | 0.724294 | 0.705168 | 0.978582 | 0.545961 | 600 | 600 | 50 | 0 |

## 5. Per-Image Mismatch Patterns

Largest fingerprint mismatches for `pil_bilinear_resize_with_pad` by checksum_sum:

| filename | android_original_width | android_original_height | wsl_original_width | wsl_original_height | abs_diff_checksum_sum | rel_diff_checksum_sum | abs_diff_checksum_sum_sq | abs_diff_tensor_mean | abs_diff_tensor_std | abs_diff_patch_mean_max | content_pad_dimensions_match | prior_best_abs_diff_c6 | prior_best_diff_topiq | prior_best_diff_koniq |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1675564674752-15.jpg | 960 | 1280 | 3024 | 4032 | 1.3147e+06 | 0.040602 | 7.5316e+07 | 2.9722 | 4.14028 | 3.10462 | True | 0.467406 | -0.79223 | 0.307186 |
| 1675564674752-17.jpg | 960 | 1280 | 3024 | 4032 | 798880 | 0.0244739 | 1.41465e+08 | 1.8055 | 3.85452 | 0.598672 | True | 0.92499 | -1.49748 | 0.414142 |
| 20240515_092725.jpg | 960 | 1280 | 3000 | 4000 | 622170 | 0.0151286 | 3.62881e+07 | 1.40677 | 2.44487 | 3.07796 | True | 1.38272 | -1.93531 | -0.100015 |
| 20250205_000515.jpg | 960 | 1280 | 3000 | 4000 | 611034 | 0.0667384 | 5.22977e+06 | 1.38121 | 1.05976 | 2.64621 | True | 3.81442 | 5.17119 | 0.668622 |
| 20240510_191716.jpg | 960 | 1280 | 3000 | 4000 | 565497 | 0.0148752 | 1.20392e+08 | 1.27783 | 3.58244 | 0.599375 | True | 0.897394 | 0.508058 | -4.15345 |
| 1675564674752-2.jpg | 960 | 1280 | 3024 | 4032 | 562172 | 0.0156392 | 5.00212e+07 | 1.27059 | 2.1861 | 1.90087 | True | 1.39257 | -2.39036 | 0.938953 |
| 20250204_170733.jpg | 960 | 1280 | 3000 | 4000 | 561244 | 0.0134274 | 3.1931e+07 | 1.26868 | 1.18605 | 4.5575 | True | 1.07742 | 0.236513 | 3.05289 |
| 1704296944756.jpg | 960 | 1280 | 3024 | 4032 | 551368 | 0.0133278 | 1.27198e+07 | 1.24611 | 1.42265 | 2.07295 | True | 1.06387 | -1.34412 | -0.399933 |
| 20250204_133404.jpg | 960 | 1705 | 2252 | 4000 | 523079 | 0.0136575 | 3.37891e+07 | 1.18257 | 0.76061 | 4.4948 | True | 1.60452 | -3.82329 | 3.42927 |
| IMG_20240519_215827_510.jpg | 960 | 1200 | 1440 | 1800 | 522950 | 0.0118334 | 8.87996e+07 | 1.18183 | 0.235864 | 1.30733 | True | 1.96743 | -1.75355 | -2.45317 |

Highest prior C6 mismatches for `pil_bilinear_resize_with_pad` rows:

| filename | android_original_width | android_original_height | wsl_original_width | wsl_original_height | abs_diff_checksum_sum | rel_diff_checksum_sum | abs_diff_checksum_sum_sq | abs_diff_tensor_mean | abs_diff_tensor_std | abs_diff_patch_mean_max | content_pad_dimensions_match | prior_best_abs_diff_c6 | prior_best_diff_topiq | prior_best_diff_koniq |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20230201_181300.jpg | 960 | 1705 | 2252 | 4000 | 192715 | 0.00504738 | 4.25441e+07 | 0.435536 | 0.12268 | 0.510453 | True | 4.62557 | -0.908623 | 5.23869 |
| 20250205_000515.jpg | 960 | 1280 | 3000 | 4000 | 611034 | 0.0667384 | 5.22977e+06 | 1.38121 | 1.05976 | 2.64621 | True | 3.81442 | 5.17119 | 0.668622 |
| IMG_2729.JPG | 960 | 1440 | 1280 | 1920 | 19209 | 0.0152607 | 1.31648e+07 | 0.0438528 | 0.778477 | 6.65662 | True | 3.36015 | -3.90812 | -2.06822 |
| 20250205_072644.jpg | 960 | 1280 | 3000 | 4000 | 266200 | 0.00637681 | 3.71209e+07 | 0.601933 | 0.188677 | 0.260792 | True | 2.66905 | 1.71867 | 4.64992 |
| 1675564423029-9.jpg | 960 | 1280 | 3024 | 4032 | 276056 | 0.00890461 | 1.71444e+08 | 0.623773 | 3.28202 | 0.265168 | True | 2.53245 | 1.50227 | -11.9568 |
| 20250129_102520.jpg | 960 | 1705 | 2252 | 4000 | 152630 | 0.0255125 | 1.21554e+07 | 0.344984 | 0.680032 | 0.390208 | True | 2.43432 | 5.26908 | -4.18676 |
| 20230201_181510.jpg | 960 | 1705 | 2252 | 4000 | 183527 | 0.00492405 | 3.70171e+07 | 0.414449 | 0.0792307 | 0.869375 | True | 2.06238 | -0.368228 | 2.06238 |
| IMG_2739.JPG | 960 | 1440 | 1280 | 1920 | 95149 | 0.0352068 | 2.4071e+07 | 0.21542 | 1.23456 | 0.98975 | True | 2.05674 | -2.37802 | -1.3071 |
| IMG_20240519_215827_510.jpg | 960 | 1200 | 1440 | 1800 | 522950 | 0.0118334 | 8.87996e+07 | 1.18183 | 0.235864 | 1.30733 | True | 1.96743 | -1.75355 | -2.45317 |
| 20240101_143158.jpg | 1280 | 960 | 4000 | 3000 | 194081 | 0.00398031 | 6.20027e+07 | 0.438391 | 1.49697 | 0.729672 | True | 1.86861 | -2.39458 | -0.621355 |

Pattern note: the report compares tensor fingerprints first, then overlays prior C6 mismatch columns. C6 mismatch can remain even when the TOPIQ tensor backend is closest because C6 also depends on KonIQ and the cap branch.

## 6. Backend Decision

Decision: B. pil_bilinear_resize_with_pad

Confirmed fact: `pil_bilinear_resize_with_pad` is the closest tested WSL backend to Android tensor fingerprints by the aggregate metrics in this run.

Confirmed fact: no tested backend is an exact Android source-decode match because original decoded dimensions match `0/50` images. The backend result is therefore a closest-tested-backend decision, not proof of exact Android tensor equivalence.

Assumption: no unlogged Android preprocessing branch exists outside the provided fingerprint rows.

## 7. Implications for C6 Parity

WSL reference parity should prefer the tensor-fingerprint winner when choosing among the four tested resize/pad backends. However, a backend-only switch is not enough to claim exact Android parity because the source decode dimensions differ before the final `384x384` resize-with-pad step.

Prior score ablation alone was ambiguous because the C6-best backend differed from the TOPIQ-score-best backend. Tensor fingerprints are stronger evidence for preprocessing equivalence than final C6 score fit, and they now point to PIL bilinear as the closest tested resize backend.

This does not imply any production scoring change. It only identifies the WSL diagnostic/reference preprocessing path that best matches Android tensors.

## 8. Next Step

Add one diagnostic WSL variant that first resizes the EXIF-transposed RGB source to Android's logged `original_decoded_width`/`original_decoded_height`, then applies the closest backend (`pil_bilinear_resize_with_pad`) and recomputes fingerprints. If that still does not collapse the checksum gaps, export one full Flutter TOPIQ input tensor for pixel-wise comparison.
