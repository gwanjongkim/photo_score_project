# TOPIQ Source-Dimension-Matched Tensor Fingerprint Diagnostic

## 1. Summary

Source-dimension matching worsened the primary checksum metric versus the previous `pil_bilinear_resize_with_pad` baseline.

Confirmed facts:
- Android rows processed: 50
- Failures: 0
- HEIF registered: True
- Source-dimension match count: 50/50
- Final content/pad dimension match count: 50/50

Key result:
- Previous mean relative checksum_sum diff: 0.01298755
- New mean relative checksum_sum diff: 0.01425503
- Delta new - previous: 0.00126748

Decision: B. Source-dimension matching does not improve parity; keep previous pil_bilinear backend and investigate another cause.

Assumption:
- The source resize from WSL full EXIF-transposed dimensions to Android logged decoded dimensions uses PIL bilinear as a diagnostic approximation of Android decode downsampling.

## 2. Inputs

- Android tensor fingerprint CSV: `data/debug/c6_parity_20260527/android_topiq_tensor_fingerprint_rows.csv`
- Previous tensor fingerprint report: `outputs/c6_topiq_tensor_fingerprint_compare_20260529/report.md`
- Previous tensor fingerprint summary: `outputs/c6_topiq_tensor_fingerprint_compare_20260529/backend_fingerprint_summary.csv`
- Previous per-image fingerprint comparison: `outputs/c6_topiq_tensor_fingerprint_compare_20260529/per_image_fingerprint_comparison.csv`
- Image directory: `test_vila`
- Failed image details: None

## 3. Method

1. Register `pillow_heif` when available.
2. Open each image with PIL.
3. Apply `ImageOps.exif_transpose`.
4. Convert to RGB.
5. Resize the EXIF-transposed RGB source to Android's logged `original_decoded_width` and `original_decoded_height`.
6. Apply PIL bilinear resize-with-pad to `384x384`, pad value `0`.
7. Convert to float32 RGB tensor in raw `0..255`.
8. Compute checksums, aggregate stats, channel stats, patch means, first/last values, and geometry fields.
9. Compare against Android fingerprints and previous `pil_bilinear_resize_with_pad` rows.

## 4. Results

| variant | mean_rel_diff_checksum_sum | mean_rel_diff_checksum_sum_sq | mean_abs_diff_tensor_mean | mean_abs_diff_tensor_std | mean_abs_diff_channel_mean | mean_abs_diff_channel_std | mean_abs_diff_patch_mean | first_12_near_total | last_12_near_total | content_pad_dimension_match_count | source_dimension_match_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pil_bilinear_resize_with_pad | 0.0129875 | 0.0144057 | 0.745724 | 1.07981 | 0.994759 | 1.1107 | 0.366151 | 600 | 600 | 50 | 0 |
| source_dimension_matched_pil_bilinear_resize_with_pad | 0.014255 | 0.0148815 | 0.759713 | 1.11185 | 1.00705 | 1.13775 | 0.371589 | 600 | 600 | 50 | 50 |
| delta_new_minus_previous | 0.00126748 | 0.000475755 | 0.0139899 | 0.0320446 | 0.0122958 | 0.0270541 | 0.00543747 | 0 | 0 | 0 | 50 |

Metric movement:
- Mean relative checksum_sum diff: 0.00126748
- Mean relative checksum_sum_sq diff: 0.00047575
- Mean tensor mean diff: 0.013990
- Mean tensor std diff: 0.032045
- Mean channel mean diff: 0.012296
- Mean patch mean diff: 0.005437

Decision reason: All key aggregate fingerprint metrics worsened.

## 5. Geometry Findings

Confirmed facts:
- The new diagnostic forces WSL source dimensions to Android logged decoded dimensions for all processed images.
- Final content/pad dimensions match Android for all processed images.
- WSL full EXIF-transposed dimensions remain available in the per-image CSV for comparison.

Example geometry from largest remaining mismatches:

| filename | android_original_width | android_original_height | wsl_full_exif_width | wsl_full_exif_height | wsl_source_matched_width | wsl_source_matched_height |
| --- | --- | --- | --- | --- | --- | --- |
| 1675564674752-15.jpg | 960 | 1280 | 3024 | 4032 | 960 | 1280 |
| 1675564674752-17.jpg | 960 | 1280 | 3024 | 4032 | 960 | 1280 |
| 20240515_092725.jpg | 960 | 1280 | 3000 | 4000 | 960 | 1280 |
| 20250205_000515.jpg | 960 | 1280 | 3000 | 4000 | 960 | 1280 |
| 20240510_191716.jpg | 960 | 1280 | 3000 | 4000 | 960 | 1280 |
| 1675564674752-2.jpg | 960 | 1280 | 3024 | 4032 | 960 | 1280 |
| 20250204_170733.jpg | 960 | 1280 | 3000 | 4000 | 960 | 1280 |
| 1704296944756.jpg | 960 | 1280 | 3024 | 4032 | 960 | 1280 |
| 20250204_133404.jpg | 960 | 1705 | 2252 | 4000 | 960 | 1705 |
| IMG_20240519_215827_510.jpg | 960 | 1200 | 1440 | 1800 | 960 | 1200 |

## 6. Per-Image Mismatch Patterns

Top remaining mismatches by new checksum_sum difference:

| filename | android_original_width | android_original_height | wsl_full_exif_width | wsl_full_exif_height | wsl_source_matched_width | wsl_source_matched_height | abs_diff_checksum_sum | rel_diff_checksum_sum | abs_diff_checksum_sum_sq | abs_diff_tensor_mean | abs_diff_tensor_std | abs_diff_patch_mean_max | previous_abs_diff_checksum_sum | delta_abs_diff_checksum_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1675564674752-15.jpg | 960 | 1280 | 3024 | 4032 | 960 | 1280 | 1.31584e+06 | 0.0406372 | 8.03003e+07 | 2.97477 | 4.22119 | 3.04733 | 1.3147e+06 | 1138 |
| 1675564674752-17.jpg | 960 | 1280 | 3024 | 4032 | 960 | 1280 | 798681 | 0.0244678 | 1.45559e+08 | 1.80505 | 3.91617 | 0.484078 | 798880 | -199 |
| 20240515_092725.jpg | 960 | 1280 | 3000 | 4000 | 960 | 1280 | 625361 | 0.0152062 | 3.78947e+07 | 1.41398 | 2.48083 | 3.06754 | 622170 | 3191 |
| 20250205_000515.jpg | 960 | 1280 | 3000 | 4000 | 960 | 1280 | 613722 | 0.067032 | 5.18763e+06 | 1.38729 | 1.06811 | 2.66704 | 611034 | 2688 |
| 20240510_191716.jpg | 960 | 1280 | 3000 | 4000 | 960 | 1280 | 564315 | 0.0148441 | 1.26441e+08 | 1.27519 | 3.68093 | 0.635836 | 565497 | -1182 |
| 1675564674752-2.jpg | 960 | 1280 | 3024 | 4032 | 960 | 1280 | 562861 | 0.0156584 | 5.44386e+07 | 1.27214 | 2.2569 | 2.01546 | 562172 | 689 |
| 20250204_170733.jpg | 960 | 1280 | 3000 | 4000 | 960 | 1280 | 561622 | 0.0134364 | 3.10015e+07 | 1.26952 | 1.20207 | 4.5575 | 561244 | 378 |
| 1704296944756.jpg | 960 | 1280 | 3024 | 4032 | 960 | 1280 | 554001 | 0.0133915 | 1.1012e+07 | 1.25206 | 1.45747 | 2.05212 | 551368 | 2633 |
| 20250204_133404.jpg | 960 | 1705 | 2252 | 4000 | 960 | 1705 | 523799 | 0.0136763 | 3.36325e+07 | 1.1842 | 0.764371 | 4.51562 | 523079 | 720 |
| IMG_20240519_215827_510.jpg | 960 | 1200 | 1440 | 1800 | 960 | 1200 | 522488 | 0.0118229 | 8.58582e+07 | 1.18082 | 0.277085 | 1.2813 | 522950 | -462 |

## 7. Decision

B. Source-dimension matching does not improve parity; keep previous pil_bilinear backend and investigate another cause.

Confirmed fact: source-dimension matching did not produce exact tensor parity. The first/last 12 values still match because these sampled positions are padded zeros, but aggregate checksums and tensor/channel/patch stats remain different.

Assumption: if Android's decode downsampling uses a different kernel or color pipeline, matching dimensions alone will not reproduce the full tensor.

## 8. Implications for C6 Parity

This diagnostic does not justify a production scoring change. For WSL Android-reference preprocessing, source-dimension matching should not replace the previous PIL bilinear baseline because the aggregate tensor-fingerprint metrics worsened. Strict C6 parity still needs tensor-level evidence rather than another scoring-side change.

The prior C6 mismatch can still be affected by KonIQ and C6 cap-branch sensitivity, so TOPIQ tensor improvements may not translate directly to C6 Max AE improvement.

## 9. Next Step

Smallest next action: export one full Flutter TOPIQ input tensor for a high remaining mismatch such as `1675564674752-15.jpg` and compare pixel-wise against both previous PIL bilinear and source-dimension-matched WSL tensors. This removes ambiguity from decode downsampling and interpolation kernels.
