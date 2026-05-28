# C6 Android vs WSL Parity Report — EXIF Fixed

## 1. Summary
Comparison between Android Flutter TFLite implementation and WSL reference implementation after applying PIL EXIF orientation transpose in the WSL reference path.
- Android Score CSV: `data/debug/c6_parity_20260527/android_c6_scores.csv`
- Image Directory: `test_vila`
- KonIQ Model: `exports/tflite/koniq_mobile.tflite`
- TOPIQ Model: `outputs/final_topiq_lite_mixed112_export_20260517/topiq_lite_mixed112_frozen_fp16.tflite`
- `pillow-heif` registered: True
- Numeric outputs finite/no NaN: True
- Scale inversion check: PASS

## 2. What Changed
- Registered `pillow-heif` when available, instead of requiring it at import time.
- Opened images with PIL, captured original EXIF Orientation and image size, applied `ImageOps.exif_transpose`, then converted the corrected image to RGB.
- Reused the same corrected tensor for both KonIQ and TOPIQ preprocessing.
- Added per-image EXIF and before/after size fields to `wsl_c6_scores.csv` and `android_vs_wsl_parity.csv`.

## 3. Image Matching
- Total images in Android CSV: 50
- Matched and processed: 50
- Failed: 0

## 4. EXIF Orientation Handling
- EXIF orientation counts:
```
exif_orientation_original
1        4
6       39
8        3
none     4
```
- Images with EXIF transpose applied: 42
- Images with size changed by transpose: 42

Fields emitted per image:
- `exif_orientation_original`
- `exif_transpose_applied`
- `image_size_before_transpose`
- `image_size_after_transpose`

Model contracts used:
### KonIQ
- Input: 224x224, Resize (Stretch)
- Normalization: pixel / 255.0
- Output: Raw (0..100)

### TOPIQ
- Input: 384x384, Resize with Pad
- Normalization: None (0..255)
- Output: Raw * 100 (0..100)

### C6 Formula
- `c6 = min(0.7 * topiq + 0.3 * koniq, koniq + 8.0)`

## 5. Per-Model Parity Metrics
| Model | MAE | Max AE | Median AE | SRCC |
| :--- | :--- | :--- | :--- | :--- |
| KonIQ | 2.0578 | 11.9568 | 1.4740 | 0.9142 |
| TOPIQ | 1.1965 | 3.5928 | 1.0157 | 0.9753 |
| **C6** | **1.3470** | **5.2387** | **0.8212** | **0.9690** |

## 6. Top Mismatches
### KonIQ
```
            filename exif_orientation_original  exif_transpose_applied image_size_before_transpose image_size_after_transpose  koniq_score_100  wsl_koniq_score_100  diff_koniq  abs_diff_koniq
 1675564423029-9.jpg                         6                    True                   4032x3024                  3024x4032            76.62            64.663193  -11.956807       11.956807
1675564423029-24.jpg                         6                    True                   4032x3024                  3024x4032            77.65            71.739838   -5.910162        5.910162
   1715855266134.jpg                         6                    True                   3088x2316                  2316x3088            65.88            60.183681   -5.696319        5.696319
 20230201_181300.jpg                         6                    True                   4000x2252                  2252x4000            53.21            58.448689    5.238689        5.238689
 1675564423029-6.jpg                         6                    True                   4032x3024                  3024x4032            73.88            68.727455   -5.152545        5.152545
 20250205_072644.jpg                         6                    True                   4000x3000                  3000x4000            58.23            62.879925    4.649925        4.649925
 20250129_102520.jpg                         6                    True                   4000x2252                  2252x4000            37.23            33.043236   -4.186764        4.186764
 20250129_103313.jpg                         6                    True                   4000x2252                  2252x4000            50.06            45.902386   -4.157614        4.157614
 20240510_191716.jpg                         6                    True                   4000x3000                  3000x4000            76.90            72.746552   -4.153448        4.153448
 20250204_133404.jpg                         6                    True                   4000x2252                  2252x4000            67.13            70.559273    3.429273        3.429273
```

### TOPIQ
```
            filename exif_orientation_original  exif_transpose_applied image_size_before_transpose image_size_after_transpose  topiq_mixed112_score_100  wsl_topiq_score_100  diff_topiq  abs_diff_topiq
 20250129_102520.jpg                         6                    True                   4000x2252                  2252x4000                     33.09            29.497197   -3.592803        3.592803
        IMG_2739.JPG                         8                    True                   1920x1280                  1280x1920                     48.57            52.048010    3.478010        3.478010
 20240505_153023.jpg                         6                    True                   4000x3000                  3000x4000                     68.19            65.209305   -2.980695        2.980695
   1715855266134.jpg                         6                    True                   3088x2316                  2316x3088                     61.31            58.410001   -2.899999        2.899999
1675564423029-24.jpg                         6                    True                   4032x3024                  3024x4032                     67.78            64.953744   -2.826256        2.826256
 20250129_103313.jpg                         6                    True                   4000x2252                  2252x4000                     51.46            48.721331   -2.738669        2.738669
 20250129_054504.jpg                         6                    True                   4000x3000                  3000x4000                     75.58            73.304200   -2.275800        2.275800
 20240510_191716.jpg                         6                    True                   4000x3000                  3000x4000                     74.89            72.734576   -2.155424        2.155424
        IMG_2729.JPG                         6                    True                   1920x1280                  1280x1920                     50.88            52.891517    2.011517        2.011517
        IMG_2785.JPG                         1                   False                   1920x1280                  1920x1280                     67.76            69.719690    1.959690        1.959690
```

### C6
```
            filename exif_orientation_original  exif_transpose_applied image_size_before_transpose image_size_after_transpose  candidate_c6_score  wsl_c6_score   diff_c6  abs_diff_c6
 20230201_181300.jpg                         6                    True                   4000x2252                  2252x4000               61.21     66.448689  5.238689     5.238689
 1675564423029-9.jpg                         6                    True                   4032x3024                  3024x4032               71.08     66.178074 -4.901926     4.901926
 20250129_102520.jpg                         6                    True                   4000x2252                  2252x4000               34.33     30.561009 -3.768991     3.768991
1675564423029-24.jpg                         6                    True                   4032x3024                  3024x4032               70.74     66.989572 -3.750428     3.750428
   1715855266134.jpg                         6                    True                   3088x2316                  2316x3088               62.68     58.942105 -3.737895     3.737895
 20250129_103313.jpg                         6                    True                   4000x2252                  2252x4000               51.04     47.875648 -3.164352     3.164352
 1675564423029-6.jpg                         6                    True                   4032x3024                  3024x4032               69.22     66.363593 -2.856407     2.856407
 20240510_191716.jpg                         6                    True                   4000x3000                  3000x4000               75.50     72.738169 -2.761831     2.761831
 20250205_072644.jpg                         6                    True                   4000x3000                  3000x4000               66.23     68.391454  2.161454     2.161454
        IMG_2785.JPG                         1                   False                   1920x1280                  1920x1280               66.95     69.032561  2.082561     2.082561
```

## 7. Remaining Error Analysis
- C6 top-5 ranking overlap: 3/5
- C6 top-10 ranking overlap: 9/10
- EXIF orientation handling removes the largest known Android-vs-WSL decode mismatch.
- Remaining C6 error is consistent with bilinear resize implementation differences and C6 branch sensitivity near the `min()` transition.
- Production replacement still requires stricter parity than log-only diagnostics.

## 8. Decision
C. FAIL, further preprocessing mismatch remains
