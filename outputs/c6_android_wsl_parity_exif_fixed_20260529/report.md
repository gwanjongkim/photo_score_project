# C6 Android vs WSL Parity Report

## 1. Summary
Comparison between Android Flutter TFLite implementation and WSL reference implementation after applying EXIF orientation in the WSL diagnostic loader.

## 2. Inputs
- Android Score CSV: `data/debug/c6_parity_20260527/android_c6_scores.csv`
- Image Directory: `test_vila`
- KonIQ Model: `exports/tflite/koniq_mobile.tflite`
- TOPIQ Model: `outputs/final_topiq_lite_mixed112_export_20260517/topiq_lite_mixed112_frozen_fp16.tflite`

## 3. Image Matching
- Total images in Android CSV: 50
- Matched and processed: 50
- Failed: 0
- Failed image details:
```
None
```

## 4. Model Contracts Used
### KonIQ
- Input: 224x224, Resize (Stretch)
- Normalization: pixel / 255.0
- Output: Raw (0..100)

### TOPIQ
- Input: 384x384, Resize with Pad
- Normalization: None (0..255)
- Output: Raw * 100 (0..100)

### Decode
- PIL image loading with `ImageOps.exif_transpose(img)` before `convert("RGB")`, resize, or model preprocessing.
- HEIF opener registered: True

### C6 Formula
- `c6 = min(0.7 * topiq + 0.3 * koniq, koniq + 8.0)`

## 5. Per-Model Parity Metrics
| Model | MAE | Max AE | Median AE | SRCC |
| :--- | :--- | :--- | :--- | :--- |
| KonIQ | 2.0578 | 11.9568 | 1.4740 | 0.9142 |
| TOPIQ | 1.1965 | 3.5928 | 1.0157 | 0.9753 |
| **C6** | **1.3470** | **5.2387** | **0.8212** | **0.9690** |

## 6. Before vs After
| Run | C6 MAE | C6 Max AE | C6 SRCC |
| :--- | :--- | :--- | :--- |
| Original | 2.3256 | 7.5531 | 0.9118 |
| New EXIF-fixed | 1.3470 | 5.2387 | 0.9690 |

## 7. Top 10 Remaining Mismatches (C6)
```
            filename exif_orientation  candidate_c6_score  wsl_c6_score  abs_diff_c6  diff_koniq  diff_topiq android_c6_branch wsl_c6_branch mismatch_attribution
 20230201_181300.jpg                6               61.21     66.448689     5.238689    5.238689    1.380332         koniq_cap     koniq_cap                KonIQ
 1675564423029-9.jpg                6               71.08     66.178074     4.901926  -11.956807   -1.882691             blend         blend                KonIQ
 20250129_102520.jpg                6               34.33     30.560998     3.769002   -4.186764   -3.592818             blend         blend                TOPIQ
1675564423029-24.jpg                6               70.74     66.989572     3.750428   -5.910162   -2.826256             blend         blend                TOPIQ
   1715855266134.jpg                6               62.68     58.942105     3.737895   -5.696319   -2.899999             blend         blend                TOPIQ
 20250129_103313.jpg                6               51.04     47.875652     3.164348   -4.157614   -2.738663             blend         blend                TOPIQ
 1675564423029-6.jpg                6               69.22     66.363593     2.856407   -5.152545   -1.879491             blend         blend                KonIQ
 20240510_191716.jpg                6               75.50     72.738173     2.761827   -4.153448   -2.155418             blend         blend                TOPIQ
 20250205_072644.jpg                6               66.23     68.391463     2.161463    4.649925    0.993551         koniq_cap         blend        Branch Switch
        IMG_2785.JPG                1               66.95     69.032561     2.082561    2.359260    1.959690             blend         blend                TOPIQ
```

## 8. Remaining Mismatch Diagnosis
- NaN present: False
- Score scale inversion: False
- All-image attribution counts:
```
mismatch_attribution
TOPIQ            33
KonIQ            15
Branch Switch     2
```
- Top-10 attribution counts:
```
mismatch_attribution
TOPIQ            6
KonIQ            3
Branch Switch    1
```
- Top-10 EXIF orientation counts:
```
exif_orientation
6    9
1    1
```
- Main remaining mismatch driver: TOPIQ

## 9. Decision
C. FAIL, more preprocessing mismatch remains

Next smallest diagnostic step: inspect the top remaining `TOPIQ` cases by exporting Android and WSL post-decode/post-resize tensors for one top mismatch and one low-mismatch control image.
