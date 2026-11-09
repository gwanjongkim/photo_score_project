# C6 Android vs WSL Parity Report

## 1. Summary
Comparison between Android Flutter TFLite implementation and WSL reference implementation.

## 2. Inputs
- Android Score CSV: `data/debug/c6_parity_20260527/android_c6_scores.csv`
- Image Directory: `test_vila`
- KonIQ Model: `exports/tflite/koniq_mobile.tflite`
- TOPIQ Model: `outputs/final_topiq_lite_mixed112_export_20260517/topiq_lite_mixed112_frozen_fp16.tflite`

## 3. Image Matching
- Total images in Android CSV: 50
- Matched and processed: 50
- Failed: 0

## 4. Model Contracts Used
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
| KonIQ | 3.8662 | 13.4384 | 2.8312 | 0.8107 |
| TOPIQ | 2.6713 | 8.7807 | 2.5297 | 0.8715 |
| **C6** | **2.3256** | **7.5531** | **1.9838** | **0.9118** |

## 6. Top Mismatches (C6)
```
              filename  candidate_c6_score  wsl_c6_score   diff_c6
   20230201_181510.jpg               59.78     67.333081  7.553081
  1675564423029-24.jpg               70.74     63.313631 -7.426369
20250204_174237(0).jpg               75.24     67.871037 -7.368963
   20250205_000515.jpg               36.14     42.551060  6.411060
   20250204_135520.jpg               72.94     68.130104 -4.809896
```

## 7. Root Cause Notes
- Minor differences (AE < 0.1) are expected due to floating point precision and interpolation implementation differences between Android (TensorFlow Lite / Flutter) and WSL (TensorFlow Python).
- Systematic offsets might indicate preprocessing (resize/pad/normalization) mismatches.
- HEIF image (IMG_20240519_215827_510.jpg) decoded via pillow-heif.

## 8. Decision
C. FAIL, preprocessing mismatch must be fixed
