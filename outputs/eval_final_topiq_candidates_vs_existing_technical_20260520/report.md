# Final Benchmark: TOPIQ-lite Candidates vs Existing Technical Models

## 2. Scaling Audit
| Model | Input | Preprocess | Raw output | score_100 conversion | Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| koniq_mobile | 224x224 | resize (stretch), /255.0 | 0..100 MOS | score = raw | metadata interpretation |
| flive_image_mobile | 224x224 | resize (stretch), /255.0 | 0..100 MOS | score = raw | metadata interpretation |
| TOPIQ mixed112 | 384x384 | resize_with_pad, NO /255.0 | 0..1 normalized | score = raw * 100 | context notes, trainer logs |
| TOPIQ ranking_lam01 | 384x384 | resize_with_pad, NO /255.0 | 0..1 normalized | score = raw * 100 | context notes, trainer logs |

## 3. Benchmark Metrics

| Dataset | Model | MAE | RMSE | SRCC | PLCC | Bias | Std Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| flive | existing_avg_technical | 4.98 | 6.54 | 0.534 | 0.646 | -3.25 | 1.19 |
| flive | flive_image_mobile | 3.12 | 4.10 | 0.630 | 0.757 | 1.08 | 0.75 |
| flive | koniq_mobile | 9.07 | 12.10 | 0.447 | 0.529 | -7.58 | 1.84 |
| flive | **topiq_lite_mixed112_frozen_fp16** | 5.02 | 6.81 | 0.469 | 0.563 | -2.36 | 1.22 |
| flive | **topiq_lite_ranking_lam01_gap05_fp16** | 6.56 | 8.93 | 0.549 | 0.627 | -3.05 | 1.78 |
| koniq | existing_avg_technical | 8.52 | 11.22 | 0.864 | 0.886 | 7.44 | 0.61 |
| koniq | flive_image_mobile | 14.26 | 18.27 | 0.688 | 0.735 | 13.71 | 0.36 |
| koniq | koniq_mobile | 5.59 | 7.19 | 0.866 | 0.891 | 1.18 | 0.92 |
| koniq | **topiq_lite_mixed112_frozen_fp16** | 5.80 | 7.73 | 0.863 | 0.884 | 2.46 | 0.85 |
| koniq | **topiq_lite_ranking_lam01_gap05_fp16** | 7.01 | 9.07 | 0.845 | 0.872 | 3.87 | 1.06 |
| spaq | existing_avg_technical | 12.87 | 15.55 | 0.875 | 0.844 | 9.05 | 0.63 |
| spaq | flive_image_mobile | 20.88 | 24.98 | 0.827 | 0.681 | 18.42 | 0.44 |
| spaq | koniq_mobile | 9.58 | 11.81 | 0.856 | 0.845 | -0.32 | 0.91 |
| spaq | **topiq_lite_mixed112_frozen_fp16** | 7.91 | 10.04 | 0.899 | 0.893 | 1.30 | 0.83 |
| spaq | **topiq_lite_ranking_lam01_gap05_fp16** | 7.73 | 10.03 | 0.890 | 0.892 | -1.20 | 0.94 |
