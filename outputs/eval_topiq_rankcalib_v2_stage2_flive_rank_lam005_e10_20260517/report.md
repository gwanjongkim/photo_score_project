# TOPIQ-RankCalib v2 Stage 1 Regression Probe Full Evaluation

## 3. Metrics Summary

| Dataset | Model | Head | MAE | RMSE | SRCC | PLCC | Bias | Std Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| flive | mixed_112_frozen | default | 5.02 | 6.82 | 0.469 | 0.563 | -2.37 | 1.22 |
| koniq | mixed_112_frozen | default | 5.79 | 7.73 | 0.863 | 0.884 | 2.45 | 0.85 |
| spaq | mixed_112_frozen | default | 7.91 | 10.04 | 0.899 | 0.893 | 1.29 | 0.83 |
| flive | ranking_lam01_gap05_e10 | default | 6.57 | 8.94 | 0.549 | 0.627 | -3.06 | 1.78 |
| koniq | ranking_lam01_gap05_e10 | default | 7.01 | 9.07 | 0.845 | 0.872 | 3.86 | 1.06 |
| spaq | ranking_lam01_gap05_e10 | default | 7.73 | 10.03 | 0.890 | 0.892 | -1.21 | 0.94 |
| flive | rankcalib_v2_stage1 | unified | 5.08 | 6.99 | 0.433 | 0.521 | -1.90 | 1.23 |
| flive | rankcalib_v2_stage1 | flive | 3.73 | 5.10 | 0.478 | 0.548 | 0.28 | 0.64 |
| flive | rankcalib_v2_stage1 | koniq | 10.40 | 12.85 | 0.435 | 0.510 | -9.55 | 1.64 |
| flive | rankcalib_v2_stage1 | spaq | 10.39 | 13.44 | 0.402 | 0.452 | -7.72 | 2.04 |
| koniq | rankcalib_v2_stage1 | unified | 6.44 | 8.63 | 0.823 | 0.846 | 2.21 | 0.88 |
| koniq | rankcalib_v2_stage1 | flive | 14.35 | 18.43 | 0.659 | 0.726 | 13.68 | 0.34 |
| koniq | rankcalib_v2_stage1 | koniq | 5.80 | 7.51 | 0.849 | 0.878 | 0.20 | 0.92 |
| koniq | rankcalib_v2_stage1 | spaq | 9.64 | 12.10 | 0.696 | 0.719 | 2.82 | 1.01 |
| spaq | rankcalib_v2_stage1 | unified | 8.69 | 11.04 | 0.878 | 0.868 | 1.60 | 0.83 |
| spaq | rankcalib_v2_stage1 | flive | 22.14 | 26.18 | 0.801 | 0.760 | 19.57 | 0.31 |
| spaq | rankcalib_v2_stage1 | koniq | 10.51 | 13.21 | 0.844 | 0.828 | 4.62 | 0.76 |
| spaq | rankcalib_v2_stage1 | spaq | 9.11 | 11.53 | 0.877 | 0.876 | 4.52 | 0.84 |
| flive | rankcalib_v2_stage2 | unified | 5.06 | 6.94 | 0.439 | 0.527 | -2.14 | 1.21 |
| flive | rankcalib_v2_stage2 | flive | 5.38 | 7.12 | 0.534 | 0.595 | -0.99 | 1.44 |
| flive | rankcalib_v2_stage2 | koniq | 10.06 | 12.53 | 0.444 | 0.519 | -9.06 | 1.67 |
| flive | rankcalib_v2_stage2 | spaq | 13.47 | 16.72 | 0.406 | 0.453 | -12.20 | 2.12 |
| koniq | rankcalib_v2_stage2 | unified | 6.42 | 8.63 | 0.829 | 0.849 | 2.54 | 0.86 |
| koniq | rankcalib_v2_stage2 | flive | 12.60 | 15.65 | 0.725 | 0.763 | 11.94 | 0.73 |
| koniq | rankcalib_v2_stage2 | koniq | 5.59 | 7.27 | 0.860 | 0.886 | 0.47 | 0.92 |
| koniq | rankcalib_v2_stage2 | spaq | 9.49 | 12.09 | 0.707 | 0.725 | -2.25 | 1.05 |
| spaq | rankcalib_v2_stage2 | unified | 8.58 | 10.89 | 0.880 | 0.871 | 1.31 | 0.83 |
| spaq | rankcalib_v2_stage2 | flive | 16.40 | 19.86 | 0.802 | 0.778 | 13.48 | 0.56 |
| spaq | rankcalib_v2_stage2 | koniq | 10.55 | 13.24 | 0.846 | 0.831 | 4.84 | 0.76 |
| spaq | rankcalib_v2_stage2 | spaq | 8.07 | 10.38 | 0.882 | 0.883 | 1.07 | 0.85 |
