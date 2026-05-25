# TOPIQ-RankCalib v2 Stage 3 Probe B Distillation Evaluation

## 3. Full Test Metrics Summary

| Dataset | Model | Head | MAE | RMSE | SRCC | PLCC | Bias | Std Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| flive | mixed_112_frozen | default | 5.02 | 6.82 | 0.469 | 0.563 | -2.37 | 1.22 |
| koniq | mixed_112_frozen | default | 5.80 | 7.73 | 0.863 | 0.884 | 2.45 | 0.85 |
| spaq | mixed_112_frozen | default | 7.91 | 10.04 | 0.899 | 0.893 | 1.30 | 0.83 |
| flive | ranking_lam01_gap05 | default | 5.17 | 7.04 | 0.496 | 0.586 | -2.27 | 1.33 |
| koniq | ranking_lam01_gap05 | default | 5.84 | 7.78 | 0.864 | 0.885 | 2.72 | 0.91 |
| spaq | ranking_lam01_gap05 | default | 7.73 | 9.84 | 0.898 | 0.895 | 0.59 | 0.86 |
| flive | rankcalib_v2_stage2 | **unified** | 4.92 | 6.73 | 0.444 | 0.533 | -1.53 | 1.21 |
| flive | rankcalib_v2_stage2 | flive | 4.26 | 5.59 | 0.509 | 0.574 | -1.23 | 0.95 |
| flive | rankcalib_v2_stage2 | koniq | 10.20 | 12.66 | 0.441 | 0.516 | -9.28 | 1.65 |
| flive | rankcalib_v2_stage2 | spaq | 13.34 | 16.45 | 0.413 | 0.460 | -12.09 | 2.08 |
| koniq | rankcalib_v2_stage2 | **unified** | 6.53 | 8.84 | 0.827 | 0.849 | 3.18 | 0.87 |
| koniq | rankcalib_v2_stage2 | flive | 12.64 | 16.24 | 0.700 | 0.748 | 11.86 | 0.49 |
| koniq | rankcalib_v2_stage2 | koniq | 5.72 | 7.41 | 0.855 | 0.882 | 0.13 | 0.93 |
| koniq | rankcalib_v2_stage2 | spaq | 9.28 | 11.81 | 0.704 | 0.723 | -1.23 | 1.02 |
| spaq | rankcalib_v2_stage2 | **unified** | 8.76 | 11.10 | 0.878 | 0.868 | 1.98 | 0.82 |
| spaq | rankcalib_v2_stage2 | flive | 19.55 | 23.21 | 0.802 | 0.769 | 16.53 | 0.39 |
| spaq | rankcalib_v2_stage2 | koniq | 10.54 | 13.23 | 0.846 | 0.830 | 4.83 | 0.76 |
| spaq | rankcalib_v2_stage2 | spaq | 8.22 | 10.59 | 0.878 | 0.878 | 0.44 | 0.82 |
| flive | rankcalib_v2_stage3_probeA | **unified** | 5.13 | 6.98 | 0.470 | 0.556 | -1.90 | 1.29 |
| flive | rankcalib_v2_stage3_probeA | flive | 4.27 | 5.60 | 0.510 | 0.575 | -1.25 | 0.96 |
| flive | rankcalib_v2_stage3_probeA | koniq | 10.19 | 12.67 | 0.442 | 0.517 | -9.26 | 1.66 |
| flive | rankcalib_v2_stage3_probeA | spaq | 13.22 | 16.36 | 0.414 | 0.461 | -11.90 | 2.09 |
| koniq | rankcalib_v2_stage3_probeA | **unified** | 6.48 | 8.69 | 0.834 | 0.855 | 3.09 | 0.89 |
| koniq | rankcalib_v2_stage3_probeA | flive | 12.60 | 16.19 | 0.702 | 0.749 | 11.81 | 0.50 |
| koniq | rankcalib_v2_stage3_probeA | koniq | 5.70 | 7.39 | 0.856 | 0.882 | 0.17 | 0.93 |
| koniq | rankcalib_v2_stage3_probeA | spaq | 9.28 | 11.80 | 0.705 | 0.724 | -1.04 | 1.03 |
| spaq | rankcalib_v2_stage3_probeA | **unified** | 8.46 | 10.78 | 0.881 | 0.874 | 1.38 | 0.85 |
| spaq | rankcalib_v2_stage3_probeA | flive | 19.48 | 23.14 | 0.803 | 0.769 | 16.47 | 0.39 |
| spaq | rankcalib_v2_stage3_probeA | koniq | 10.47 | 13.16 | 0.847 | 0.831 | 4.74 | 0.77 |
| spaq | rankcalib_v2_stage3_probeA | spaq | 8.20 | 10.57 | 0.877 | 0.878 | 0.52 | 0.83 |
| flive | **rankcalib_v2_stage3_probeB** | **unified** | 5.15 | 7.01 | 0.471 | 0.557 | -2.00 | 1.29 |
| flive | **rankcalib_v2_stage3_probeB** | flive | 4.27 | 5.60 | 0.510 | 0.575 | -1.25 | 0.96 |
| flive | **rankcalib_v2_stage3_probeB** | koniq | 10.19 | 12.67 | 0.442 | 0.517 | -9.26 | 1.66 |
| flive | **rankcalib_v2_stage3_probeB** | spaq | 13.23 | 16.36 | 0.414 | 0.461 | -11.91 | 2.09 |
| koniq | **rankcalib_v2_stage3_probeB** | **unified** | 6.46 | 8.66 | 0.834 | 0.855 | 3.03 | 0.89 |
| koniq | **rankcalib_v2_stage3_probeB** | flive | 12.60 | 16.20 | 0.701 | 0.749 | 11.81 | 0.50 |
| koniq | **rankcalib_v2_stage3_probeB** | koniq | 5.70 | 7.39 | 0.856 | 0.882 | 0.16 | 0.93 |
| koniq | **rankcalib_v2_stage3_probeB** | spaq | 9.28 | 11.80 | 0.705 | 0.724 | -1.06 | 1.03 |
| spaq | **rankcalib_v2_stage3_probeB** | **unified** | 8.46 | 10.78 | 0.880 | 0.873 | 1.29 | 0.84 |
| spaq | **rankcalib_v2_stage3_probeB** | flive | 19.49 | 23.15 | 0.803 | 0.769 | 16.48 | 0.39 |
| spaq | **rankcalib_v2_stage3_probeB** | koniq | 10.48 | 13.17 | 0.846 | 0.831 | 4.75 | 0.77 |
| spaq | **rankcalib_v2_stage3_probeB** | spaq | 8.21 | 10.58 | 0.877 | 0.878 | 0.52 | 0.83 |
