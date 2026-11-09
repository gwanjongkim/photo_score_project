# TOPIQ Label Dry-Run Inference Report

- selected TOPIQ model: topiq_nr
- available TOPIQ model names: topiq_fr, topiq_fr-pipal, topiq_iaa, topiq_iaa_res50, topiq_nr, topiq_nr-face, topiq_nr-face-v1, topiq_nr-flive, topiq_nr-spaq, topiq_nr_swin-face
- torch version: 2.5.1+cu121
- cuda availability: True
- total attempted: 9000
- total succeeded: 9000
- total failed: 0
- topiq_raw min: 0.04038259
- topiq_raw max: 0.82348019
- topiq_raw mean: 0.46632712
- overall SRCC vs quality_score_0_100: 0.73945328
- overall PLCC vs quality_score_0_100: 0.78509719

## Dataset Metrics

| dataset | n | SRCC vs quality_score_0_100 | PLCC vs quality_score_0_100 |
|---|---:|---:|---:|
| flive | 3000 | 0.28918339 | 0.32825478 |
| koniq10k | 3000 | 0.90392817 | 0.89968384 |
| spaq | 3000 | 0.82830590 | 0.83136926 |
