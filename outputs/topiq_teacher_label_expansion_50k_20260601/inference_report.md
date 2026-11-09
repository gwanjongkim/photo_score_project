# TOPIQ 50k Label Expansion Inference Report

- selected TOPIQ model: topiq_nr
- available TOPIQ model names: topiq_fr, topiq_fr-pipal, topiq_iaa, topiq_iaa_res50, topiq_nr, topiq_nr-face, topiq_nr-face-v1, topiq_nr-flive, topiq_nr-spaq, topiq_nr_swin-face
- torch version: 2.5.1+cu121
- cuda availability: True
- total attempted: 50000
- total succeeded: 50000
- total failed: 0
- topiq_raw min: 0.03039709
- topiq_raw max: 0.85269964
- topiq_raw mean: 0.49680206
- overall SRCC vs quality_score_0_100: 0.58485690
- overall PLCC vs quality_score_0_100: 0.69483953

## Dataset Metrics

| dataset | n | SRCC vs quality_score_0_100 | PLCC vs quality_score_0_100 |
|---|---:|---:|---:|
| flive | 32000 | 0.32518731 | 0.35436761 |
| koniq10k | 9000 | 0.90131478 | 0.89579223 |
| spaq | 9000 | 0.82251088 | 0.82580296 |
