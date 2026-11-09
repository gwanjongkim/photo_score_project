# TOPIQ PyIQA Teacher Validation Report

- selected TOPIQ model: topiq_nr
- torch version: 2.5.1+cu121
- cuda availability: True
- total attempted: 900
- total succeeded: 900
- total failed: 0
- topiq_raw min: 0.04682437
- topiq_raw max: 0.80431986
- topiq_raw mean: 0.46138373
- overall SRCC vs quality_score_0_100: 0.69626484
- overall PLCC vs quality_score_0_100: 0.76897425
- score direction conclusion: higher topiq_raw appears higher-is-better
- recommendation: NEEDS_TARGET_GUARD_DESIGN_FIRST

## Dataset Metrics

| dataset | n | SRCC vs quality_score_0_100 | PLCC vs quality_score_0_100 |
|---|---:|---:|---:|
| flive | 300 | 0.25095968 | 0.31502268 |
| koniq10k | 300 | 0.85657898 | 0.87700276 |
| spaq | 300 | 0.80304544 | 0.81400952 |

## Score Direction

Higher-is-better is considered plausible only if correlations remain positive at overall and dataset level.

## Top 20 Possible False Positives

| rank | dataset | split | quality_score_0_100 | topiq_raw | rank_gap | image_path |
|---:|---|---|---:|---:|---:|---|
| 1 | flive | val | 50.14844907 | 0.65807128 | 0.66407119 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/blur_dataset/motion0176.jpg |
| 2 | spaq | test | 36.60000000 | 0.61265641 | 0.65072303 | /home/omen_pc1/photo_score_project/data/raw/spaq/10997.jpg |
| 3 | spaq | test | 54.10000000 | 0.67102355 | 0.64738598 | /home/omen_pc1/photo_score_project/data/raw/spaq/10775.jpg |
| 4 | spaq | test | 57.80000000 | 0.69375676 | 0.64404894 | /home/omen_pc1/photo_score_project/data/raw/spaq/10830.jpg |
| 5 | flive | val | 52.26123045 | 0.65463912 | 0.63737486 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/patches/AVA__833668.jpg_patch_3.jpg |
| 6 | flive | test | 48.50266433 | 0.63510394 | 0.63181313 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/patches/AVA__864117.jpg_patch_3.jpg |
| 7 | flive | test | 52.22895142 | 0.64950132 | 0.62847608 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/patches/AVA__910372.jpg_patch_2.jpg |
| 8 | spaq | test | 53.80000000 | 0.64492053 | 0.60789766 | /home/omen_pc1/photo_score_project/data/raw/spaq/11123.jpg |
| 9 | spaq | test | 61.20000000 | 0.69374049 | 0.58731924 | /home/omen_pc1/photo_score_project/data/raw/spaq/10858.jpg |
| 10 | flive | test | 55.53833126 | 0.63850105 | 0.57842047 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/patches/AVA__650563.jpg_patch_3.jpg |
| 11 | spaq | test | 58.40000000 | 0.66425818 | 0.57563960 | /home/omen_pc1/photo_score_project/data/raw/spaq/11022.jpg |
| 12 | flive | val | 60.67171733 | 0.67587930 | 0.57285873 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/blur_dataset/motion0255.jpg |
| 13 | spaq | test | 61.70000000 | 0.67515129 | 0.54949944 | /home/omen_pc1/photo_score_project/data/raw/spaq/10762.jpg |
| 14 | flive | val | 48.18843152 | 0.59950960 | 0.54838710 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/patches/AVA__793930.jpg_patch_3.jpg |
| 15 | spaq | test | 57.60000000 | 0.63214368 | 0.54282536 | /home/omen_pc1/photo_score_project/data/raw/spaq/10798.jpg |
| 16 | spaq | test | 27.30000000 | 0.53287774 | 0.52224694 | /home/omen_pc1/photo_score_project/data/raw/spaq/11023.jpg |
| 17 | flive | test | 47.67329874 | 0.58781403 | 0.51835373 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/patches/AVA__501778.jpg_patch_1.jpg |
| 18 | spaq | test | 53.30000000 | 0.60321933 | 0.51557286 | /home/omen_pc1/photo_score_project/data/raw/spaq/10512.jpg |
| 19 | spaq | test | 64.00000000 | 0.68374449 | 0.51279199 | /home/omen_pc1/photo_score_project/data/raw/spaq/10773.jpg |
| 20 | spaq | test | 64.10000000 | 0.68237442 | 0.50889878 | /home/omen_pc1/photo_score_project/data/raw/spaq/10738.jpg |

## Top 20 Possible False Negatives

| rank | dataset | split | quality_score_0_100 | topiq_raw | rank_gap | image_path |
|---:|---|---|---:|---:|---:|---|
| 1 | flive | test | 80.95953958 | 0.34593096 | -0.69632925 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/patches/AVA__834014.jpg_patch_2.jpg |
| 2 | flive | val | 78.96761016 | 0.34711114 | -0.65739711 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/patches/JPEGImages__2012_003435.jpg_patch_1.jpg |
| 3 | flive | val | 71.41795856 | 0.08532328 | -0.63626251 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/patches/out_of_focus0397.jpg_patch_2.jpg |
| 4 | spaq | test | 77.00000000 | 0.33529317 | -0.63181313 | /home/omen_pc1/photo_score_project/data/raw/spaq/10533.jpg |
| 5 | flive | val | 79.13560013 | 0.36750484 | -0.62736374 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/patches/EMOTIC__COCO_train2014_000000280980.jpg_patch_1.jpg |
| 6 | flive | test | 77.17858322 | 0.34823459 | -0.62068966 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/patches/AVA__954182.jpg_patch_2.jpg |
| 7 | spaq | test | 87.30000000 | 0.43845358 | -0.60511680 | /home/omen_pc1/photo_score_project/data/raw/spaq/11043.jpg |
| 8 | flive | val | 76.99583726 | 0.36224324 | -0.58954394 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/patches/VOC2012__2009_002059.jpg_patch_2.jpg |
| 9 | flive | val | 80.04198843 | 0.41905591 | -0.57842047 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/patches/EMOTIC__COCO_train2014_000000459589.jpg_patch_1.jpg |
| 10 | flive | val | 79.23313394 | 0.40848634 | -0.57508343 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/patches/EMOTIC__COCO_val2014_000000464515.jpg_patch_1.jpg |
| 11 | flive | val | 78.68077932 | 0.40349522 | -0.57508343 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/patches/JPEGImages__2009_002990.jpg_patch_2.jpg |
| 12 | spaq | test | 85.10000000 | 0.45132545 | -0.56840934 | /home/omen_pc1/photo_score_project/data/raw/spaq/10207.jpg |
| 13 | flive | val | 74.65579622 | 0.31680802 | -0.56840934 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/voc_emotic_ava/AVA__310505.jpg |
| 14 | flive | val | 77.66781890 | 0.39953795 | -0.56284761 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/patches/VOC2012__2008_008644.jpg_patch_1.jpg |
| 15 | flive | val | 76.86558406 | 0.38848522 | -0.55394883 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/patches/EMOTIC__COCO_train2014_000000429144.jpg_patch_2.jpg |
| 16 | spaq | test | 72.00000000 | 0.21045348 | -0.55172414 | /home/omen_pc1/photo_score_project/data/raw/spaq/10983.jpg |
| 17 | flive | val | 79.16329218 | 0.42619404 | -0.55061179 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/patches/VOC2012__2008_003047.jpg_patch_2.jpg |
| 18 | flive | val | 74.84697359 | 0.34590420 | -0.53948832 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/patches/EMOTIC__COCO_train2014_000000060035.jpg_patch_2.jpg |
| 19 | flive | val | 76.19518816 | 0.38928679 | -0.52947720 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/patches/EMOTIC__COCO_train2014_000000559055.jpg_patch_1.jpg |
| 20 | flive | val | 75.38814900 | 0.36416996 | -0.52836485 | /home/omen_pc1/photo_score_project/data/raw/paq2piq/voc_emotic_ava/VOC2012__2010_002294.jpg |
