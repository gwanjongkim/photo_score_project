# DistortionGuard-IQA v1 Stage B Evaluation

## 1. Summary
PARTIAL PASS

- Candidate: outputs/distortionguard_stageB2_partial_unfreeze_b3_from_stageB_e10_20260524
- Model load method: final_model.keras
- Decision: B. Add hard-FP guard

## 2. Training Result
- best_epoch: 10
- val_srcc: 0.7179
- val_plcc: 0.8531
- val_mae_100: 6.61
- val_rmse_100: 9.18
- mode_collapse: False
- Stage A transfer: loaded=3, skipped=6, mismatched=0

## 3. Test Metrics
| Dataset | Model | MAE | RMSE | SRCC | PLCC | Bias | Avg ms |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| flive | distortionguard_stageB | 5.12 | 7.39 | 0.3985 | 0.4695 | -2.33 | 9.74 |
| flive | koniq_mobile | 9.07 | 12.10 | 0.4472 | 0.5288 | -7.58 | 9.55 |
| flive | flive_mobile | 3.12 | 4.10 | 0.6303 | 0.7567 | 1.08 | 9.15 |
| flive | existing_avg | 4.98 | 6.54 | 0.5337 | 0.6461 | -3.25 | 0.00 |
| flive | topiq_mixed112 | 5.02 | 6.81 | 0.4688 | 0.5631 | -2.36 | 44.86 |
| flive | topiq_ranking | 6.56 | 8.93 | 0.5492 | 0.6269 | -3.05 | 44.73 |
| koniq | distortionguard_stageB | 7.30 | 9.78 | 0.8029 | 0.8059 | 3.10 | 10.13 |
| koniq | koniq_mobile | 5.59 | 7.19 | 0.8661 | 0.8913 | 1.18 | 14.20 |
| koniq | flive_mobile | 14.26 | 18.27 | 0.6881 | 0.7354 | 13.71 | 13.33 |
| koniq | existing_avg | 8.52 | 11.22 | 0.8639 | 0.8861 | 7.44 | 0.00 |
| koniq | topiq_mixed112 | 5.80 | 7.73 | 0.8625 | 0.8837 | 2.46 | 48.02 |
| koniq | topiq_ranking | 7.01 | 9.07 | 0.8448 | 0.8716 | 3.87 | 49.15 |
| spaq | distortionguard_stageB | 9.22 | 11.54 | 0.8497 | 0.8510 | 0.68 | 10.47 |
| spaq | koniq_mobile | 9.58 | 11.81 | 0.8555 | 0.8451 | -0.32 | 12.15 |
| spaq | flive_mobile | 20.88 | 24.98 | 0.8267 | 0.6815 | 18.42 | 11.63 |
| spaq | existing_avg | 12.87 | 15.55 | 0.8749 | 0.8436 | 9.05 | 0.00 |
| spaq | topiq_mixed112 | 7.91 | 10.04 | 0.8992 | 0.8931 | 1.30 | 46.81 |
| spaq | topiq_ranking | 7.73 | 10.03 | 0.8897 | 0.8924 | -1.20 | 46.90 |

## 4. Hard-FP v2 Behavior
| Model | Mean | Median | Count >55 | Count >65 |
| :--- | ---: | ---: | ---: | ---: |
| distortionguard_stageB | 63.68 | 67.45 | 36 | 28 |
| koniq_mobile | 47.84 | 47.88 | 5 | 0 |
| flive_mobile | 74.19 | 74.41 | 44 | 43 |
| existing_avg | 61.02 | 61.63 | 41 | 7 |
| topiq_mixed112 | 65.35 | 68.42 | 40 | 28 |
| topiq_ranking | 58.12 | 62.29 | 27 | 17 |
| techiqa_stage4 | 68.07 | 69.92 | 41 | 32 |

Top over-scored DistortionGuard hard-FP images:

| Rank | Filename | Score | Image path |
| ---: | :--- | ---: | :--- |
| 1 | AVA__281607.jpg | 74.79 | /home/omen_pc1/photo_score_project/data/raw/flive/voc_emotic_ava/AVA__281607.jpg |
| 2 | EMOTIC__3ejdmzqxrcxiglybzz.jpg | 74.24 | /home/omen_pc1/photo_score_project/data/raw/flive/voc_emotic_ava/EMOTIC__3ejdmzqxrcxiglybzz.jpg |
| 3 | EMOTIC__COCO_train2014_000000440339.jpg | 72.63 | /home/omen_pc1/photo_score_project/data/raw/flive/voc_emotic_ava/EMOTIC__COCO_train2014_000000440339.jpg |
| 4 | JPEGImages__2008_001456.jpg | 72.57 | /home/omen_pc1/photo_score_project/data/raw/flive/voc_emotic_ava/JPEGImages__2008_001456.jpg |
| 5 | EMOTIC__caaozx7olrfcc4laiu.jpg | 72.50 | /home/omen_pc1/photo_score_project/data/raw/flive/voc_emotic_ava/EMOTIC__caaozx7olrfcc4laiu.jpg |
| 6 | EMOTIC__sun_aihqexaleoggapet.jpg | 72.25 | /home/omen_pc1/photo_score_project/data/raw/flive/voc_emotic_ava/EMOTIC__sun_aihqexaleoggapet.jpg |
| 7 | motion0089.jpg | 72.19 | /home/omen_pc1/photo_score_project/data/raw/flive/blur_dataset/motion0089.jpg |
| 8 | EMOTIC__ayo9ttwo44ekls4v7n.jpg | 71.86 | /home/omen_pc1/photo_score_project/data/raw/flive/voc_emotic_ava/EMOTIC__ayo9ttwo44ekls4v7n.jpg |
| 9 | EMOTIC__sun_afkzndehpwgjdxcc.jpg | 71.46 | /home/omen_pc1/photo_score_project/data/raw/flive/voc_emotic_ava/EMOTIC__sun_afkzndehpwgjdxcc.jpg |
| 10 | EMOTIC__2d8m251io0nvtitrlm.jpg | 71.29 | /home/omen_pc1/photo_score_project/data/raw/flive/voc_emotic_ava/EMOTIC__2d8m251io0nvtitrlm.jpg |

## 5. Comparison to Existing Models
- flive: DistortionGuard SRCC 0.3985 / PLCC 0.4695 vs best local baseline flive_mobile SRCC 0.6303 / PLCC 0.7567.
- koniq: DistortionGuard SRCC 0.8029 / PLCC 0.8059 vs best local baseline koniq_mobile SRCC 0.8661 / PLCC 0.8913.
- spaq: DistortionGuard SRCC 0.8497 / PLCC 0.8510 vs best local baseline topiq_mixed112 SRCC 0.8992 / PLCC 0.8931.
- hard-FP v2: DistortionGuard mean 63.68 vs existing_avg mean 61.02.
- hard-FP v2: DistortionGuard mean 63.68 vs TechIQA-Guard Stage 4 mean 68.07.
- TechIQA-Guard Stage 4 MOS test-set predictions were not available in the local baseline summaries.

## 6. Decision
B. Add hard-FP guard

## 7. Recommended Next Step
Keep the Stage B candidate for comparison, then add a hard-FP guard before any export work.
