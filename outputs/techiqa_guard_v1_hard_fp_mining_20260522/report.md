# TechIQA-Guard v1 Hard False-Positive Mining Report

## 1. Summary
The hard false-positive dataset expansion has identified **1,374 suspicious candidates** across FLIVE, KonIQ, SPAQ, and smoke test datasets. These images exhibit significant score gaps where general IQA models (Mixed112, Ranking) score them highly (often >70) while existing technical guards (KonIQ Mobile, FLIVE Mobile) score them much lower (often <50).

## 2. Current Problem
Stage 5 training failed to maintain false-positive protection because the existing confirmed hard-FP set was too small (only 3 unique images). The ranking loss signal from 10,000 FLIVE pairs completely overwhelmed the guard loss signal from these few samples. To generalize the "technical guard" behavior, a diverse set of at least 50-100 confirmed hard false-positives is required.

## 3. Sources Found
- `outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/`
- `outputs/eval_techiqa_guard_v1_stage4_fpdata_lam005_20260520/`
- `outputs/eval_techiqa_guard_v1_stage5*`
- `data/processed/techiqa_guard/hard_false_positive.csv`

## 4. Candidate Mining Rules
- **Existing Combined Score**: `average(koniq_mobile, flive_image_mobile)`
- **Candidate**: `Mixed112 - Existing >= 5.0` OR `Ranking_Lam01 - Existing >= 5.0`
- **Strong Candidate**: `Mixed112 - Existing >= 8.0` OR `Ranking_Lam01 - Existing >= 8.0`

## 5. Candidate Counts
- **Total Mined Rows**: 6,567
- **Hard-FP Candidates (Delta >= 5)**: 1,374
- **Strong Hard-FP Candidates (Delta >= 8)**: 504
- **Confirmed Hard-FP (Manual/Existing)**: 6

| Dataset | Candidate Count |
| :--- | :--- |
| FLIVE | 1,081 |
| KonIQ | 142 |
| SPAQ | 88 |
| Smoke | 60 |
| Hard-FP | 3 |

## 6. Top 30 Visual Review Queue (Sample)
These images show the highest positive delta (overscoring by general IQA).

| Filename | Existing Score | Delta Stage5-Stage4 | Source |
| :--- | :--- | :--- | :--- |
| AVA__781250.jpg | 31.01 | -0.05 | flive |
| AVA__928726.jpg | 33.75 | 0.34 | flive |
| AVA__830694.jpg | 20.94 | 2.90 | flive |
| AVA__42058.jpg | 46.86 | 2.21 | flive |
| motion0220.jpg | 34.49 | -0.04 | flive |
| AVA__504050.jpg | 51.24 | 1.21 | flive |
| EMOTIC__2d8m251io0nvtitrlm.jpg | 56.21 | 1.63 | flive |
| EMOTIC__4dqqr157rtt73iykdv.jpg | 52.21 | 3.06 | flive |
| EMOTIC__0yxxaznlgm6deps44j.jpg | 52.22 | 4.12 | flive |
| AVA__60509.jpg | 51.04 | 0.68 | flive |

## 7. Confirmed Hard-FP Set
- `20230201_181300.jpg` (Manual User Flagged)
- `1675342165226-13.jpg` (Manual User Flagged)
- `1675342165226-3.jpg` (Manual User Flagged)

## 8. Missing Images
- **Total Missing**: 0 (All candidates found on local disk)

## 9. Recommendation
**B. Need user visual review first.**

The mining process has provided a large pool of suspicious candidates. A visual audit of the `visual_review_queue.csv` is necessary to confirm which of these images are truly "bad quality" technically. Once 50-100 images are confirmed, we can rerun Stage 5 training with oversampling of this set.
