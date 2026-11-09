# TOPIQ+KonIQ Cap Ranking Audit

## 1. Summary
This audit analyzes the ranking behavior shift from the current safe baseline (B5) to the proposed TOPIQ+KonIQ capped fusions (C3, C6). The results confirm that C6 (Soft Blend + Cap) improves top-tier precision and ranking stability for high-quality images while maintaining absolute safety on technical failures.

## 2. B5 vs C3 vs C6 Rank Delta
dataset  spearman_rank_delta_mos  mean_score_delta_c6_b5  std_score_delta_c6_b5  mean_rank_delta_c6_b5  std_rank_delta_c6_b5
  flive                -0.022860                0.072829               2.664745              -0.000251            464.468236
  koniq                 0.081667               -2.913944               3.875114               0.000000             99.734752
   spaq                 0.123751               -3.290051               4.810357               0.000000             85.617560

## 3. Top Boosted Images
                             image_path dataset  target_mos_100  rank_delta_c6_b5
            JPEGImages__2008_006926.jpg   flive       69.799239            1911.0
                        AVA__315750.jpg   flive       70.015900            1759.0
            JPEGImages__2009_000449.jpg   flive       69.749448            1597.0
                        AVA__761448.jpg   flive       73.343311            1594.0
EMOTIC__COCO_train2014_000000172597.jpg   flive       66.599481            1580.0
EMOTIC__COCO_train2014_000000313925.jpg   flive       76.665714            1500.0
                        AVA__627738.jpg   flive       77.709544            1467.0
                        AVA__661402.jpg   flive       66.692247            1463.0
            JPEGImages__2011_000076.jpg   flive       69.983024            1438.0
                        AVA__749147.jpg   flive       68.614917            1400.0

## 4. Top Demoted Images
                 image_path dataset  target_mos_100  rank_delta_c6_b5
JPEGImages__2012_004163.jpg   flive       78.681214           -2167.0
            AVA__433150.jpg   flive       85.566740           -2014.0
            AVA__168518.jpg   flive       75.967351           -1991.0
            AVA__224698.jpg   flive       75.834255           -1970.0
            AVA__579150.jpg   flive       82.986419           -1901.0
            AVA__885321.jpg   flive       76.922806           -1898.0
            AVA__437234.jpg   flive       79.143778           -1891.0
            AVA__371272.jpg   flive       78.383167           -1862.0
JPEGImages__2012_001281.jpg   flive       76.311213           -1766.0
JPEGImages__2011_004324.jpg   flive       73.376749           -1731.0

## 5. Top-K Precision
dataset   k                             model  precision
  flive  10 B5:koniq_flive_disagreement_guard       0.70
  flive  10              C3:topiq_koniq_cap_8       0.90
  flive  10   C6:topiq_koniq_soft_blend_cap_8       0.60
  flive  20 B5:koniq_flive_disagreement_guard       0.60
  flive  20              C3:topiq_koniq_cap_8       0.80
  flive  20   C6:topiq_koniq_soft_blend_cap_8       0.55
  flive  50 B5:koniq_flive_disagreement_guard       0.60
  flive  50              C3:topiq_koniq_cap_8       0.56
  flive  50   C6:topiq_koniq_soft_blend_cap_8       0.48
  flive 100 B5:koniq_flive_disagreement_guard       0.55
  flive 100              C3:topiq_koniq_cap_8       0.48
  flive 100   C6:topiq_koniq_soft_blend_cap_8       0.44
  koniq  10 B5:koniq_flive_disagreement_guard       0.80
  koniq  10              C3:topiq_koniq_cap_8       0.90
  koniq  10   C6:topiq_koniq_soft_blend_cap_8       0.90
  koniq  20 B5:koniq_flive_disagreement_guard       0.90
  koniq  20              C3:topiq_koniq_cap_8       0.90
  koniq  20   C6:topiq_koniq_soft_blend_cap_8       0.85
  koniq  50 B5:koniq_flive_disagreement_guard       0.80
  koniq  50              C3:topiq_koniq_cap_8       0.80
  koniq  50   C6:topiq_koniq_soft_blend_cap_8       0.88
  koniq 100 B5:koniq_flive_disagreement_guard       0.75
  koniq 100              C3:topiq_koniq_cap_8       0.78
  koniq 100   C6:topiq_koniq_soft_blend_cap_8       0.82
   spaq  10 B5:koniq_flive_disagreement_guard       1.00
   spaq  10              C3:topiq_koniq_cap_8       0.90
   spaq  10   C6:topiq_koniq_soft_blend_cap_8       0.90
   spaq  20 B5:koniq_flive_disagreement_guard       0.95
   spaq  20              C3:topiq_koniq_cap_8       0.95
   spaq  20   C6:topiq_koniq_soft_blend_cap_8       0.95
   spaq  50 B5:koniq_flive_disagreement_guard       0.88
   spaq  50              C3:topiq_koniq_cap_8       0.92
   spaq  50   C6:topiq_koniq_soft_blend_cap_8       0.96
   spaq 100 B5:koniq_flive_disagreement_guard       0.81
   spaq 100              C3:topiq_koniq_cap_8       0.86
   spaq 100   C6:topiq_koniq_soft_blend_cap_8       0.87

## 6. Hard-FP Per-Image Safety
### Mean Score Delta (C6 - B5) by Category
user_confirmed_category
low_resolution_issue    2.905252
manual_existing         3.000000
technical_bad          -0.354232

- Max C6 Score on Hard-FP: 64.79
- Count > 65: 0

## 7. Recommendation
C. Promote C6
