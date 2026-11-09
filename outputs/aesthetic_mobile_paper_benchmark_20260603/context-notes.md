# Context Notes - Mobile Aesthetic Paper Benchmark

## Decisions
- Using `data/processed/aadb/test.csv` for AADB labels as it's already pre-processed into a clean format.
- Using `external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv` for ICAA17K labels.
- Using `external/icaa_official_repo/SPAQ_code/dataset/SPAQ/spaqTest.csv` for SPAQ labels.
- NIMA AVA ground truth: `vote_1` to `vote_10` in `ground_truth_dataset.csv` are probabilities. Mean score = sum(prob_i * i).

## Issues & Fixes
- **NIMA gt_score**: Initially tried to sum `vote_1..10` but `data/processed/ava/test.csv` contains counts, not probabilities. Fixed to use `mean_score` column directly.
- **A-LAMP score range**: Metadata confirms A-LAMP v2 output is in 0..1 range. Threshold for binary classification set to 0.5.
- **ICAA SPAQ Benchmark**: Correlation is lower (SRCC 0.32) because SPAQ labels are for "Colorfulness" while ICAA model predicts "Color Aesthetics".
- **AVA Test Set**: Using 25,552 images from `data/processed/ava/test.csv` which is a standard test split.
- **A-LAMP Patch Protocol**: Implemented center + 4 corners proxy protocol as original adaptive patches logic was not available.
