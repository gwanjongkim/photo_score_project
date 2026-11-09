# Paper-Protocol Technical IQA Audit for A-cut

## 1. Executive Summary
Local datasets can reproduce actual paper protocols for **KonIQ-10k** and **SPAQ**. For **PaQ-2-PiQ (FLIVE)**, reproduction is partial because the official 30k/7.7k/1.8k split file is missing locally, although images and scores are available. **AVA**, **AADB**, and **ICAA17K** are confirmed as aesthetic datasets and should be excluded from the technical IQA benchmark.

## 2. Paper Protocols
- **MUSIQ protocol**:
  - KonIQ-10k: Random 80% train, 20% test, average 10 runs.
  - SPAQ: Official train/test split, resize shorter side to 512, use image + MOS.
  - PaQ-2-PiQ: Official 30k train, 7.7k validation, 1.8k test split.
- **PaQ-2-PiQ original protocol**:
  - Same 30k/7.7k/1.8k split as above.
  - Training/validation pictures padded to 640x640 with white borders.
- **Common NR-IQA 70/10/20 protocol**:
  - Random 70/10/20 train/val/test split.
  - Fallback if official splits are missing.

## 3. Local Dataset Audit

### koniq10k
- **Files found**: `images/` directory, `koniq10k_scores_and_distributions.csv`, `koniq10k_indicators.csv`.
- **Annotations found**: `MOS` and `SD` columns in `koniq10k_scores_and_distributions.csv`.
- **Split found**: No fixed split (protocol requires random sampling).
- **Score columns found**: `MOS` (Scale 1-5).
- **Protocol reproduction**: YES.

### spaq
- **Files found**: images in `data/raw/spaq/`, official split CSVs in `external/icaa_official_repo/SPAQ_code/dataset/SPAQ/`.
- **Annotations found**: `MOS` in `spaqTrain.csv` and `spaqTest.csv`.
- **Split found**: Official train (10,000) and test (1,125).
- **Score columns found**: `MOS` (Scale 1-10).
- **Protocol reproduction**: YES.

### paq2piq / flive
- **Files found**: `labels_image.csv`, `labels_patch.csv`, `patches_rebuilt.csv`.
- **Annotations found**: `mos` column in `labels_image.csv`.
- **Split found**: Only 80/10/10 in `data/processed/`. Official 30k/7.7k/1.8k split file is MISSING.
- **Score columns found**: `mos` (Scale 0-100).
- **Protocol reproduction**: PARTIAL (Missing official split file).

### aadb
- **Files found**: `AADBinfo.mat`, `images/`.
- **Protocol reproduction**: NO (Confirmed aesthetic dataset).

### ava
- **Files found**: `ground_truth_dataset.csv`, `images/` (via symlink to `/home/omen_pc1/ava`).
- **Protocol reproduction**: NO (Confirmed aesthetic dataset).

### icaa17k
- **Files found**: `complementary/`, `monochromatic/`.
- **Protocol reproduction**: NO (Confirmed aesthetic dataset).

## 4. Reproducibility Matrix
(See `protocol_reproducibility_matrix.csv` for details)

| Dataset | Protocol | Exact Possible | Missing Items | Include in Benchmark |
|---------|----------|----------------|---------------|----------------------|
| KonIQ-10k | MUSIQ | YES | None | YES |
| SPAQ | MUSIQ | YES | None | YES |
| PaQ-2-PiQ | MUSIQ | PARTIAL | 30k/7.7k/1.8k split file | YES |
| FLIVE | NR-IQA | YES | None | YES (as fallback) |

## 5. Recommended Protocol Order
1. **MUSIQ protocol for KonIQ-10k and SPAQ**: These are ready and fully reproducible.
2. **NR-IQA 70/10/20 for FLIVE**: Use as fallback until the official PaQ-2-PiQ split is recovered.
3. **PaQ-2-PiQ original protocol**: Postpone until official split file is restored.

## 6. Next Action
**B. Perform deeper audit on missing split/annotation files**
- Focus on locating or reconstructing the 30,000 / 7,700 / 1,800 split for PaQ-2-PiQ.
- Check if any other external tool or script in the repo can generate this split.

## 7. Forbidden Actions
- No training performed.
- No custom splits created for datasets with official protocols.
- Aesthetic datasets (AVA, AADB, ICAA17K) excluded from technical benchmark.
- Production `technical_score` unchanged.
