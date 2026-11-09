# MDIQA FLIVE Image-Test Sanity Report

## 1. Executive Summary
- Final decision: `MDIQA_FLIVE_PROMISING_BUT_NEEDS_FULL_TEST`.
- N evaluated: `1000`.
- Elapsed seconds: `32.4`.

## 2. Inputs
- Input manifest: `/home/omen_pc1/photo_score_project/outputs/research_protocol_iqa_manifests_20260530/benchmark_all_research_protocol_iqa.csv`.
- Output directory: `/home/omen_pc1/photo_score_project/outputs/mdiqa_flive_image_test_sanity_20260601`.

## 3. CUDA Preflight
- torch version: `2.12.0+cu130`.
- CUDA version: `13.0`.
- CUDA available: `True`.
- Device count: `1`.
- Device name: `NVIDIA GeForce RTX 4070 SUPER`.
- cuDNN version: `92000`.

## 4. FLIVE Filtering Rule
- Rule: `FLIVE only, content_type == image, split == test`.

## 5. Dataset Counts
- Valid candidates found: `1000`.
- Sample target: `1000`.
- Successfully evaluated: `1000`.

## 6. MDIQA Variant Used
- Variant: `mdiqa_xd`.
- Checkpoint: `mdiqa_shared_5t_4a_mbv3_koniq_xd.pth`.

## 7. Prediction Distribution
- Raw min/max/mean: `0.1702 / 0.8678 / 0.6483`.
- 0..100 min/max/mean: `17.02 / 86.78 / 64.83`.
- MOS min/max/mean: `21.78 / 89.84 / 71.81`.

## 8. Metrics Against MOS
- **All (n=1000):**
  - SRCC: `0.5666`
  - PLCC: `0.6715`
  - MAE (0..100): `8.0792`
  - Linear Aligned MAE: `4.3041`
- **Subset (n=100):**
  - SRCC: `0.5665`
  - PLCC: `0.5865`

## 9. Comparison Against Production and Paper
- Production FLIVE SRCC: `0.574`
- Paper FLIVE SRCC: `0.612`
- Delta vs Production: `-0.0074`
- Delta vs Paper: `-0.0454`

## 10. Sensitivity Checks
- SRCC(pred, MOS): `0.5666`
- SRCC(-pred, MOS): `-0.5666`
- SRCC(pred, 100-MOS): `-0.5666`
- Source group aggregation: N/A (all images unique).

## 11. Failure Cases
- None.

## 12. Final Decision
MDIQA_FLIVE_PROMISING_BUT_NEEDS_FULL_TEST

## 13. Recommended Next Step
- Check if larger sample improves SRCC stability.