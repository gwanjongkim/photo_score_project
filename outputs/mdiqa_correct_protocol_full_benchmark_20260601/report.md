# MDIQA Correct-Protocol Full Raw Benchmark Report

## 1. Executive Summary
- Final decision: `MDIQA_KONIQ_SPAQ_ONLY_PARTIAL_CANDIDATE`.
- N evaluated: `23102`.
- Elapsed seconds: `528.2`.

## 2. Inputs
- Input manifest: `/home/omen_pc1/photo_score_project/outputs/research_protocol_iqa_manifests_20260530/benchmark_all_research_protocol_iqa.csv`.
- Output directory: `/home/omen_pc1/photo_score_project/outputs/mdiqa_correct_protocol_full_benchmark_20260601`.

## 3. CUDA Preflight
- torch version: `2.12.0+cu130`.
- CUDA version: `13.0`.
- CUDA available: `True`.
- Device count: `1`.
- Device name: `NVIDIA GeForce RTX 4070 SUPER`.
- cuDNN version: `92000`.

## 4. Checkpoint Inventory
- koniq10k (mdiqa_koniq): `mdiqa_shared_5t_4a_mbv3_koniq_seed1.pth` [PRESENT]
- spaq (mdiqa_xd): `mdiqa_shared_5t_4a_mbv3_koniq_xd.pth` [PRESENT]
- flive (mdiqa_xd): `mdiqa_shared_5t_4a_mbv3_koniq_xd.pth` [PRESENT]

## 5. Dataset Filtering Protocol
- KonIQ-10k: `split == test`
- SPAQ: `split == test`
- FLIVE: `content_type == image` AND `split == test` (strictly no patches)

## 6. Dataset Counts
| Dataset | Total | Kept | Excl Patch | Excl Non-Test | Missing/Unread |
|---|---|---|---|---|---|
| koniq10k | 100730 | 20150 | 0 | 80580 | 0 |
| spaq | 11125 | 1125 | 0 | 10000 | 0 |
| flive | 159240 | 1827 | 119430 | 37983 | 0 |

## 7. Prediction Distributions
- **flive**: raw `0.1667..0.8678` (mean `0.6465`); 0..100 mean `64.65`
- **koniq10k**: raw `0.4346..0.6906` (mean `0.5980`); 0..100 mean `59.80`
- **spaq**: raw `0.1180..0.7999` (mean `0.4363`); 0..100 mean `43.63`

## 8. Metrics Against MOS
| Dataset | n | SRCC | PLCC | MAE (0..100) | Aligned MAE |
|---|---|---|---|---|---|
| flive | 1827 | `0.5692` | `0.6637` | `8.3321` | `4.3171` |
| koniq10k | 20150 | `0.9464` | `0.9568` | `6.3065` | `2.4365` |
| spaq | 1125 | `0.9136` | `0.9187` | `7.9342` | `6.6582` |

## 9. Comparison Against Production Baseline
- flive: `0.5692` vs `0.574` (delta `-0.0048`)
- koniq10k: `0.9464` vs `0.934` (delta `+0.0124`)
- spaq: `0.9136` vs `0.873` (delta `+0.0406`)

## 10. FLIVE Comparison Against Paper
- FLIVE SRCC: `0.5692` vs paper `0.612` (delta `-0.0428`)

## 11. Sensitivity Checks
- **flive**:
  - SRCC(pred, MOS): `0.5692`
  - SRCC(-pred, MOS): `-0.5692`
  - SRCC(pred, 100-MOS): `-0.5692`
- **koniq10k**:
  - SRCC(pred, MOS): `0.9464`
  - SRCC(-pred, MOS): `-0.9464`
  - SRCC(pred, 100-MOS): `-0.9464`
- **spaq**:
  - SRCC(pred, MOS): `0.9136`
  - SRCC(-pred, MOS): `-0.9136`
  - SRCC(pred, 100-MOS): `-0.9136`

## 12. Failure Cases
- None.

## 13. Final Decision
MDIQA_KONIQ_SPAQ_ONLY_PARTIAL_CANDIDATE

## 14. Recommended Next Step
- Use MDIQA as teacher for KonIQ/SPAQ only; keep existing FLIVE teacher or labels.