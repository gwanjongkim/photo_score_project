# MDIQA Variant 100-image Sanity Benchmark Report

## 1. Executive Summary
- Final decision: `MDIQA_PARTIAL_TEACHER_CANDIDATE`.
- Evaluated variants: `mdiqa_koniq, mdiqa_xd, mdiqa_unified`.
- Evaluated 300 sampled images with 900 predictions.
- Elapsed seconds: `26.3`.
- Scope stayed benchmark-only: no training, TFLite export, Flutter edits, production scoring edits, distillation, teacher-label creation, or full benchmark.

## 2. Inputs
- Input manifest: `/home/omen_pc1/photo_score_project/outputs/research_protocol_iqa_manifests_20260530/benchmark_all_research_protocol_iqa.csv`.
- Output directory: `/home/omen_pc1/photo_score_project/outputs/mdiqa_variant_sanity_20260601`.
- Seed: `20260601`.
- Sample target: `100` per dataset.
- Batch size: `8`.

## 3. Checkpoint Inventory
- Inventory CSV: `/home/omen_pc1/photo_score_project/outputs/mdiqa_variant_sanity_20260601/checkpoint_inventory.csv`.
- Missing required variants: `none`.
- Missing shared checkpoints: `0`.

## 4. CUDA / Environment Notes
- CUDA report: `/home/omen_pc1/photo_score_project/outputs/mdiqa_cuda_diagnosis_20260601/report.md`.
- torch: `2.12.0+cu130`.
- torch.version.cuda: `13.0`.
- torch.cuda.is_available: `True`.
- torch.cuda.device_count: `1`.
- device used: `cuda`.
- device name: `NVIDIA GeForce RTX 4070 SUPER`.
- Use clean command pattern: `env -u LD_LIBRARY_PATH ./.venv_mdiqa/bin/python scripts/technical_iqa/run_mdiqa_variant_sanity_benchmark.py`.

## 5. Dataset Sampling
- FLIVE: 100 images.
  - test: 4
  - train: 79
  - val: 17
- KonIQ-10k: 100 images.
  - test: 17
  - train: 83
- SPAQ: 100 images.
  - test: 9
  - train: 91

## 6. Variant Prediction Distributions
- mdiqa_koniq / FLIVE: raw min=0.4582, max=0.6690, mean=0.5347, std=0.0549; 0..100 mean=53.4651
- mdiqa_koniq / KonIQ-10k: raw min=0.4619, max=0.6824, mean=0.6069, std=0.0459; 0..100 mean=60.6916
- mdiqa_koniq / SPAQ: raw min=0.4470, max=0.6468, mean=0.5473, std=0.0623; 0..100 mean=54.7335
- mdiqa_unified / FLIVE: raw min=0.4555, max=0.6646, mean=0.5302, std=0.0549; 0..100 mean=53.0190
- mdiqa_unified / KonIQ-10k: raw min=0.4635, max=0.6677, mean=0.5984, std=0.0445; 0..100 mean=59.8448
- mdiqa_unified / SPAQ: raw min=0.4386, max=0.6362, mean=0.5382, std=0.0623; 0..100 mean=53.8157
- mdiqa_xd / FLIVE: raw min=0.2067, max=0.8099, mean=0.4293, std=0.1585; 0..100 mean=42.9327
- mdiqa_xd / KonIQ-10k: raw min=0.2160, max=0.8001, mean=0.6239, std=0.1248; 0..100 mean=62.3929
- mdiqa_xd / SPAQ: raw min=0.1248, max=0.7301, mean=0.4429, std=0.1903; 0..100 mean=44.2875

## 7. Dataset-wise Metrics
| Variant | Dataset | n | SRCC | PLCC | MAE raw*100 | Linear MAE | Failures |
|---|---|---:|---:|---:|---:|---:|---:|
| mdiqa_koniq | FLIVE | 100 | 0.0328 | 0.0323 | 18.1318 | 4.8772 | 0 |
| mdiqa_koniq | KonIQ-10k | 100 | 0.9452 | 0.9566 | 6.7471 | 2.1843 | 0 |
| mdiqa_koniq | SPAQ | 100 | 0.9303 | 0.9266 | 12.9421 | 5.9757 | 0 |
| mdiqa_unified | FLIVE | 100 | 0.0282 | 0.0306 | 18.5365 | 4.8786 | 0 |
| mdiqa_unified | KonIQ-10k | 100 | 0.9339 | 0.9503 | 7.3594 | 2.2946 | 0 |
| mdiqa_unified | SPAQ | 100 | 0.9311 | 0.9254 | 12.8009 | 6.0781 | 0 |
| mdiqa_xd | FLIVE | 100 | 0.0676 | 0.0647 | 29.1213 | 4.8952 | 0 |
| mdiqa_xd | KonIQ-10k | 100 | 0.9310 | 0.9541 | 3.8562 | 2.1183 | 0 |
| mdiqa_xd | SPAQ | 100 | 0.9341 | 0.9265 | 7.6517 | 5.9728 | 0 |

## 8. Best Variant by Dataset
| Dataset | Best Variant | Best SRCC | Production SRCC | Delta |
|---|---|---:|---:|---:|
| FLIVE | mdiqa_xd | 0.0676 | 0.574 | -0.5064 |
| KonIQ-10k | mdiqa_koniq | 0.9452 | 0.934 | 0.0112 |
| SPAQ | mdiqa_xd | 0.9341 | 0.873 | 0.0611 |

## 9. Comparison Against Production Baseline
- FLIVE: best `mdiqa_xd` SRCC 0.0676 vs production 0.574 (delta -0.5064).
- KonIQ-10k: best `mdiqa_koniq` SRCC 0.9452 vs production 0.934 (delta 0.0112).
- SPAQ: best `mdiqa_xd` SRCC 0.9341 vs production 0.873 (delta 0.0611).

## 10. Failure Cases
- None.

## 11. Final Decision
MDIQA_PARTIAL_TEACHER_CANDIDATE

## 12. Recommended Next Step
- Keep only datasets whose best variant beats production for the next bounded check; do not create labels yet.
