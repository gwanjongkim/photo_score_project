# MDIQA Official Variant 100-image Sanity Benchmark Report

## 1. Executive Summary
- Final decision: `MDIQA_PARTIAL_TEACHER_CANDIDATE`.
- Evaluated variants: `koniq_seed1, koniq_xd, net_best`.
- Evaluated 300 sampled images with 900 predictions.
- Elapsed seconds: `466.3`.
- Scope stayed benchmark-only: no training, TFLite export, Flutter edits, production scoring edits, distillation, teacher-label creation, or full benchmark.

## 2. Inputs
- Input manifest: `/home/omen_pc1/photo_score_project/outputs/research_protocol_iqa_manifests_20260530/benchmark_all_research_protocol_iqa.csv`.
- Output directory: `/home/omen_pc1/photo_score_project/outputs/mdiqa_official_variant_sanity_20260601`.
- Seed: `20260601`.
- Sample target: `100` per dataset.
- Batch size: `8`.

## 3. Checkpoint Inventory
- Inventory CSV: `/home/omen_pc1/photo_score_project/outputs/mdiqa_official_variant_sanity_20260601/checkpoint_inventory.csv`.
- Missing required variants: `none`.
- Missing shared checkpoints: `0`.

## 4. CUDA / Environment Notes
- CUDA report: `/home/omen_pc1/photo_score_project/outputs/mdiqa_cuda_diagnosis_20260601/report.md`.
- torch: `2.12.0+cu130`.
- torch.version.cuda: `13.0`.
- torch.cuda.is_available: `False`.
- torch.cuda.device_count: `0`.
- device used: `cpu`.
- device name: ``.
- Use clean command pattern: `env -u LD_LIBRARY_PATH ./.venv_mdiqa/bin/python scripts/technical_iqa/run_mdiqa_official_variant_sanity_benchmark.py`.

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
- koniq_seed1 / FLIVE: raw min=0.4583, max=0.6691, mean=0.5347, std=0.0549; 0..100 mean=53.4662
- koniq_seed1 / KonIQ-10k: raw min=0.4619, max=0.6824, mean=0.6070, std=0.0460; 0..100 mean=60.6990
- koniq_seed1 / SPAQ: raw min=0.4469, max=0.6470, mean=0.5474, std=0.0623; 0..100 mean=54.7390
- koniq_xd / FLIVE: raw min=0.2068, max=0.8101, mean=0.4295, std=0.1586; 0..100 mean=42.9464
- koniq_xd / KonIQ-10k: raw min=0.2162, max=0.8002, mean=0.6242, std=0.1248; 0..100 mean=62.4163
- koniq_xd / SPAQ: raw min=0.1249, max=0.7304, mean=0.4431, std=0.1904; 0..100 mean=44.3066
- net_best / FLIVE: raw min=0.4555, max=0.6647, mean=0.5302, std=0.0549; 0..100 mean=53.0205
- net_best / KonIQ-10k: raw min=0.4635, max=0.6677, mean=0.5985, std=0.0445; 0..100 mean=59.8474
- net_best / SPAQ: raw min=0.4386, max=0.6361, mean=0.5382, std=0.0624; 0..100 mean=53.8158

## 7. Dataset-wise Metrics
| Variant | Dataset | n | SRCC | PLCC | MAE raw*100 | Linear MAE | Failures |
|---|---|---:|---:|---:|---:|---:|---:|
| koniq_seed1 | FLIVE | 100 | 0.0334 | 0.0323 | 18.1307 | 4.8772 | 0 |
| koniq_seed1 | KonIQ-10k | 100 | 0.9451 | 0.9565 | 6.7414 | 2.1858 | 0 |
| koniq_seed1 | SPAQ | 100 | 0.9307 | 0.9266 | 12.9390 | 5.9754 | 0 |
| koniq_xd | FLIVE | 100 | 0.0698 | 0.0647 | 29.1117 | 4.8952 | 0 |
| koniq_xd | KonIQ-10k | 100 | 0.9310 | 0.9540 | 3.8416 | 2.1202 | 0 |
| koniq_xd | SPAQ | 100 | 0.9343 | 0.9265 | 7.6398 | 5.9730 | 0 |
| net_best | FLIVE | 100 | 0.0279 | 0.0304 | 18.5353 | 4.8785 | 0 |
| net_best | KonIQ-10k | 100 | 0.9339 | 0.9503 | 7.3581 | 2.2955 | 0 |
| net_best | SPAQ | 100 | 0.9310 | 0.9255 | 12.7989 | 6.0762 | 0 |

## 8. Best Variant by Dataset
| Dataset | Best Variant | Best SRCC | Production SRCC | Delta |
|---|---|---:|---:|---:|
| FLIVE | koniq_xd | 0.0698 | 0.574 | -0.5042 |
| KonIQ-10k | koniq_seed1 | 0.9451 | 0.934 | 0.0111 |
| SPAQ | koniq_xd | 0.9343 | 0.873 | 0.0613 |

## 9. Comparison Against Production Baseline
- FLIVE: best `koniq_xd` SRCC 0.0698 vs production 0.574 (delta -0.5042).
- KonIQ-10k: best `koniq_seed1` SRCC 0.9451 vs production 0.934 (delta 0.0111).
- SPAQ: best `koniq_xd` SRCC 0.9343 vs production 0.873 (delta 0.0613).

## 10. Failure Cases
- None.

## 11. Final Decision
MDIQA_PARTIAL_TEACHER_CANDIDATE

## 12. Recommended Next Step
- Keep only datasets whose best variant beats production for the next bounded check; do not create labels yet.
