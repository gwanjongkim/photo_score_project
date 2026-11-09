# MDIQA 100-image Sanity Benchmark Report

## 1. Executive Summary
- Final decision: `MDIQA_NEEDS_VARIANT_CHECKPOINTS`.
- Evaluated 300 sampled images with 300 predictions.
- Elapsed seconds: `10.7`.
- Scope stayed benchmark-only: no training, TFLite export, Flutter edits, production scoring edits, distillation, or teacher-label creation.

## 2. Inputs
- Input manifest: `/home/omen_pc1/photo_score_project/outputs/research_protocol_iqa_manifests_20260530/benchmark_all_research_protocol_iqa.csv`.
- Output directory: `/home/omen_pc1/photo_score_project/outputs/mdiqa_sanity_20260601`.
- Seed: `20260601`.
- Sample target: `100` per dataset.
- Batch size: `8`.

## 3. CUDA Diagnosis Summary
- CUDA report: `/home/omen_pc1/photo_score_project/outputs/mdiqa_cuda_diagnosis_20260601/report.md`.
- torch: `2.12.0+cu130`.
- torch.version.cuda: `13.0`.
- torch.cuda.is_available: `True`.
- torch.cuda.device_count: `1`.
- device used: `cuda`.
- device name: `NVIDIA GeForce RTX 4070 SUPER`.
- Torch reinstall needed: `no`.

## 4. Checkpoint Used
- Variant: `Koniq MDIQA checkpoint set`.
- fusion: `/home/omen_pc1/photo_score_project/MDIQA/checkpoints/mdiqa_shared_5t_4a_mbv3_koniq_seed1.pth` (19930987 bytes)
- technology_backbone: `/home/omen_pc1/photo_score_project/MDIQA/checkpoints/net_freeze_backbone.pth` (144754849 bytes)
- technology_head0: `/home/omen_pc1/photo_score_project/MDIQA/checkpoints/net_best_koniq10k_val_head0_head0.pth` (36428809 bytes)
- technology_head1: `/home/omen_pc1/photo_score_project/MDIQA/checkpoints/net_best_koniq10k_val_head1_head1.pth` (36428809 bytes)
- technology_head2: `/home/omen_pc1/photo_score_project/MDIQA/checkpoints/net_best_koniq10k_val_head2_head2.pth` (36428809 bytes)
- technology_head3: `/home/omen_pc1/photo_score_project/MDIQA/checkpoints/net_best_koniq10k_val_head3_head3.pth` (36428809 bytes)
- technology_head4: `/home/omen_pc1/photo_score_project/MDIQA/checkpoints/net_best_koniq10k_val_head4_head4.pth` (36428809 bytes)
- aesthetic_color: `/home/omen_pc1/photo_score_project/MDIQA/checkpoints/net_best_PARA_val_color.pth` (290502209 bytes)

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

## 6. Raw Prediction Distributions
- FLIVE: raw min=0.4582, max=0.6690, mean=0.5347, std=0.0549; 0..100 mean=53.4651
- KonIQ-10k: raw min=0.4619, max=0.6824, mean=0.6069, std=0.0459; 0..100 mean=60.6916
- SPAQ: raw min=0.4470, max=0.6468, mean=0.5473, std=0.0623; 0..100 mean=54.7335

## 7. Dataset-wise Metrics
| Dataset | n | SRCC | PLCC | MAE raw*100 | Linear MAE | Failures |
|---|---:|---:|---:|---:|---:|---:|
| FLIVE | 100 | 0.0328 | 0.0323 | 18.1318 | 4.8772 | 0 |
| KonIQ-10k | 100 | 0.9452 | 0.9566 | 6.7471 | 2.1843 | 0 |
| SPAQ | 100 | 0.9303 | 0.9266 | 12.9421 | 5.9757 | 0 |

## 8. Comparison Against Production Baseline
| Dataset | Production SRCC | MDIQA Koniq SRCC | Delta |
|---|---:|---:|---:|
| FLIVE | 0.574 | 0.0328 | -0.5412 |
| KonIQ-10k | 0.934 | 0.9452 | 0.0112 |
| SPAQ | 0.873 | 0.9303 | 0.0573 |

## 9. Failure Cases
- None.

## 10. Final Decision
MDIQA_NEEDS_VARIANT_CHECKPOINTS

## 11. Recommended Next Step
- Download/check SPAQ and FLIVE MDIQA checkpoints before judging MDIQA as a multi-dataset teacher.
