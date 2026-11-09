# Source register

Retrieved 2026-07-11. “Verified” means the cited page/PDF/repository was opened during this audit. Exact URLs are retained so Phase 3 can reproduce every lookup.

| ID | Method | Source | Type | Exact location used | URL | Status | Confidence |
|---|---|---|---|---|---|---|---|
| S01 | NIMA | *NIMA: Neural Image Assessment*, Talebi & Milanfar | arXiv v2 / IEEE TIP 2018 paper | Sec. II-C; Sec. III; Eq. 1; Tables I, III–V | https://arxiv.org/abs/1709.05424 | VERIFIED_PRIMARY_SOURCE | HIGH |
| S02 | AVA | *AVA: A Large-Scale Database for Aesthetic Visual Analysis*, Murray, Marchesotti, Perronnin | CVPR 2012 paper | dataset construction and benchmark protocol; also method-paper descriptions where split is method-defined | https://ieeexplore.ieee.org/document/6247954 | SOURCE_INACCESSIBLE | MEDIUM |
| S03 | AADB | *Photo Aesthetics Ranking Network with Attributes and Content Adaptation*, Kong et al. | ECCV 2016 paper | Secs. 3, 4.1–4.5, 5.1–5.3; Eq. 1; Tables 2, 3, 6, 7 | https://www.ics.uci.edu/~fowlkes/papers/kslmf-eccv16.pdf | VERIFIED_PRIMARY_SOURCE | HIGH |
| S04 | RGNet | *Composition-Aware Image Aesthetics Assessment*, Liu et al. | WACV 2020 paper | Secs. 3.1–3.3, 4.1–4.3; Eqs. 1–5; Tables 1, 4, 5 | https://openaccess.thecvf.com/content_WACV_2020/papers/Liu_Composition-Aware_Image_Aesthetics_Assessment_WACV_2020_paper.pdf | VERIFIED_PRIMARY_SOURCE | HIGH |
| S05 | A-LAMP | *A-Lamp: Adaptive Layout-Aware Multi-Patch Deep Convolutional Neural Network for Photo Aesthetic Assessment*, Ma, Liu, Chen | CVPR 2017 paper | Secs. 3, 3.1.1, 3.1.2, 3.2, 4; Eqs. 1–8; Tables 1–2 | https://openaccess.thecvf.com/content_cvpr_2017/papers/Ma_A-Lamp_Adaptive_Layout-Aware_CVPR_2017_paper.pdf | VERIFIED_PRIMARY_SOURCE | HIGH |
| S06 | ICAA/DAT | *Thinking Image Color Aesthetics Assessment: Models, Datasets and Benchmarks*, He et al. | ICCV 2023 paper | Secs. 3.1–3.2, 4.1–4.2, 5.1; Eqs. 1–7; Figs. 2–4; Tables 2–6 | https://openaccess.thecvf.com/content/ICCV2023/papers/He_Thinking_Image_Color_Aesthetics_Assessment_Models_Datasets_and_Benchmarks_ICCV_2023_paper.pdf | VERIFIED_PRIMARY_SOURCE | HIGH |
| S07 | ICAA/DAT | Author repository | official code/data | `ICAA17K_code/`, `models/`, `dataset.py`, `main_nni.py`, `config.yml`; README run instructions | https://github.com/woshidandan/Image-Color-Aesthetics-and-Quality-Assessment | VERIFIED_OFFICIAL_CODE | HIGH |
| S08 | MUSIQ | *MUSIQ: Multi-Scale Image Quality Transformer*, Ke et al. | ICCV 2021 paper | Secs. 3.1–3.4, 4.1–4.3; Eq. 3; Fig. 2; Tables 1–4, 6–9 | https://openaccess.thecvf.com/content/ICCV2021/papers/Ke_MUSIQ_Multi-Scale_Image_Quality_Transformer_ICCV_2021_paper.pdf | VERIFIED_PRIMARY_SOURCE | HIGH |
| S09 | MUSIQ | Google Research `musiq` | author-maintained official code | package README and source tree | https://github.com/google-research/google-research/tree/master/musiq | VERIFIED_OFFICIAL_CODE | HIGH |
| S10 | KonIQ/KonCept512 | *KonIQ-10k: An Ecologically Valid Database for Deep Learning of BIQA*, Hosu et al. | IEEE TIP 2020 / arXiv paper | dataset protocol; KonCept512 architecture and Tables 7–10 | https://arxiv.org/abs/1910.06180 | VERIFIED_PRIMARY_SOURCE | HIGH |
| S11 | KonIQ | KonIQ-10k Database | official dataset documentation | downloads and score schema, lines headed “Downloads” and “scores file header” | https://database.mmsp-kn.de/koniq-10k-database.html | VERIFIED_DATASET_DOCUMENTATION | HIGH |
| S12 | FLIVE/PaQ-2-PiQ | *From Patches to Pictures (PaQ-2-PiQ): Mapping the Perceptual Space of Picture Quality*, Ying et al. | CVPR 2020 paper | Secs. 3–5; Figs. 3–5; dataset and result tables | https://openaccess.thecvf.com/content_CVPR_2020/papers/Ying_From_Patches_to_Pictures_PaQ-2-PiQ_Mapping_the_Perceptual_Space_of_CVPR_2020_paper.pdf | VERIFIED_PRIMARY_SOURCE | HIGH |
| S13 | FLIVE/PaQ-2-PiQ | `baidut/PaQ-2-PiQ` | author official code | README, `paq2piq/`, training/evaluation notebooks and released weights | https://github.com/baidut/PaQ-2-PiQ | VERIFIED_OFFICIAL_CODE | HIGH |
| S14 | MDIQA | *MDIQA: Unified Image Quality Assessment for Multi-dimensional Evaluation and Restoration*, Yao et al. | arXiv v1, 2025 | abstract/method/experiments; paper PDF link was opened only at abstract page | https://arxiv.org/abs/2508.16887 | VERIFIED_PRIMARY_SOURCE | MEDIUM |
| S15 | MDIQA | `YaoShunyu19/MDIQA` | author official code | README; `MDIQA/train_shared.py`; `train_MDIQA.py`; `options_MDIQA/**`; checkpoint instructions | https://github.com/YaoShunyu19/MDIQA | VERIFIED_OFFICIAL_CODE | HIGH |
| S16 | TOPIQ | *TOPIQ: A Top-down Approach from Semantics to Distortions for IQA*, Chen et al. | arXiv / IEEE TIP 2024 paper | Secs. III–IV; cross-scale attention; result tables | https://arxiv.org/abs/2308.03060 | VERIFIED_PRIMARY_SOURCE | HIGH |
| S17 | TOPIQ | `chaofengc/IQA-PyTorch` | author official code | `pyiqa/archs/topiq_arch.py`, metric configuration and pretrained-weight registry | https://github.com/chaofengc/IQA-PyTorch | VERIFIED_OFFICIAL_CODE | HIGH |
| S18 | Stage 5 / teacher-student | *Distilling the Knowledge in a Neural Network*, Hinton, Vinyals, Dean | arXiv 2015 paper | Secs. 2–3, softened targets and temperature | https://arxiv.org/abs/1503.02531 | SOURCE_INACCESSIBLE | MEDIUM |

No primary paper identity was found for the inventory labels `Stage 5`, `C6`, `RankCalib`, `TechIQA-Guard`, or `DistortionGuard`. They are not aliases for S18 merely because they involve students or calibration.

## Source-completeness review

| Check | Result |
|---|---|
| Every essential requirement primary-sourced? | Yes for NIMA, AADB, RGNet, A-LAMP, ICAA, MUSIQ, KonCept512, PaQ-2-PiQ and TOPIQ. MDIQA exact heads/recipe remain unresolved. Stage 5 has no verified paper identity, so no essentials were invented. |
| Any requirement inferred from local implementation? | No. Inventory supplied names only. |
| Paper/code differences recorded? | Yes, in `source_disagreements.md`; remaining file-level gaps are listed in `unresolved_paper_questions.md`. |
| Inaccessible sources marked? | Yes; none is marked verified. |
| Local results confused with paper results? | No. The results reference contains cited literature values only. |
| Exact locations recorded? | Yes where retrieved; missing code symbols/table cells are explicitly unresolved. |

Overall limitation: MDIQA’s full numeric recipe, TOPIQ code-level settings, ICAA head arity, and several legacy author repositories need further retrieval before exact reproduction.
