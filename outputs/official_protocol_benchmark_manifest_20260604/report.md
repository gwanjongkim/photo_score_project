# Official Protocol Benchmark Manifest Audit Report

## 1. Executive Summary
The official-protocol benchmark manifest (Track B) is **complete and ready for model inference**. We have successfully identified official/repository-provided split definitions for KonIQ-10k, SPAQ, and FLIVE. Local image assets (100% availability) and labels/MOS have been matched to these splits. A combined official test manifest containing **10,448 rows** (including FLIVE patches) or **4,967 image-only rows** has been constructed.

**Final Decision: OFFICIAL_PROTOCOL_MANIFEST_READY_FOR_INFERENCE**

## 2. Sources Inspected
- `MDIQA/datasets/meta_info/meta_info_KonIQ10kDataset.csv`
- `outputs/research_protocol_iqa_manifests_20260530/benchmark_spaq_official_split.csv`
- `outputs/paq2piq_flive_toolbox_split_verify_20260530/meta_info_FLIVEDataset.csv`
- `data/raw/koniq10k/images/`
- `data/raw/spaq/`
- `data/raw/paq2piq/`

## 3. Official Split Inventory
| Dataset | Source Split File | Protocol Name |
| :--- | :--- | :--- |
| **KonIQ-10k** | `MDIQA/datasets/meta_info/meta_info_KonIQ10kDataset.csv` | Official 70/10/20 Split |
| **SPAQ** | `outputs/research_protocol_iqa_manifests_20260530/benchmark_spaq_official_split.csv` | Official 80/20 Random Split |
| **FLIVE** | `outputs/paq2piq_flive_toolbox_split_verify_20260530/meta_info_FLIVEDataset.csv` | PyIQA Standardized Train/Val/Test |

## 4. Dataset-specific Protocols
- **KonIQ-10k**: Official three-way split (Train: 7058, Val: 1000, Test: 2015).
- **SPAQ**: Two-way split (Train: 10000, Test: 1125).
- **FLIVE**: Standardized split (Train: 121024, Val: 30908, Test: 7308).
    - Image Test: 1827
    - Patch Test: 5481

## 5. Official Benchmark Manifest Construction
A total of **10,448** rows were selected for the official test manifest.
- **KonIQ-10k**: 2,015 rows
- **SPAQ**: 1,125 rows
- **FLIVE (Images)**: 1,827 rows
- **FLIVE (Patches)**: 5,481 rows

## 6. Image and Label Availability
- **Images**: 100% (10,448/10,448) confirmed on disk.
- **Labels (MOS)**: 100% available in source metadata files.

## 7. Internal vs Official Benchmark Overlap
Overlap with the internal Track A (1,413 rows) is **small (113 rows)**.
- **Reason**: The internal benchmark test set (sampled from the 14,105 distillation manifest) primarily uses images from the official **Train** or **Val** splits of the respective datasets.
- **KonIQ Overlap Analysis**: Out of 901 internal test rows, only 180 correspond to the official Test split; others are from Train (637) or Val (84).

## 8. Risks and Ambiguities
- **SPAQ Seed**: SPAQ splits are often re-generated randomly. The identified 1125 test rows match the MDIQA/PyIQA standard, but slight variations exist in literature.
- **FLIVE Patches**: Inference on 119k patches (Train+Val+Test) is time-consuming; the initial benchmark will prioritize image-only inference (4,967 images total).

## 9. Final Decision
**OFFICIAL_PROTOCOL_MANIFEST_READY_FOR_INFERENCE**

## 10. Recommended Next Step
Run MDIQA and Production inference on the **4,967 unique images** in the official test manifest to generate standardized performance metrics.
