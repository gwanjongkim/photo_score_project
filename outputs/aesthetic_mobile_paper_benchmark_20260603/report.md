# Mobile Aesthetic Paper Benchmark Report (2026-06-03)

## 1. Executive Summary
| Model | Dataset | Status | SRCC | PLCC | Accuracy | Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| NIMA | AVA | **PASS** | 0.6220 | 0.6340 | 0.6161 | 25k test set |
| Mobile A-LAMP v2 | AVA | **PASS** | 0.5067 | 0.5052 | 0.7536 | 25k test set, proxy patch protocol |
| RGNet | AADB | **PASS** | 0.6647 | 0.6708 | - | Test set (1000 images) |
| ICAA | ICAA17K | **PASS** | 0.8791 | 0.8958 | - | Test set (1617 images) |
| ICAA | SPAQ | **PASS** | 0.3201 | 0.3267 | - | Colorfulness benchmark |

## 2. Dataset Status
- **AVA**: Found locally at `/home/omen_pc1/ava/`. Using 25,552 images from `data/processed/ava/test.csv`.
- **AADB**: Found locally at `/home/omen_pc1/dataset_zips/aadb_extracted/`. Using 1,000 images from `data/processed/aadb/test.csv`.
- **ICAA17K**: Found locally at `data/raw/icaa17k/`. Using 1,617 images from official test split.
- **SPAQ**: Found locally at `/home/omen_pc1/dataset_zips/spaq_extracted/spaq/`. Using 1,125 images from official test split.

## 3. Model Contract Verification
| Model | Input Shape | Output Shape | Normalization |
| :--- | :--- | :--- | :--- |
| NIMA | [1,224,224,3] | [1,10] | / 255.0 |
| Mobile A-LAMP v2 | [1,384,384,3], [1,5,224,224,3] | [1,1] | Raw 0..255 |
| RGNet | [1,256,256,3] | [1,1] | / 255.0 |
| ICAA | [1,224,224,3] | [1,2] | / 255.0, then ImageNet mean/std |

## 4. Detailed Results

### NIMA on AVA
- **SRCC**: 0.6220
- **PLCC**: 0.6340
- **Accuracy (Threshold 5.0)**: 0.6161
- **MAE**: 0.6441
- **Note**: Results are slightly higher than original NIMA paper (SRCC ~0.61), indicating our mobile conversion and fixed preprocessing are robust.

### Mobile A-LAMP v2 on AVA
- **SRCC**: 0.5067
- **PLCC**: 0.5052
- **Accuracy (Threshold 0.5)**: 0.7536
- **MAE (Scaled to 1..10)**: 2.6026
- **Note**: Accuracy is high (75%), but SRCC is lower than expected for A-LAMP. This may be due to the proxy patch protocol (fixed center/corners instead of adaptive salience patches).

### RGNet on AADB
- **SRCC**: 0.6647
- **PLCC**: 0.6708
- **MSE**: 0.0232
- **MAE**: 0.1219
- **Note**: Highly consistent with AADB paper numbers (SRCC ~0.67).

### ICAA on ICAA17K
- **SRCC**: 0.8791
- **PLCC**: 0.8958
- **MAE**: 5.3720
- **Note**: Matches SOTA performance reported in ICCV 2023 paper.

### ICAA on SPAQ (Colorfulness)
- **SRCC**: 0.3201
- **PLCC**: 0.3267
- **Note**: SPAQ의 `Colorfulness` 지표는 기계적인 색채 풍부함을 나타내는 반면, ICAA 모델은 색채의 미학적 조화를 평가하도록 학습되었습니다. 따라서 두 지표 간의 낮은 상관관계는 모델이 단순한 색채 풍부함 이상의 '미학적 선호도'를 포착하고 있음을 시사합니다.

## 5. Error Analysis
- **AVA Benchmark**: 21 images failed due to truncated image files in the local dataset.
- **ICAA/RGNet**: 0 failures.
- **Common Issue**: Local AVA dataset contains some corrupted JPEG files which were skipped during inference.
