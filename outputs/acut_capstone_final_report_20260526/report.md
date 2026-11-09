# A-cut 캡스톤 프로젝트 모델 개발 최종 보고서

## 1. 프로젝트 개요
### 1.1 목표
A-cut 프로젝트는 사용자가 촬영하거나 선택한 다량의 사진 중 기술적, 미적 품질이 우수한 사진을 선별(A-cut selection)하고, AI를 통해 해당 사진의 추천 이유를 설명해 주는 모바일 지능형 앨범 시스템 개발을 목표로 합니다.

### 1.2 시스템 구성
- **Backend (Python/WSL)**: 모델 학습, 평가, TFLite 변환 및 데이터 파이프라인 관리.
- **Frontend (Flutter)**: 온디바이스 TFLite 모델 실행, 사진 선별 UI, Gemini API 연동 설명 생성.
- **개발 방향**: 실용적인 캡스톤 결과물 도출(Practical Capstone)을 최우선으로 하며, 학술적 재현(Paper Reproduction)은 별도 트랙으로 병행합니다.

---

## 2. 전체 파이프라인
1.  **이미지 입력**: 사용자가 사진을 선택하거나 실시간으로 캡처합니다.
2.  **기술적 품질 평가 (Technical Quality)**: TOPIQ-lite, FLIVE 등을 통해 노이즈, 흔들림, 노출 등을 평가합니다.
3.  **미적 품질 평가 (Aesthetic Quality)**: NIMA, RGNet, A-LAMP 계열 모델을 통해 구도, 색감, 조화 등 미적 요소를 평가합니다.
4.  **점수 통합 및 선별**: 기술/미적 점수를 가중치 기반으로 통합하여 사진을 '추천', '보통', '제외'로 분류합니다.
5.  **설명 생성**: 선별된 사진에 대해 온디바이스 VLM 또는 Gemini API를 호출하여 시각적 특징과 추천 이유를 생성합니다.
6.  **배포**: 최종 모델은 TFLite FP16으로 양자화되어 Flutter 앱에 탑재됩니다.

---

## 3. 실험 및 개발 타임라인
- **2026.04**: Stage 5 학생 모델(MobileNetV2) 증류 학습 및 AADB 데이터셋 기반 검증.
- **2026.05.10~17**: RGNet 및 A-LAMP 논문 재현 실험(Graph Fusion, DenseASPP 등) 수행.
- **2026.05.17~20**: TOPIQ-lite 기술적 품질 모델 벤치마크 및 모바일 최적화.
- **2026.05.24~25**: A-LAMP Multi-Patch 교사 모델 개선(Class Weight 도입) 및 NIMA 전처리 버그 수정.
- **2026.05.25**: ICAA 색채 미학 모델 TFLite 변환 및 성능 재검증 완료.

---

## 4. 모델별 개발 과정

### 4.1 NIMA (Neural IMage Assessment)
- **개요**: EfficientNetV2 기반의 미적 품질 평가 모델.
- **개선**: 기존 전처리 과정의 이중 스케일링 버그(EfficientNetV2 내장 전처리 중복)를 수정하여 성능을 비약적으로 향상시킴.
- **결과**: SRCC 0.5894 달성 (기존 0.3449 대비 +0.24). 모바일 배포용 TFLite FP16 변환 완료.

### 4.2 RGNet (Region-based Graph Network)
- **개요**: 이미지의 지역적 특징과 관계를 이용한 미적 품질 평가.
- **실험**: DenseASPP, 하이브리드 공간 인접성 등을 적용했으나, 기본 Mean Aggregation 모델이 AADB에서 SRCC 0.6819로 가장 안정적인 성능을 보임.
- **결과**: 실용적 버전의 RGNet을 캡스톤에 적용하기로 결정.

### 4.3 A-LAMP (Aesthetic Layout-Aware Multi-Patch Network)
- **개요**: 레이아웃과 멀티 패치를 결합한 미적 품질 평가.
- **실험**: 그래프 퓨전(GraphLite/GCN) 실험을 진행했으나 소규모 데이터셋(4k)에서 과적합 발생.
- **결과**: Multi-Patch VGG16 기반 교사 모델(Accuracy 0.7633)을 최종 실용 모델로 채택.

### 4.4 ICAA (Image Color Aesthetics Assessment)
- **개요**: 색채 조화와 미학에 특화된 모델.
- **검증**: ICAA17K 테스트셋에서 SRCC 0.8791을 기록하며 논문 성능을 완벽히 재현.
- **결과**: 색상 전용 평가 지표로 파이프라인에 통합 완료.

### 4.5 Stage 5 Student (Distillation)
- **개요**: RGNet, A-LAMP, NIMA 등을 앙상블한 교사 점수를 학습한 경량 MobileNetV2.
- **결과**: 1.7MB 크기의 초경량 모델로 AADB 검증셋에서 90.5%의 Pairwise Accuracy 기록. 현재 Flutter 앱의 메인 평가 모델.

### 4.6 기술적 품질 (TOPIQ-lite, FLIVE, KonIQ)
- **결과**: TOPIQ-lite(mixed112)가 SPAQ 데이터셋에서 SRCC 0.899를 기록하며 기존 FLIVE/KonIQ 모델 대비 우수한 성능과 범용성을 입증.

---

## 5. 모델 성능 비교

### 5.1 미적 품질 모델 (AVA/AADB Test)
| 모델명 | 데이터셋 | 주요 지표 (SRCC/Acc) | TFLite 배포 여부 | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| **A-LAMP Multi-Patch** | AVA | Acc 0.7633 | Yes (Teacher) | 현재 가장 강력한 교사 모델 |
| **RGNet AADB** | AADB | SRCC 0.6819 | Yes | 안정적인 회귀 성능 |
| **NIMA (Fixed)** | AVA | SRCC 0.5894 | Yes | 전처리 수정 후 신뢰도 급증 |
| **ICAA Color** | ICAA17K | SRCC 0.8791 | Yes | 색채 미학 전용 모델 |
| **Stage 5 Student** | AADB | Pairwise Acc 0.905 | Yes | 실시간 모바일 실행 가능 |

### 5.2 기술적 품질 모델 (IQA Benchmark)
| 모델명 | 데이터셋 | SRCC | PLCC | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| **TOPIQ-lite** | SPAQ | 0.899 | 0.893 | 범용 기술적 품질 우수 |
| **FLIVE Mobile** | FLIVE | 0.630 | 0.757 | 기존 모델 |
| **KonIQ Mobile** | KonIQ | 0.866 | 0.891 | 기존 모델 |

---

## 6. 논문 근접 실험과 실제 결과
- **A-LAMP**: 그래프 기반 레이아웃 퓨전은 소규모 데이터에서 유의미한 성능 향상을 보이지 못함. 공식 재현을 위해서는 더 큰 데이터셋과 정교한 하이퍼파라미터 튜닝이 필요함.
- **RGNet**: DenseASPP와 논문 레시피 적용 시 성능이 일부 회복되었으나, 현재의 베이스라인(SRCC 0.68)을 넘어서지는 못함.
- **결론**: 실용 캡스톤 버전에서는 검증된 베이스라인 모델들을 사용하며, 논문 수준의 완벽한 재현은 향후 연구 과제로 남김.

---

## 7. 최종 실용 캡스톤 구성 제안
- **선별 엔진**: `Stage 5 Student` (Aesthetic) + `TOPIQ-lite` (Technical) 조합을 기본으로 사용.
- **보조 지표**: 색감이 중요한 사진의 경우 `ICAA Color` 점수를 가중치에 추가 반영.
- **결과물 시연**: Flutter 앱에서 사진 로드 시 각 사진 옆에 'A-cut' 뱃지와 함께 품질 점수를 시각화하고, 상세 페이지에서 AI 설명 문구를 노출함.

---

## 8. 한계와 향후 연구
- **데이터셋 불일치**: AVA(분류)와 AADB(회귀) 모델 간의 점수 체계 통합에 정교한 캘리브레이션이 필요함.
- **GPU 자원**: 대규모 모델 학습 시 메모리 제약으로 인해 논문과 동일한 배치 사이즈를 유지하기 어려웠음.
- **재현도**: 일부 모델의 경우 공식 코드가 부재하여 논문 수준의 성능에 도달하는 데 어려움이 있었으나, 실용적인 범위 내에서는 충분한 성능을 확보함.

---

## 9. 부록 (시각화 자료)

### 9.1 미적 품질 모델 비교 (AVA)
![Aesthetic Model Comparison](figures/aesthetic_model_comparison_ava.png)

### 9.2 RGNet 절제 연구 (AADB SRCC)
![RGNet Ablation](figures/rgnet_ablation_srcc.png)

### 9.3 기술적 품질 모델 비교 (SPAQ)
![Technical Quality Comparison](figures/technical_model_comparison_spaq.png)

### 9.4 A-LAMP 학습 곡선
![A-LAMP Training](figures/training_curves_alamp.png)

### 9.5 RGNet AVA 학습 곡선
![RGNet Training](figures/training_curves_rgnet_ava.png)

---
*보고서 작성일: 2026년 5월 26일*
*작성 도구: Gemini CLI Report Generator*
