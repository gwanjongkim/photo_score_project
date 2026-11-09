# A-cut 캡스톤 주차별 연구일지

## 0. 작성 기준
- 이 문서는 프로젝트의 최종 성과뿐만 아니라 주차별 의사결정, 실험 과정, 실패 사례 및 해결 방안을 포함한 연구 기록입니다.
- 모든 내용은 `outputs/` 및 `experiments/` 디렉토리의 실제 파일과 로그를 기반으로 작성되었습니다.
- 날짜가 명시되지 않은 일부 항목은 디렉토리 생성일 및 전후 맥락을 통해 추정하였습니다.

---

## 1. 전체 연구 흐름 요약
본 프로젝트는 모바일 환경에서 최적의 사진(A-cut)을 선별하기 위한 딥러닝 모델 개발을 목표로 합니다. 초기에는 복잡한 논문 구조의 완벽한 재현에 집중했으나, 데이터셋의 제약과 모바일 배포 효율성을 고려하여 **'실용적 캡스톤(Practical Capstone)'** 방향으로 선회하였습니다. 이를 통해 경량화된 학생 모델(Stage 5)을 성공적으로 구축하고, 기존 상용 모델(NIMA)의 심각한 버그를 해결하며 실질적인 서비스 가능성을 확보하였습니다.

---

## 2. 주차별 연구일지

### 1주차: 2026.04.20 ~ 2026.04.26
#### 연구 목표
- 모바일 온디바이스에서 실행 가능한 경량 미적 품질 평가 모델 구축.
#### 수행 내용
- 기존 고성능 교사 모델(RGNet, A-LAMP, NIMA)의 앙상블 점수를 학습 데이터로 사용하는 지식 증류(Knowledge Distillation) 수행.
- MobileNetV2 기반의 경량 학생 모델(Stage 5) 설계 및 학습.
#### 주요 결과
- **Stage 5 Student**: AADB 검증셋에서 Pairwise Accuracy 90.56% 달성.
- TFLite FP16 변환 시 모델 크기 약 1.7MB로 최적화 성공.
#### 산출물
- 근거: `outputs/student_conservative_aesthetic_full_aadb_20260423/handoff.md`
#### 다음 주 계획
- VLM(Gemini API)을 활용한 추가 고품질 교사 데이터 확보 방안 탐색.

### 2주차: 2026.04.27 ~ 2026.05.03
#### 연구 목표
- VLM을 활용한 미적 품질 평가 교사 레이블링 시스템 구축.
#### 수행 내용
- Gemini API를 호출하여 이미지에 대한 미적 점수 및 설명(Reasoning) 데이터 수집.
- Gemma 모바일 모델의 온디바이스 배포 가능성 검토.
#### 실험/구현 내용
- `outputs/gemini_api_teacher_aesthetic3_20260427`: 100개 배치 단위의 고스펙 레이블링 실험 수행.
#### 주요 결과
- VLM이 인간의 미적 관점과 유사한 상세 피드백을 생성함을 확인.
#### 실패/문제점
- Gemini API 호출 비용 및 속도 문제로 대규모 데이터셋 구축에는 한계가 있음을 인지.
#### 산출물
- 근거: `outputs/gemini_api_teacher_aesthetic3_20260427`
#### 다음 주 계획
- 학술적 기반 강화를 위한 RGNet/A-LAMP 논문 재현 베이스라인 수립.

### 3주차: 2026.05.04 ~ 2026.05.10
#### 연구 목표
- RGNet 및 A-LAMP 논문 기반의 강력한 미적 품질 베이스라인 수립.
#### 수행 내용
- AADB(회귀) 및 AVA(분류) 데이터셋을 활용한 RGNet 학습.
- AVA 사전 학습 후 AADB 미세 조정(Fine-tuning) 파이프라인 구축.
#### 주요 결과
- **RGNet AADB**: SRCC 0.6819 기록 (실용적 베이스라인 확보).
- **RGNet AVA**: Accuracy 0.7697 기록.
#### 해결 또는 판단
- 논문의 복잡한 그래프 구조 대신 Mean Aggregation 방식을 사용하는 것이 회귀 작업에서 더 안정적임을 확인.
#### 산출물
- 근거: `outputs/rgnet_paper_consolidated_20260511/rgnet_paper_oriented_summary.md`
#### 다음 주 계획
- A-LAMP의 핵심인 레이아웃 인지형 그래프 퓨전 실험 착수.

### 4주차: 2026.05.11 ~ 2026.05.17
#### 연구 목표
- 레이아웃 정보를 활용한 A-LAMP 그래프 퓨전 모델 성능 검증.
#### 수행 내용
- 4096개 샘플 서브셋을 활용한 GraphLite 및 GraphGCN 퓨전 실험.
- ICAA(색채 미학) 모델의 PyTorch 가중치를 TF Native로 재구현.
#### 실험/구현 내용
- `outputs/alamp_paper_fusion_4096_final_report.md`
#### 실패/문제점
- 그래프 퓨전 모델이 학습 데이터에 극도로 과적합(Train AUC 0.99, Val AUC 0.70)되어 일반화 성능 확보에 실패.
#### 해결 또는 판단
- 데이터 부족 및 그래프 특징의 변별력 한계로 인해 실용적 모델에서는 그래프 퓨전 트랙을 보류하기로 결정.
#### 산출물
- 근거: `outputs/alamp_paper_fusion_4096_final_report.md`
#### 다음 주 계획
- 기술적 품질 평가 모델의 현대화를 위한 TOPIQ-lite 벤치마크 수행.

### 5주차: 2026.05.18 ~ 2026.05.24
#### 연구 목표
- 최신 기술적 품질 평가 모델(TOPIQ-lite) 도입 및 교사 모델 편향 수정.
#### 수행 내용
- TOPIQ-lite, FLIVE, KonIQ 모델 간의 성능 비교 벤치마크.
- A-LAMP Multi-Patch 교사 모델의 심각한 양성 편향(Positive Bias) 분석.
#### 주요 결과
- **TOPIQ-lite**: SPAQ 데이터셋에서 SRCC 0.899를 기록하며 압도적 성능 입증.
- A-LAMP 교사 모델의 재학습 계획(Class Weight 및 Backbone Unfreeze) 수립.
#### 산출물
- 근거: `outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/report.md`
#### 다음 주 계획
- 오랜 성능 저하 원인이었던 NIMA 모델의 버그 수정 및 최종 성능 검증.

### 6주차: 2026.05.25 ~ 2026.05.26
#### 연구 목표
- 주요 모델의 최종 성능 고도화 및 TFLite 배포 패키징.
#### 수행 내용
- **NIMA 혁신**: EfficientNetV2의 중복 전처리 버그를 발견 및 수정하여 SRCC 0.59 달성 (+0.24 향상).
- **ICAA 검증**: 색채 미학 모델의 공식 성능(SRCC 0.88) 완벽 재현 확인.
- **RGNet 고도화**: DenseASPP 및 논문 레시피 적용 실험 수행 (결과는 베이스라인 미도달).
#### 주요 결과
- NIMA 모델이 실용적 순위 산출 도구로 사용 가능한 수준까지 성능 회복.
- ICAA 모델의 TFLite FP16 변환 및 파리티(Parity) 확인 완료.
#### 산출물
- 근거: `outputs/nima_fixed_preproc_eval_20260525/report.md`
- 근거: `outputs/acut_capstone_final_report_20260526/report.md`
#### 최종 정리
- 실용적 캡스톤을 위한 모든 핵심 모델(Aesthetic, Technical, Color)의 라인업 완성 및 검증 완료.

---

## 3. 모델별 누적 변화 요약

- **NIMA**: (초기) 성능 미흡 → (중기) 원인 불명 → (최종) 전처리 버그 수정 후 성능 폭발 (SRCC 0.59).
- **RGNet**: (초기) 기본형 수립 → (중기) AADB/AVA 분화 → (최종) DenseASPP 시도했으나 기본형을 최종 실용 모델로 채택.
- **A-LAMP**: (초기) Multi-Patch 구조 → (중기) 그래프 퓨전 시도(실패) → (최종) Multi-Patch 교사 개선 트랙으로 회귀.
- **ICAA**: (초기) 공식 가중치 활용 → (중기) TF Native 재구현 → (최종) TFLite 배포 준비 완료 (SRCC 0.88).
- **Stage 5 Student**: (초기) 초기 증류 모델 → (중기) 보류 → (최종) 현재 Flutter 앱의 가장 안정적인 모바일 스코어러.

---

## 4. 주차별 성과 표

| 주차 | 기간 | 핵심 목표 | 수행 작업 | 주요 결과 | 산출물 경로 |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 1 | 04.20-04.26 | 모바일 경량 모델 | Stage 5 증류 학습 | Pairwise Acc 0.905 | `outputs/student_...` |
| 2 | 04.27-05.03 | VLM 교사 탐색 | Gemini API 레이블링 | 상세 미적 피드백 확보 | `outputs/gemini_...` |
| 3 | 05.04-05.10 | 베이스라인 수립 | RGNet/A-LAMP 학습 | SRCC 0.68, Acc 0.77 | `outputs/rgnet_...` |
| 4 | 05.11-05.17 | 그래프 퓨전 실험 | Graph Fusion / ICAA TF | 퓨전 과적합 확인 | `outputs/alamp_...` |
| 5 | 05.18-05.24 | 기술 품질 고도화 | TOPIQ-lite 벤치마크 | SPAQ SRCC 0.899 | `outputs/eval_final_...` |
| 6 | 05.25-05.26 | 최종 검증/배포 | NIMA 수정, ICAA 재검증 | NIMA 성능 정상화 | `outputs/nima_fixed_...` |

---

## 5. 최종 정리
- **연구 성과**: 모바일 환경에서 사용 가능한 고성능 미적/기술적 품질 평가 파이프라인을 구축함.
- **완성도**: 실용 캡스톤 관점에서 TFLite 모델 및 Flutter 연동 준비가 90% 이상 완료됨.
- **미완료 사항**: A-LAMP 및 RGNet의 완벽한 논문 성능 재현은 데이터 및 하드웨어 제약으로 인해 향후 과제로 남음.
- **우선순위**: 현 시점에서는 모델 추가 학습보다 구축된 모델들의 Flutter 앱 내 사용자 경험(UX) 최적화와 안정적 배포에 집중할 것을 권장함.
