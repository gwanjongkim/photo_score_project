# MDIQA Mobile Student IQA Distillation Final No-Go Report

## 1. Executive Summary

MDIQA Mobile Student IQA 증류 프로젝트에 대한 최종 분석 결과, 해당 라인을 **No-Go(중단)**로 결정하였다. 주요 근거는 다음과 같다.

- **최종 결정**: `MDIQA_STUDENT_FINAL_NO_GO_RESEARCH_REFERENCE_ONLY`.
- **Teacher 성능**: MDIQA Teacher는 0.8227 SRCC로 매우 강력한 성능을 입증하였다.
- **Student 성능**: 50k 데이터로 학습된 최선의 B0 Student는 0.7033 SRCC에 그쳤다.
- **비교 우위 실패**: 기존 14k Student(0.7436) 및 현재 프로덕션 KonIQ+FLIVE(0.7168)를 유의미하게 능가하지 못했다.
- **교정 시도 실패**: 데이터셋 가중치, 랭킹 손실, 레이블 스케일링 등 모든 개선 시도가 성능 향상에 실패하였다.
- **프로덕션 권고**: 기존 KonIQ+FLIVE 스택을 유지하며, MDIQA Student는 연구용 참조 자료로만 활용한다.

## 2. Original Goal and Decision Criteria

본 프로젝트의 목표는 MDIQA Teacher의 우수한 품질 측정 능력을 모바일에서 실행 가능한 EfficientNetV2B0 Student 모델로 증류하여, 현재 앱에서 사용 중인 기술 품질 측정(Technical Quality) 스택을 대체하거나 성능을 개선하는 것이었다.

성공 기준은 다음과 같았다.
1. 기존 프로덕션 KonIQ+FLIVE(SRCC 0.7168) 이상의 성능 확보.
2. 이전 시도인 14k Student(SRCC 0.7436) 이상의 성능 확보.
3. Severe False Positive 발생 억제 및 안정적인 점수 분포 확보.

## 3. Evidence Sources Inspected

본 보고서 작성을 위해 다음의 리포트 및 데이터를 검토하였다.

- **Teacher 벤치마크**: `outputs/official_protocol_mdiqa_signal_benchmark_20260604/report.md`.
- **학생 모델 실험**:
    - Phase 16 (B0 Baseline): `outputs/mobile_student_iqa_50k_controlled_b0_20260608/report.md`.
    - Phase 19 (Weighting): `outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/report.md`.
    - Phase 21 (Ranking): `outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609/report.md`.
    - Phase 23 (Scaling): `outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609/report.md`.
- **데이터 및 코드 오딧**:
    - 파이프라인 무결성: `outputs/mdiqa_student_pre_final_audit_20260609/report.md`.
    - 데이터 품질: `outputs/mdiqa_distillation_50k_full_targets_20260608/report.md`.
- **프로덕션 기준**: `outputs/official_protocol_technical_signal_benchmark_20260604/official_signal_metrics_overall.csv`.

## 4. MDIQA Teacher and Pipeline Validity

MDIQA Teacher 및 추론 파이프라인은 기술적으로 유효함이 확인되었다.
- MDIQA Teacher는 공식 벤치마크에서 0.8227 SRCC를 기록하며 금본위제(Gold Standard)임을 입증했다.
- 데이터셋별 변종(Variant) 라우팅 및 전처리가 정확하게 수행되었다.
- 따라서 Student의 성능 저조는 Teacher 파이프라인의 결함이 아닌 증류 과정 자체의 한계로 판단된다.

## 5. Student Model and Data Validity

Student 모델 및 학습 데이터 역시 기술적인 결함은 발견되지 않았다.
- EfficientNetV2B0 아키텍처는 모바일 배포에 적합하며 TFLite 변환이 가능함이 확인되었다.
- 50,000건의 학습 데이터는 누수(Leakage)나 중복이 없는 깨끗한 상태였다.
- 학습 로그 분석 결과, 모델의 손실 함수(Loss)는 정상적으로 하락하였으며 붕괴(Collapse) 현상도 없었다.

## 6. Experiment Timeline and Results

| Stage | Purpose | Official SRCC | Decision |
|---|---:|---:|---|
| MDIQA teacher | reference | 0.8227 | keep as reference |
| 14k student | prior mobile student | 0.7436 | stronger than 50k line |
| production KonIQ+FLIVE | current production | 0.7168 | keep |
| Phase 16 unweighted 50k B0 | baseline 50k student | 0.7033 | below production |
| Phase 19 W5-v2 | corrected dataset weighting | 0.5163 | no-go |
| Phase 21 ranking smoke | rank-loss test | 0.4803 | no-go |
| Phase 23 S2 KonIQ x2 | final label-scaling smoke | 0.5404 | final no-go |

## 7. Why Each Corrective Attempt Failed

### 7.1 Unweighted 50k B0
학습은 성공적으로 진행되었으나, 기존 14k 학생 모델보다 낮은 0.7033 SRCC를 기록했다. 이는 데이터 양이 늘어남에 따라 Teacher의 복잡한 타겟을 학습하는 난이도가 상승했기 때문으로 풀이된다.

### 7.2 Dataset Weighting (W5-v2)
데이터셋별 기여도를 30:35:35로 조정하려 했으나, 오히려 SRCC가 0.51 수준으로 대폭 하락했다. 이는 특정 데이터셋의 가중치를 높이는 것이 전체적인 일반화 성능을 저해했음을 보여준다.

### 7.3 Ranking Loss
랭킹 손실 함수를 도입하여 품질 순서를 학습시키려 했으나 SRCC 0.48로 가장 낮은 성능을 보였다. 이는 Teacher의 랭킹 자체가 불안정하거나 Student가 랭킹과 회귀를 동시에 학습하기에 용량이 부족했음을 시사한다.

### 7.4 KonIQ Label Scaling (S2)
KonIQ의 좁은 레이블 범위를 2배로 확장하여 학습시켰으며, 이전 개선 시도들보다는 높은 0.54를 기록했으나 여전히 베이스라인에는 크게 못 미쳤다. 레이블 압착이 주요 원인 중 하나임은 확인되었으나, 그것만으로는 성능을 완전히 회복할 수 없었다.

## 8. Root Cause Assessment

- **증류의 한계**: 강력한 Teacher 모델의 지식을 매우 가벼운 B0 모델 하나에 담기에는 정보의 손실이 너무 컸다.
- **데이터셋 도메인 갭**: 서로 다른 성격의 세 데이터셋을 하나의 모델로 통합하면서 발생하는 충돌을 해결하지 못했다.
- **FLIVE 정렬 문제**: FLIVE 데이터셋에서의 Teacher 예측값과 실제 MOS 간의 정렬이 KonIQ나 SPAQ에 비해 약해, 학생 모델이 잘못된 방향으로 학습될 가능성이 있었다.
- **모델 용량**: B1/B2로의 확장을 고려할 수 있으나, 현재의 격차를 고려할 때 모델 크기만으로 해결될 수준을 넘어선 것으로 판단된다.

## 9. Production Decision

- **프로덕션 대체 불가**: MDIQA Student는 현재 프로덕션 스택을 대체할 만큼 우수하지 않다.
- **현행 유지**: 기존의 KonIQ+FLIVE 기술 품질 스택을 그대로 유지한다.
- **TFLite/Flutter 통합 중단**: 해당 학생 모델의 배포용 변환 및 앱 통합을 수행하지 않는다.
- **MDIQA Teacher 보존**: MDIQA Teacher는 향후 연구를 위한 성능 상한선(Ceiling) 참조용으로만 유지한다.

## 10. Research Takeaways for Thesis / Capstone Report

본 실험 결과는 캡스톤 프로젝트 및 논문에서 다음과 같은 논거로 활용될 수 있다.
- 강력한 교사가 반드시 우수한 학생을 보장하지 않음을 증명하는 사례.
- 실제 서비스 적용 시 성능뿐만 아니라 배포 가능성 및 검증된 기준과의 비교가 얼마나 중요한지를 보여줌.
- 부정적 결과(Negative Result)를 통해 기존 시스템의 견고함을 입증하고 자원 낭비를 방지한 의사결정 사례.

## 11. Final Decision

**MDIQA_STUDENT_FINAL_NO_GO_RESEARCH_REFERENCE_ONLY**

## 12. Next Actions

1. MDIQA 학생 모델 관련 모든 출력물을 연구 아카이브로 분류 및 저장한다.
2. 관련 README에 해당 실험 라인이 프로덕션용이 아님을 명시한다.
3. 앱 내 기술 품질 측정 로직은 기존 KonIQ+FLIVE를 유지한다.
4. 추가적인 연구 계획이 수립되지 않는 한 B1/B2 학습을 진행하지 않는다.
