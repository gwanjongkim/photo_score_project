# 최종 수정된 캡스톤 논문 목차 (Final Thesis Outline)

**논문 가제:** 온디바이스 사진 선택을 위한 모바일 제약 기반의 미적·기술적 품질 평가 모델 최적화 및 앙상블 시스템 구축 (Optimization and Ensemble of IQA/IAA Models under Mobile Constraints for On-Device Best-Shot Selection)

## 1. 서론 (Introduction)
*   **연구 배경:** 스마트폰 다중 촬영(Burst/Continuous) 사진 중 최고의 사진(A-cut)을 자동 선별하는 시스템의 필요성.
*   **문제 제기:** 서버 비용 한계와 사용자 개인정보 보호를 위해 모바일 기기 내부(On-Device)에서 즉각적 추론이 필요하나, 기존 SOTA 모델들은 너무 무겁고 모바일 환경(TFLite) 지원이 부족함.
*   **연구 기여:** 모바일 제약 속에서 동작 가능한 구조로 모델을 최적화(A-LAMP 단순화 등)하고, 단일 모델의 불안정성/편향성을 다중 경량 모델의 양자화 및 앙상블(Ensemble)로 극복한 시스템 통합 사례를 제시함.

## 2. 관련 연구 (Related Work)
*   **기술적 이미지 화질 평가 (Technical IQA):** KonIQ, FLIVE, TOPIQ 모델 동향.
*   **미적 이미지 화질 평가 (Aesthetic IAA):** NIMA, RGNet, A-LAMP 모델과 그 모바일 한계점.

## 3. 온디바이스 선별을 위한 시스템 최적화 (Proposed System & Optimization)
*   **3.1 모바일 배포 제약과 모델 선택 (Deployment Constraints):** 다중 패치 및 무거운 연산자를 가진 모델들의 한계 파악.
*   **3.2 구조적 단순화 및 재구현 (Structural Simplifications):**
    *   `Mobile A-LAMP v2`: 그래프 신경망/객체탐지를 제거하고 MobileNetV3와 Self-attention으로 경량화한 모바일 친화적 구조 제안.
    *   `NIMA / RGNet`: 전처리(PIL Resize, Normalization) 정합성 복원 및 양자화.
*   **3.3 프레임워크 한계 돌파 (Engineering Workarounds):** PyTorch 모델(ICAA)의 TFLite 변환 실패를 Native TF 포팅으로 극복한 사례 소개.

## 4. 앙상블 및 편향성 제어 실험 (Experiments & System Tuning)
*   **4.1 단일 증류 모델의 한계 (Ablation: Stage 5 Student):** 거대 지식을 단일 모델로 압축한 Stage 5 시도가 정성적 평가(랭킹 붕괴)에서 어떻게 실패했는지 분석.
*   **4.2 기술적 품질의 편향성 완화 (TQA Ensemble):** 단일 FLIVE 모델의 극심한 양수 편향(+18.42)을 KonIQ 모델과의 앙상블로 보정(+7.71)한 정량적 증거 제시.
*   **4.3 미적 품질 앙상블의 가중치 최적화 (AQA Ensemble Lab):** 경량화된 NIMA, RGNet, Mobile A-LAMP 3종의 로컬 가중치 탐색(`forWeights2`) 기반 앙상블 실험 결과.

## 5. 논의 및 한계 (Discussion & Limitations)
*   **속도(Latency)와 성능의 트레이드오프 (The ICAA Exclusion Case):** ICAA 모델이 TFLite 변환 및 높은 정확도에도 불구하고 실제 디바이스의 Latency 제약에 걸려 최종 모바일 프로덕션에서 기각된 사례 분석. 모바일 온디바이스 AI 시스템 통합 시 겪게 되는 현실적 한계 고찰.

## 6. 결론 (Conclusion)
*   성공적인 SOTA 확보보다는 실용성과 안정성을 택한 엔지니어링 성과 요약. 온디바이스 환경에서 사진 선별 파이프라인의 구축 기준 제시.