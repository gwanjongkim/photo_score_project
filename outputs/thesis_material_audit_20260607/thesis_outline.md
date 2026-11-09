# 모바일 온디바이스 사진 선택을 위한 품질 평가 모델 논문 개요

## 1. 서론 (Introduction)
*   **배경:** 스마트폰 카메라 성능 향상으로 사용자의 사진 촬영량이 급증함에 따라, 베스트 샷(Best-shot)을 자동으로 추천해주는 시스템의 필요성 대두.
*   **문제 정의:** 기술적 품질(초점, 노출)과 미적 품질(구도, 색감)을 동시에 평가해야 하며, 이를 오프라인(On-device) 모바일 환경에서 실시간에 가깝게 처리해야 함. 기존의 크고 복잡한 모델(SOTA)은 모바일 직접 배포가 불가능함.
*   **제안 방법:** A-cut 시스템. 기술적 품질은 모델 앙상블 기법으로 편향을 최소화하고, 미적 품질은 거대 교사 모델의 지식을 모바일 경량 모델로 증류(Distillation)하여 배포.
*   **참고 자료:** `docs/capstone_artifact_map_20260605.md`, `docs/mobile_deployment_notes.md`

## 2. 관련 연구 (Related Work)
*   **기술적 이미지 화질 평가 (Technical IQA):** KonIQ, FLIVE, TOPIQ 모델의 특징과 한계점.
*   **미적 이미지 화질 평가 (Aesthetic IAA):** NIMA, A-LAMP, RGNet, ICAA 모델 구조 및 한계(다중 패치 입력, 커스텀 레이어 등 모바일 배포 제약).
*   **참고 자료:** `outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/report.md`

## 3. 제안 시스템 구조 (Proposed System Architecture)
### 3.1 기술적 화질 평가 앙상블 설계
*   모바일 넷 기반의 단일 모델(KonIQ, FLIVE)을 TFLite로 배포 가능함을 확인.
*   단일 모델의 편향성 문제(FLIVE의 +18.42 bias)를 해결하기 위한 가중치 앙상블(4/7 KonIQ + 3/7 FLIVE) 설계.
*   **참고 자료:** `outputs/capstone_technical_iqa_baseline_summary_20260530/evidence_table.csv`

### 3.2 미적 화질 평가 지식 증류 (Knowledge Distillation)
*   A-LAMP(다중 패치) 및 RGNet의 성능은 우수하나 모바일 탑재가 불가능함에 따라 Teacher 모델로 활용.
*   Student 모델(Stage 5) 구조 설계 및 학습. (Pairwise Accuracy 최적화).
*   **참고 자료:** `outputs/acut_capstone_final_report_20260526/model_metrics_summary.csv`

### 3.3 TFLite 변환 및 모바일 최적화 엔지니어링
*   PyTorch → ONNX → TFLite 변환 실패 사례(ICAA)와 Native TensorFlow 재구현 과정.
*   안드로이드 플랫폼과 WSL 백엔드 간의 엄격한 Parity(오차율 검증) 시스템.
*   **참고 자료:** `experiments/icaa_tf_native/reports/`, `outputs/c6_android_wsl_parity_20260527/`

## 4. 실험 결과 (Experiments)
### 4.1 기술적 품질 모델 성능
*   KonIQ vs FLIVE vs 앙상블 vs TOPIQ 성능(SRCC) 및 Bias 완화 효과 비교.
*   **참고 자료:** `outputs/capstone_technical_iqa_baseline_summary_20260530/evidence_table.csv`

### 4.2 미적 품질 모델 및 증류 성능
*   A-LAMP, NIMA, RGNet, ICAA 단일 모델 성능 비교.
*   Stage 5 Student 모델의 성능 유지 여부 (Pairwise Accuracy 0.905 달성 확인).
*   **참고 자료:** `outputs/acut_capstone_final_report_20260526/model_metrics_summary.csv`

## 5. 토의 및 한계 (Discussion & Limitations)
*   A-LAMP Graph Fusion 실험 실패 원인 분석 (과적합 문제).
*   안드로이드 디바이스 배포 시 메모리/추론 속도(Latency) 트레이드오프.

## 6. 결론 (Conclusion)
*   시스템 요약 및 실용적 가치 강조. 모바일 한계를 고려한 파이프라인 구성의 중요성.