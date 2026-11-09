# A-cut 시스템 모바일 배포 및 화질/미적 평가 모델 분석 리포트

## 1. 요약 (Executive Summary)
본 리포트는 A-cut 모바일 사진 선택 시스템 구축을 위해 진행된 다양한 기술적 화질 평가(IQA) 및 미적 화질 평가(IAA) 모델의 실험 결과를 종합한 것입니다. WSL 환경에서 수행된 모델 학습, 지식 증류(Knowledge Distillation), TFLite 모바일 변환, 그리고 안드로이드-WSL 간의 Parity 검증 결과를 포함합니다. 

이 레포지토리의 증거들은 단순히 "SOTA(State-of-the-Art) 모델을 만들었다"는 주장보다는, **"복잡한 다중 입력 모델(A-LAMP, RGNet 등)을 어떻게 모바일 환경(TFLite)에 제약 내에서 최적화하고 통합하였는가"**를 보여주는 훌륭한 실증적 소프트웨어 공학 및 응용 논문 자료로 활용될 수 있습니다.

## 2. 논문 작성 가능 방향 (Thesis Angle)
**추천 논문 제목:** "모바일 온디바이스 사진 선택을 위한 기술적·미적 품질 평가 모델의 실증적 분석 및 최적화 시스템" (Empirical analysis of IQA/IAA model deployment for mobile best-shot selection)

본 레포지토리의 증거를 바탕으로 제안하는 논문의 주요 기여도는 다음과 같습니다:
1. **기술적 품질 평가의 앙상블 안정성 입증:** 단일 최신 모델(TOPIQ-lite, C6) 도입 대신, KonIQ와 FLIVE 앙상블(4:3 비율)이 모바일 환경에서 편향성(Bias)을 상쇄하고 강건한 성능을 냄을 입증.
2. **복잡한 미적 평가 모델의 모바일 지식 증류:** 연산량이 큰 A-LAMP, RGNet 등의 모델 앙상블을 단일 Student 모델(Stage 5)로 증류하여 모바일 배포 제약을 극복.
3. **배포 안정성 및 Parity 확보:** PyTorch 모델의 TFLite 변환 실패(ICAA) 분석과 Native TF 재구현, 그리고 WSL-Android 간의 엄격한 추론 Parity 검증 체계 구축.

## 3. 검증된 주요 기여도 및 증거 (Confirmed Contributions)

### 3.1 기술적 품질 평가 (Technical Quality Assessment - TQA)
- **실험 모델:** KonIQ, FLIVE, TOPIQ-lite (mixed112), C6, MDIQA
- **데이터셋:** SPAQ, KonIQ, FLIVE
- **주요 결과:** 
  - TOPIQ-lite와 C6 모델을 평가했으나, 프로덕션 환경에서는 기존의 **KonIQ(4/7) + FLIVE(3/7) 앙상블 공식**을 유지하기로 결정함. (`evidence_table.csv`)
  - 이유는 단일 모델(예: FLIVE)이 스마트폰 데이터에 대해 심각한 편향성(+18.42 Bias)을 가지는 반면, 앙상블이 이를 효과적으로 완화(+7.71)하기 때문임.
  - C6 모델은 WSL과 안드로이드 간의 Parity가 검증되었으나(`outputs/c6_android_wsl_parity_20260527/`), 앙상블보다 성능이 뛰어나지 않아 **Log-only** 상태로 남김.

### 3.2 미적 품질 평가 (Aesthetic Quality Assessment - AQA)
- **실험 모델:** A-LAMP, RGNet, NIMA, ICAA Color, Stage 5 Student
- **데이터셋:** AVA, AADB, ICAA17K
- **주요 결과:**
  - A-LAMP 다중 패치 모델과 RGNet은 우수한 성능(A-LAMP 정확도 0.7633, RGNet SRCC 0.6819)을 보이나, 다중 입력과 커스텀 레이어로 인해 모바일 직접 배포가 불가능함.
  - 이를 해결하기 위해 **Stage 5 Student 모델**로 증류(Distillation)를 수행하여 AADB 데이터셋 기준 Pairwise Accuracy 0.905, SRCC 0.8586을 달성, 모바일 대리(Proxy) 모델로 활용. (`model_metrics_summary.csv`)
  - ICAA Color 모델은 SRCC 0.8791로 높은 색채 미적 평가 성능을 기록함. NIMA는 전처리 버그 수정 후 랭킹 성능이 크게 향상됨(SRCC 0.5894).

### 3.3 모바일 배포 및 최적화 (Mobile Deployment / TFLite)
- **주요 결과:**
  - 단일 입력 모델인 KonIQ와 FLIVE-image는 모바일용으로 성공적으로 변환 및 도입 완료 (`docs/mobile_deployment_notes.md`).
  - PyTorch 기반 ICAA 모델을 ONNX를 거쳐 TFLite로 변환하는 시도가 실패하여, 이를 TensorFlow로 Native 재구현하는 엔지니어링 과정을 거침 (`experiments/icaa_tf_native/`).
  - 안드로이드 앱 구현 시 성능 평가 및 Parity 검증을 위해 `android_gemma_latency_harness` 등을 통한 시스템 레벨의 검증 완료.

### 3.4 실패 및 기각된 실험 (Failed/Rejected Experiments)
- **A-LAMP GraphLite / GraphGCN Fusion:** AVA 4096 데이터로 융합을 시도했으나 과적합(Overfitting) 발생으로 기각됨 (ROC-AUC ~0.706).
- **RGNet DenseASPP Ablation:** 논문 레시피를 근사하여 시도했으나 베이스라인보다 낮은 성능(SRCC 0.6518)을 기록하여 기각.
- 실패 사례는 제한된 데이터와 모바일 제약 환경에서의 모델 설계 한계를 논의하는 데 귀중한 자료(Limitation)로 쓰임.

## 4. 논문 활용 가능 테이블 (Usable Tables)
- **Table 1:** 기술적 품질 모델 비교 (SPAQ 데이터셋 상의 편향성 및 SRCC 비교, 앙상블 우수성 증명)
- **Table 2:** 미적 품질 모델 벤치마크 (A-LAMP, RGNet, Stage 5 Student 간의 성능 및 파라미터/배포 가능성 비교)
- **Table 3:** 안드로이드 vs WSL Parity 오차율 분석 (TFLite 변환 성공을 증명하는 테이블)

## 5. 결론 및 최종 판단
본 프로젝트 데이터는 단순히 "새로운 딥러닝 구조를 제안"하는 논문보다는, **소프트웨어 공학 및 응용 관점에서 시스템 통합 논문**으로 작성하는 것이 가장 안전하고 경쟁력 있습니다. 모바일 환경의 제약, 모델 편향성 보정(Ensemble), 지식 증류를 통한 최적화 과정이 탄탄한 실험 증거로 뒷받침되고 있습니다.