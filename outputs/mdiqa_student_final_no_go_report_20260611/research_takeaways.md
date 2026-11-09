# MDIQA Mobile Student IQA Distillation Research Takeaways

본 연구에서는 강력한 성능을 가진 MDIQA Teacher 모델을 경량화된 EfficientNetV2B0 Student 모델로 증류(Distillation)하여 기존의 KonIQ+FLIVE 기반 프로덕션 기술 품질 측정 스택을 대체하거나 개선하려는 시도를 수행하였다.

### 주요 연구 결과 및 시사점

1. **강력한 교사가 반드시 우수한 학생을 보장하지 않음**: MDIQA Teacher는 공식 벤치마크에서 0.8227 SRCC라는 압도적인 성능을 보였으나, 이를 50k 규모의 데이터로 학습시킨 B0 Student 모델은 0.70-0.74 수준의 성능에 머물렀다. 이는 복잡한 트랜스포머 기반의 Teacher 지식을 단일 경량 CNN 구조로 완벽하게 전이하는 데 한계가 있음을 시사한다.

2. **데이터셋 간 스케일 차이의 복합성**: KonIQ-10k, SPAQ, FLIVE 등 서로 다른 MOS(Mean Opinion Score) 범위를 가진 데이터셋을 하나의 모델로 통합 학습할 때 발생하는 스케일 압착(Label Compression) 현상이 주요 병목 지점으로 확인되었다. 특히 KonIQ 데이터의 경우 Teacher의 예측값이 매우 좁은 범위에 집중되어 Student가 세밀한 품질 차이를 학습하기 어려웠다.

3. **교정 가설의 한계**: 데이터셋 가중치 조정(Weighting), 랭킹 손실 함수(Ranking Loss) 도입, 타겟 레이블 스케일링(Label Scaling) 등 여러 교정 전략을 테스트하였으나, 공식 SRCC 지표를 유의미하게 개선하지 못하거나 오히려 하락시키는 결과를 초래했다. 이는 모델의 용량 부족보다는 증류 타겟의 품질과 데이터셋 간의 정렬 문제가 더 근본적인 원인임을 보여준다.

4. **네거티브 결과의 실무적 가치**: 본 실험의 실패(No-Go) 결정은 현재 프로덕션에서 사용 중인 KonIQ+FLIVE 스택이 실질적으로 얼마나 견고하고 신뢰할 수 있는지를 역설적으로 증명한다. 검증되지 않은 새로운 모델로의 교체 대신, 기존의 검증된 스택을 유지하는 결정을 데이터 기반으로 내릴 수 있게 되었다.

### 결론

MDIQA Mobile Student 라인은 현재의 학습 설정 및 모델 구조하에서는 프로덕션 대체 경로로서 부적합하다고 판단된다. MDIQA Teacher는 향후 연구를 위한 최상위 참조 모델(Research Reference)로 활용하며, 프로덕션 기술 품질 측정은 기존의 안정적인 스택을 유지한다.
