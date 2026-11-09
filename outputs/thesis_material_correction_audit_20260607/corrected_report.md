# A-cut 논문 방향 수정 및 재감사 리포트

## 1. 수정 배경 (What Changed from Previous Audit)
이전 감사에서는 `Stage 5 Student` 모델(모바일넷 기반의 지식 증류 모델)이 높은 성능 지표(Pairwise Acc 0.905)와 핸드오프 문서(`pozy_app_stage5_aesthetic_student_handoff.md`)를 근거로 최종 모바일 배포용 미적(Aesthetic) 통합 모델이라고 잘못 해석했습니다.

**수정된 사실:** 프로젝트 오너의 확인 결과, Stage 5 Student는 최종 통합 모델로 채택되지 않았습니다. 통합 증류된 점수가 실제 서비스 통합 과정에서 불안정하거나 이상한 결과(Unreliable/strange)를 보였기 때문입니다. 이에 따라 프로젝트의 최종 방향은 **"개별 미적 평가 모델(NIMA, RGNet, A-LAMP, ICAA)을 각각 경량화/양자화하여 개별 배포하고, 추론 시점에 가중치 앙상블을 수행하는 방식"**으로 선회(`forWeights2` 툴 및 문서 참조)하였습니다.

## 2. 수정된 논문 작성 방향 (Corrected Thesis Angle)
**추천 논문 제목:** "온디바이스 사진 선택을 위한 다중 미적 평가 모델의 양자화 및 모바일 앙상블 시스템 구축" (On-Device Ensemble of Quantized Aesthetic Assessment Models for Mobile Best-Shot Selection)

새로운 논문은 "최고의 단일 모델을 만들었다"가 아니라, **"다양한 특성을 가진 미적 평가 모델들을 모바일 제약 내에서 어떻게 개별적으로 최적화하고 통합(Ensemble)했는지, 그리고 그 과정에서 겪은 엔지니어링적 트레이드오프"**에 집중해야 합니다.

### 안전하게 주장할 수 있는 논점 (Safe Claims):
1. **개별 모델의 모바일 최적화 및 양자화:** NIMA의 전처리 버그 수정, PyTorch 기반 ICAA의 TensorFlow Native 재구현을 통한 TFLite 변환 성공 등 모델별 맞춤형 배포 전략.
2. **앙상블을 통한 안정성 확보:** Stage 5와 같은 단일 증류 모델 접근법의 한계를 지적하고(Ablation Study), 개별 모델 점수를 합산(가중치: RGNet 0.4, NIMA 0.2, A-LAMP 0.2, ICAA 0.2)하는 방식이 질적(Qualitative)으로 더 안정적임을 실험 도구(`forWeights2`)를 통해 입증.
3. **무거운 모델 구조의 과감한 변형:** A-LAMP의 복잡한 그래프 연산과 동적 패치 추출을 제거하고, MobileNetV3 기반의 듀얼 브랜치 구조(`mobile_alamp_v2`)로 재설계하여 모바일 동작을 가능하게 함.

### 피해야 할 위험한 주장 (Unsafe Claims to Avoid):
*   "Stage 5 증류 모델이 기존 앙상블을 완벽히 대체하여 상용 배포되었다." (거짓: 안정성 문제로 폐기 또는 실험적 시도로만 남음).
*   "우리가 구현한 A-LAMP가 원본 논문의 SOTA 성능을 넘어섰다." (거짓: 모바일에 맞게 구조를 크게 단순화했으므로 성능 하락이 수반되었을 가능성이 큼. 모바일 구동 가능성에 의의를 둬야 함).

## 3. 추천 논문 목차 (Recommended Final Thesis Structure)
1. **서론 (Introduction):** 온디바이스 사진 선택의 한계와 다각적 미적 평가의 필요성.
2. **관련 연구 (Related Work):** IQA 및 IAA 모델들의 동향. 기존 거대 모델들의 모바일 한계점.
3. **제안 시스템: 양자화 기반 개별 앙상블 아키텍처 (Proposed System):**
   - Stage 5 단일 증류 모델 접근법과 그 실패/한계 분석 (Ablation).
   - 다중 모델 개별 양자화 및 TFLite 최적화 (NIMA 전처리 고도화, ICAA TF Native 구현).
   - 모바일용 경량 A-LAMP (`mobile_alamp_v2`) 재설계.
4. **실험 및 시스템 평가 (Experiments & System Evaluation):**
   - 기술적 품질(KonIQ+FLIVE) 보정 및 앙상블 효과 (이전 감사 내용 유지).
   - 로컬 가중치 실험(`forWeights2`) 기반의 앙상블 가중치 최적화 결과.
   - 각 TFLite 모델의 사이즈 및 안드로이드-WSL 정합성(Parity).
5. **토의 (Discussion):** 단일 증류 vs 개별 앙상블 간의 트레이드오프 (정량적 지표 vs 실제 유저 체감 품질).
6. **결론 (Conclusion):** 실용적인 모바일 사진 평가 프레임워크 완성.

## 4. A-LAMP 단독 논문 게재 가능성 판단
`src/models/mobile_alamp_v2.py`에 구현된 구조 변경(그래프/Object Detection 제거, MobileNetV3 도입 등)은 훌륭한 엔지니어링 성과입니다. 하지만 모델 자체의 수식적 창의성(Novelty) 보다는 "모바일을 위한 구조 다이어트(Engineering reduction)"에 가깝습니다. 
**판단:** **단독 메인 논문이나 SOTA를 주장하는 논문으로는 부족**하며, 메인 캡스톤 논문의 핵심 챕터(제안 시스템 챕터 내)나 Appendix, 또는 "소프트웨어 공학/응용 어플리케이션" 학회의 짧은 시스템 논문(Short Paper) 정도로 활용하는 것이 보수적이고 안전합니다.