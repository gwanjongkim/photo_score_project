# Stage 5 Student 모델 재평가 (Stage 5 Reassessment)

이전 감사에서는 Stage 5를 최종 모델로 오인하였으나, 프로젝트 오너의 피드백에 따라 이를 정정합니다. Stage 5 모델의 개발 기록은 폐기할 것이 아니라, 논문의 **"한계 분석 (Ablation Study) 및 설계 의사결정 과정"**을 설명하는 매우 가치 있는 증거 자료로 활용되어야 합니다.

## 1. Stage 5의 원래 목적 (Intended Goal)
*   프로젝트 초기, A-LAMP 및 RGNet과 같은 무겁고 복잡한 다중 입력 모델들을 모바일 기기(Flutter/TFLite 환경)에 그대로 올리기 불가능하다고 판단함.
*   이에 대한 돌파구로, 복수의 교사(Teacher) 모델이 내놓는 미적 앙상블 점수(`teacher_aesthetic_score`)를 정답지로 삼아, 단일 MobileNetV2 기반의 경량 모델 하나로 지식 증류(Knowledge Distillation)를 시도함.
*   증거 파일: `docs/pozy_app_stage5_aesthetic_student_handoff.md`

## 2. 기록된 성과 지표 (Reported Metrics)
*   **크기:** 3.41MB (초소형 TFLite 변환 성공)
*   **평가 메트릭:** AADB 검증셋 기준 Pairwise Accuracy 0.90568, SRCC 0.8449.
*   정량적 지표상으로는 TFLite 변환도 매끄러웠고 점수 일치도도 나쁘지 않았습니다.

## 3. 최종 솔루션에서 배제된 이유 (Why it is NOT the final solution)
*   **정성적 불안정성 (Qualitative Unreliability):** 정량 지표(Pairwise Acc)가 준수하게 나왔음에도 불구하고, 실제 앱 통합 연동 및 다양한 실제 사진들에 대해 테스트했을 때 랭킹이 예상과 다르게 튀거나 직관과 동떨어진 결과(Strange/Unreliable behavior)를 보였습니다.
*   **단일 프록시 모델의 한계:** 복잡한 그래프 모델(A-LAMP)과 해상도에 민감한 모델(RGNet)의 서로 다른 판단 기준을 고작 3MB짜리 단일 MobileNetV2 하나로 욱여넣으려다 보니, 특정 도메인이나 왜곡에 대해 점수 체계가 붕괴되는 현상이 발생했습니다.

## 4. 새로운 방향과 증거 (The Pivot)
*   결국, Stage 5 단일 모델 증류를 포기하고, 모델을 통합하기보다는 **각 미적 평가 모델(NIMA, RGNet, A-LAMP, ICAA)을 개별적으로 모바일 스펙(FP16)에 맞게 깎아서(양자화/구조축소) 4개의 TFLite 파일로 배포**하는 정공법으로 회귀했습니다.
*   평가 시에는 디바이스 단에서 각각의 TFLite를 구동한 후 가중치(`aesthetic_weight_lab.yaml` 기반)로 더하는 앙상블 체계를 채택하여 정성적 안정성을 확보했습니다.
*   증거 파일: `forWeights2/HANDOFF.md`, `forWeights2/READ.md`

## 5. 논문에서의 활용 방안
Stage 5 실패 사례는 시스템 논문에서 훌륭한 **토의(Discussion) 주제**가 됩니다.
"연산 비용을 아끼기 위해 앙상블을 단일 모델로 증류하는 기법(Knowledge Distillation)은 정량적 지표는 만족시킬 수 있으나, 미적 평가와 같이 도메인 특성이 복잡하고 모호한 Task에서는 정성적 안정성을 크게 훼손할 위험이 있다. 따라서 실용적인 모바일 사진 선택 시스템에서는 다소 메모리/추론 비용이 들더라도 개별 최적화된 모델 여러 개를 추론 시점에 앙상블하는 방식이 실제 품질 확보에 유리하다."라는 결론을 이끌어낼 수 있습니다.