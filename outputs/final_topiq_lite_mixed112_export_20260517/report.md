# TOPIQ-lite mixed_112_frozen Final Candidate TFLite Export Report

## 1. Summary
**PASS**

`mixed_112_frozen` 모델을 FP32 및 FP16 TFLite로 성공적으로 변환하였습니다. 변환 과정에서 Builtin-only 옵션을 사용하여 Flex나 Select TF Ops에 대한 의존성 없이 변환되었으며, 수치적 Parity 체크 결과 오차가 매우 낮아(FP16 기준 Max Diff < 0.07) 모바일 배포에 적합함이 확인되었습니다.

## 2. Export Results
| Model | Size | Builtin-only | Flex | Erfc | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| mixed112_frozen_fp32 | 23.95 MB | Yes | None | None | SUCCESS |
| mixed112_frozen_fp16 | 11.99 MB | Yes | None | None | SUCCESS |

## 3. Tensor Signatures
- **Input:** `[1, 384, 384, 3]`, `float32`, RGB (pad resize)
- **Output:** `[1, 1]`, `float32`, Norm MOS [0, 1] (score_100 = out * 100)

## 4. Op Analysis
- **Unique Ops:** CONV_2D, DEPTHWISE_CONV_2D, ADD, MUL, FULLY_CONNECTED, RELU6, HARD_SWISH, MEAN, RESHAPE, LOGISTIC 등 EfficientNetV2-B0 기반의 표준 Builtin Op들로 구성됨.
- **Select TF Ops / Flex:** 없음 (추가 라이브러리 링크 불필요).
- **Erfc:** 없음.

## 5. Parity Results (Keras vs TFLite, 0-100 Scale)
| Dataset | FP32 max diff | FP16 max diff | FP16 mean diff | PASS? |
| :--- | :---: | :---: | :---: | :---: |
| SPAQ | 0.019 | 0.063 | 0.022 | YES |
| KonIQ | 0.017 | 0.036 | 0.015 | YES |
| FLIVE | 0.010 | 0.020 | 0.010 | YES |

- FP16 최대 오차가 0.063점으로, 허용 기준인 0.5점보다 훨씬 낮은 우수한 정밀도를 보여줍니다.

## 6. Replacement Implication
- **최적의 단일 모델:** 현재까지 개발된 모델 중 SPAQ, KonIQ, FLIVE 전반에 걸쳐 가장 균형 잡힌 성능을 보여주는 단일 모델입니다.
- **배포 가능성:** TFLite 변환이 깔끔하게 완료되었으며 용량도 12MB 수준으로 모바일 배포에 매우 유리합니다.
- **성능 제약 사항:** FLIVE 랭킹 성능(SRCC ~0.46)은 여전히 전용 모델(0.62)에 미치지 못합니다. 이는 단일 모델 통합에 따른 트레이드오프이며, 실제 서비스 도입 시 수용 가능 여부에 대한 최종 판단이 필요합니다.

## 7. Recommended Next Step
**Android smoke test with mixed_112 FP16**
- 실제 안드로이드 기기에서 추론 속도(Latency)와 결과값이 예상대로 나오는지 검증하는 Smoke Test를 권장합니다.

## 8. Files Created
- `outputs/final_topiq_lite_mixed112_export_20260517/topiq_lite_mixed112_frozen_fp16.tflite`
- `outputs/final_topiq_lite_mixed112_export_20260517/report.md`
- `outputs/final_topiq_lite_mixed112_export_20260517/parity_*.csv`

## 9. Final GO / NO-GO
**GO (Deployment Validation)**
기술적 변환 및 정밀도 검증은 완벽합니다. 이제 모바일 환경에서의 실제 동작 확인이 필요합니다.
