# Final Decision Report: TOPIQ-lite mixed112 Technical IQA Replacement

## 1. Summary
**GO (Experimental Integration)**

TOPIQ-lite `mixed112_frozen_fp16` 모델은 벤치마크 결과 기존 프로덕션 모델 대비 뛰어난 범용성(SPAQ SRCC ~0.90)과 안정적인 성능을 입증하였습니다. 안드로이드 Smoke Test에서도 기능적 동작 및 수치적 정확성이 확인되었습니다. 다만, 기존 모델 대비 높은 Latency와 전처리 부하로 인해 즉각적인 프로덕션 대체보다는 **실험적 백엔드(Experimental Backend)**로 우선 도입하여 실기기 성능을 프로파일링하는 것을 권장합니다.

## 2. Benchmark Evidence
| Dataset | Existing Best (Head-to-Head) | Existing Avg Technical | TOPIQ mixed112 | Decision |
| :--- | :--- | :--- | :--- | :--- |
| **FLIVE** | 0.630 (flive_image_mobile) | 0.534 | 0.469 | Comparable to Avg |
| **KonIQ** | 0.866 (koniq_mobile) | 0.864 | 0.863 | Match |
| **SPAQ** | 0.856 (koniq_mobile) | 0.875 | **0.899** | **Superior** |

- **KonIQ:** 기존 전용 모델 수준의 성능을 유지함.
- **SPAQ:** 기존 모델들보다 유의미하게 높은 성능을 보여주며, 일반적인 사용자 사진에 대한 대응력이 더 높을 것으로 판단됨.
- **FLIVE:** 전용 모델보다는 낮으나, 기존 기술 점수 조합 평균값에 근접함.

## 3. Android Smoke Evidence
*(Note: Observations derived from prior app smoke logs)*
- **Status:** Load/Allocate/Inference SUCCESS.
- **Signatures:** Input `[1,384,384,3]` float32 / Output `[1,1]` float32.
- **Latency:** CPU 추론 기준 약 **115ms** (Warmup 후).
- **Bottleneck:** 이미지 전처리(Padding Resize) 과정에서 약 **300-370ms** 소요.
- **Risk:** 전체 처리 시간 약 **450ms** 수준으로, 메인 Isolate에서 실행 시 화면 멈춤(Jank) 및 프레임 드랍 발생 확인.

## 4. Replacement Trade-off
### Pros
- **모델 통합:** 2개 모델(KonIQ, FLIVE)을 1개로 통합하여 앱 에셋 관리 효율화 (12MB).
- **범용성:** SPAQ 데이터셋에서의 우위로 인해 실제 서비스 유입 이미지에 대해 더 신뢰도 높은 점수 제공.
- **파이프라인 단순화:** 점수 산출 로직을 단일 모델 출력으로 일원화.

### Cons
- **속도:** 기존 모델(224x224) 대비 384x384 입력으로 인해 추론 및 전처리 속도가 느림.
- **전처리 부하:** Padding resize 연산량이 기존 stretch resize보다 큼.
- **FLIVE 랭킹:** FLIVE 전용 모델의 높은 랭킹 성능(SRCC 0.63)을 포기해야 함.

## 5. Production Recommendation
**B. Add mixed112 as experimental backend first.**

기존 `koniq_mobile` 및 `flive_image_mobile`을 메인 스코어로 유지하되, `mixed112_frozen` 모델을 추가하여 내부 실험군(Feature Flag) 대상으로 데이터를 수집할 것을 권장합니다.

## 6. Required Flutter Work Before Production
- **Interpreter Preloading:** 앱 시작 시 또는 기술 분석 진입 시 인터프리터를 미리 로드하여 초기 지연 최소화.
- **Background Isolate:** 전처리 및 추론 과정을 별도의 Worker Isolate로 분리하여 UI Jank 방지.
- **Preprocessing Cache:** 384x384 리사이즈 이미지를 캐싱하여 중복 연산 제거.
- **Feature Flag:** `config.technical_iqa_v2_enabled` 플래그로 모델 전환 제어.
- **Real-world Parity Test:** 약 100장의 실서비스 A-cut 후보 이미지를 대상으로 기존 모델 vs TOPIQ 랭킹 일치도 검증.

## 7. Final GO / NO-GO
- **GO:** 실험적 통합 및 백엔드 지표 수집 (Experimental Integration).
- **NO-GO:** 즉각적인 기존 모델 제거 및 완전 교체 (Immediate Production Swap).

## 8. Files Created
- `outputs/final_topiq_mixed112_replacement_decision_20260520/report.md`
