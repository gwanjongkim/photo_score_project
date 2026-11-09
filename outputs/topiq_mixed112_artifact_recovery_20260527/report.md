# TOPIQ Mixed112 Artifact Recovery Report

## 1. Summary
TOPIQ Mixed112 TFLite 아티팩트를 성공적으로 복구하고 기술 사양을 확인하였습니다. 이 모델은 EfficientNetV2-B0 기반의 TOPIQ-lite 구조를 가지며, 2026년 5월 17일에 수출되었습니다. SPAQ, KonIQ, FLIVE 데이터셋이 혼합된 `mixed_112` 데이터셋으로 학습되었으며, 범용적인 기술 화질 평가 성능이 우수함이 확인되었습니다.

## 2. Located Artifacts
- **FP16 Model:** `outputs/final_topiq_lite_mixed112_export_20260517/topiq_lite_mixed112_frozen_fp16.tflite` (12 MB)
- **FP32 Model:** `outputs/final_topiq_lite_mixed112_export_20260517/topiq_lite_mixed112_frozen_fp32.tflite` (24 MB)
- **Training Source:** `outputs/mixed_replay_topiq_lite_112_frozen_e15_20260517`
- **Export Record:** `outputs/final_topiq_lite_mixed112_export_20260517/report.md`

## 3. TFLite Tensor Details
- **Input Tensor:**
  - Name: `serving_default_topiq_input:0`
  - Shape: `[1, 384, 384, 3]`
  - Dtype: `float32`
  - Quantization: None (Floating point)
- **Output Tensor:**
  - Name: `StatefulPartitionedCall_1:0`
  - Shape: `[1, 1]`
  - Dtype: `float32`
  - Quantization: None (Floating point)

## 4. Historical Runtime Evidence
- **Export Date:** 2026-05-17
- **Op Sets:** TFLITE_BUILTINS only (Flex/Select ops 불필요)
- **Accuracy Parity:** FP16 모델의 Max Diff가 0.063 미만으로 Keras 모델과 매우 일치함.
- **Performance:** 
  - SPAQ SRCC: 0.899
  - KonIQ SRCC: 0.863

## 5. Preprocessing Contract
- **Color Space:** RGB
- **Resizing:** `resize_with_pad` (검은색 패딩 사용)
- **Target Size:** 384 x 384
- **Normalization:** **None** (모델 내부에 Preprocessing 레이어가 포함되어 있어 `[0, 255]` 범위의 float32 입력을 직접 받음)
- **Flutter Implementation Hint:** 이미지를 384x384로 패딩 리사이즈한 후, 0~255 범위의 float32 배열로 변환하여 입력.

## 6. Output Contract
- **Raw Range:** `[0, 1]` (Sigmoid 활성화 함수 적용됨)
- **Score Scale:** 0-100 (Raw Output * 100)
- **Interpretation:** 100점에 가까울수록 높은 화질.

## 7. Metadata Draft
(아래 `topiq_lite_mixed112.metadata.json` 파일 내용 참조)

## 8. Flutter Copy Instructions
1. `topiq_lite_mixed112_frozen_fp16.tflite` 파일을 Flutter 프로젝트의 `assets/models/` 폴더로 복사합니다.
2. `pubspec.yaml`에 자산을 등록합니다.
3. `topiq_lite_mixed112.metadata.json`의 내용을 참고하여 전처리 및 후처리를 구현합니다.

## 9. Blockers
- 없음. 모든 기술 사양이 확인됨.

## 10. Recommendation
- 현재 확인된 `topiq_lite_mixed112_frozen_fp16.tflite`는 모바일 배포에 가장 적합한 모델로 판단됩니다.
- C6 로깅 테스트를 위해 해당 모델을 Flutter 앱에 통합하는 것을 권장합니다.
