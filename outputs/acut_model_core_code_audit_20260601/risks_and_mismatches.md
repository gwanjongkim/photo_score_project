# 리스크 및 코드 불일치 분석 보고서

## 1. 확인된 불일치 (Confirmed Mismatches)

### 1.1 Mobile A-LAMP v2 전처리 (0..1 vs 0..255)
- **증상**: `src/infer/predict_quality_bundle.py` (L619) 및 `src/export/alamp_tflite.py` (L128)에서는 이미지를 `/255.0`으로 나누어 0~1 범위로 입력하고 있음.
- **리스크**: `forWeights2/models/aesthetic/MODEL_CARD.md`와 `configs/aesthetic_weight_lab.yaml`에 따르면 `mobile_alamp_v2_fp16.tflite` 모델은 **0..255 (Raw Float)** 입력을 기대하며 내부 MobileNetV3 레이어에서 전처리를 수행함.
- **영향**: 0..1 입력을 줄 경우 점수가 비정상적으로 낮게 나오거나 성능이 크게 저하될 수 있음.

### 1.2 기술 모델 가중치 (Hardcoded vs Intended)
- **증상**: `src/infer/select_best_shots.py`의 `DEFAULT_SELECTOR_WEIGHTS`에는 KonIQ 0.32, FLIVE 0.58 등으로 복잡하게 분산되어 있음.
- **결정 사항**: 2026년 5월 30일자 리서치 보고서 (`outputs/research_protocol_production_technical_eval_20260530/report.md`)에서는 **KonIQ 4/7, FLIVE 3/7** 단순 결합을 프로덕션 공식으로 정의함.
- **영향**: 기본 파이프라인 실행 시 최신 리서치 결과가 아닌 이전 가중치가 적용될 위험이 있음.

## 2. 의심되는 리스크 (Suspected Risks)

### 2.1 ICAA 출력 인덱스 오설정
- **내용**: ICAA TFLite 모델은 `[MOS, Color]` 벡터를 출력함. 프로덕션에서 필요한 "색상 미학 점수"는 Index 1임.
- **현황**: `forWeights2/configs/aesthetic_weight_lab.yaml` 등 일부 설정 파일에 `score_index: 0`으로 설정된 사례가 발견됨. 이는 전체 미학 점수(MOS)를 가져오게 되어 색상 특화 평가가 누락될 수 있음.

### 2.2 RGNet 256 Resize 제약
- **내용**: RGNet 모델은 256x256 입력을 기반으로 훈련되었으며, PIL Resize 방식에 민감함.
- **현황**: 모바일(Flutter) 구현 시 플랫폼 기본 리사이즈 라이브러리가 PIL과 미세하게 다를 경우 점수 오차가 발생할 수 있음 (Parity 검증 필요).

## 3. 누락된 증거 및 추가 확인 사항

### 3.1 Flutter 프로젝트 연동 코드
- WSL 레포지토리 내에는 Python 기반의 모델 관리 코드만 존재함.
- **필수 확인**: Flutter 프로젝트 내 `lib/` 또는 `assets/` 설정에서 위에서 언급한 전처리(A-LAMP 0-255) 및 출력 인덱스(ICAA Index 1)가 올바르게 구현되어 있는지 확인해야 함.

### 3.2 TOPIQ-lite 프로덕션 전환 시점
- 리서치 상으로는 TOPIQ-lite가 우수하나, `run_acut_pipeline.py`의 기본 모델 리스트에는 아직 포함되지 않음. 전환을 위한 명시적인 Config 업데이트가 필요함.
