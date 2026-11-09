# A-cut 이미지 평가 모델 핵심 코드 감사 보고서

## 1. 감사 목적
- 현재 A-cut 프로젝트에서 사용하는 미적(Aesthetic) 모델과 기술(Technical) 모델의 핵심 코드 위치 및 구현 상세 확인
- Production 모델과 Research Candidate 모델의 명확한 구분
- 전처리, 추론, 후처리, 가중치 결합(Fusion) 로직의 소스 코드 기반 검증

## 2. 현재 사용 모델 목록

### 2.1 Production/Practical Aesthetic Models
- **NIMA**: 미적 점수 분포를 기반으로 평균 점수 계산.
- **RGNet**: 회귀 기반 미적 점수 모델 (PIL Resize 전처리 기반).
- **Mobile A-LAMP v2**: Multi-patch 기반 미적 모델 (Raw 0-255 입력 사용).
- **ICAA**: 색상 기반 미적 모델 (Output Index 1이 Color Aesthetic 점수).

### 2.2 Production Technical Models
- **KonIQ**: 기술적 품질 평가 모델.
- **FLIVE**: 기술적 품질 평가 모델 (Image 및 Patch 기반).
- **Technical Fusion**: KonIQ (4/7) + FLIVE (3/7) 가중치 결합 (최신 결정 사항).

### 2.3 Research / Log-only Candidates
- **TOPIQ-lite**: 최신 고성능 IQA 모델 후보.
- **C6 (TOPIQ+KonIQ Soft-Cap)**: 기술 점수 보정 모델 후보.
- **DistortionGuard**: 왜곡 감지 및 방어 모델 후보.

## 3. 핵심 코드 위치 요약표

| 모델/시스템 | 용도 | 로더 파일 | 전처리 함수 | 추론 파일/함수 | 후처리/가중치 로직 | 확신도 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pipeline Hub** | 전체 랭킹 | `run_acut_pipeline.py` | - | `select_best_shots.py` | `rank_records` | High |
| **NIMA** | Aesthetic | `predict_quality_bundle.py` | `preprocess_image_pil` | `predict_nima.py` | `distribution_mean_score` | High |
| **RGNet** | Aesthetic | `predict_quality_bundle.py` | `preprocess_image_pil` | `predict_rgnet.py` | Scalar output (0-1) | High |
| **A-LAMP v2** | Aesthetic | `predict_quality_bundle.py` | `prepare_alamp_inputs` | `predict_alamp.py` | Scalar output | High |
| **ICAA** | Color | `vector_tflite` (YAML) | 0-1 Normalization | - | Output Index 1 (Color) | High |
| **KonIQ** | Technical | `predict_quality_bundle.py` | `preprocess_image_pil` | `predict_single` | `score / 100.0` | High |
| **FLIVE** | Technical | `predict_quality_bundle.py` | `make_patch_batch` | `predict_batch` | `score / 100.0` | High |
| **Final Fusion**| 통합 점수 | `select_best_shots.py` | - | - | `DEFAULT_SELECTOR_WEIGHTS` | High |

## 4. 미적 모델 코드 분석

### 4.1 NIMA
- **모델 파일**: `nima_mobile_fixed_preproc_fp16.tflite`
- **전처리**: 256 Resize -> 224 Center Crop -> /255.0 (0~1 범위).
- **출력**: `[1, 10]` 크기의 확률 분포.
- **점수 변환**: `sum(p_i * i)` 로 평균 점수(MOS) 계산 후 `(mean - 1) / 9.0` 으로 0-1 정규화.
- **코드 위치**: `src/models/nima_distribution.py`, `src/infer/predict_nima.py`.

### 4.2 RGNet
- **모델 파일**: `rgnet_pil_resize_aadb_fp16.tflite`
- **전처리**: PIL Bilinear Resize (256x256) -> /255.0.
- **출력**: 0~1 범위의 Scalar 미적 점수.
- **코드 위치**: `src/models/rgnet.py`, `src/infer/predict_quality_bundle.py` (L586).

### 4.3 Mobile A-LAMP v2
- **모델 파일**: `mobile_alamp_v2_fp16.tflite`
- **입력 구조**: Global View (384x384) + 5 Patches (224x224).
- **전처리 중요 사항**: **나누기 255를 하지 않은 Raw 0~255 Float 입력** 사용 (내부 MobileNetV3 전처리 포함).
- **코드 위치**: `src/models/alamp.py`, `src/datasets/native_size_dataset.py` (L156).
- **리스크**: `predict_quality_bundle.py` (L619)에서는 `/255.0`을 수행하고 있어, TFLite 모델용 전처리와 서버용 `.keras` 모델 전처리가 다를 가능성 확인 필요.

### 4.4 ICAA
- **모델 파일**: `icaa_dat_tf_native_fp16.tflite`
- **전처리**: 224x224 Resize -> /255.0.
- **출력 중요 사항**: 벡터 형태 출력 `[MOS, Color]`. **Color Aesthetic 점수는 Index 1** 사용.
- **코드 위치**: `forWeights2/configs/aesthetic_weight_lab.yaml` 및 관련 리서치 보고서 (`outputs/icaa_performance_reverification_20260525/report.md`).

## 5. 기술 모델 코드 분석

### 5.1 KonIQ
- **모델 파일**: `koniq_mobile.tflite`
- **전처리**: 224x224 Resize -> /255.0.
- **점수 정규화**: `raw_score / 100.0`.
- **코드 위치**: `src/infer/predict_quality_bundle.py` (L228).

### 5.2 FLIVE
- **모델 파일**: `flive_image_mobile.tflite`
- **전처리**: Image (224x224) 및 50% ratio Patches 추출.
- **점수 정규화**: `raw_score / 100.0`.
- **코드 위치**: `src/infer/predict_quality_bundle.py` (L229-231).

### 5.3 KonIQ+FLIVE Production Technical Score
- **공식**: `(4/7 * KonIQ) + (3/7 * FLIVE_image)`
- **증거**: `outputs/research_protocol_production_technical_eval_20260530/production_formula_audit.md`.
- **참고**: `select_best_shots.py`의 기본 가중치는 현재 KonIQ(0.32) + FLIVE(0.58) + MUSIQ(0.10)로 설정되어 있으나, 최신 결정은 4/7 & 3/7임.

## 6. Research / Candidate Technical Models

### 6.1 TOPIQ-lite
- **상태**: 고성능 IQA 후보. `topiq_lite_mixed112_frozen_fp16.tflite`로 변환 완료.
- **코드**: `scripts/technical_iqa/` 산하 실험 코드.

### 6.2 C6 TOPIQ+KonIQ Soft-Cap
- **상태**: 기술 점수 상한선(Soft-cap) 적용 연구용.
- **코드**: `run_wsl_c6_parity_with_orientation.py`.

### 6.3 DistortionGuard
- **상태**: 특정 왜곡(Blur, Noise 등) 감지용 연구 모델.
- **코드**: `check_severe.py`, `severe_fp_dist.py`.

## 7. Score Fusion / Final Ranking Pipeline

- **결합 위치**: `src/infer/select_best_shots.py` 의 `rank_records` 함수.
- **과정**: 
  1. `compute_selector_row`: 개별 모델 점수 정규화 및 컴포넌트(Aesthetic/Technical) 점수 합산.
  2. `apply_optional_pairwise_rerank`: 상위 Pool 내 Pairwise 모델 기반 재정렬.
  3. `apply_optional_vila_rerank`: VILA(LLM) 기반 미적 재정렬.
  4. `apply_optional_diversity_rerank`: 중복 이미지 제거(Lightweight similarity).
- **임계값**: `diversity_threshold` (0.82), `hard_duplicate_threshold` (0.88).

## 8. 리스크 및 검증 항목

- **A-LAMP 전처리**: TFLite 모델은 0~255 입력을 기대하지만, 일부 Python 코드(`predict_quality_bundle.py`)에서 `/255.0`을 수행하는 불일치가 감지됨. Flutter 연동 시 반드시 Raw 0-255 확인 필요.
- **ICAA 출력 인덱스**: `aesthetic_weight_lab.yaml`에는 `score_index: 0`(MOS)으로 설정된 경우가 있으나, 프로덕션 요구사항인 "Color Score"는 **Index 1**임.
- **기술 가중치 불일치**: `select_best_shots.py`의 하드코딩된 기본값과 5월 30일 리서치 결정 사항(4/7, 3/7) 간의 차이가 존재함.
- **Flutter 연동**: WSL 환경의 Python 코드와 Flutter 프로젝트 내의 `model_config.dart` 또는 추론 로직 간 동기화 여부 확인 필수.

## 9. 권장 차기 조치
- `src/infer/select_best_shots.py`의 `DEFAULT_SELECTOR_WEIGHTS`를 최신 4/7, 3/7 결정 사항으로 업데이트 검토.
- `predict_quality_bundle.py`의 A-LAMP 전처리 로직을 TFLite 모델 계약(0~255)에 맞게 통일.
- ICAA 연동 시 인덱스 1(Color) 사용을 강제하는 명시적 코드 주석 또는 설정 추가.
