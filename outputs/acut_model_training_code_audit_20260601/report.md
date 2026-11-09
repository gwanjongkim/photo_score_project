# A-cut 사용 모델 학습 코드 추적 보고서

## 1. 감사 목적
- 현재 프로덕션 환경에서 사용 중인 이미지 평가 모델(Aesthetic & Technical)의 WSL/Python 기반 학습, 평가, 내보내기(Export) 경로를 역추적하여 데이터 계보(Provenance)를 확인합니다.
- 단순 추론 코드가 아닌, 모델이 생성된 기점(Training script)과 당시의 설정(Config)을 식별하는 것이 주 목적입니다.

## 2. 현재 추적 대상 모델 목록

| 모델 명 | 최종 TFLite 경로 | 학습/재구축 코드 | 체크포인트 (Checkpoint) | 확신도 |
| :--- | :--- | :--- | :--- | :--- |
| **NIMA** | `nima_mobile_fixed_preproc_fp16.tflite` | `src/train/train_nima.py` | `checkpoints/nima_ava_gpu_fixed_preproc_20260525/` | High |
| **RGNet** | `rgnet_pil_resize_aadb_fp16.tflite` | `src/train/train_rgnet_paper_v1_aadb.py` | `outputs/rgnet_v1_aadb_pil_resize_retrain_20260524/` | High |
| **A-LAMP v2**| `mobile_alamp_v2_fp16.tflite` | `src/train/train_mobile_alamp_v2.py` | `outputs/mobile_alamp_v2_full_ava_20260519/` | High |
| **ICAA** | `icaa_dat_tf_native_fp16.tflite` | `experiments/icaa_tf_native/scripts/stage_i_export_savedmodel.py` | `weights/icaa_official/e_30_ICAA17K...pth` | High |
| **KonIQ** | `koniq_mobile.tflite` | `src/train/train_regression.py` | `checkpoints/technical_koniq_gpu/` | High |
| **FLIVE** | `flive_image_mobile.tflite` | `src/train/train_regression.py` | `checkpoints/technical_flive_image_gpu/` | High |

---

## 3. 모델별 상세 추적 결과

### 3.1 NIMA (Aesthetic Distribution)
- **학습 스크립트**: `src/train/train_nima.py`
- **학습 데이터**: `data/processed/ava/train_cleaned.csv` (AVA Dataset)
- **전처리 특징**: EfficientNetV2의 `include_preprocessing=False` 설정으로 이중 리사이징 버그를 수정하고 [0, 1] 입력을 사용함.
- **체크포인트**: `checkpoints/nima_ava_gpu_fixed_preproc_20260525/best.weights.h5`
- **TFLite 내보내기**: `src/export/export_tflite.py --preset nima`
- **근거**: `outputs/nima_preprocessing_fix_20260525/report.md` 에 상세 학습 명령어가 기록되어 있음.

### 3.2 RGNet (Aesthetic Regression)
- **학습 스크립트**: `src/train/train_rgnet_paper_v1_aadb.py`
- **학습 데이터**: `data/processed/aadb/train.csv` (AADB Dataset)
- **전처리 특징**: PIL Bilinear 리사이즈를 명시적으로 사용하는 `pil_bilinear` 백엔드 기반 재학습.
- **체크포인트**: `outputs/rgnet_v1_aadb_pil_resize_retrain_20260524/best.weights.h5`
- **TFLite 내보내기**: `scripts/export_rgnet_paper_aadb_tflite.py`
- **근거**: `outputs/rgnet_v1_aadb_pil_resize_retrain_20260524/train_summary.json` 에 모델 아키텍처 및 하이퍼파라미터 상세 기록.

### 3.3 Mobile A-LAMP v2 (Multi-patch Aesthetic)
- **학습 스크립트**: `src/train/train_mobile_alamp_v2.py`
- **학습 데이터**: `outputs/alamp_v4_full_ava_20260517/subsets/train_patch_boxes_full_v4.jsonl` (AVA 기반 패치 추출 데이터)
- **전처리 특징**: 0~255 범위의 Raw Pixel 입력을 사용하며 MobileNetV3 내부 전처리 레이어 활용.
- **체크포인트**: `outputs/mobile_alamp_v2_full_ava_20260519/best_val_auc_model.keras`
- **TFLite 내보내기**: `src/export/export_alamp_tflite.py`
- **근거**: `outputs/mobile_alamp_v2_full_ava_20260519/summary.json` 에 실행된 전체 `command` 라인이 포함됨.

### 3.4 ICAA (Color Aesthetic)
- **재구축 스크립트**: `experiments/icaa_tf_native/scripts/stage_i_export_savedmodel.py`
- **근원 가중치**: 공식 PyTorch 가중치(`weights/icaa_official/e_30_ICAA17K...pth`)
- **수행 방식**: PyTorch 가중치를 TF Native 구현체로 로드하여 SavedModel로 저장한 뒤 TFLite로 변환 (WSL에서 새로 학습되지 않음).
- **TFLite 내보내기**: `experiments/icaa_tf_native/scripts/stage_k_convert_tflite_fp16.py`
- **근거**: `outputs/icaa_tf_native_savedmodel_20260516_134439/stage_i_savedmodel_report.json` 에 변환 과정 기록.

### 3.5 KonIQ / FLIVE (Technical Quality)
- **학습 스크립트**: `src/train/train_regression.py --task technical`
- **학습 데이터**: KonIQ-10k 및 FLIVE 데이터셋.
- **전처리 특징**: 224x224 리사이즈, [0, 1] 정규화.
- **체크포인트**: `checkpoints/technical_koniq_gpu/best.weights.h5` 등.
- **TFLite 내보내기**: `src/export/export_tflite.py --preset koniq / flive_image`
- **근거**: `outputs/audit_model_paper_basis_20260506/audit_report.md` 내 모델 이력 섹션.

---

## 4. 학습 코드와 추론 코드 구분

1. **학습 코드 (`src/train/`)**: 
   - 모델을 처음부터 훈련하거나 특정 데이터셋으로 파인튜닝하는 역할.
   - 출력물: `.keras` (전체 모델) 또는 `.weights.h5` (가중치).
2. **평가 및 검증 코드 (`scripts/` 또는 `src/infer/`)**:
   - 학습된 모델의 성능(SRCC, MAE 등)을 확인하고 TFLite 변환 후의 정확도 일치성(Parity)을 검증.
3. **내보내기 코드 (`src/export/`)**:
   - 학습 완료된 모델을 모바일 배포용 TFLite 포맷으로 변환.
4. **런타임 추론 코드 (`src/infer/`)**:
   - 실제 앱 서비스 환경과 유사하게 단일 이미지 또는 폴더에 대해 점수를 산출하는 프로덕션용 코드.

---

## 5. 불확실한 항목

- **KonIQ/FLIVE의 정확한 학습 명령어**: `train_regression.py`를 사용한 것은 확실하나, 학습 시점에 실행된 로그 파일이 `outputs/`에 명시적인 날짜 기반 폴더로 남아있지 않아 정확한 Epoch 수나 LR 등은 체크포인트 메타데이터를 직접 확인해야 함.
- **ICAA 파인튜닝 여부**: 현재 ICAA는 공식 가중치를 TF Native로 재구축하여 사용 중이며, WSL 내부 데이터셋을 통한 추가 파인튜닝 기록은 발견되지 않음.

---

## 6. 보고서용 요약 요약

- **NIMA**는 `src/train/train_nima.py`를 통해 AVA 데이터셋 기반으로 학습되었으며, 전처리 버그가 수정된 2026-05-25일자 체크포인트에서 TFLite로 변환되었습니다.
- **RGNet**은 PIL 리사이즈 불일치를 해결하기 위해 `src/train/train_rgnet_paper_v1_aadb.py`로 AADB 데이터셋에 대해 재학습되었습니다.
- **Mobile A-LAMP v2**는 `src/train/train_mobile_alamp_v2.py`를 사용하여 AVA 기반 패치 데이터셋으로 학습된 근거가 확인되었습니다.
- **ICAA**는 공식 PyTorch 가중치를 TF Native 아키텍처로 이식(Stage I)한 후 TFLite로 변환(Stage K)한 산출물입니다.
- **KonIQ / FLIVE**는 공용 회귀 학습 코드인 `src/train/train_regression.py`를 사용하여 각각의 기술 품질 데이터셋으로 학습되었습니다.
