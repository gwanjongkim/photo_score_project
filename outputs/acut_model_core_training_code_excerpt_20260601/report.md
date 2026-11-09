# A-cut 모델별 핵심 학습 코드 발췌 보고서

## 1. 목적
- 현재 프로덕션 환경에서 사용하는 이미지 평가 모델들의 핵심 Python 코드를 발췌하여, 모델 생성 및 변환 과정을 기술적으로 문서화합니다.
- 전체 소스 코드가 아닌, 아키텍처 정의, 전처리 규약, 학습 루프, 내보내기 로직 등 모델의 본질을 설명하는 핵심 스니펫을 중심으로 구성합니다.

## 2. 모델별 핵심 코드 요약표

| 모델 명 | 용도 | 학습/변환 스크립트 | 아키텍처 발췌 | 전처리 발췌 | 학습 발췌 | 내보내기 발췌 | 확신도 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NIMA** | 미적 분포 | `train_nima.py` | 10-bin Softmax | Center Crop | EMD Loss | Preset | High |
| **RGNet** | 미적 회귀 | `train_rgnet_paper_v1_aadb.py` | GNN/ASPP | PIL Bilinear | MSE | Script | High |
| **A-LAMP v2** | 멀티 패치 | `train_mobile_alamp_v2.py` | Dual-branch | Saliency | Class Weight | Script | High |
| **ICAA** | 색상 미학 | `stage_i_export_savedmodel.py` | TF Native | ImageNet Norm | (Rebuild) | FP16 | High |
| **KonIQ** | 기술 품질 | `train_regression.py` | MobileNetV2 | [0..1] Norm | MSE (Argparse) | Preset | High |
| **FLIVE** | 기술 품질 | `train_regression.py` | MobileNetV2 | [0..1] Norm | MSE (Argparse) | Preset | High |

---

## 3. NIMA 핵심 코드
### 3.1 모델 구조
NIMA는 EfficientNetV2B0 백본 위에 10개의 점수 빈(Bin)을 예측하는 Softmax 헤드를 결합합니다.
- 파일: `src/models/nima_distribution.py`
- 핵심: `outputs = tf.keras.layers.Dense(10, activation="softmax", dtype="float32")(x)`

### 3.2 데이터셋/전처리
- 파일: `src/datasets/ava_distribution_dataset.py`
- 특징: 256 리사이즈 후 224 센터 크롭을 수행하며, `convert_image_dtype`을 통해 [0, 1] 범위를 보장합니다.

### 3.3 학습 설정 및 학습 루프
- 파일: `src/train/train_nima.py`
- 손실 함수: `emd_loss` (Earth Mover's Distance)
- 평가 지표: `mean_score_mae`

### 3.4 TFLite Export
- 파일: `src/export/export_tflite.py`
- 프리셋: `nima` (FP16 최적화 적용)

---

## 4. RGNet 핵심 코드
> **참고**: 현재 배포된 핵심 자산은 `rgnet_pil_resize_aadb_fp16.tflite`이며, 이는 논문 재현 목적의 부가 실험과는 구분되는 실무용 모델입니다.

### 4.1 모델 구조
ASPP(Atrous Spatial Pyramid Pooling)와 지역 기반 그래프 컨볼루션을 사용하여 이미지의 구조적 미적 요소를 학습합니다.
- 파일: `src/models/rgnet_paper_v1.py`

### 4.2 전처리
모바일 환경과의 일치성을 위해 학습 시에도 PIL Bilinear 리사이즈를 강제합니다.
- 파일: `src/train/train_rgnet_paper_v1_aadb.py`

### 4.3 학습 설정
- 손실 함수: MSE
- 특징: AADB 데이터셋의 점수 범위([0, 1])에 맞춘 Sigmoid 출력층 사용.

---

## 5. Mobile A-LAMP v2 핵심 코드
### 5.1 모델 구조
전체 이미지와 5개의 주요 패치를 동시에 입력받아 특징을 융합합니다.
- 파일: `src/models/mobile_alamp_v2.py`

### 5.2 전처리
에지 에너지 및 분산을 활용한 Saliency 기반 적응형 패치 추출 로직이 포함되어 있습니다.
- 파일: `src/datasets/native_size_dataset.py`

### 5.3 학습 루프
- 특징: `class_weight`를 적용하여 데이터 불균형을 해소합니다.

---

## 6. ICAA 핵심 코드
### 6.1 TF Native 모델 재구축
공식 PyTorch 가중치를 로드하기 위해 TF Native로 동일한 아키텍처를 재현했습니다. WSL 환경에서 새로 학습된 것이 아니라, 기존 모델의 가중치를 TensorFlow 환경으로 이식한 산출물입니다.
- 파일: `experiments/icaa_tf_native/scripts/stage_i_export_savedmodel.py`
- 내부 구현: `TFICAA17KDAT` (내부 구현 파일 별도 확인 필요)

### 6.2 공식 가중치 로드
- 특징: `load_from_pytorch_state_dict` 메서드를 통해 레이어별 가중치 전이(Transfer) 수행.

### 6.3 SavedModel/TFLite 변환
- 특징: 변환 직후 랜덤 입력 및 실제 이미지를 사용하여 원본 모델과의 Parity(일치성)를 엄격히 검증합니다.

---

## 7. KonIQ / 8. FLIVE 핵심 코드
기술 품질 모델은 MobileNetV2 백본을 기반으로 하며, 범용 회귀 학습 코드를 공유합니다.
- 아키텍처 파일: `src/models/technical_regressor.py`
- 학습 파일: `src/train/train_regression.py` (Argparse 기반 태스크 및 데이터셋 선택 로직 포함)
- 특징: MSE Loss를 기반으로 스칼라 품질 점수를 직접 예측합니다.
- 내보내기: `src/export/export_tflite.py`의 `koniq` 및 `flive_image` 프리셋 사용.

---

## 9. 공통 Export 코드
대부분의 모델은 `src/export/export_tflite.py`를 통해 규격화된 방식으로 배포됩니다.
- 특징: `TFLiteConverter.from_keras_model` 사용, FP16 최적화, 메타데이터 동시 생성.

---

## 10. 발췌 한계
- **세부 로그 부족**: 일부 모델의 경우 학습 당시의 구체적인 하이퍼파라미터 로그 파일이 발췌문에 모두 포함되지는 않았습니다.
- **아키텍처 복잡성**: RGNet 및 A-LAMP의 커스텀 레이어 전체 코드는 양이 방대하여 핵심 연산 부분만 포함되었습니다.
