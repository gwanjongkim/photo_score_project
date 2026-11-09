# FLIVE Core Code Snippets

FLIVE 모델은 KonIQ와 동일한 아키텍처를 기반으로 하지만, 이미지 및 패치 기반의 품질 데이터셋(FLIVE)으로 학습되었습니다.

### 1. 학습 설정 (Training Task)
KonIQ와 마찬가지로 `src/train/train_regression.py`를 사용하되, 입력 데이터셋(`--train_csv`)으로 FLIVE 경로를 지정합니다.

```python
# src/train/train_regression.py (FLIVE specific run)
# Example execution logic (interpreted from pipeline artifacts)
# --task technical --train_csv data/processed/flive/train.csv --target_col mos
```

### 2. 모델 구조 (Architecture)
KonIQ와 공유하는 `build_technical_regressor`를 사용합니다 (MobileNetV2 backbone).

```python
# src/models/technical_regressor.py
model = build_technical_regressor(input_shape=(224, 224, 3))
```

### 3. 데이터셋 및 전처리 (Preprocessing)
이미지 전체 뷰를 기반으로 224x224 리사이즈 및 [0, 1] 정규화를 수행합니다.

```python
# src/datasets/csv_dataset.py
# FLIVE 이미지 데이터 로드 및 정규화
img = decode_img(path, image_size=(224, 224), ...)
```

### 4. 학습 루프 (Training)
MSE Loss를 사용하여 FLIVE 데이터셋의 주관적 품질 점수(MOS)를 회귀하도록 학습합니다.

```python
# src/train/train_regression.py
model.fit(train_ds, validation_data=val_ds, ...)
```

### 5. TFLite Export
`src/export/export_tflite.py`에서 `flive_image` 프리셋을 사용하여 내보냅니다.
