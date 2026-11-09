# KonIQ Core Code Snippets

KonIQ 모델은 이미지의 기술적 품질 점수(Technical Quality)를 예측하는 회귀 모델입니다.

### 1. 학습 설정 및 태스크 선택 (Training Task Selection)
`src/train/train_regression.py`를 통해 기술 품질 모델로 설정하여 학습을 진행합니다.

```python
# src/train/train_regression.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["aadb", "technical"], required=True)
    # ...
    args = parser.parse_args()

    if args.task == "aadb":
        model = build_aadb_regressor(...)
    else:
        # KonIQ/FLIVE 등 기술 모델용 빌더 호출
        model = build_technical_regressor((args.image_size, args.image_size, 3))
```

### 2. 모델 구조 (Architecture)
`src/models/technical_regressor.py`에서 정의된 모델은 MobileNetV2를 백본으로 사용합니다.

```python
# src/models/technical_regressor.py
def build_technical_regressor(input_shape=(224, 224, 3)):
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    # ...
    # 기술 점수 스칼라 값 출력 (MOS-like)
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)
    return tf.keras.Model(inputs, outputs, name="technical_regressor")
```

### 3. 데이터셋 및 전처리 (Preprocessing)
공용 회귀 데이터셋 코드를 사용하여 224x224 리사이즈 및 [0, 1] 정규화를 수행합니다.

```python
# src/datasets/csv_dataset.py
def decode_img(path, image_size=(224, 224), training=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32) # [0, 1] 범위 강제
    # ...
    return img
```

### 4. 학습 루프 (Training Loop)
MSE(Mean Squared Error) 손실 함수를 사용하여 실제 MOS 점수와의 오차를 최소화합니다.

```python
# src/train/train_regression.py
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="mse",
    metrics=["mae"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    callbacks=callbacks
)
```

### 5. TFLite Export
`src/export/export_tflite.py`에서 `koniq` 프리셋을 사용하여 내보냅니다. 이 프리셋은 내부적으로 `score / 100.0` 정규화 가이드를 포함합니다.
