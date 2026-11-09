# NIMA Core Code Snippets

NIMA 모델은 이미지의 미적 품질 점수 분포(1..10)를 예측하는 모델입니다.

### 1. 모델 구조 (Architecture)
`src/models/nima_distribution.py`에서 정의된 NIMA의 핵심은 10개의 빈(bin)을 가진 Softmax 출력층입니다.

```python
# src/models/nima_distribution.py
def build_nima_distribution_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    backbone_weights: str | None = "imagenet",
) -> tf.keras.Model:
    base = _build_efficientnetv2_backbone(input_shape=input_shape, weights=backbone_weights)

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    # 10개 점수(1..10)에 대한 확률 분포 출력
    outputs = tf.keras.layers.Dense(10, activation="softmax", dtype="float32")(x)
    return tf.keras.Model(inputs, outputs, name="nima_distribution")
```

### 2. 전처리 (Preprocessing)
NIMA는 EfficientNetV2 백본을 사용하며, 이중 스케일링 버그를 방지하기 위해 `include_preprocessing=False` 설정을 사용하고 [0, 1] 범위를 직접 입력받습니다.

```python
# src/datasets/ava_distribution_dataset.py
def _decode_image(path: tf.Tensor, image_size: tuple[int, int], training: bool) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    # [0, 1] 범위로 변환
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    # 256 리사이즈 후 224 센터 크롭
    resize_h = tf.maximum(256, image_size[0])
    resize_w = tf.maximum(256, image_size[1])
    img = tf.image.resize(img, [resize_h, resize_w], method="bilinear")
    if training:
        img = tf.image.random_crop(img, [image_size[0], image_size[1], 3])
    else:
        img = tf.image.resize_with_crop_or_pad(img, image_size[0], image_size[1])
    return img
```

### 3. 학습 루프 및 손실 함수 (Training & Loss)
NIMA는 분포 간의 거리를 최적화하기 위해 Earth Mover's Distance(EMD) 손실 함수를 사용합니다.

```python
# src/models/nima_distribution.py
@tf.keras.utils.register_keras_serializable(package="photo_score_project")
def emd_loss(y_true: tf.Tensor, y_pred: tf.Tensor, r: int = 2) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    cdf_true = tf.cumsum(y_true, axis=-1)
    cdf_pred = tf.cumsum(y_pred, axis=-1)
    samplewise = tf.reduce_mean(tf.pow(tf.abs(cdf_true - cdf_pred), r), axis=-1)
    return tf.pow(samplewise, 1.0 / r)
```

### 4. TFLite Export
`src/export/export_tflite.py`에서 `nima` 프리셋을 사용하여 FP16 변환을 수행합니다.
(공통 Export 코드 참조)
