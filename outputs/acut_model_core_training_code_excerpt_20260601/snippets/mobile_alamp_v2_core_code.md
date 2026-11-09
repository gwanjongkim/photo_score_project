# Mobile A-LAMP v2 Core Code Snippets

Mobile A-LAMP v2 모델은 전역 뷰(Global View)와 국소 패치(Patches)를 결합하여 미적 품질을 평가합니다.

### 1. 모델 구조 (Architecture)
`src/models/mobile_alamp_v2.py`에서 정의된 모델은 두 개의 MobileNetV3Small 백본을 사용하는 듀얼 브랜치 구조입니다.

```python
# src/models/mobile_alamp_v2.py
def build_mobile_alamp_v2_model(...):
    # Inputs: Patches & Full Image
    patches = tf.keras.Input(shape=(patch_count, patch_size, patch_size, 3), name="patches")
    full_image = tf.keras.Input(shape=(image_size, image_size, 3), name="full_image")

    # Shared Backbones: MobileNetV3Small
    patch_backbone = _build_mobilenetv3small(...)
    layout_backbone = _build_mobilenetv3small(...)

    # Fusion
    fused = tf.keras.layers.Concatenate(name="patch_layout_concat")([patch_feature, layout_feature])
    score = tf.keras.layers.Dense(1, activation="sigmoid", name="score")(fused)
    return tf.keras.Model(inputs={"patches": patches, "full_image": full_image}, outputs=score)
```

### 2. 전처리 (Preprocessing)
A-LAMP는 이미지의 특징적인 부분을 추출하기 위해 Saliency Map 기반의 적응형 패치 제안 알고리즘을 사용합니다.

```python
# src/datasets/native_size_dataset.py
def prepare_alamp_inputs(image, global_size=384, patch_size=224, num_patches=5):
    # 전체 이미지 리사이즈 (Aspect Ratio 보존)
    global_view = resize_with_pad(image, target_size=global_size)
    # Saliency 기반 패치 영역 제안
    boxes, _ = propose_adaptive_boxes(image=image, num_patches=num_patches)
    # 이미지에서 제안된 영역 크롭 및 리사이즈
    patches = tf.image.crop_and_resize(image[None, ...], boxes, ...)
    return global_view, patches
```

### 3. 학습 설정 (Training)
학습 데이터셋의 불균형을 해결하기 위해 클래스 가중치를 적용합니다.

```python
# src/train/train_mobile_alamp_v2.py
def _class_weights_from_summary(label_summary, mode):
    # Positive/Negative 비율에 따른 가중치 계산
    return {
        0: total / (2.0 * negative_count),
        1: total / (2.0 * positive_count),
    }

# 학습 실행
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    class_weight=class_weight,
    ...
)
```

### 4. TFLite Export
`src/export/export_alamp_tflite.py`를 통해 다중 입력(Multi-input)을 지원하는 TFLite 모델로 내보냅니다.
