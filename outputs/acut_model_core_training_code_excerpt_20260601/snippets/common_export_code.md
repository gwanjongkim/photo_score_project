# Common TFLite Export Code Snippets

A-cut 프로젝트의 다수 모델(NIMA, KonIQ, FLIVE, AADB 등)은 공용 TFLite 내보내기 스크립트를 공유합니다.

### 1. Keras 모델 TFLite 변환 (FP16 최적화)
학습된 Keras 모델을 로드하여 FP16 정밀도로 최적화된 TFLite 파일로 변환합니다.

```python
# src/export/export_tflite.py
def convert_keras_model(model: tf.keras.Model, mode: str) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # SELECT_TF_OPS가 필요한 경우 대응
    if mode == "select_tf_ops":
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
    return converter.convert()

# FP16 최적화 (TFLitePreset build 시 적용)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
```

### 2. 메타데이터 생성
변환 시 모델의 입력 크기, 정규화 방식, 출력 의미 등을 JSON 메타데이터로 함께 저장하여 앱 연동 시 참고합니다.

```python
# src/export/export_tflite.py
metadata = {
    **preset.to_metadata(),
    "source": {
        "checkpoint_path": str(checkpoint_path),
        "weights_path": str(weights_path),
    },
    "export": {
        "used_conversion_mode": used_mode,
        "tflite_path": str(output_path),
    },
}
```

### 3. 주요 프리셋 (Presets)
각 모델별 고유 설정을 프리셋으로 관리합니다.

```python
# src/export/tflite_presets.py
PRESETS: dict[str, TFLitePreset] = {
    "koniq": TFLitePreset(
        name="koniq",
        input_size=224,
        normalization="pixel_value / 255.0",
        output_interpretation="Raw MOS-like score.",
        ...
    ),
    "nima": TFLitePreset(
        name="nima",
        input_size=224,
        output_shape=(1, 10),
        output_interpretation="Softmax distribution.",
        ...
    ),
}
```
