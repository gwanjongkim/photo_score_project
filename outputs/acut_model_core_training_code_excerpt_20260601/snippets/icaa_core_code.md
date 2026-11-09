# ICAA Core Code Snippets

ICAA 모델은 공식 PyTorch 가중치를 TF Native 구현체로 재구축(Rebuild)하여 TFLite로 변환되었습니다. WSL 환경에서 새로 학습된 것이 아니라, 기존 모델의 가중치를 TensorFlow 환경으로 이식한 산출물입니다.

### 1. TF Native 모델 재구축 (Architecture Rebuild)
`TFICAA17KDAT` 클래스를 사용하여 PyTorch 가중치를 로드합니다 (내부 구현 파일 별도 확인 필요).

```python
# experiments/icaa_tf_native/scripts/stage_i_export_savedmodel.py
def run():
    # 1. PyTorch 체크포인트 로드
    checkpoint_obj = load_checkpoint(torch, checkpoint_path)
    state_dict, _ = extract_state_dict(checkpoint_obj)

    # 2. TF Native 모델 인스턴스화
    TFICAA17KDAT = import_tf_dat(root)
    model_tf = TFICAA17KDAT(epsilon=1e-5)
    model_tf.build_for_input_shape((1, 224, 224, 3))
    
    # 3. 가중치 전이 (PyTorch -> TensorFlow)
    model_tf.load_from_pytorch_state_dict(state_dict)

    # 4. SavedModel로 내보내기
    tf.saved_model.save(module, str(saved_model_path), ...)
```

### 2. 전처리 (Preprocessing)
ICAA는 ImageNet 평균 및 표준편차를 사용하여 정규화를 수행합니다.

```python
# experiments/icaa_tf_native/scripts/stage_i_export_savedmodel.py
def preprocess_real_image(path: Path):
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    image = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return (array - mean) / std
```

### 3. TFLite FP16 변환 (Export)
SavedModel을 FP16 TFLite로 변환하고 동등성(Parity)을 검증합니다.

```python
# experiments/icaa_tf_native/scripts/stage_k_convert_tflite_fp16.py
def build_fp16_converter(sm_model, infer, use_select_tf_ops):
    concrete_func = model_func.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], sm_model)
    # FP16 최적화 설정
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter
```
