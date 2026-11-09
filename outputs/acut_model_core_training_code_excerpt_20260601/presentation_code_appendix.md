# 부록: A-cut 모델 학습 및 변환 핵심 코드

본 부록은 A-cut 프로젝트에서 사용되는 주요 이미지 평가 모델들의 핵심 구현 코드를 요약하여 제공합니다.

## 1. NIMA (Aesthetic Distribution)
- **목적**: 10단계 점수 분포 예측을 통한 미적 품질 산출
- **학습 스크립트**: `src/train/train_nima.py`
- **핵심 코드**:
```python
# 10개 빈(bin)을 가진 Softmax 출력층 정의
outputs = tf.keras.layers.Dense(10, activation="softmax", dtype="float32")(x)

# Earth Mover's Distance 손실 함수
def emd_loss(y_true, y_pred, r=2):
    cdf_true = tf.cumsum(y_true, axis=-1)
    cdf_pred = tf.cumsum(y_pred, axis=-1)
    return tf.pow(tf.reduce_mean(tf.pow(tf.abs(cdf_true - cdf_pred), r), axis=-1), 1.0/r)
```

## 2. RGNet (Aesthetic Regression)
- **목적**: 그래프 신경망 기반 지역적 관계 학습 및 미적 회귀
- **학습 스크립트**: `src/train/train_rgnet_paper_v1_aadb.py`
- **핵심 코드**:
```python
# ASPP(Atrous Spatial Pyramid Pooling)를 통한 다중 스케일 컨텍스트 추출
context_map = ASPPContextModule(filters=256, dilation_rates=(1, 3, 6, 12, 18))(feature_map)

# 코사인 유사도 기반 인접 행렬 생성 및 그래프 컨볼루션
adjacency = V1RegionSimilarityAdjacency(temperature=0.25)(node_features)
graph_context = V1ResidualGraphConvolution(units=256)((node_features, adjacency))
```

## 3. Mobile A-LAMP v2 (Multi-patch Aesthetic)
- **목적**: 전역 이미지와 국소 패치의 적응형 결합을 통한 미적 평가
- **학습 스크립트**: `src/train/train_mobile_alamp_v2.py`
- **핵심 코드**:
```python
# Saliency 기반 적응형 패치 영역 제안
def prepare_alamp_inputs(image, global_size=384, patch_size=224):
    global_view = resize_with_pad(image, target_size=global_size)
    boxes, _ = propose_adaptive_boxes(image=image, num_patches=5)
    patches = tf.image.crop_and_resize(image[None, ...], boxes, ...)
    return global_view, patches
```

## 4. ICAA (Color Aesthetic)
- **목적**: 색상 특화 미적 품질 평가 (공식 가중치 이식)
- **변환 스크립트**: `experiments/icaa_tf_native/scripts/stage_i_export_savedmodel.py`
- **핵심 코드**:
```python
# PyTorch 가중치를 TF Native 구현체로 로드
model_tf = TFICAA17KDAT(epsilon=1e-5)
model_tf.load_from_pytorch_state_dict(state_dict)

# ImageNet 정규화 적용
mean = np.asarray([0.485, 0.456, 0.406])
std = np.asarray([0.229, 0.224, 0.225])
array = (array / 255.0 - mean) / std
```

## 5. KonIQ / FLIVE (Technical Quality)
- **목적**: 기술적 이미지 품질(노이즈, 블러 등) 회귀 예측
- **학습 스크립트**: `src/train/train_regression.py`
- **핵심 코드**:
```python
# Argparse 기반 태스크 및 데이터셋 선택 (KonIQ/FLIVE 공용)
parser.add_argument("--task", choices=["aadb", "technical"], required=True)
if args.task == "technical":
    model = build_technical_regressor(...)

# MSE 손실 함수 기반 학습
model.compile(optimizer=Adam(1e-4), loss="mse")
model.fit(train_ds, validation_data=val_ds, ...)
```

## 6. 공통 TFLite 변환 (Export)
- **핵심 코드**:
```python
# FP16 정밀도 최적화 및 TFLite 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
```
te_model = converter.convert()
```
