# RGNet Core Code Snippets

RGNet 모델은 그래프 신경망(GNN)을 활용하여 이미지의 지역적 관계를 학습하는 미적 품질 회귀 모델입니다.

> **Warning**: 현재 프로덕션(Flutter)에서 사용되는 핵심 자산은 `rgnet_pil_resize_aadb_fp16.tflite`입니다. 이는 논문 재현 목적의 실험(DenseASPP, Cascaded DenseASPP 등)과는 구분되는 **실무용(Practical) 모델**입니다. 본 스니펫은 실제 배포된 RGNet 모델의 학습 및 내보내기 계보를 설명합니다.

### 1. 모델 구조 (Architecture)
`src/models/rgnet_paper_v1.py`에서 정의된 RGNet의 핵심은 ASPP와 그래프 컨볼루션 레이어입니다.

```python
# src/models/rgnet_paper_v1.py
def build_rgnet_paper_v1_model(...):
    # Backbone: DenseNet121
    backbone = _build_densenet121_backbone(input_shape=input_shape, weights=backbone_weights)
    
    # Context Module: ASPP
    context_map = ASPPContextModule(filters=int(region_dim), dilation_rates=dilation_rates)(feature_map)
    
    # Adjacency: Region Similarity
    adjacency = V1RegionSimilarityAdjacency(temperature=graph_temperature)(node_features)
    
    # Graph Convolution Blocks
    graph_context = node_features
    for block_index in range(int(graph_blocks)):
        graph_context = V1ResidualGraphConvolution(units=int(graph_units))((graph_context, adjacency))
    
    # Region Scores & Aggregation
    region_scores = tf.keras.layers.Dense(1, activation="sigmoid")(region_context)
    outputs = RegionScoreAggregation(aggregation=aggregation, lse_r=lse_r)(region_scores)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
```

### 2. 전처리 (Preprocessing)
RGNet은 PIL Bilinear 리사이즈 결과와의 일치성을 위해 학습 시에도 동일한 백엔드를 사용합니다.

```python
# src/train/train_rgnet_paper_v1_aadb.py
def _decode_image_pil_bilinear(contents: tf.Tensor, image_size: int) -> np.ndarray:
    from PIL import Image
    with Image.open(BytesIO(contents.numpy())) as image:
        # PIL Bilinear 리사이즈 강제
        image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0
```

### 3. 학습 설정 (Training)
회귀 문제를 위해 MSE 손실 함수를 사용합니다.

```python
# src/train/train_rgnet_paper_v1_aadb.py
model.compile(
    optimizer=optimizer,
    loss="mse",
    metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
)
```

### 4. TFLite Export
`scripts/export_rgnet_paper_aadb_tflite.py`를 통해 FP16 TFLite 모델을 생성합니다.

```python
# scripts/export_rgnet_paper_aadb_tflite.py
def convert_model(model: tf.keras.Model, output_path: Path, precision: str) -> dict[str, object]:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if precision == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_bytes = converter.convert()
    output_path.write_bytes(tflite_bytes)
```
