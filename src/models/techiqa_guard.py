# TechIQA-Guard v1 단일 출력 기술 품질 모델 구조를 정의한다.
from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


DEFAULT_FEATURE_LAYERS = {
    "low": "block3b_add",
    "mid": "block5c_add",
    "high": "top_activation",
}


@dataclass(frozen=True)
class FeatureSpec:
    level: str
    layer_name: str
    shape: tuple[int | None, ...]


def _shape_tuple(tensor: tf.Tensor) -> tuple[int | None, ...]:
    return tuple(None if dim is None else int(dim) for dim in tensor.shape[1:])


def _spatial_dims(spec: FeatureSpec) -> tuple[int, int]:
    height, width = spec.shape[0], spec.shape[1]
    if height is None or width is None:
        raise ValueError(
            f"Feature layer {spec.layer_name} has dynamic spatial shape {spec.shape}; "
            "TechIQA-Guard v1 expects fixed feature map sizes for TFLite-friendly resizing."
        )
    return int(height), int(width)


def _build_efficientnetv2b0(input_shape: tuple[int, int, int]) -> tf.keras.Model:
    return tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        include_preprocessing=True,
        input_shape=input_shape,
    )


def _feature_specs(backbone: tf.keras.Model) -> dict[str, FeatureSpec]:
    specs: dict[str, FeatureSpec] = {}
    missing: list[str] = []
    for level, layer_name in DEFAULT_FEATURE_LAYERS.items():
        try:
            layer = backbone.get_layer(layer_name)
        except ValueError:
            missing.append(layer_name)
            continue
        specs[level] = FeatureSpec(
            level=level,
            layer_name=layer_name,
            shape=_shape_tuple(layer.output),
        )

    if missing:
        preview = ", ".join(layer.name for layer in backbone.layers[:40])
        add_layers = ", ".join(layer.name for layer in backbone.layers if layer.name.endswith("_add"))
        raise ValueError(
            f"EfficientNetV2B0 is missing required TechIQA-Guard feature layers {missing}. "
            f"Expected layer map: {DEFAULT_FEATURE_LAYERS}. "
            f"First layers in this install: {preview}. "
            f"Available add layers: {add_layers}"
        )
    return specs


def _feature_extractor(
    input_shape: tuple[int, int, int],
    backbone_trainable: bool,
) -> tuple[tf.keras.Model, dict[str, FeatureSpec]]:
    backbone = _build_efficientnetv2b0(input_shape=input_shape)
    specs = _feature_specs(backbone)
    extractor = tf.keras.Model(
        inputs=backbone.input,
        outputs=[
            backbone.get_layer(specs["low"].layer_name).output,
            backbone.get_layer(specs["mid"].layer_name).output,
            backbone.get_layer(specs["high"].layer_name).output,
        ],
        name="techiqa_guard_backbone",
    )
    extractor.trainable = bool(backbone_trainable)
    return extractor, specs


def _guided_attention(
    source: tf.Tensor,
    target: tf.Tensor,
    target_spec: FeatureSpec,
    name: str,
) -> tf.Tensor:
    target_height, target_width = _spatial_dims(target_spec)
    attention = tf.keras.layers.Conv2D(
        1,
        kernel_size=1,
        padding="same",
        activation=None,
        name=f"{name}_attention_logits",
    )(source)
    attention = tf.keras.layers.Activation(
        "sigmoid",
        name=f"{name}_attention_sigmoid",
    )(attention)
    attention = tf.keras.layers.Resizing(
        target_height,
        target_width,
        interpolation="bilinear",
        name=f"{name}_attention_resize",
    )(attention)
    return tf.keras.layers.Multiply(name=f"{name}_guided")([target, attention])


def _patch_weighted_quality_pool(feature: tf.Tensor, name: str) -> tf.Tensor:
    height = feature.shape[1]
    width = feature.shape[2]
    if height is None or width is None:
        raise ValueError(f"{name} needs fixed spatial dimensions for weighted pooling.")
    position_count = int(height) * int(width)

    score_map = tf.keras.layers.Conv2D(
        1,
        kernel_size=1,
        padding="same",
        activation=None,
        name=f"{name}_score_map",
    )(feature)
    weight_logits = tf.keras.layers.Conv2D(
        1,
        kernel_size=1,
        padding="same",
        activation=None,
        name=f"{name}_weight_logits",
    )(feature)
    scores = tf.keras.layers.Reshape(
        (position_count, 1),
        name=f"{name}_score_flat",
    )(score_map)
    weights = tf.keras.layers.Reshape(
        (position_count, 1),
        name=f"{name}_weight_flat",
    )(weight_logits)
    weights = tf.keras.layers.Softmax(axis=1, name=f"{name}_spatial_softmax")(weights)
    weighted_scores = tf.keras.layers.Multiply(name=f"{name}_weighted_scores")(
        [scores, weights]
    )
    return tf.keras.layers.Lambda(
        lambda x: tf.reduce_sum(x, axis=1),
        output_shape=(1,),
        name=f"{name}_weighted_quality",
    )(weighted_scores)


def _defect_branch(
    guided_low: tf.Tensor,
    guided_mid: tf.Tensor,
    head_units: int,
) -> tf.Tensor:
    low_defect = tf.keras.layers.Conv2D(
        32,
        kernel_size=3,
        padding="same",
        activation="swish",
        name="low_defect_conv",
    )(guided_low)
    low_defect = tf.keras.layers.Conv2D(
        32,
        kernel_size=1,
        padding="same",
        activation="swish",
        name="low_defect_pointwise",
    )(low_defect)
    low_defect = tf.keras.layers.GlobalAveragePooling2D(name="low_defect_gap")(low_defect)

    mid_defect = tf.keras.layers.Conv2D(
        48,
        kernel_size=3,
        padding="same",
        activation="swish",
        name="mid_defect_conv",
    )(guided_mid)
    mid_defect = tf.keras.layers.Conv2D(
        48,
        kernel_size=1,
        padding="same",
        activation="swish",
        name="mid_defect_pointwise",
    )(mid_defect)
    mid_defect = tf.keras.layers.GlobalAveragePooling2D(name="mid_defect_gap")(mid_defect)

    defect = tf.keras.layers.Concatenate(name="defect_feature_concat")(
        [low_defect, mid_defect]
    )
    return tf.keras.layers.Dense(
        max(head_units // 2, 32),
        activation="swish",
        name="defect_dense",
    )(defect)


def build_techiqa_guard_v1(
    input_shape: tuple[int, int, int] = (384, 384, 3),
    backbone_trainable: bool = False,
    dropout: float = 0.25,
    head_units: int = 256,
    name: str = "techiqa_guard_v1",
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32, name="image")
    backbone, specs = _feature_extractor(
        input_shape=input_shape,
        backbone_trainable=backbone_trainable,
    )
    low_feature, mid_feature, high_feature = backbone(inputs)

    guided_mid = _guided_attention(
        source=high_feature,
        target=mid_feature,
        target_spec=specs["mid"],
        name="high_to_mid",
    )
    guided_low = _guided_attention(
        source=guided_mid,
        target=low_feature,
        target_spec=specs["low"],
        name="mid_to_low",
    )

    high_quality = _patch_weighted_quality_pool(high_feature, name="high")
    mid_quality = _patch_weighted_quality_pool(guided_mid, name="mid_guided")
    low_quality = _patch_weighted_quality_pool(guided_low, name="low_guided")

    high_vector = tf.keras.layers.GlobalAveragePooling2D(name="high_gap")(high_feature)
    mid_vector = tf.keras.layers.GlobalAveragePooling2D(name="mid_guided_gap")(guided_mid)
    low_vector = tf.keras.layers.GlobalAveragePooling2D(name="low_guided_gap")(guided_low)
    defect_vector = _defect_branch(
        guided_low=guided_low,
        guided_mid=guided_mid,
        head_units=head_units,
    )

    fused = tf.keras.layers.Concatenate(name="techiqa_feature_concat")(
        [
            high_quality,
            mid_quality,
            low_quality,
            high_vector,
            mid_vector,
            low_vector,
            defect_vector,
        ]
    )
    fused = tf.keras.layers.Dense(
        head_units,
        activation="swish",
        name="technical_head_dense",
    )(fused)
    fused = tf.keras.layers.Dropout(dropout, name="technical_head_dropout")(fused)
    fused = tf.keras.layers.Dense(
        max(head_units // 2, 1),
        activation="swish",
        name="technical_head_dense_half",
    )(fused)
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        dtype="float32",
        name="technical_score",
    )(fused)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    model.techiqa_feature_specs = specs
    return model


__all__ = ["build_techiqa_guard_v1"]
