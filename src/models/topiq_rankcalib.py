# TOPIQ-RankCalib v2 실험 모델 구조를 정의한다.
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


@tf.keras.utils.register_keras_serializable(package="TopiqRankCalib")
class SpatialReduceSum(tf.keras.layers.Layer):
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(inputs, axis=1)

    def compute_output_shape(self, input_shape: tuple[int | None, ...]) -> tuple[int | None, ...]:
        return (input_shape[0], input_shape[2])


def _shape_tuple(tensor: tf.Tensor) -> tuple[int | None, ...]:
    return tuple(None if dim is None else int(dim) for dim in tensor.shape[1:])


def _spatial_dims(spec: FeatureSpec) -> tuple[int, int]:
    height, width = spec.shape[0], spec.shape[1]
    if height is None or width is None:
        raise ValueError(
            f"Feature layer {spec.layer_name} has dynamic spatial shape {spec.shape}; "
            "TOPIQ-RankCalib v2 expects fixed feature map sizes."
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
        layer_preview = ", ".join(layer.name for layer in backbone.layers[:30])
        raise ValueError(
            f"EfficientNetV2B0 is missing required TOPIQ-RankCalib layers {missing}. "
            f"First layers in this install: {layer_preview}"
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
        name="topiq_rankcalib_backbone",
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
    return SpatialReduceSum(name=f"{name}_weighted_quality")(weighted_scores)


def _head(
    features: tf.Tensor,
    head_units: int,
    dropout: float,
    name: str,
) -> tf.Tensor:
    x = tf.keras.layers.Dense(
        head_units,
        activation="swish",
        name=f"{name}_dense",
    )(features)
    x = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")(x)
    return tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        dtype="float32",
        name=name,
    )(x)


def build_topiq_rankcalib_v2(
    input_shape: tuple[int, int, int] = (384, 384, 3),
    backbone_trainable: bool = False,
    dropout: float = 0.3,
    head_units: int = 256,
    name: str = "topiq_rankcalib_v2",
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

    fused = tf.keras.layers.Concatenate(name="rankcalib_feature_concat")(
        [
            high_quality,
            mid_quality,
            low_quality,
            high_vector,
            mid_vector,
            low_vector,
        ]
    )
    outputs = {
        "unified": _head(fused, head_units=head_units, dropout=dropout, name="unified"),
        "spaq": _head(fused, head_units=head_units, dropout=dropout, name="spaq"),
        "koniq": _head(fused, head_units=head_units, dropout=dropout, name="koniq"),
        "flive": _head(fused, head_units=head_units, dropout=dropout, name="flive"),
    }

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    model.topiq_feature_specs = specs
    return model


def list_topiq_rankcalib_layers(
    input_shape: tuple[int, int, int] = (384, 384, 3),
) -> list[tuple[int, str, tuple[int | None, ...]]]:
    backbone = _build_efficientnetv2b0(input_shape=input_shape)
    rows: list[tuple[int, str, tuple[int | None, ...]]] = []
    for index, layer in enumerate(backbone.layers):
        shape = _shape_tuple(layer.output)
        rows.append((index, layer.name, shape))
        print(f"{index:03d} {layer.name}: {shape}")

    print("\nTOPIQ-RankCalib selected feature layers:")
    for level, spec in _feature_specs(backbone).items():
        print(f"{level}: {spec.layer_name} {spec.shape}")
    return rows


__all__ = ["build_topiq_rankcalib_v2"]
