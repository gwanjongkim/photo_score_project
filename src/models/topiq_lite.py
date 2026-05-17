# TOPIQ-lite v1 모델의 교차 스케일 주의 구조를 정의한다.
from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


SUPPORTED_BACKBONES = {"efficientnetv2b0"}
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


def _normalise_weights(weights: str | None) -> str | None:
    if weights is None:
        return None
    text = str(weights).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return text


def _shape_tuple(tensor: tf.Tensor) -> tuple[int | None, ...]:
    return tuple(None if dim is None else int(dim) for dim in tensor.shape[1:])


def _require_spatial_shape(spec: FeatureSpec) -> tuple[int, int]:
    height, width = spec.shape[0], spec.shape[1]
    if height is None or width is None:
        raise ValueError(
            f"Feature layer {spec.layer_name} has dynamic spatial shape {spec.shape}; "
            "TOPIQ-lite v1 expects fixed feature map sizes for TFLite-friendly resizing."
        )
    return int(height), int(width)


def _build_efficientnetv2b0(
    input_shape: tuple[int, int, int],
    weights: str | None,
) -> tf.keras.Model:
    return tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights=_normalise_weights(weights),
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
            f"EfficientNetV2B0 is missing required TOPIQ-lite feature layers {missing}. "
            f"First layers in this install: {layer_preview}"
        )
    return specs


def _feature_extractor(
    input_shape: tuple[int, int, int],
    weights: str | None,
    freeze_backbone: bool,
) -> tuple[tf.keras.Model, dict[str, FeatureSpec]]:
    backbone = _build_efficientnetv2b0(input_shape=input_shape, weights=weights)
    specs = _feature_specs(backbone)
    extractor = tf.keras.Model(
        inputs=backbone.input,
        outputs=[
            backbone.get_layer(specs["low"].layer_name).output,
            backbone.get_layer(specs["mid"].layer_name).output,
            backbone.get_layer(specs["high"].layer_name).output,
        ],
        name="topiq_backbone",
    )
    extractor.trainable = not freeze_backbone
    return extractor, specs


def build_topiq_lite(
    input_shape: tuple[int, int, int] = (384, 384, 3),
    backbone_name: str = "efficientnetv2b0",
    weights: str | None = "imagenet",
    dropout_rate: float = 0.3,
    dense_units: int = 256,
    freeze_backbone: bool = False,
) -> tf.keras.Model:
    backbone_key = backbone_name.lower()
    if backbone_key not in SUPPORTED_BACKBONES:
        raise ValueError(
            "TOPIQ-lite v1 currently supports only backbone_name='efficientnetv2b0'."
        )

    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32, name="topiq_input")
    backbone, specs = _feature_extractor(
        input_shape=input_shape,
        weights=weights,
        freeze_backbone=freeze_backbone,
    )
    low_feature, mid_feature, high_feature = backbone(inputs)

    mid_height, mid_width = _require_spatial_shape(specs["mid"])
    low_height, low_width = _require_spatial_shape(specs["low"])

    high_attention = tf.keras.layers.Conv2D(
        1,
        kernel_size=1,
        padding="same",
        activation="sigmoid",
        name="high_to_mid_attention",
    )(high_feature)
    high_attention_resized = tf.keras.layers.Resizing(
        mid_height,
        mid_width,
        interpolation="bilinear",
        name="high_to_mid_attention_resize",
    )(high_attention)
    mid_guided = tf.keras.layers.Multiply(name="mid_guided_by_high")(
        [mid_feature, high_attention_resized]
    )

    mid_attention = tf.keras.layers.Conv2D(
        1,
        kernel_size=1,
        padding="same",
        activation="sigmoid",
        name="mid_to_low_attention",
    )(mid_guided)
    mid_attention_resized = tf.keras.layers.Resizing(
        low_height,
        low_width,
        interpolation="bilinear",
        name="mid_to_low_attention_resize",
    )(mid_attention)
    low_guided = tf.keras.layers.Multiply(name="low_guided_by_mid")(
        [low_feature, mid_attention_resized]
    )

    high_vector = tf.keras.layers.GlobalAveragePooling2D(name="high_gap")(high_feature)
    mid_vector = tf.keras.layers.GlobalAveragePooling2D(name="mid_guided_gap")(mid_guided)
    low_vector = tf.keras.layers.GlobalAveragePooling2D(name="low_guided_gap")(low_guided)
    fused = tf.keras.layers.Concatenate(name="topiq_feature_concat")(
        [high_vector, mid_vector, low_vector]
    )
    fused = tf.keras.layers.Dense(
        dense_units,
        activation="swish",
        name="quality_head",
    )(fused)
    fused = tf.keras.layers.Dropout(dropout_rate, name="quality_dropout")(fused)
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        dtype="float32",
        name="normalized_mos",
    )(fused)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="topiq_lite_v1")
    model.topiq_feature_specs = specs
    return model


def list_efficientnetv2b0_layers(
    input_shape: tuple[int, int, int] = (384, 384, 3),
) -> list[tuple[int, str, tuple[int | None, ...]]]:
    backbone = _build_efficientnetv2b0(input_shape=input_shape, weights=None)
    rows: list[tuple[int, str, tuple[int | None, ...]]] = []
    for index, layer in enumerate(backbone.layers):
        shape = _shape_tuple(layer.output)
        rows.append((index, layer.name, shape))
        print(f"{index:03d} {layer.name}: {shape}")

    print("\nTOPIQ-lite selected feature layers:")
    for level, spec in _feature_specs(backbone).items():
        print(f"{level}: {spec.layer_name} {spec.shape}")
    return rows


__all__ = ["build_topiq_lite", "list_efficientnetv2b0_layers"]
