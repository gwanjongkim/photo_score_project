# DistortionGuard-IQA v1 모델과 Stage A 표현 전이 유틸리티를 정의한다.
from __future__ import annotations

from pathlib import Path
from typing import Any

import tensorflow as tf


IMAGE_SIZE = 384
NUM_DISTORTION_TYPES = 10
BACKBONE_LAYER_NAME = "efficientnetv2-b0"


def _shape_list(weights: list[Any]) -> list[list[int]]:
    return [list(weight.shape) for weight in weights]


def _build_efficientnetv2b0(
    input_shape: tuple[int, int, int],
    weights: str | None,
) -> tf.keras.Model:
    return tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights=weights,
        include_preprocessing=True,
        input_shape=input_shape,
    )


def _shared_stagea_representation(
    inputs: tf.Tensor,
    input_shape: tuple[int, int, int],
    weights: str | None,
    backbone_trainable: bool,
    freeze_batch_norm: bool,
    dropout: float,
    head_units: int,
) -> tuple[tf.Tensor, tf.keras.Model]:
    backbone = _build_efficientnetv2b0(input_shape=input_shape, weights=weights)
    backbone.trainable = bool(backbone_trainable)
    if freeze_batch_norm:
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    features = backbone(inputs)
    gap = tf.keras.layers.GlobalAveragePooling2D(name="stagea_gap")(features)
    gmp = tf.keras.layers.GlobalMaxPooling2D(name="stagea_gmp")(features)
    pooled = tf.keras.layers.Concatenate(name="stagea_pool_concat")([gap, gmp])
    x = tf.keras.layers.Dense(
        head_units,
        activation="swish",
        name="stagea_dense",
    )(pooled)
    x = tf.keras.layers.Dropout(dropout, name="stagea_dropout")(x)
    embedding = tf.keras.layers.Dense(
        max(head_units // 2, 64),
        activation="swish",
        name="stagea_embedding",
    )(x)
    return embedding, backbone


def _build_stagea_transfer_model(
    input_shape: tuple[int, int, int],
    dropout: float,
    head_units: int,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32, name="image")
    embedding, _ = _shared_stagea_representation(
        inputs=inputs,
        input_shape=input_shape,
        weights=None,
        backbone_trainable=True,
        freeze_batch_norm=False,
        dropout=dropout,
        head_units=head_units,
    )
    distortion_type = tf.keras.layers.Dense(
        NUM_DISTORTION_TYPES,
        activation="softmax",
        dtype="float32",
        name="distortion_type",
    )(embedding)
    severity = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        dtype="float32",
        name="severity",
    )(embedding)
    quality_proxy = tf.keras.layers.Dense(
        1,
        activation=None,
        dtype="float32",
        name="quality_proxy",
    )(embedding)
    return tf.keras.Model(
        inputs=inputs,
        outputs=[distortion_type, severity, quality_proxy],
        name="distortionguard_stageA_synthetic",
    )


def load_stageA_weights_into_iqa_model(
    model: tf.keras.Model,
    stageA_weights: str | Path,
    input_shape: tuple[int, int, int],
    dropout: float,
    head_units: int,
) -> dict[str, Any]:
    weights_path = Path(stageA_weights).expanduser()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Stage A weights file not found: {weights_path}")

    stagea_model = _build_stagea_transfer_model(
        input_shape=input_shape,
        dropout=dropout,
        head_units=head_units,
    )
    stagea_model.load_weights(str(weights_path))

    stagea_layers = {layer.name: layer for layer in stagea_model.layers}
    target_layers = {layer.name: layer for layer in model.layers}
    loaded_layers: list[dict[str, Any]] = []
    skipped_layers: list[dict[str, Any]] = []
    mismatched_layers: list[dict[str, Any]] = []

    for layer in model.layers:
        target_weights = layer.get_weights()
        if not target_weights:
            continue
        source_layer = stagea_layers.get(layer.name)
        if source_layer is None:
            skipped_layers.append(
                {
                    "layer": layer.name,
                    "reason": "target_layer_not_in_stageA",
                    "target_weight_shapes": _shape_list(target_weights),
                }
            )
            continue
        source_weights = source_layer.get_weights()
        source_shapes = _shape_list(source_weights)
        target_shapes = _shape_list(target_weights)
        if source_shapes != target_shapes:
            mismatched_layers.append(
                {
                    "layer": layer.name,
                    "reason": "shape_mismatch",
                    "source_weight_shapes": source_shapes,
                    "target_weight_shapes": target_shapes,
                }
            )
            continue
        layer.set_weights(source_weights)
        loaded_layers.append(
            {
                "layer": layer.name,
                "weight_shapes": target_shapes,
                "weight_count": len(target_weights),
            }
        )

    skipped_stageA_layers = [
        {
            "layer": layer.name,
            "reason": "stageA_aux_or_absent_from_stageB",
            "source_weight_shapes": _shape_list(layer.get_weights()),
        }
        for layer in stagea_model.layers
        if layer.get_weights() and layer.name not in target_layers
    ]
    report = {
        "stageA_weights": str(weights_path),
        "loaded_layer_count": int(len(loaded_layers)),
        "skipped_layer_count": int(len(skipped_layers) + len(skipped_stageA_layers)),
        "mismatched_layer_count": int(len(mismatched_layers)),
        "loaded_layers": loaded_layers,
        "skipped_layers": skipped_layers,
        "skipped_stageA_layers": skipped_stageA_layers,
        "mismatched_layers": mismatched_layers,
    }
    return report


def build_distortionguard_iqa_v1(
    input_shape: tuple[int, int, int] = (IMAGE_SIZE, IMAGE_SIZE, 3),
    stageA_weights: str | Path | None = None,
    backbone_trainable: bool = False,
    freeze_batch_norm: bool = True,
    dropout: float = 0.25,
    head_units: int = 256,
    name: str = "distortionguard_iqa_v1",
) -> tf.keras.Model:
    weights = None if stageA_weights else "imagenet"
    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32, name="image")
    embedding, backbone = _shared_stagea_representation(
        inputs=inputs,
        input_shape=input_shape,
        weights=weights,
        backbone_trainable=backbone_trainable,
        freeze_batch_norm=freeze_batch_norm,
        dropout=dropout,
        head_units=head_units,
    )
    x = tf.keras.layers.Dense(
        head_units,
        activation="swish",
        name="technical_head_dense",
    )(embedding)
    x = tf.keras.layers.Dropout(dropout, name="technical_head_dropout")(x)
    x = tf.keras.layers.Dense(
        max(head_units // 2, 1),
        activation="swish",
        name="technical_head_dense_half",
    )(x)
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        dtype="float32",
        name="technical_score",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    model.distortionguard_backbone = BACKBONE_LAYER_NAME
    model.distortionguard_stage = "stageB_authentic_iqa"
    if stageA_weights:
        report = load_stageA_weights_into_iqa_model(
            model=model,
            stageA_weights=stageA_weights,
            input_shape=input_shape,
            dropout=dropout,
            head_units=head_units,
        )
    else:
        report = {
            "stageA_weights": None,
            "loaded_layer_count": 0,
            "skipped_layer_count": 0,
            "mismatched_layer_count": 0,
            "loaded_layers": [],
            "skipped_layers": [],
            "skipped_stageA_layers": [],
            "mismatched_layers": [],
        }
    model.stageA_weight_load_report = report
    if freeze_batch_norm:
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
    return model


def efficientnet_block_name(layer_name: str) -> str | None:
    parts = layer_name.split("_", 1)
    prefix = parts[0]
    suffix = prefix[5:]
    if (
        prefix.startswith("block")
        and len(suffix) >= 2
        and suffix[:-1].isdigit()
        and suffix[-1].isalpha()
    ):
        return prefix
    return None


def selected_top_blocks(backbone: tf.keras.Model, count: int) -> list[str]:
    if count <= 0:
        return []
    blocks: list[str] = []
    for layer in backbone.layers:
        block = efficientnet_block_name(layer.name)
        if block is not None and block not in blocks:
            blocks.append(block)
    if not blocks:
        raise ValueError("No EfficientNet block layers were found in DistortionGuard.")
    if count > len(blocks):
        raise ValueError(
            f"--unfreeze_top_blocks={count} exceeds available EfficientNet blocks ({len(blocks)})."
        )
    return blocks[-count:]


def layer_param_count(layer: tf.keras.layers.Layer) -> int:
    return int(sum(tf.keras.backend.count_params(weight) for weight in layer.weights))


def configure_distortionguard_trainability(
    model: tf.keras.Model,
    freeze_backbone: bool,
    unfreeze_top_blocks: int,
    freeze_batch_norm: bool,
) -> dict[str, Any]:
    backbone = model.get_layer(BACKBONE_LAYER_NAME)
    selected_blocks = selected_top_blocks(backbone, unfreeze_top_blocks)
    selected_block_set = set(selected_blocks)
    partial_unfreeze = unfreeze_top_blocks > 0

    for layer in model.layers:
        if layer is not backbone:
            layer.trainable = True

    trainable_backbone_layers: list[str] = []
    frozen_batch_norm_layers: list[str] = []
    if partial_unfreeze:
        backbone.trainable = True
        for layer in backbone.layers:
            block = efficientnet_block_name(layer.name)
            is_top_projection = layer.name in {"top_conv", "top_activation"}
            should_train = (block in selected_block_set) or is_top_projection
            if freeze_batch_norm and isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
                frozen_batch_norm_layers.append(layer.name)
            else:
                layer.trainable = bool(should_train)
            if layer.trainable:
                trainable_backbone_layers.append(layer.name)
    else:
        backbone.trainable = not freeze_backbone
        for layer in backbone.layers:
            if freeze_batch_norm and isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
                frozen_batch_norm_layers.append(layer.name)
            if layer.trainable:
                trainable_backbone_layers.append(layer.name)

    trainable_top_level_layers = [
        layer.name for layer in model.layers if layer.trainable and layer is not backbone
    ]
    all_trainable_names = [
        *[f"{BACKBONE_LAYER_NAME}/{name}" for name in trainable_backbone_layers],
        *trainable_top_level_layers,
    ]
    return {
        "freeze_backbone": bool(freeze_backbone),
        "partial_unfreeze": bool(partial_unfreeze),
        "unfreeze_top_blocks": int(unfreeze_top_blocks),
        "selected_top_blocks": selected_blocks,
        "freeze_batch_norm": bool(freeze_batch_norm),
        "batch_norm_frozen_count": int(len(frozen_batch_norm_layers)),
        "batch_norm_trainable_count": int(
            sum(
                1
                for layer in backbone.layers
                if isinstance(layer, tf.keras.layers.BatchNormalization) and layer.trainable
            )
        ),
        "trainable_backbone_layers": trainable_backbone_layers,
        "trainable_top_level_layers": trainable_top_level_layers,
        "trainable_layers": all_trainable_names,
        "trainable_backbone_param_estimate": int(
            sum(layer_param_count(layer) for layer in backbone.layers if layer.trainable)
        ),
        "trainable_params": int(
            sum(tf.keras.backend.count_params(weight) for weight in model.trainable_weights)
        ),
        "non_trainable_params": int(
            sum(tf.keras.backend.count_params(weight) for weight in model.non_trainable_weights)
        ),
    }


__all__ = [
    "BACKBONE_LAYER_NAME",
    "build_distortionguard_iqa_v1",
    "configure_distortionguard_trainability",
    "load_stageA_weights_into_iqa_model",
]
