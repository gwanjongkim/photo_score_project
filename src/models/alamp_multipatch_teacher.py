# A-LAMP 멀티패치 teacher 기준 모델을 정의한다.
from __future__ import annotations

from typing import Any

import tensorflow as tf


MODEL_VARIANT = "A-LAMP Multi-Patch teacher baseline"
MODEL_DESCRIPTION = "VGG16 shared-patch Multi-Patch teacher with orderless mean+max aggregation"
REPRODUCTION_CLAIM = "not full A-LAMP reproduction"
PATCH_PROJECTION_MODES = {"gap", "flatten_dense"}


def normalize_backbone_weights(weights: str | None) -> str | None:
    if weights is None:
        return None
    if str(weights).strip().lower() in {"none", "null", "false", "random"}:
        return None
    return str(weights)


def normalize_unfreeze_from_layer(unfreeze_from_layer: str | None) -> str | None:
    if unfreeze_from_layer is None:
        return None
    value = str(unfreeze_from_layer).strip()
    if not value or value.lower() in {"none", "null", "false"}:
        return None
    return value


def normalize_patch_projection_mode(patch_projection_mode: str) -> str:
    mode = str(patch_projection_mode).strip().lower()
    if mode not in PATCH_PROJECTION_MODES:
        raise ValueError(
            f"patch_projection_mode must be one of {sorted(PATCH_PROJECTION_MODES)}, got {patch_projection_mode!r}."
        )
    return mode


def _set_vgg16_backbone_trainability(
    backbone: tf.keras.Model,
    *,
    backbone_trainable: bool,
    unfreeze_from_layer: str | None = None,
) -> None:
    unfreeze_from_layer = normalize_unfreeze_from_layer(unfreeze_from_layer)
    if not backbone_trainable:
        backbone.trainable = False
        for layer in backbone.layers:
            layer.trainable = False
        return

    backbone.trainable = True
    if unfreeze_from_layer is None:
        for layer in backbone.layers:
            layer.trainable = True
        return

    matched_layers = 0
    for layer in backbone.layers:
        layer.trainable = layer.name.startswith(unfreeze_from_layer)
        if layer.trainable:
            matched_layers += 1
    if matched_layers == 0:
        raise ValueError(f"No VGG16 layers matched unfreeze_from_layer={unfreeze_from_layer!r}.")


@tf.keras.utils.register_keras_serializable(package="photo_score_project")
class MergePatchBatch(tf.keras.layers.Layer):
    def __init__(self, patch_count: int, patch_size: int, channels: int = 3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.patch_count = int(patch_count)
        self.patch_size = int(patch_size)
        self.channels = int(channels)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]
        return tf.reshape(
            inputs,
            [batch_size * self.patch_count, self.patch_size, self.patch_size, self.channels],
        )

    def compute_output_shape(self, input_shape: tuple[int | None, ...]) -> tuple[None, int, int, int]:
        return (None, self.patch_size, self.patch_size, self.channels)

    def get_config(self) -> dict[str, object]:
        config = dict(super().get_config())
        config.update(
            {
                "patch_count": self.patch_count,
                "patch_size": self.patch_size,
                "channels": self.channels,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="photo_score_project")
class RestorePatchBatch(tf.keras.layers.Layer):
    def __init__(self, patch_count: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.patch_count = int(patch_count)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0] // self.patch_count
        feature_dim = tf.shape(inputs)[-1]
        return tf.reshape(inputs, [batch_size, self.patch_count, feature_dim])

    def compute_output_shape(self, input_shape: tuple[int | None, ...]) -> tuple[None, int, int | None]:
        return (None, self.patch_count, input_shape[-1])

    def get_config(self) -> dict[str, object]:
        config = dict(super().get_config())
        config.update({"patch_count": self.patch_count})
        return config


@tf.keras.utils.register_keras_serializable(package="photo_score_project")
class OrderlessMeanMaxAggregation(tf.keras.layers.Layer):
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        mean_features = tf.reduce_mean(inputs, axis=1)
        max_features = tf.reduce_max(inputs, axis=1)
        return tf.concat([mean_features, max_features], axis=-1)

    def compute_output_shape(self, input_shape: tuple[int | None, ...]) -> tuple[None, int | None]:
        feature_dim = input_shape[-1]
        return (None, None if feature_dim is None else int(feature_dim) * 2)

    def get_config(self) -> dict[str, object]:
        return dict(super().get_config())


def build_alamp_multipatch_teacher_model(
    *,
    patch_count: int = 5,
    patch_size: int = 224,
    backbone_weights: str | None = "imagenet",
    backbone_trainable: bool = False,
    unfreeze_from_layer: str | None = None,
    patch_projection_mode: str = "gap",
    patch_feature_dim: int = 512,
    head_units: int = 256,
    head_layers: int = 1,
    head_dropout: float | None = None,
    dropout_rate: float = 0.5,
) -> tf.keras.Model:
    backbone_weights = normalize_backbone_weights(backbone_weights)
    unfreeze_from_layer = normalize_unfreeze_from_layer(unfreeze_from_layer)
    patch_projection_mode = normalize_patch_projection_mode(patch_projection_mode)
    patch_feature_dim = int(patch_feature_dim)
    head_units = int(head_units)
    head_layers = int(head_layers)
    head_dropout = float(dropout_rate if head_dropout is None else head_dropout)
    if patch_feature_dim <= 0:
        raise ValueError("patch_feature_dim must be positive.")
    if head_units <= 0:
        raise ValueError("head_units must be positive.")
    if head_layers <= 0:
        raise ValueError("head_layers must be positive.")

    patches = tf.keras.Input(
        shape=(patch_count, patch_size, patch_size, 3),
        name="patches",
    )
    x = MergePatchBatch(
        patch_count=patch_count,
        patch_size=patch_size,
        name="merge_patch_batch",
    )(patches)

    backbone = tf.keras.applications.VGG16(
        include_top=False,
        weights=backbone_weights,
        input_shape=(patch_size, patch_size, 3),
        name="vgg16_backbone",
    )
    _set_vgg16_backbone_trainability(
        backbone,
        backbone_trainable=bool(backbone_trainable),
        unfreeze_from_layer=unfreeze_from_layer,
    )

    x = backbone(x)
    if patch_projection_mode == "gap":
        x = tf.keras.layers.GlobalAveragePooling2D(name="patch_gap")(x)
        effective_patch_feature_dim = 512
        patch_feature_description = "VGG16 include_top=False + GlobalAveragePooling2D"
    else:
        x = tf.keras.layers.Flatten(name="patch_flatten")(x)
        x = tf.keras.layers.Dense(
            patch_feature_dim,
            activation="relu",
            name="patch_projection_dense",
        )(x)
        effective_patch_feature_dim = patch_feature_dim
        patch_feature_description = "VGG16 include_top=False + Flatten + shared Dense projection"
    x = RestorePatchBatch(patch_count=patch_count, name="restore_patch_batch")(x)
    x = OrderlessMeanMaxAggregation(name="orderless_mean_max_aggregation")(x)
    if head_layers == 1:
        x = tf.keras.layers.Dense(head_units, activation="relu", name="teacher_dense")(x)
        x = tf.keras.layers.Dropout(head_dropout, name="teacher_dropout")(x)
    else:
        for index in range(head_layers):
            x = tf.keras.layers.Dense(head_units, activation="relu", name=f"teacher_dense_{index + 1}")(x)
            x = tf.keras.layers.Dropout(head_dropout, name=f"teacher_dropout_{index + 1}")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="probability")(x)

    model = tf.keras.Model(inputs=patches, outputs=output, name="alamp_multipatch_teacher")
    model._alamp_multipatch_teacher_config = {  # type: ignore[attr-defined]
        "model_variant": MODEL_VARIANT,
        "description": MODEL_DESCRIPTION,
        "reproduction_claim": REPRODUCTION_CLAIM,
        "patch_count": int(patch_count),
        "patch_size": int(patch_size),
        "backbone": "VGG16",
        "backbone_weights": backbone_weights,
        "backbone_trainable": bool(backbone_trainable),
        "unfreeze_from_layer": unfreeze_from_layer,
        "patch_projection_mode": patch_projection_mode,
        "patch_feature_dim": int(effective_patch_feature_dim),
        "patch_feature_dim_requested": int(patch_feature_dim),
        "patch_feature": patch_feature_description,
        "aggregation": "orderless mean features concatenated with orderless max features",
        "head_units": int(head_units),
        "head_layers": int(head_layers),
        "head_dropout": float(head_dropout),
        "dropout_rate": float(head_dropout),
    }
    return model


def set_vgg16_backbone_trainable(model: tf.keras.Model, trainable: bool) -> None:
    backbone = model.get_layer("vgg16_backbone")
    _set_vgg16_backbone_trainability(backbone, backbone_trainable=bool(trainable))


def set_vgg16_backbone_trainability(
    model: tf.keras.Model,
    *,
    backbone_trainable: bool,
    unfreeze_from_layer: str | None = None,
) -> None:
    backbone = model.get_layer("vgg16_backbone")
    _set_vgg16_backbone_trainability(
        backbone,
        backbone_trainable=bool(backbone_trainable),
        unfreeze_from_layer=unfreeze_from_layer,
    )


def summarize_vgg16_trainability(model: tf.keras.Model) -> dict[str, Any]:
    backbone = model.get_layer("vgg16_backbone")
    trainable_layers = [layer.name for layer in backbone.layers if layer.trainable]
    frozen_layers = [layer.name for layer in backbone.layers if not layer.trainable]
    return {
        "backbone_name": backbone.name,
        "trainable": bool(backbone.trainable),
        "trainable_layer_count": len(trainable_layers),
        "frozen_layer_count": len(frozen_layers),
        "trainable_layers": trainable_layers,
        "frozen_layers": frozen_layers,
        "model_trainable_weight_count": len(model.trainable_weights),
        "model_non_trainable_weight_count": len(model.non_trainable_weights),
    }


def get_alamp_multipatch_teacher_custom_objects() -> dict[str, object]:
    return {
        "MergePatchBatch": MergePatchBatch,
        "RestorePatchBatch": RestorePatchBatch,
        "OrderlessMeanMaxAggregation": OrderlessMeanMaxAggregation,
    }
