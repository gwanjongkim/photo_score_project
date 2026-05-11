# A-LAMP 논문 지향 AVA 이진 분류 모델을 정의한다.
"""A-LAMP-paper-oriented approximation for AVA binary classification.

This module is isolated from the practical app-oriented A-LAMP implementation.
It defines an A-LAMP-style AVA classification model for the A-LAMP-paper-AVA-v0
track. It is not an official reproduction.
"""

from __future__ import annotations

import sys
from typing import Any

import tensorflow as tf

MODEL_VARIANT = "A-LAMP-paper-AVA-v0"
MODEL_DESCRIPTION = "A-LAMP-paper-oriented approximation"
STYLE_DESCRIPTION = "A-LAMP-style AVA classification model"
SUPPORTED_VARIANTS = {"v0_a", "v0_b"}
DEFAULT_LAYOUT_FEATURE_DIM = 10
_UNSET = object()


def _normalize_backbone_weights(weights: str | None) -> str | None:
    if weights is None:
        return None
    if str(weights).lower() in {"none", "null", "false", "random"}:
        return None
    return str(weights)


def _model_cfg(config: dict[str, Any] | None, key: str, default: Any) -> Any:
    if not config:
        return default
    model_section = config.get("model", {})
    if isinstance(model_section, dict) and key in model_section:
        return model_section[key]
    return config.get(key, default)


def _bool_cfg(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _build_vgg16_backbone(
    input_shape: tuple[int, int, int],
    weights: str | None = "imagenet",
    trainable: bool = False,
    name: str = "vgg16",
) -> tf.keras.Model:
    weights = _normalize_backbone_weights(weights)
    try:
        backbone = tf.keras.applications.VGG16(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            name=name,
        )
    except Exception as exc:
        if weights is not None:
            print(
                f"VGG16 {weights} weights unavailable, falling back to random init: {exc}",
                file=sys.stderr,
            )
            backbone = tf.keras.applications.VGG16(
                include_top=False,
                weights=None,
                input_shape=input_shape,
                name=name,
            )
        else:
            raise
    backbone.trainable = trainable
    return backbone


@tf.keras.utils.register_keras_serializable(package="photo_score_project")
class Vgg16UnitPreprocess(tf.keras.layers.Layer):
    """Converts [0, 1] RGB tensors to VGG16 ImageNet preprocessing."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = tf.cast(inputs, tf.float32)
        return tf.keras.applications.vgg16.preprocess_input(inputs * 255.0)

    def get_config(self) -> dict[str, object]:
        return dict(super().get_config())


@tf.keras.utils.register_keras_serializable(package="photo_score_project")
class WeightedPatchPooling(tf.keras.layers.Layer):
    """Pools patch features with learned per-patch weights."""

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        patch_features, patch_attention = inputs
        return tf.reduce_sum(patch_features * patch_attention, axis=1)

    def get_config(self) -> dict[str, object]:
        return dict(super().get_config())


@tf.keras.utils.register_keras_serializable(package="photo_score_project")
class MergePatchBatch(tf.keras.layers.Layer):
    """Merges fixed patch and batch dimensions before shared patch encoding."""

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
    """Restores fixed patch sequences after shared patch encoding."""

    def __init__(self, patch_count: int, feature_dim: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.patch_count = int(patch_count)
        self.feature_dim = int(feature_dim)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        flat_count = tf.shape(inputs)[0]
        batch_size = flat_count // self.patch_count
        return tf.reshape(inputs, [batch_size, self.patch_count, self.feature_dim])

    def compute_output_shape(self, input_shape: tuple[int | None, ...]) -> tuple[None, int, int]:
        return (None, self.patch_count, self.feature_dim)

    def get_config(self) -> dict[str, object]:
        config = dict(super().get_config())
        config.update(
            {
                "patch_count": self.patch_count,
                "feature_dim": self.feature_dim,
            }
        )
        return config


def _build_shared_patch_encoder(
    patch_size: int,
    backbone_weights: str | None,
    train_backbone: bool,
    feature_dim: int,
    dropout: float,
) -> tf.keras.Model:
    patch_input = tf.keras.Input(shape=(patch_size, patch_size, 3), name="patch_image")
    x = Vgg16UnitPreprocess(name="vgg16_preprocess")(patch_input)
    backbone = _build_vgg16_backbone(
        input_shape=(patch_size, patch_size, 3),
        weights=backbone_weights,
        trainable=train_backbone,
        name="shared_patch_vgg16_backbone",
    )
    x = backbone(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="patch_gap")(x)
    x = tf.keras.layers.Dense(feature_dim, activation="relu", name="patch_feature_dense")(x)
    x = tf.keras.layers.Dropout(dropout, name="patch_feature_dropout")(x)
    return tf.keras.Model(patch_input, x, name="shared_vgg16_patch_encoder")


def _build_fixed_patch_sequence_features(
    patch_input: tf.Tensor,
    patch_count: int,
    patch_size: int,
    backbone_weights: str | None,
    train_backbone: bool,
    feature_dim: int,
    dropout: float,
) -> tf.Tensor:
    x = MergePatchBatch(
        patch_count=patch_count,
        patch_size=patch_size,
        name="v0_b_patch_batch_merge",
    )(patch_input)
    x = Vgg16UnitPreprocess(name="v0_b_patch_vgg16_preprocess")(x)
    backbone = _build_vgg16_backbone(
        input_shape=(patch_size, patch_size, 3),
        weights=backbone_weights,
        trainable=train_backbone,
        name="v0_b_patch_vgg16_backbone",
    )
    x = backbone(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="v0_b_patch_gap")(x)
    x = tf.keras.layers.Dense(feature_dim, activation="relu", name="v0_b_patch_feature_dense")(x)
    x = tf.keras.layers.Dropout(dropout, name="v0_b_patch_feature_dropout")(x)
    return RestorePatchBatch(
        patch_count=patch_count,
        feature_dim=feature_dim,
        name="v0_b_patch_batch_restore",
    )(x)


def _build_global_vgg16_branch(
    global_input: tf.Tensor,
    global_size: int,
    backbone_weights: str | None,
    train_backbone: bool,
    feature_dim: int,
    dropout: float,
) -> tf.Tensor:
    x = Vgg16UnitPreprocess(name="v0_b_global_vgg16_preprocess")(global_input)
    backbone = _build_vgg16_backbone(
        input_shape=(global_size, global_size, 3),
        weights=backbone_weights,
        trainable=train_backbone,
        name="v0_b_global_vgg16_backbone",
    )
    x = backbone(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="v0_b_global_gap")(x)
    x = tf.keras.layers.Dropout(dropout, name="v0_b_global_dropout")(x)
    return tf.keras.layers.Dense(feature_dim, activation="relu", name="v0_b_global_dense")(x)


def build_alamp_paper_ava_model(
    config: dict[str, Any] | None = None,
    *,
    variant: str | None = None,
    patch_count: int | None = None,
    image_size_patch: int | None = None,
    image_size_global: int | None = None,
    backbone: str | None = None,
    backbone_weights: str | None | object = _UNSET,
    shared_patch_backbone: bool | None = None,
    include_global_branch: bool | None = None,
    include_layout_features: bool | None = None,
    layout_feature_dim: int | None = None,
    train_backbone: bool | None = None,
    feature_dim: int | None = None,
    dropout: float | None = None,
) -> tf.keras.Model:
    """Builds the isolated A-LAMP-paper-AVA-v0 approximation model."""

    variant = str(variant or _model_cfg(config, "variant", "v0_a")).lower()
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported A-LAMP-paper-AVA-v0 variant: {variant}")

    patch_count = int(patch_count or _model_cfg(config, "patch_count", 5))
    image_size_patch = int(image_size_patch or _model_cfg(config, "image_size_patch", _model_cfg(config, "patch_size", 224)))
    image_size_global = int(image_size_global or _model_cfg(config, "image_size_global", _model_cfg(config, "global_size", 384)))
    backbone = str(backbone or _model_cfg(config, "backbone", "VGG16"))
    if backbone.lower() != "vgg16":
        raise ValueError(f"Unsupported backbone for A-LAMP-paper-AVA-v0: {backbone}")

    backbone_weights_value = _model_cfg(config, "backbone_weights", "imagenet") if backbone_weights is _UNSET else backbone_weights
    backbone_weights = _normalize_backbone_weights(backbone_weights_value if isinstance(backbone_weights_value, str) else None)
    shared_patch_backbone = _bool_cfg(
        shared_patch_backbone
        if shared_patch_backbone is not None
        else _model_cfg(config, "shared_patch_backbone", True)
    )
    if not shared_patch_backbone:
        raise ValueError("A-LAMP-paper-AVA-v0 requires shared_patch_backbone=true")

    include_global_branch = _bool_cfg(
        include_global_branch
        if include_global_branch is not None
        else _model_cfg(config, "include_global_branch", variant == "v0_b")
    )
    include_layout_features = _bool_cfg(
        include_layout_features
        if include_layout_features is not None
        else _model_cfg(config, "include_layout_features", variant == "v0_b")
    )
    if variant == "v0_a":
        include_global_branch = False
        include_layout_features = False
    if variant == "v0_b":
        include_global_branch = True
        include_layout_features = True

    layout_feature_dim = int(layout_feature_dim or _model_cfg(config, "layout_feature_dim", DEFAULT_LAYOUT_FEATURE_DIM))
    train_backbone = _bool_cfg(train_backbone if train_backbone is not None else _model_cfg(config, "train_backbone", False))
    feature_dim = int(feature_dim or _model_cfg(config, "feature_dim", 256))
    dropout = float(dropout if dropout is not None else _model_cfg(config, "dropout", 0.3))

    patch_input = tf.keras.Input(
        shape=(patch_count, image_size_patch, image_size_patch, 3),
        name="patches",
    )
    inputs: dict[str, tf.keras.Input] = {"patches": patch_input}

    if variant == "v0_b":
        layout_input = tf.keras.Input(
            shape=(patch_count, layout_feature_dim),
            name="layout_features",
        )
        inputs["layout_features"] = layout_input
        global_input = tf.keras.Input(
            shape=(image_size_global, image_size_global, 3),
            name="global_view",
        )
        inputs["global_view"] = global_input

        patch_features = _build_fixed_patch_sequence_features(
            patch_input=patch_input,
            patch_count=patch_count,
            patch_size=image_size_patch,
            backbone_weights=backbone_weights,
            train_backbone=train_backbone,
            feature_dim=feature_dim,
            dropout=dropout,
        )
        patch_summary = tf.keras.layers.GlobalAveragePooling1D(name="v0_b_patch_feature_average")(patch_features)
        patch_summary = tf.keras.layers.Dense(feature_dim, activation="relu", name="multi_patch_subnet_dense")(patch_summary)

        global_features = _build_global_vgg16_branch(
            global_input=global_input,
            global_size=image_size_global,
            backbone_weights=backbone_weights,
            train_backbone=train_backbone,
            feature_dim=feature_dim,
            dropout=dropout,
        )

        layout_features = tf.keras.layers.Flatten(name="v0_b_layout_flatten")(layout_input)
        layout_features = tf.keras.layers.Dense(64, activation="relu", name="v0_b_layout_dense1")(layout_features)
        layout_features = tf.keras.layers.Dropout(dropout, name="v0_b_layout_dropout")(layout_features)
        layout_features = tf.keras.layers.Dense(64, activation="relu", name="v0_b_layout_dense2")(layout_features)

        fused = tf.keras.layers.Concatenate(name="alamp_paper_ava_fusion")(
            [patch_summary, global_features, layout_features]
        )
    else:
        patch_encoder = _build_shared_patch_encoder(
            patch_size=image_size_patch,
            backbone_weights=backbone_weights,
            train_backbone=train_backbone,
            feature_dim=feature_dim,
            dropout=dropout,
        )
        patch_features = tf.keras.layers.TimeDistributed(patch_encoder, name="shared_patch_column")(patch_input)
        attention_hidden = tf.keras.layers.Dense(128, activation="tanh", name="patch_attention_hidden")(patch_features)
        attention_logits = tf.keras.layers.Dense(1, name="patch_attention_logits")(attention_hidden)
        attention = tf.keras.layers.Softmax(axis=1, name="patch_attention")(attention_logits)
        patch_summary = WeightedPatchPooling(name="patch_weighted_pool")([patch_features, attention])
        fused = tf.keras.layers.Dense(feature_dim, activation="relu", name="multi_patch_subnet_dense")(patch_summary)

    fused = tf.keras.layers.Dropout(dropout, name="classifier_dropout1")(fused)
    fused = tf.keras.layers.Dense(256, activation="relu", name="classifier_dense")(fused)
    fused = tf.keras.layers.Dropout(max(0.1, dropout * 0.5), name="classifier_dropout2")(fused)
    output = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="probability")(fused)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=output,
        name=f"alamp_paper_ava_{variant}",
    )
    model._alamp_paper_ava_config = {  # type: ignore[attr-defined]
        "model_variant": MODEL_VARIANT,
        "variant": variant,
        "description": MODEL_DESCRIPTION,
        "style": STYLE_DESCRIPTION,
        "backbone": backbone,
        "backbone_weights_requested": backbone_weights,
        "train_backbone": train_backbone,
        "patch_count": patch_count,
        "image_size_patch": image_size_patch,
        "image_size_global": image_size_global,
        "shared_patch_backbone": True,
        "include_global_branch": include_global_branch,
        "include_layout_features": include_layout_features,
        "layout_feature_dim": layout_feature_dim if include_layout_features else 0,
        "v0_b_fixed_shape_fusion": variant == "v0_b",
        "v0_b_time_distributed_removed": variant == "v0_b",
        "exact_object_global_attribute_graph": "not implemented in v0",
        "exact_saliency_map_pipeline": "not implemented in v0",
    }
    return model


def get_alamp_paper_ava_custom_objects() -> dict[str, object]:
    return {
        "MergePatchBatch": MergePatchBatch,
        "RestorePatchBatch": RestorePatchBatch,
        "Vgg16UnitPreprocess": Vgg16UnitPreprocess,
        "WeightedPatchPooling": WeightedPatchPooling,
    }
