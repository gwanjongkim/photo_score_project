"""
Refined practical A-Lamp-style aesthetic model.

Based on:
- A-Lamp: Adaptive Layout-Aware Multi-Patch deep convolutional neural network for photo aesthetic assessment

Faithful parts:
- separate global layout-aware branch and multi-patch branch
- aspect-preserving global view for composition/layout cues
- local multi-patch detail branch
- learned patch importance weighting before final fusion

Approximated parts:
- uses practical pretrained backbones available in this repo ecosystem
- layout-awareness is implemented with explicit coordinate and valid-area cues rather than the paper's exact machinery
- adaptive patch proposal remains deterministic heuristic search, though stronger than simple Sobel energy

Expected inputs:
- global_view: [B, global_size, global_size, 3]
- patches: [B, num_patches, patch_size, patch_size, 3]

Expected outputs:
- scalar aesthetic prediction suitable for AADB normalized scores or AVA mean scores
"""

from __future__ import annotations

import sys

import tensorflow as tf


def _build_backbone(
    input_shape: tuple[int, int, int],
    family: str,
    weights: str | None = "imagenet",
) -> tf.keras.Model:
    builders = {
        "efficientnetv2b0": tf.keras.applications.EfficientNetV2B0,
        "mobilenetv2": tf.keras.applications.MobileNetV2,
    }
    builder = builders[family]
    extra_kwargs = {"include_preprocessing": False} if family == "efficientnetv2b0" else {}
    if weights is None:
        return builder(include_top=False, weights=None, input_shape=input_shape, **extra_kwargs)
    try:
        return builder(include_top=False, weights=weights, input_shape=input_shape, **extra_kwargs)
    except Exception as exc:
        print(f"{family} imagenet weights unavailable, falling back to random init: {exc}", file=sys.stderr)
        return builder(include_top=False, weights=None, input_shape=input_shape, **extra_kwargs)


@tf.keras.utils.register_keras_serializable(package="photo_score_project")
class LayoutCueAugmentation(tf.keras.layers.Layer):
    def call(self, image: tf.Tensor) -> tf.Tensor:
        batch = tf.shape(image)[0]
        height = tf.shape(image)[1]
        width = tf.shape(image)[2]

        ys = tf.linspace(-1.0, 1.0, height)
        xs = tf.linspace(-1.0, 1.0, width)
        grid_y, grid_x = tf.meshgrid(ys, xs, indexing="ij")
        coords = tf.stack([grid_y, grid_x], axis=-1)
        coords = tf.broadcast_to(coords[None, ...], [batch, height, width, 2])
        coords = tf.cast(coords, self.compute_dtype)

        valid_mask = tf.cast(tf.reduce_max(image, axis=-1, keepdims=True) > 0.0, self.compute_dtype)
        return tf.concat([image, coords, valid_mask], axis=-1)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="photo_score_project")
class WeightedPatchPooling(tf.keras.layers.Layer):
    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        patch_features, patch_attention = inputs
        return tf.reduce_sum(patch_features * patch_attention, axis=1)

    def get_config(self):
        return super().get_config()


def _build_patch_encoder(
    patch_size: int,
    backbone_weights: str | None = "imagenet",
    rescale_unit_to_signed: bool = True,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(patch_size, patch_size, 3))
    base = _build_backbone((patch_size, patch_size, 3), family="mobilenetv2", weights=backbone_weights)
    x_in = (
        tf.keras.layers.Rescaling(2.0, offset=-1.0, name="patch_unit_to_signed")(inputs)
        if rescale_unit_to_signed
        else inputs
    )
    x = base(x_in, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    return tf.keras.Model(inputs, x, name="alamp_patch_encoder")


def build_alamp_model(
    global_size: int = 384,
    patch_size: int = 224,
    num_patches: int = 5,
    backbone_weights: str | None = "imagenet",
    rescale_unit_to_signed: bool = True,
) -> tf.keras.Model:
    global_input = tf.keras.Input(shape=(global_size, global_size, 3), name="global_view")
    patch_input = tf.keras.Input(shape=(num_patches, patch_size, patch_size, 3), name="patches")

    global_backbone = _build_backbone(
        (global_size, global_size, 3),
        family="efficientnetv2b0",
        weights=backbone_weights,
    )
    global_in = (
        tf.keras.layers.Rescaling(2.0, offset=-1.0, name="global_unit_to_signed")(global_input)
        if rescale_unit_to_signed
        else global_input
    )
    global_visual = global_backbone(global_in, training=False)
    global_visual = tf.keras.layers.GlobalAveragePooling2D(name="global_visual_pool")(global_visual)

    layout_tensor = LayoutCueAugmentation(name="layout_cues")(global_input)
    layout_branch = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(layout_tensor)
    layout_branch = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(layout_branch)
    layout_branch = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(layout_branch)
    layout_branch = tf.keras.layers.Conv2D(192, 3, strides=2, padding="same", activation="relu")(layout_branch)
    layout_branch = tf.keras.layers.GlobalAveragePooling2D(name="layout_pool")(layout_branch)

    global_branch = tf.keras.layers.Concatenate(name="global_branch_concat")([global_visual, layout_branch])
    global_branch = tf.keras.layers.Dropout(0.25)(global_branch)
    global_branch = tf.keras.layers.Dense(256, activation="relu", name="global_branch_dense")(global_branch)

    patch_encoder = _build_patch_encoder(
        patch_size=patch_size,
        backbone_weights=backbone_weights,
        rescale_unit_to_signed=rescale_unit_to_signed,
    )
    patch_features = tf.keras.layers.TimeDistributed(patch_encoder, name="patch_encoder_td")(patch_input)

    global_context = tf.keras.layers.Dense(256, activation="relu", name="patch_context_proj")(global_branch)
    repeated_global = tf.keras.layers.RepeatVector(num_patches, name="patch_context_repeat")(global_context)
    patch_context = tf.keras.layers.Concatenate(name="patch_context_concat")([patch_features, repeated_global])
    patch_context = tf.keras.layers.Dense(192, activation="tanh", name="patch_attention_hidden")(patch_context)
    patch_attention_logits = tf.keras.layers.Dense(1, name="patch_attention_logits")(patch_context)
    patch_attention = tf.keras.layers.Softmax(axis=1, name="patch_attention")(patch_attention_logits)

    patch_summary = WeightedPatchPooling(name="patch_weighted_pool")([patch_features, patch_attention])
    patch_branch = tf.keras.layers.Dense(256, activation="relu", name="patch_branch_dense")(patch_summary)

    fused = tf.keras.layers.Concatenate(name="alamp_fusion")([global_branch, patch_branch])
    fused = tf.keras.layers.Dropout(0.3)(fused)
    fused = tf.keras.layers.Dense(256, activation="relu", name="fusion_dense")(fused)
    fused = tf.keras.layers.Dropout(0.2)(fused)
    output = tf.keras.layers.Dense(1, activation="linear", dtype="float32", name="alamp_score")(fused)

    return tf.keras.Model(
        inputs={"global_view": global_input, "patches": patch_input},
        outputs=output,
        name="alamp_refined",
    )
