"""
Siamese pairwise aesthetics comparator.

Based on:
- Image Aesthetic Assessment Based on Pairwise Comparison (ICCV 2019)

Faithful parts:
- shared feature extractor for two images
- pairwise comparison head
- output is a relative-aesthetics relation over a pair

Approximated parts:
- ternary classification is used: A better, similar, B better
- uses a practical EfficientNetV2B0 backbone with offline fallback
- comparator head predicts relation probabilities rather than the paper's full unified objective stack
"""

from __future__ import annotations

import sys

import tensorflow as tf


def _build_backbone(input_shape: tuple[int, int, int]) -> tf.keras.Model:
    try:
        return tf.keras.applications.EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
    except Exception as exc:
        print(f"EfficientNetV2B0 imagenet weights unavailable, falling back to random init: {exc}", file=sys.stderr)
        return tf.keras.applications.EfficientNetV2B0(
            include_top=False,
            weights=None,
            input_shape=input_shape,
        )


@tf.keras.utils.register_keras_serializable(package="photo_score_project")
class AbsoluteDifference(tf.keras.layers.Layer):
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.math.abs(inputs)

    def get_config(self):
        return super().get_config()


def build_pairwise_comparator(input_shape: tuple[int, int, int] = (224, 224, 3)) -> tf.keras.Model:
    backbone = _build_backbone(input_shape=input_shape)

    image_a = tf.keras.Input(shape=input_shape, name="image_a")
    image_b = tf.keras.Input(shape=input_shape, name="image_b")

    encoder = tf.keras.Sequential(
        [
            backbone,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation="relu"),
        ],
        name="pairwise_encoder",
    )

    feat_a = encoder(image_a)
    feat_b = encoder(image_b)

    diff = tf.keras.layers.Subtract(name="feat_diff")([feat_a, feat_b])
    abs_diff = AbsoluteDifference(name="feat_abs_diff")(diff)
    prod = tf.keras.layers.Multiply(name="feat_prod")([feat_a, feat_b])
    pair_features = tf.keras.layers.Concatenate(name="pair_features")([feat_a, feat_b, diff, abs_diff, prod])
    pair_features = tf.keras.layers.Dropout(0.2)(pair_features)
    pair_features = tf.keras.layers.Dense(256, activation="relu", name="pair_head_dense")(pair_features)
    output = tf.keras.layers.Dense(3, activation="softmax", dtype="float32", name="relation")(pair_features)

    return tf.keras.Model({"image_a": image_a, "image_b": image_b}, output, name="pairwise_comparator")
