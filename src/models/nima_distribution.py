"""
Paper-faithful NIMA distribution head.

Based on:
- NIMA: Neural Image Assessment

Faithful parts:
- predicts a 10-bin score distribution
- exposes Earth Mover's Distance loss
- computes mean score from the predicted distribution

Approximated parts:
- uses an EfficientNetV2B0 backbone for repo practicality instead of the exact paper backbone variants

Expected inputs:
- batch of RGB images with shape [B, H, W, 3] scaled to [0, 1]

Expected outputs:
- batch of score distributions with shape [B, 10]
"""

from __future__ import annotations

import tensorflow as tf


def _assert_no_internal_rescaling(model: tf.keras.Model) -> None:
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Rescaling):
            raise RuntimeError(
                "NIMA EfficientNetV2B0 must not include an internal Rescaling layer when fed [0,1] images."
            )


def _build_efficientnetv2_backbone(
    input_shape: tuple[int, int, int],
    weights: str | None,
) -> tf.keras.Model:
    try:
        # Keras EfficientNetV2 normally includes preprocessing by default.
        # This project feeds [0,1] NIMA images, so include_preprocessing=False is
        # required to avoid a second 1/255 rescale inside the backbone.
        base = tf.keras.applications.EfficientNetV2B0(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            include_preprocessing=False,
        )
    except TypeError as exc:
        raise RuntimeError(
            "This TensorFlow/Keras EfficientNetV2B0 does not support include_preprocessing=False. "
            "Do not retrain NIMA until the model-side preprocessing contract can be fixed."
        ) from exc
    _assert_no_internal_rescaling(base)
    return base


def build_nima_distribution_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    backbone_weights: str | None = "imagenet",
) -> tf.keras.Model:
    base = _build_efficientnetv2_backbone(input_shape=input_shape, weights=backbone_weights)

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax", dtype="float32")(x)
    return tf.keras.Model(inputs, outputs, name="nima_distribution")


@tf.keras.utils.register_keras_serializable(package="photo_score_project")
def emd_loss(y_true: tf.Tensor, y_pred: tf.Tensor, r: int = 2) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    cdf_true = tf.cumsum(y_true, axis=-1)
    cdf_pred = tf.cumsum(y_pred, axis=-1)
    samplewise = tf.reduce_mean(tf.pow(tf.abs(cdf_true - cdf_pred), r), axis=-1)
    return tf.pow(samplewise, 1.0 / r)


def distribution_mean_score(distribution: tf.Tensor) -> tf.Tensor:
    weights = tf.range(1, 11, dtype=tf.float32)
    return tf.reduce_sum(tf.cast(distribution, tf.float32) * weights[None, :], axis=-1)


@tf.keras.utils.register_keras_serializable(package="photo_score_project")
def mean_score_mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    true_mean = distribution_mean_score(y_true)
    pred_mean = distribution_mean_score(y_pred)
    return tf.reduce_mean(tf.abs(true_mean - pred_mean))
