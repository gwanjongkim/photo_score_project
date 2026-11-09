from __future__ import annotations

import sys

import tensorflow as tf


def _build_mobilenet_v2_backbone(
    input_shape: tuple[int, int, int],
    alpha: float,
    weights: str | None,
    weights_path: str | None,
) -> tuple[tf.keras.Model, bool]:
    weights_are_pretrained = False
    final_weights = weights

    if weights_path:
        print(f"Attempting to load backbone weights from local path: {weights_path}")
        final_weights = weights_path
    elif weights:
        print(f"Attempting to load backbone weights from Keras default: {weights!r}")

    if not final_weights:
        print("Building backbone with random initialization (no pretrained weights).")
        return tf.keras.applications.MobileNetV2(
            include_top=False,
            weights=None,
            input_shape=input_shape,
            alpha=alpha,
        ), False

    try:
        model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights=final_weights,
            input_shape=input_shape,
            alpha=alpha,
        )
        weights_are_pretrained = True
        print(f"Successfully loaded backbone weights: {final_weights}")
        return model, True
    except Exception as exc:
        print("=" * 80, file=sys.stderr)
        print(" " * 20 + "CRITICAL PRETRAINED WEIGHTS WARNING", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(
            f"MobileNetV2 weights {final_weights!r} could not be loaded. "
            f"Falling back to RANDOM INITIALIZATION.",
            file=sys.stderr,
        )
        print(f"Error: {exc}", file=sys.stderr)
        print("The resulting model will likely have poor performance.", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        return tf.keras.applications.MobileNetV2(
            include_top=False,
            weights=None,
            input_shape=input_shape,
            alpha=alpha,
        ), False


def build_mobile_student_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    alpha: float = 0.5,
    head_units: int = 128,
    dropout_rate: float = 0.2,
    backbone_trainable: bool = True,
    backbone_weights: str | None = "imagenet",
    backbone_weights_path: str | None = None,
) -> tuple[tf.keras.Model, bool]:
    inputs = tf.keras.Input(shape=input_shape, name="image")
    backbone, weights_were_loaded = _build_mobilenet_v2_backbone(
        input_shape=input_shape,
        alpha=float(alpha),
        weights=backbone_weights,
        weights_path=backbone_weights_path,
    )
    backbone.trainable = bool(backbone_trainable)

    features = backbone(inputs)
    features = tf.keras.layers.GlobalAveragePooling2D(name="student_global_pool")(features)
    features = tf.keras.layers.Dropout(float(dropout_rate), name="student_dropout")(features)
    features = tf.keras.layers.Dense(int(head_units), activation="relu", name="student_head_dense")(features)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="teacher_score")(features)
    return tf.keras.Model(inputs, outputs, name="stage5_student_mobilenetv2"), weights_were_loaded


def build_distillation_loss(
    loss_name: str,
    huber_delta: float,
    ranking_weight: float,
    ranking_margin: float,
):
    loss_name = str(loss_name).strip().lower()
    if loss_name == "mae":
        base_loss = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    elif loss_name == "mse":
        base_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    else:
        base_loss = tf.keras.losses.Huber(delta=float(huber_delta), reduction=tf.keras.losses.Reduction.NONE)

    ranking_weight = float(ranking_weight)
    ranking_margin = float(ranking_margin)

    def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(tf.reshape(y_true, [-1, 1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1, 1]), tf.float32)
        pointwise_loss = tf.reduce_mean(base_loss(y_true, y_pred))
        if ranking_weight <= 0.0:
            return pointwise_loss

        true_diff = y_true - tf.transpose(y_true)
        pred_diff = y_pred - tf.transpose(y_pred)
        valid_mask = tf.cast(tf.abs(true_diff) >= ranking_margin, tf.float32)
        signed_margin = tf.sign(true_diff) * pred_diff
        ranking_loss = tf.nn.relu(ranking_margin - signed_margin)
        ranking_loss = tf.reduce_sum(ranking_loss * valid_mask) / tf.maximum(tf.reduce_sum(valid_mask), 1.0)
        return pointwise_loss + ranking_weight * ranking_loss

    return _loss


def build_optimizer(learning_rate: float, weight_decay: float):
    learning_rate = float(learning_rate)
    weight_decay = float(weight_decay)
    if weight_decay > 0.0 and hasattr(tf.keras.optimizers, "AdamW"):
        return tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)
