# DistortionGuard-IQA v1 Stage A 합성 왜곡 사전학습을 실행하는 스크립트.
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf


IMAGE_SIZE = 384
NUM_DISTORTION_TYPES = 10
BACKBONE_NAME = "efficientnetv2b0"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def _configure_tensorflow() -> None:
    tf.keras.mixed_precision.set_global_policy("float32")
    gpus = tf.config.list_physical_devices("GPU")
    print("Visible GPUs:", gpus)
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            print("memory growth setup failed:", exc)
    print("mixed precision policy:", tf.keras.mixed_precision.global_policy().name)


def _command_for_summary() -> str:
    command = " ".join([sys.executable, *sys.argv])
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        return f"PYTHONPATH={pythonpath} {command}"
    return command


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_training_log(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _count_trainable_params(model: tf.keras.Model) -> tuple[int, int]:
    trainable = int(
        np.sum([tf.keras.backend.count_params(weight) for weight in model.trainable_weights])
    )
    non_trainable = int(
        np.sum([tf.keras.backend.count_params(weight) for weight in model.non_trainable_weights])
    )
    return trainable, non_trainable


def build_distortionguard_stageA(
    input_shape: tuple[int, int, int] = (IMAGE_SIZE, IMAGE_SIZE, 3),
    num_distortion_types: int = NUM_DISTORTION_TYPES,
    freeze_backbone: bool = False,
    weights: str | None = "imagenet",
    dropout_rate: float = 0.25,
    embedding_units: int = 256,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32, name="image")
    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights=weights,
        include_preprocessing=True,
        input_shape=input_shape,
    )
    backbone.trainable = not freeze_backbone

    features = backbone(inputs)
    gap = tf.keras.layers.GlobalAveragePooling2D(name="stagea_gap")(features)
    gmp = tf.keras.layers.GlobalMaxPooling2D(name="stagea_gmp")(features)
    pooled = tf.keras.layers.Concatenate(name="stagea_pool_concat")([gap, gmp])
    x = tf.keras.layers.Dense(
        embedding_units,
        activation="swish",
        name="stagea_dense",
    )(pooled)
    x = tf.keras.layers.Dropout(dropout_rate, name="stagea_dropout")(x)
    embedding = tf.keras.layers.Dense(
        max(embedding_units // 2, 64),
        activation="swish",
        name="stagea_embedding",
    )(x)

    distortion_type = tf.keras.layers.Dense(
        num_distortion_types,
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

    model = tf.keras.Model(
        inputs=inputs,
        outputs=[distortion_type, severity, quality_proxy],
        name="distortionguard_stageA_synthetic",
    )
    model.distortionguard_stage = "stageA_synthetic_pretraining"
    model.distortionguard_backbone = BACKBONE_NAME
    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DistortionGuard-IQA v1 Stage A on synthetic distortion data."
    )
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--pairs_csv", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--pair_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--init_weights")
    parser.add_argument("--type_lambda", type=float, default=1.0)
    parser.add_argument("--severity_lambda", type=float, default=1.0)
    parser.add_argument("--pair_lambda", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--max_steps_per_epoch", type=int, default=500)
    parser.add_argument("--max_val_samples", type=int, default=2048)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.pair_batch_size <= 0:
        raise ValueError("--pair_batch_size must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive.")
    if args.type_lambda < 0.0:
        raise ValueError("--type_lambda must be non-negative.")
    if args.severity_lambda < 0.0:
        raise ValueError("--severity_lambda must be non-negative.")
    if args.pair_lambda < 0.0:
        raise ValueError("--pair_lambda must be non-negative.")
    if args.tau <= 0.0:
        raise ValueError("--tau must be positive.")
    if args.max_steps_per_epoch <= 0:
        raise ValueError("--max_steps_per_epoch must be positive for bounded Stage A runs.")
    if args.max_val_samples <= 0:
        raise ValueError("--max_val_samples must be positive.")


def _path_exists(path_text: Any) -> bool:
    if path_text is None:
        return False
    text = str(path_text)
    return bool(text) and text.lower() != "nan" and Path(text).is_file()


def _split_source_keys(source_keys: pd.Series, seed: int, val_fraction: float = 0.2) -> tuple[set[str], set[str]]:
    keys = sorted({str(value) for value in source_keys.dropna().astype(str) if str(value)})
    if len(keys) < 2:
        return set(keys), set()
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(keys, dtype=object)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_keys = set(str(value) for value in shuffled[:val_count])
    train_keys = set(str(value) for value in shuffled[val_count:])
    return train_keys, val_keys


def _load_manifest(
    manifest_csv: str | Path,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    path = Path(manifest_csv)
    df = pd.read_csv(path)
    required = {
        "distorted_image_path",
        "distortion_type",
        "distortion_type_id",
        "severity",
        "severity_norm",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required manifest columns: {missing}")

    original_count = len(df)
    df = df.copy()
    if "generation_status" in df.columns:
        df = df.loc[df["generation_status"].astype(str).eq("generated")].copy()
    df["distorted_image_path"] = df["distorted_image_path"].astype(str)
    exists = df["distorted_image_path"].map(_path_exists)
    df = df.loc[exists].copy()
    df["distortion_type_id"] = pd.to_numeric(df["distortion_type_id"], errors="coerce")
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce")
    df["severity_norm"] = pd.to_numeric(df["severity_norm"], errors="coerce")
    valid_label = (
        df["distortion_type_id"].between(0, NUM_DISTORTION_TYPES - 1)
        & df["severity"].between(1, 5)
        & df["severity_norm"].between(0.0, 1.0)
    )
    df = df.loc[valid_label].copy()
    if len(df) == 0:
        raise ValueError(f"{path} has no usable generated rows.")

    if "source_image_path" in df.columns:
        train_keys, val_keys = _split_source_keys(df["source_image_path"], seed=seed)
        if val_keys:
            train_df = df.loc[df["source_image_path"].astype(str).isin(train_keys)].copy()
            val_df = df.loc[df["source_image_path"].astype(str).isin(val_keys)].copy()
        else:
            train_df = df.copy()
            val_df = df.head(0).copy()
    else:
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        val_count = max(1, int(round(len(shuffled) * 0.2)))
        val_df = shuffled.iloc[:val_count].copy()
        train_df = shuffled.iloc[val_count:].copy()

    if len(val_df) == 0:
        val_df = train_df.sample(n=min(len(train_df), 512), random_state=seed).copy()

    summary = {
        "path": str(path),
        "original_count": int(original_count),
        "usable_count": int(len(df)),
        "dropped_count": int(original_count - len(df)),
        "train_count": int(len(train_df)),
        "val_count": int(len(val_df)),
        "distortion_type_counts": {
            str(key): int(value)
            for key, value in df["distortion_type"].value_counts().sort_index().items()
        },
        "severity_counts": {
            str(int(key)): int(value)
            for key, value in df["severity"].value_counts().sort_index().items()
        },
        "source_count": int(df["source_image_path"].nunique()) if "source_image_path" in df.columns else None,
        "val_source_count": int(val_df["source_image_path"].nunique()) if "source_image_path" in val_df.columns else None,
    }
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), summary


def _load_pairs(
    pairs_csv: str | Path,
    train_sources: set[str],
    val_sources: set[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    path = Path(pairs_csv)
    df = pd.read_csv(path)
    required = {"distorted_a_path", "distorted_b_path", "label", "severity_a", "severity_b", "distortion_type"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required pair columns: {missing}")

    original_count = len(df)
    df = df.copy()
    df["distorted_a_path"] = df["distorted_a_path"].astype(str)
    df["distorted_b_path"] = df["distorted_b_path"].astype(str)
    exists_a = df["distorted_a_path"].map(_path_exists)
    exists_b = df["distorted_b_path"].map(_path_exists)
    df = df.loc[exists_a & exists_b].copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.loc[df["label"].isin([0, 1])].copy()
    df["sign"] = (2.0 * df["label"].astype("float32")) - 1.0
    if len(df) == 0:
        raise ValueError(f"{path} has no usable pair rows.")

    if "source_image_path" in df.columns and train_sources and val_sources:
        source_text = df["source_image_path"].astype(str)
        train_df = df.loc[source_text.isin(train_sources)].copy()
        val_df = df.loc[source_text.isin(val_sources)].copy()
    else:
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        val_count = max(1, int(round(len(shuffled) * 0.2)))
        val_df = shuffled.iloc[:val_count].copy()
        train_df = shuffled.iloc[val_count:].copy()

    if len(train_df) == 0:
        raise ValueError(f"{path} produced no train pairs after splitting.")
    if len(val_df) == 0:
        val_df = train_df.sample(n=min(len(train_df), 512), random_state=seed).copy()

    summary = {
        "path": str(path),
        "original_count": int(original_count),
        "usable_count": int(len(df)),
        "dropped_count": int(original_count - len(df)),
        "train_count": int(len(train_df)),
        "val_count": int(len(val_df)),
        "label_counts": {
            str(int(key)): int(value)
            for key, value in df["label"].value_counts().sort_index().items()
        },
        "distortion_type_counts": {
            str(key): int(value)
            for key, value in df["distortion_type"].value_counts().sort_index().items()
        },
    }
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), summary


def _decode_resize_with_pad(path: tf.Tensor) -> tf.Tensor:
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)
    image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
    return image


def _make_manifest_dataset(
    df: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            df["distorted_image_path"].astype(str).tolist(),
            df["distortion_type_id"].astype("int32").to_numpy(),
            df["severity_norm"].astype("float32").to_numpy(),
        )
    )
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(df), 8192),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def _map_fn(
        path: tf.Tensor,
        distortion_type_id: tf.Tensor,
        severity_norm: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return (
            _decode_resize_with_pad(path),
            tf.cast(distortion_type_id, tf.int32),
            tf.reshape(tf.cast(severity_norm, tf.float32), (1,)),
        )

    return (
        dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def _make_pair_dataset(
    df: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            df["distorted_a_path"].astype(str).tolist(),
            df["distorted_b_path"].astype(str).tolist(),
            df["sign"].astype("float32").to_numpy(),
        )
    )
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(df), 8192),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def _map_fn(path_a: tf.Tensor, path_b: tf.Tensor, sign: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return (
            _decode_resize_with_pad(path_a),
            _decode_resize_with_pad(path_b),
            tf.reshape(tf.cast(sign, tf.float32), (1,)),
        )

    return (
        dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def _inspect_manifest_batch(dataset: tf.data.Dataset) -> dict[str, Any]:
    for images, type_ids, severities in dataset.take(1):
        image_values = images.numpy()
        return {
            "image_shape": list(image_values.shape),
            "image_dtype": str(image_values.dtype),
            "image_min": float(np.min(image_values)),
            "image_max": float(np.max(image_values)),
            "image_mean": float(np.mean(image_values)),
            "image_std": float(np.std(image_values)),
            "type_shape": list(type_ids.numpy().shape),
            "type_min": int(np.min(type_ids.numpy())),
            "type_max": int(np.max(type_ids.numpy())),
            "severity_shape": list(severities.numpy().shape),
            "severity_min": float(np.min(severities.numpy())),
            "severity_max": float(np.max(severities.numpy())),
        }
    raise RuntimeError("Manifest dataset produced no batches.")


def _supervised_losses(
    type_loss_fn: tf.keras.losses.Loss,
    severity_loss_fn: tf.keras.losses.Loss,
    type_labels: tf.Tensor,
    severity_labels: tf.Tensor,
    type_probs: tf.Tensor,
    severity_pred: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    type_loss = type_loss_fn(type_labels, type_probs)
    severity_loss = severity_loss_fn(severity_labels, severity_pred)
    return type_loss, severity_loss


def _ranking_loss(
    quality_a: tf.Tensor,
    quality_b: tf.Tensor,
    signs: tf.Tensor,
    tau: float,
) -> tf.Tensor:
    signs = tf.reshape(tf.cast(signs, tf.float32), (-1, 1))
    return tf.reduce_mean(
        tf.nn.softplus(-signs * (quality_a - quality_b) / tf.cast(tau, tf.float32))
    )


def _pair_accuracy(quality_a: tf.Tensor, quality_b: tf.Tensor, signs: tf.Tensor) -> tf.Tensor:
    signs = tf.reshape(tf.cast(signs, tf.float32), (-1, 1))
    margins = signs * (quality_a - quality_b)
    return tf.reduce_mean(tf.cast(margins > 0.0, tf.float32))


@tf.function
def _train_supervised_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    type_loss_fn: tf.keras.losses.Loss,
    severity_loss_fn: tf.keras.losses.Loss,
    images: tf.Tensor,
    type_labels: tf.Tensor,
    severity_labels: tf.Tensor,
    type_lambda: float,
    severity_lambda: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as tape:
        type_probs, severity_pred, _ = model(images, training=True)
        type_loss, severity_loss = _supervised_losses(
            type_loss_fn=type_loss_fn,
            severity_loss_fn=severity_loss_fn,
            type_labels=type_labels,
            severity_labels=severity_labels,
            type_probs=type_probs,
            severity_pred=severity_pred,
        )
        pair_loss = tf.constant(0.0, dtype=tf.float32)
        total_loss = (
            tf.cast(type_lambda, tf.float32) * type_loss
            + tf.cast(severity_lambda, tf.float32) * severity_loss
        )
    gradients = tape.gradient(total_loss, model.trainable_variables)
    grad_pairs = [
        (gradient, variable)
        for gradient, variable in zip(gradients, model.trainable_variables)
        if gradient is not None
    ]
    if grad_pairs:
        optimizer.apply_gradients(grad_pairs)
    type_acc = tf.reduce_mean(
        tf.cast(tf.equal(tf.cast(type_labels, tf.int64), tf.argmax(type_probs, axis=-1)), tf.float32)
    )
    severity_mae = tf.reduce_mean(tf.abs(severity_labels - severity_pred))
    return type_loss, severity_loss, pair_loss, total_loss, type_acc, severity_mae


@tf.function
def _train_with_pair_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    type_loss_fn: tf.keras.losses.Loss,
    severity_loss_fn: tf.keras.losses.Loss,
    images: tf.Tensor,
    type_labels: tf.Tensor,
    severity_labels: tf.Tensor,
    pair_images_a: tf.Tensor,
    pair_images_b: tf.Tensor,
    pair_signs: tf.Tensor,
    type_lambda: float,
    severity_lambda: float,
    pair_lambda: float,
    tau: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as tape:
        type_probs, severity_pred, _ = model(images, training=True)
        type_loss, severity_loss = _supervised_losses(
            type_loss_fn=type_loss_fn,
            severity_loss_fn=severity_loss_fn,
            type_labels=type_labels,
            severity_labels=severity_labels,
            type_probs=type_probs,
            severity_pred=severity_pred,
        )
        _, _, quality_a = model(pair_images_a, training=True)
        _, _, quality_b = model(pair_images_b, training=True)
        pair_loss = _ranking_loss(quality_a, quality_b, pair_signs, tau=tau)
        total_loss = (
            tf.cast(type_lambda, tf.float32) * type_loss
            + tf.cast(severity_lambda, tf.float32) * severity_loss
            + tf.cast(pair_lambda, tf.float32) * pair_loss
        )
    gradients = tape.gradient(total_loss, model.trainable_variables)
    grad_pairs = [
        (gradient, variable)
        for gradient, variable in zip(gradients, model.trainable_variables)
        if gradient is not None
    ]
    if grad_pairs:
        optimizer.apply_gradients(grad_pairs)
    type_acc = tf.reduce_mean(
        tf.cast(tf.equal(tf.cast(type_labels, tf.int64), tf.argmax(type_probs, axis=-1)), tf.float32)
    )
    severity_mae = tf.reduce_mean(tf.abs(severity_labels - severity_pred))
    pair_acc = _pair_accuracy(quality_a, quality_b, pair_signs)
    return type_loss, severity_loss, pair_loss, total_loss, type_acc, severity_mae, pair_acc


def _evaluate_supervised(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    type_loss_fn: tf.keras.losses.Loss,
    severity_loss_fn: tf.keras.losses.Loss,
    max_batches: int,
) -> dict[str, Any]:
    type_loss_metric = tf.keras.metrics.Mean()
    severity_loss_metric = tf.keras.metrics.Mean()
    type_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    severity_mae_metric = tf.keras.metrics.MeanAbsoluteError()
    batch_count = 0
    sample_count = 0
    for images, type_labels, severity_labels in dataset.take(max_batches):
        type_probs, severity_pred, _ = model(images, training=False)
        type_loss, severity_loss = _supervised_losses(
            type_loss_fn=type_loss_fn,
            severity_loss_fn=severity_loss_fn,
            type_labels=type_labels,
            severity_labels=severity_labels,
            type_probs=type_probs,
            severity_pred=severity_pred,
        )
        type_loss_metric.update_state(type_loss)
        severity_loss_metric.update_state(severity_loss)
        type_acc_metric.update_state(type_labels, type_probs)
        severity_mae_metric.update_state(severity_labels, severity_pred)
        batch_count += 1
        sample_count += int(tf.shape(images)[0].numpy())
    return {
        "type_loss": float(type_loss_metric.result().numpy()),
        "severity_loss": float(severity_loss_metric.result().numpy()),
        "type_accuracy": float(type_acc_metric.result().numpy()),
        "severity_mae": float(severity_mae_metric.result().numpy()),
        "batch_count": int(batch_count),
        "sample_count": int(sample_count),
    }


def _evaluate_pairs(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    max_batches: int,
    tau: float,
) -> dict[str, Any]:
    pair_loss_metric = tf.keras.metrics.Mean()
    pair_acc_metric = tf.keras.metrics.Mean()
    batch_count = 0
    sample_count = 0
    for images_a, images_b, signs in dataset.take(max_batches):
        _, _, quality_a = model(images_a, training=False)
        _, _, quality_b = model(images_b, training=False)
        pair_loss = _ranking_loss(quality_a, quality_b, signs, tau=tau)
        pair_acc = _pair_accuracy(quality_a, quality_b, signs)
        pair_loss_metric.update_state(pair_loss)
        pair_acc_metric.update_state(pair_acc)
        batch_count += 1
        sample_count += int(tf.shape(images_a)[0].numpy())
    return {
        "pair_loss": float(pair_loss_metric.result().numpy()),
        "pair_accuracy": float(pair_acc_metric.result().numpy()),
        "batch_count": int(batch_count),
        "sample_count": int(sample_count),
    }


def _feature_collapse_check(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    max_batches: int,
) -> dict[str, Any]:
    probe = tf.keras.Model(
        inputs=model.input,
        outputs=[
            model.get_layer("stagea_embedding").output,
            model.get_layer("quality_proxy").output,
        ],
        name="distortionguard_stageA_feature_probe",
    )
    embeddings: list[np.ndarray] = []
    quality_scores: list[np.ndarray] = []
    sample_count = 0
    for images, _, _ in dataset.take(max_batches):
        embedding, quality = probe(images, training=False)
        embeddings.append(embedding.numpy())
        quality_scores.append(quality.numpy().reshape(-1))
        sample_count += int(tf.shape(images)[0].numpy())
    if not embeddings:
        return {"sample_count": 0, "possible_feature_collapse": None}

    embedding_array = np.concatenate(embeddings, axis=0)
    quality_array = np.concatenate(quality_scores, axis=0)
    dim_std = np.std(embedding_array, axis=0)
    mean_dim_std = float(np.mean(dim_std))
    quality_std = float(np.std(quality_array))
    return {
        "sample_count": int(sample_count),
        "embedding_shape": list(embedding_array.shape),
        "embedding_dim_std_mean": mean_dim_std,
        "embedding_dim_std_min": float(np.min(dim_std)),
        "embedding_dim_std_max": float(np.max(dim_std)),
        "quality_proxy_min": float(np.min(quality_array)),
        "quality_proxy_max": float(np.max(quality_array)),
        "quality_proxy_mean": float(np.mean(quality_array)),
        "quality_proxy_std": quality_std,
        "possible_feature_collapse": bool(mean_dim_std < 1.0e-4 or quality_std < 1.0e-5),
    }


def _loss_delta(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    if len(rows) < 2 or key not in rows[0] or key not in rows[-1]:
        return {"key": key, "available": False}
    first = rows[0][key]
    last = rows[-1][key]
    if first is None or last is None:
        return {"key": key, "available": False}
    return {
        "key": key,
        "available": True,
        "first": float(first),
        "last": float(last),
        "delta": float(last - first),
        "decreased": bool(last < first),
    }


def _write_model_summary(model: tf.keras.Model, path: Path) -> None:
    lines: list[str] = []
    model.summary(print_fn=lines.append, expand_nested=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _args_for_summary(args: argparse.Namespace) -> dict[str, Any]:
    payload = vars(args).copy()
    if payload.get("init_weights") is None:
        payload.pop("init_weights", None)
    return payload


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    _set_seed(args.seed)
    _configure_tensorflow()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, manifest_summary = _load_manifest(args.manifest_csv, seed=args.seed)
    train_sources = (
        set(train_df["source_image_path"].astype(str).tolist())
        if "source_image_path" in train_df.columns
        else set()
    )
    val_sources = (
        set(val_df["source_image_path"].astype(str).tolist())
        if "source_image_path" in val_df.columns
        else set()
    )
    pair_train_df, pair_val_df, pair_summary = _load_pairs(
        args.pairs_csv,
        train_sources=train_sources,
        val_sources=val_sources,
        seed=args.seed,
    )

    train_dataset = _make_manifest_dataset(
        train_df,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_dataset = _make_manifest_dataset(
        val_df,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    pair_train_dataset = _make_pair_dataset(
        pair_train_df,
        batch_size=args.pair_batch_size,
        shuffle=True,
        seed=args.seed,
    )
    pair_val_dataset = _make_pair_dataset(
        pair_val_df,
        batch_size=args.pair_batch_size,
        shuffle=False,
        seed=args.seed,
    )

    supervised_steps = max(1, int(np.ceil(len(train_df) / args.batch_size)))
    pair_steps = max(1, int(np.ceil(len(pair_train_df) / args.pair_batch_size)))
    full_steps_per_epoch = supervised_steps if args.pair_lambda == 0.0 else min(supervised_steps, pair_steps)
    steps_per_epoch = min(full_steps_per_epoch, args.max_steps_per_epoch)
    max_val_batches = max(1, int(np.ceil(args.max_val_samples / args.batch_size)))
    max_pair_val_batches = max(1, int(np.ceil(args.max_val_samples / args.pair_batch_size)))

    model = build_distortionguard_stageA(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_distortion_types=NUM_DISTORTION_TYPES,
        freeze_backbone=args.freeze_backbone,
        weights="imagenet",
    )
    if args.init_weights:
        init_weights_path = Path(args.init_weights).expanduser()
        if not init_weights_path.is_file():
            raise FileNotFoundError(f"Initial weights file not found: {init_weights_path}")
        model.load_weights(str(init_weights_path))
        print(f"Loaded initial weights from: {init_weights_path}")
    trainable_params, non_trainable_params = _count_trainable_params(model)
    _write_model_summary(model, out_dir / "model_summary.txt")

    type_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    severity_loss_fn = tf.keras.losses.Huber(delta=0.1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    print(
        json.dumps(
            {
                "manifest": manifest_summary,
                "pairs": pair_summary,
                "batch_inspection": _inspect_manifest_batch(train_dataset),
                "steps_per_epoch": steps_per_epoch,
                "full_steps_per_epoch": full_steps_per_epoch,
                "max_steps_per_epoch": args.max_steps_per_epoch,
                "max_val_samples": args.max_val_samples,
                "freeze_backbone": bool(args.freeze_backbone),
                "trainable_params": trainable_params,
                "non_trainable_params": non_trainable_params,
            },
            indent=2,
            sort_keys=True,
        )
    )

    best_val_total_loss = float("inf")
    best_epoch = 0
    best_weights_path = out_dir / "best.weights.h5"
    log_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        manifest_iter = iter(train_dataset.repeat())
        pair_iter = iter(pair_train_dataset.repeat())
        type_loss_metric = tf.keras.metrics.Mean()
        severity_loss_metric = tf.keras.metrics.Mean()
        pair_loss_metric = tf.keras.metrics.Mean()
        total_loss_metric = tf.keras.metrics.Mean()
        type_acc_metric = tf.keras.metrics.Mean()
        severity_mae_metric = tf.keras.metrics.Mean()
        pair_acc_metric = tf.keras.metrics.Mean()

        for step in range(steps_per_epoch):
            images, type_labels, severity_labels = next(manifest_iter)
            if args.pair_lambda > 0.0:
                pair_images_a, pair_images_b, pair_signs = next(pair_iter)
                (
                    type_loss,
                    severity_loss,
                    pair_loss,
                    total_loss,
                    type_acc,
                    severity_mae,
                    pair_acc,
                ) = _train_with_pair_step(
                    model=model,
                    optimizer=optimizer,
                    type_loss_fn=type_loss_fn,
                    severity_loss_fn=severity_loss_fn,
                    images=images,
                    type_labels=type_labels,
                    severity_labels=severity_labels,
                    pair_images_a=pair_images_a,
                    pair_images_b=pair_images_b,
                    pair_signs=pair_signs,
                    type_lambda=args.type_lambda,
                    severity_lambda=args.severity_lambda,
                    pair_lambda=args.pair_lambda,
                    tau=args.tau,
                )
            else:
                type_loss, severity_loss, pair_loss, total_loss, type_acc, severity_mae = _train_supervised_step(
                    model=model,
                    optimizer=optimizer,
                    type_loss_fn=type_loss_fn,
                    severity_loss_fn=severity_loss_fn,
                    images=images,
                    type_labels=type_labels,
                    severity_labels=severity_labels,
                    type_lambda=args.type_lambda,
                    severity_lambda=args.severity_lambda,
                )
                pair_acc = tf.constant(0.0, dtype=tf.float32)

            type_loss_metric.update_state(type_loss)
            severity_loss_metric.update_state(severity_loss)
            pair_loss_metric.update_state(pair_loss)
            total_loss_metric.update_state(total_loss)
            type_acc_metric.update_state(type_acc)
            severity_mae_metric.update_state(severity_mae)
            pair_acc_metric.update_state(pair_acc)
            if (step + 1) % 25 == 0 or (step + 1) == steps_per_epoch:
                print(
                    f"epoch {epoch}/{args.epochs} step {step + 1}/{steps_per_epoch} "
                    f"loss={float(total_loss_metric.result().numpy()):.6f} "
                    f"type_acc={float(type_acc_metric.result().numpy()):.4f} "
                    f"severity_mae={float(severity_mae_metric.result().numpy()):.4f} "
                    f"pair_acc={float(pair_acc_metric.result().numpy()):.4f}"
                )

        val_supervised = _evaluate_supervised(
            model=model,
            dataset=val_dataset,
            type_loss_fn=type_loss_fn,
            severity_loss_fn=severity_loss_fn,
            max_batches=max_val_batches,
        )
        val_pairs = _evaluate_pairs(
            model=model,
            dataset=pair_val_dataset,
            max_batches=max_pair_val_batches,
            tau=args.tau,
        )
        val_total_loss = (
            args.type_lambda * val_supervised["type_loss"]
            + args.severity_lambda * val_supervised["severity_loss"]
            + args.pair_lambda * val_pairs["pair_loss"]
        )
        row = {
            "epoch": int(epoch),
            "train_type_loss": float(type_loss_metric.result().numpy()),
            "train_severity_loss": float(severity_loss_metric.result().numpy()),
            "train_pair_loss": float(pair_loss_metric.result().numpy()),
            "train_total_loss": float(total_loss_metric.result().numpy()),
            "train_type_accuracy": float(type_acc_metric.result().numpy()),
            "train_severity_mae": float(severity_mae_metric.result().numpy()),
            "train_pair_accuracy": float(pair_acc_metric.result().numpy()),
            "val_type_loss": float(val_supervised["type_loss"]),
            "val_severity_loss": float(val_supervised["severity_loss"]),
            "val_pair_loss": float(val_pairs["pair_loss"]),
            "val_total_loss": float(val_total_loss),
            "val_type_accuracy": float(val_supervised["type_accuracy"]),
            "val_severity_mae": float(val_supervised["severity_mae"]),
            "val_pair_accuracy": float(val_pairs["pair_accuracy"]),
            "val_supervised_samples": int(val_supervised["sample_count"]),
            "val_pair_samples": int(val_pairs["sample_count"]),
        }
        log_rows.append(row)
        print("epoch summary:", json.dumps(row, sort_keys=True))

        if val_total_loss < best_val_total_loss:
            best_val_total_loss = float(val_total_loss)
            best_epoch = epoch
            model.save_weights(best_weights_path)
            print(f"Saved best weights: {best_weights_path}")

    final_model_path = out_dir / "final_model.keras"
    model.save(final_model_path)
    _write_training_log(out_dir / "training_log.csv", log_rows)
    feature_check = _feature_collapse_check(
        model=model,
        dataset=val_dataset,
        max_batches=max_val_batches,
    )

    summary = {
        "command": _command_for_summary(),
        "args": _args_for_summary(args),
        "stage": "DistortionGuard-IQA v1 Stage A synthetic pretraining",
        "not_final_iqa_regression": True,
        "not_teacher_student_distillation": True,
        "backbone": BACKBONE_NAME,
        "input_contract": {
            "shape": [None, IMAGE_SIZE, IMAGE_SIZE, 3],
            "dtype": "float32",
            "pixel_range": "0..255",
            "external_divide_by_255": False,
            "include_preprocessing": True,
        },
        "manifest": manifest_summary,
        "pairs": pair_summary,
        "steps": {
            "supervised_steps_full": int(supervised_steps),
            "pair_steps_full": int(pair_steps),
            "full_steps_per_epoch": int(full_steps_per_epoch),
            "steps_per_epoch": int(steps_per_epoch),
            "max_val_batches": int(max_val_batches),
            "max_pair_val_batches": int(max_pair_val_batches),
        },
        "parameter_counts": {
            "trainable": int(trainable_params),
            "non_trainable": int(non_trainable_params),
        },
        "best_epoch": int(best_epoch),
        "best_val_total_loss": float(best_val_total_loss),
        "final_metrics": log_rows[-1] if log_rows else {},
        "loss_decrease": {
            "train_total_loss": _loss_delta(log_rows, "train_total_loss"),
            "val_total_loss": _loss_delta(log_rows, "val_total_loss"),
            "train_type_loss": _loss_delta(log_rows, "train_type_loss"),
            "train_severity_loss": _loss_delta(log_rows, "train_severity_loss"),
            "train_pair_loss": _loss_delta(log_rows, "train_pair_loss"),
        },
        "feature_collapse_check": feature_check,
        "artifacts": {
            "best_weights": str(best_weights_path),
            "final_model": str(final_model_path),
            "training_log": str(out_dir / "training_log.csv"),
            "training_summary": str(out_dir / "training_summary.json"),
            "model_summary": str(out_dir / "model_summary.txt"),
        },
    }
    _write_json(out_dir / "training_summary.json", summary)
    print("Saved:", best_weights_path)
    print("Saved:", final_model_path)
    print("Saved:", out_dir / "training_log.csv")
    print("Saved:", out_dir / "training_summary.json")
    print("Saved:", out_dir / "model_summary.txt")
    print("Final metrics:", json.dumps(summary["final_metrics"], sort_keys=True))
    print("Feature collapse check:", json.dumps(feature_check, sort_keys=True))


if __name__ == "__main__":
    main()
