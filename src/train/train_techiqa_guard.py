# TechIQA-Guard v1 단일 출력 모델을 직접 학습하는 스크립트.
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

from src.models.techiqa_guard import build_techiqa_guard_v1


IMAGE_SIZE = 384
MISSING_GUARD_SCORE = -1.0


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"1", "true", "yes", "y"})


def _guard_source(df: pd.DataFrame) -> pd.Series:
    guard = pd.Series(np.nan, index=df.index, dtype="float32")
    for column in ["guard_source_score_100", "existing_combined_score", "flive_teacher_score"]:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            guard = guard.where(guard.notna(), values)
    return guard.fillna(MISSING_GUARD_SCORE).astype("float32")


def _dataset_counts(df: pd.DataFrame, dataset_col: str) -> dict[str, int]:
    if dataset_col not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[dataset_col].fillna("missing").value_counts().items()}


def _load_manifest(
    csv_path: str | Path,
    target_col: str,
    dataset_col: str,
    max_samples: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(csv_path)
    df = pd.read_csv(path)
    required = {"image_path", target_col}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise ValueError(f"{path} is missing required columns: {missing_cols}")

    original_count = len(df)
    df = df.copy()
    df["image_path"] = df["image_path"].astype(str)
    if "missing_image" in df.columns:
        preverified_missing = _bool_series(df["missing_image"])
    else:
        preverified_missing = pd.Series(False, index=df.index)
    exists = df["image_path"].map(lambda p: bool(p) and Path(p).is_file())
    missing_image_mask = preverified_missing | ~exists
    df = df.loc[~missing_image_mask].copy()

    targets = pd.to_numeric(df[target_col], errors="coerce")
    missing_target_mask = targets.isna()
    df = df.loc[~missing_target_mask].copy()
    targets = targets.loc[~missing_target_mask].clip(0.0, 1.0).astype("float32")
    df[target_col] = targets

    if "hard_false_positive" in df.columns:
        df["hard_false_positive_float"] = _bool_series(df["hard_false_positive"]).astype("float32")
    else:
        df["hard_false_positive_float"] = 0.0
    df["guard_source_score_train"] = _guard_source(df)

    if max_samples is not None and max_samples > 0:
        df = df.head(max_samples).copy()

    summary = {
        "path": str(path),
        "columns": list(pd.read_csv(path, nrows=0).columns),
        "original_count": int(original_count),
        "used_count": int(len(df)),
        "dropped_missing_image": int(missing_image_mask.sum()),
        "dropped_missing_target": int(missing_target_mask.sum()),
        "counts_by_dataset": _dataset_counts(df, dataset_col),
        "hard_false_positive_count": int(df["hard_false_positive_float"].sum()),
        "target_min": None if len(df) == 0 else float(df[target_col].min()),
        "target_max": None if len(df) == 0 else float(df[target_col].max()),
        "target_mean": None if len(df) == 0 else float(df[target_col].mean()),
    }
    if len(df) == 0:
        raise ValueError(f"{path} has no usable rows after filtering.")
    return df.reset_index(drop=True), summary


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


def _make_regression_dataset(
    df: pd.DataFrame,
    target_col: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            df["image_path"].astype(str).tolist(),
            df[target_col].astype("float32").to_numpy(),
            df["hard_false_positive_float"].astype("float32").to_numpy(),
            df["guard_source_score_train"].astype("float32").to_numpy(),
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
        target: tf.Tensor,
        hard_flag: tf.Tensor,
        guard_score: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        image = _decode_resize_with_pad(path)
        return (
            image,
            tf.reshape(tf.cast(target, tf.float32), (1,)),
            tf.reshape(tf.cast(hard_flag, tf.float32), (1,)),
            tf.reshape(tf.cast(guard_score, tf.float32), (1,)),
        )

    return (
        dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def _load_pair_frame(pair_csv: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(pair_csv)
    df = pd.read_csv(path)
    required = {"image_path_a", "image_path_b"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required pair columns: {missing}")
    if "sign" not in df.columns:
        if {"score_a", "score_b"}.issubset(df.columns):
            score_a = pd.to_numeric(df["score_a"], errors="coerce")
            score_b = pd.to_numeric(df["score_b"], errors="coerce")
            df["sign"] = np.where(score_a >= score_b, 1.0, -1.0)
        else:
            raise ValueError(f"{path} needs sign or score_a/score_b columns for ranking loss.")

    original_count = len(df)
    exists_a = df["image_path_a"].astype(str).map(lambda p: bool(p) and Path(p).is_file())
    exists_b = df["image_path_b"].astype(str).map(lambda p: bool(p) and Path(p).is_file())
    df = df.loc[exists_a & exists_b].copy()
    df["sign"] = pd.to_numeric(df["sign"], errors="coerce")
    df = df.loc[df["sign"].isin([-1, 1])].copy()
    if len(df) == 0:
        raise ValueError(f"{path} has no usable pairs after filtering.")
    summary = {
        "path": str(path),
        "original_count": int(original_count),
        "used_count": int(len(df)),
        "dropped_missing_or_invalid": int(original_count - len(df)),
        "sign_counts": {str(k): int(v) for k, v in df["sign"].value_counts().items()},
    }
    return df.reset_index(drop=True), summary


def _make_pair_dataset(
    df: pd.DataFrame,
    batch_size: int,
    seed: int,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            df["image_path_a"].astype(str).tolist(),
            df["image_path_b"].astype(str).tolist(),
            df["sign"].astype("float32").to_numpy(),
        )
    )
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


def _inspect_batch(dataset: tf.data.Dataset) -> dict[str, Any]:
    for images, targets, hard_flags, guard_scores in dataset.take(1):
        image_values = images.numpy()
        target_values = targets.numpy()
        guard_values = guard_scores.numpy()
        return {
            "image_shape": list(image_values.shape),
            "image_dtype": str(image_values.dtype),
            "image_min": float(np.min(image_values)),
            "image_max": float(np.max(image_values)),
            "image_mean": float(np.mean(image_values)),
            "image_std": float(np.std(image_values)),
            "target_shape": list(target_values.shape),
            "target_min": float(np.min(target_values)),
            "target_max": float(np.max(target_values)),
            "target_mean": float(np.mean(target_values)),
            "hard_false_positive_count": int(np.sum(hard_flags.numpy() > 0.5)),
            "guard_score_valid_count": int(np.sum(guard_values >= 0.0)),
        }
    raise RuntimeError("Dataset produced no batches.")


def _manual_huber(y_true: tf.Tensor, y_pred: tf.Tensor, delta: float = 0.1) -> tf.Tensor:
    error = y_pred - y_true
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, tf.cast(delta, tf.float32))
    linear = abs_error - quadratic
    loss = 0.5 * tf.square(quadratic) + tf.cast(delta, tf.float32) * linear
    return tf.reduce_mean(loss)


def _plcc_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    true_centered = y_true - tf.reduce_mean(y_true)
    pred_centered = y_pred - tf.reduce_mean(y_pred)
    denom = tf.sqrt(tf.reduce_sum(tf.square(true_centered)) * tf.reduce_sum(tf.square(pred_centered)))
    corr = tf.cond(
        denom > 1.0e-6,
        lambda: tf.reduce_sum(true_centered * pred_centered) / denom,
        lambda: tf.constant(1.0, dtype=tf.float32),
    )
    return 1.0 - corr


def _false_positive_loss(
    y_pred: tf.Tensor,
    hard_flags: tf.Tensor,
    guard_scores_100: tf.Tensor,
    margin_100: float,
) -> tf.Tensor:
    pred_100 = tf.reshape(tf.cast(y_pred, tf.float32), [-1]) * 100.0
    hard = tf.reshape(tf.cast(hard_flags, tf.float32), [-1]) > 0.5
    guard = tf.reshape(tf.cast(guard_scores_100, tf.float32), [-1])
    valid = guard >= 0.0
    mask = tf.logical_and(hard, valid)
    penalty = tf.nn.relu(pred_100 - guard - tf.cast(margin_100, tf.float32))
    masked = tf.boolean_mask(penalty, mask)
    return tf.cond(
        tf.size(masked) > 0,
        lambda: tf.reduce_mean(masked),
        lambda: tf.constant(0.0, dtype=tf.float32),
    )


@tf.function
def _train_regression_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    images: tf.Tensor,
    targets: tf.Tensor,
    hard_flags: tf.Tensor,
    guard_scores: tf.Tensor,
    huber_lambda: float,
    plcc_lambda: float,
    false_positive_lambda: float,
    false_positive_margin: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        huber = _manual_huber(targets, predictions)
        plcc = _plcc_loss(targets, predictions)
        false_positive = _false_positive_loss(
            y_pred=predictions,
            hard_flags=hard_flags,
            guard_scores_100=guard_scores,
            margin_100=false_positive_margin,
        )
        total = (
            tf.cast(huber_lambda, tf.float32) * huber
            + tf.cast(plcc_lambda, tf.float32) * plcc
            + tf.cast(false_positive_lambda, tf.float32) * false_positive
        )
    gradients = tape.gradient(total, model.trainable_variables)
    grad_pairs = [
        (gradient, variable)
        for gradient, variable in zip(gradients, model.trainable_variables)
        if gradient is not None
    ]
    optimizer.apply_gradients(grad_pairs)
    return huber, plcc, false_positive, total


@tf.function
def _train_with_pair_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    images: tf.Tensor,
    targets: tf.Tensor,
    hard_flags: tf.Tensor,
    guard_scores: tf.Tensor,
    pair_images_a: tf.Tensor,
    pair_images_b: tf.Tensor,
    pair_signs: tf.Tensor,
    huber_lambda: float,
    plcc_lambda: float,
    rank_lambda: float,
    false_positive_lambda: float,
    false_positive_margin: float,
    tau: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        huber = _manual_huber(targets, predictions)
        plcc = _plcc_loss(targets, predictions)
        false_positive = _false_positive_loss(
            y_pred=predictions,
            hard_flags=hard_flags,
            guard_scores_100=guard_scores,
            margin_100=false_positive_margin,
        )
        pred_a = model(pair_images_a, training=True)
        pred_b = model(pair_images_b, training=True)
        signs = tf.reshape(tf.cast(pair_signs, tf.float32), (-1, 1))
        rank = tf.reduce_mean(
            tf.nn.softplus(-signs * (pred_a - pred_b) / tf.cast(tau, tf.float32))
        )
        total = (
            tf.cast(huber_lambda, tf.float32) * huber
            + tf.cast(plcc_lambda, tf.float32) * plcc
            + tf.cast(rank_lambda, tf.float32) * rank
            + tf.cast(false_positive_lambda, tf.float32) * false_positive
        )
    gradients = tape.gradient(total, model.trainable_variables)
    grad_pairs = [
        (gradient, variable)
        for gradient, variable in zip(gradients, model.trainable_variables)
        if gradient is not None
    ]
    optimizer.apply_gradients(grad_pairs)
    return huber, plcc, rank, false_positive, total


def _safe_corr(target: np.ndarray, pred: np.ndarray, method: str) -> float | None:
    if len(target) < 2 or np.std(target) <= 1.0e-12 or np.std(pred) <= 1.0e-12:
        return None
    try:
        if method == "spearman":
            from scipy.stats import spearmanr

            value = spearmanr(target, pred).correlation
        else:
            from scipy.stats import pearsonr

            value = pearsonr(target, pred).statistic
    except Exception:
        return None
    return None if np.isnan(value) else float(value)


def _huber_np(target: np.ndarray, pred: np.ndarray, delta: float = 0.1) -> float:
    error = pred - target
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return float(np.mean(0.5 * np.square(quadratic) + delta * linear))


def _evaluate(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    huber_lambda: float,
    plcc_lambda: float,
    false_positive_lambda: float,
    false_positive_margin: float,
) -> dict[str, Any]:
    predictions: list[float] = []
    targets: list[float] = []
    hard_flags: list[float] = []
    guard_scores: list[float] = []
    for images, batch_targets, batch_hard_flags, batch_guard_scores in dataset:
        batch_predictions = model(images, training=False).numpy().reshape(-1)
        predictions.extend(float(v) for v in batch_predictions)
        targets.extend(float(v) for v in batch_targets.numpy().reshape(-1))
        hard_flags.extend(float(v) for v in batch_hard_flags.numpy().reshape(-1))
        guard_scores.extend(float(v) for v in batch_guard_scores.numpy().reshape(-1))

    pred = np.asarray(predictions, dtype=np.float32)
    target = np.asarray(targets, dtype=np.float32)
    hard = np.asarray(hard_flags, dtype=np.float32)
    guard = np.asarray(guard_scores, dtype=np.float32)
    if len(pred) == 0:
        raise RuntimeError("Validation dataset produced no predictions.")

    huber = _huber_np(target, pred)
    plcc_metric = _safe_corr(target, pred, method="pearson")
    plcc_loss = 0.0 if plcc_metric is None else 1.0 - plcc_metric
    valid_fp = (hard > 0.5) & (guard >= 0.0)
    if np.any(valid_fp):
        false_positive = float(np.mean(np.maximum(pred[valid_fp] * 100.0 - guard[valid_fp] - false_positive_margin, 0.0)))
    else:
        false_positive = 0.0
    total_loss = huber_lambda * huber + plcc_lambda * plcc_loss + false_positive_lambda * false_positive

    pred_std = float(np.std(pred))
    target_std = float(np.std(target))
    std_ratio = None if target_std <= 0.0 else pred_std / target_std
    mae = float(np.mean(np.abs(pred - target)))
    rmse = float(np.sqrt(np.mean(np.square(pred - target))))
    bias = float(np.mean(pred - target))
    return {
        "sample_count": int(len(pred)),
        "val_loss": float(total_loss),
        "val_huber_loss": float(huber),
        "val_plcc_loss": float(plcc_loss),
        "val_false_positive_loss": float(false_positive),
        "val_mae": mae,
        "val_rmse": rmse,
        "val_srcc": _safe_corr(target, pred, method="spearman"),
        "val_plcc": plcc_metric,
        "val_bias": bias,
        "val_pred_mean": float(np.mean(pred)),
        "val_pred_std": pred_std,
        "val_target_mean": float(np.mean(target)),
        "val_target_std": target_std,
        "val_std_ratio": std_ratio,
        "mode_collapse": bool(pred_std < 0.01),
        "val_mae_100": mae * 100.0,
        "val_rmse_100": rmse * 100.0,
        "val_bias_100": bias * 100.0,
    }


def _feature_summary(model: tf.keras.Model) -> dict[str, Any]:
    specs = getattr(model, "techiqa_feature_specs", {})
    return {
        level: {
            "layer_name": spec.layer_name,
            "shape": list(spec.shape),
        }
        for level, spec in specs.items()
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Directly train TechIQA-Guard v1.")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--target_col", default="normalized_mos")
    parser.add_argument("--dataset_col", default="dataset")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--head_units", type=int, default=256)
    parser.add_argument("--huber_lambda", type=float, default=1.0)
    parser.add_argument("--plcc_lambda", type=float, default=0.1)
    parser.add_argument("--rank_lambda", type=float, default=0.0)
    parser.add_argument("--false_positive_lambda", type=float, default=0.0)
    parser.add_argument("--false_positive_margin", type=float, default=5.0)
    parser.add_argument("--pair_csv")
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--max_steps_per_epoch", type=int, default=500)
    parser.add_argument("--max_val_samples", type=int, default=2048)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect_batch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.tau <= 0.0:
        raise ValueError("--tau must be positive.")
    if args.rank_lambda > 0.0 and not args.pair_csv:
        raise ValueError("--pair_csv is required when --rank_lambda > 0.")

    _set_seed(args.seed)
    _configure_tensorflow()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, train_summary = _load_manifest(
        args.train_csv,
        target_col=args.target_col,
        dataset_col=args.dataset_col,
    )
    val_limit = args.max_val_samples if args.max_val_samples > 0 else None
    val_df, val_summary = _load_manifest(
        args.val_csv,
        target_col=args.target_col,
        dataset_col=args.dataset_col,
        max_samples=val_limit,
    )
    print("Train CSV summary:")
    print(json.dumps(train_summary, indent=2, sort_keys=True))
    print("Val CSV summary:")
    print(json.dumps(val_summary, indent=2, sort_keys=True))

    train_dataset = _make_regression_dataset(
        train_df,
        target_col=args.target_col,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_dataset = _make_regression_dataset(
        val_df,
        target_col=args.target_col,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    if args.inspect_batch:
        print("Batch inspection:")
        print(json.dumps(_inspect_batch(train_dataset), indent=2, sort_keys=True))
        return

    pair_summary = None
    pair_dataset = None
    if args.rank_lambda > 0.0:
        pair_df, pair_summary = _load_pair_frame(args.pair_csv)
        pair_dataset = _make_pair_dataset(pair_df, batch_size=args.batch_size, seed=args.seed)

    full_steps_per_epoch = max(1, int(np.ceil(len(train_df) / args.batch_size)))
    steps_per_epoch = full_steps_per_epoch
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(full_steps_per_epoch, args.max_steps_per_epoch)

    model = build_techiqa_guard_v1(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        backbone_trainable=not args.freeze_backbone,
        dropout=args.dropout,
        head_units=args.head_units,
    )
    feature_layers = _feature_summary(model)
    with (out_dir / "model_summary.txt").open("w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(line + "\n"))

    run_setup = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "full_steps_per_epoch": int(full_steps_per_epoch),
        "steps_per_epoch": int(steps_per_epoch),
        "freeze_backbone": bool(args.freeze_backbone),
        "trainable_params": int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)),
        "non_trainable_params": int(sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)),
        "feature_layers": feature_layers,
        "rank_active": bool(args.rank_lambda > 0.0),
        "false_positive_active": bool(args.false_positive_lambda > 0.0),
    }
    print("Run setup:")
    print(json.dumps(run_setup, indent=2, sort_keys=True))

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    best_val_loss = float("inf")
    best_epoch = 0
    best_val_srcc = None
    best_val_srcc_epoch = None
    best_weights_path = out_dir / "best.weights.h5"
    log_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_iter = iter(train_dataset.repeat())
        pair_iter = iter(pair_dataset.repeat()) if pair_dataset is not None else None
        huber_metric = tf.keras.metrics.Mean()
        plcc_metric = tf.keras.metrics.Mean()
        rank_metric = tf.keras.metrics.Mean()
        fp_metric = tf.keras.metrics.Mean()
        total_metric = tf.keras.metrics.Mean()

        for step in range(steps_per_epoch):
            images, targets, hard_flags, guard_scores = next(train_iter)
            if pair_iter is not None:
                pair_images_a, pair_images_b, pair_signs = next(pair_iter)
                huber, plcc, rank, false_positive, total = _train_with_pair_step(
                    model=model,
                    optimizer=optimizer,
                    images=images,
                    targets=targets,
                    hard_flags=hard_flags,
                    guard_scores=guard_scores,
                    pair_images_a=pair_images_a,
                    pair_images_b=pair_images_b,
                    pair_signs=pair_signs,
                    huber_lambda=args.huber_lambda,
                    plcc_lambda=args.plcc_lambda,
                    rank_lambda=args.rank_lambda,
                    false_positive_lambda=args.false_positive_lambda,
                    false_positive_margin=args.false_positive_margin,
                    tau=args.tau,
                )
            else:
                huber, plcc, false_positive, total = _train_regression_step(
                    model=model,
                    optimizer=optimizer,
                    images=images,
                    targets=targets,
                    hard_flags=hard_flags,
                    guard_scores=guard_scores,
                    huber_lambda=args.huber_lambda,
                    plcc_lambda=args.plcc_lambda,
                    false_positive_lambda=args.false_positive_lambda,
                    false_positive_margin=args.false_positive_margin,
                )
                rank = tf.constant(0.0, dtype=tf.float32)

            huber_metric.update_state(huber)
            plcc_metric.update_state(plcc)
            rank_metric.update_state(rank)
            fp_metric.update_state(false_positive)
            total_metric.update_state(total)

            if (step + 1) % 25 == 0 or (step + 1) == steps_per_epoch:
                print(
                    f"epoch {epoch}/{args.epochs} step {step + 1}/{steps_per_epoch} "
                    f"loss={float(total_metric.result().numpy()):.6f} "
                    f"huber={float(huber_metric.result().numpy()):.6f} "
                    f"plcc_loss={float(plcc_metric.result().numpy()):.6f} "
                    f"rank={float(rank_metric.result().numpy()):.6f} "
                    f"fp={float(fp_metric.result().numpy()):.6f}"
                )

        val_metrics = _evaluate(
            model=model,
            dataset=val_dataset,
            huber_lambda=args.huber_lambda,
            plcc_lambda=args.plcc_lambda,
            false_positive_lambda=args.false_positive_lambda,
            false_positive_margin=args.false_positive_margin,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(total_metric.result().numpy()),
            "train_huber_loss": float(huber_metric.result().numpy()),
            "train_plcc_loss": float(plcc_metric.result().numpy()),
            "train_rank_loss": float(rank_metric.result().numpy()),
            "train_false_positive_loss": float(fp_metric.result().numpy()),
            **val_metrics,
        }
        log_rows.append(row)
        print("epoch summary:")
        print(json.dumps(row, indent=2, sort_keys=True))

        current_srcc = val_metrics.get("val_srcc")
        if current_srcc is not None and (best_val_srcc is None or current_srcc > best_val_srcc):
            best_val_srcc = float(current_srcc)
            best_val_srcc_epoch = epoch
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = float(val_metrics["val_loss"])
            best_epoch = epoch
            model.save_weights(best_weights_path)
            print(f"Saved best weights: {best_weights_path}")

    final_model_path = out_dir / "final_model.keras"
    model.save(final_model_path)
    _write_training_log(out_dir / "training_log.csv", log_rows)

    summary = {
        "command": _command_for_summary(),
        "args": vars(args),
        "artifacts": {
            "best_weights": str(best_weights_path),
            "final_model": str(final_model_path),
            "training_log": str(out_dir / "training_log.csv"),
            "training_summary": str(out_dir / "training_summary.json"),
            "model_summary": str(out_dir / "model_summary.txt"),
        },
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_srcc": best_val_srcc,
        "best_val_srcc_epoch": best_val_srcc_epoch,
        "feature_layers": feature_layers,
        "final_metrics": log_rows[-1] if log_rows else {},
        "model_params": {
            "trainable": run_setup["trainable_params"],
            "non_trainable": run_setup["non_trainable_params"],
        },
        "pair_summary": pair_summary,
        "run_setup": run_setup,
        "train_csv": train_summary,
        "val_csv": val_summary,
    }
    _write_json(out_dir / "training_summary.json", summary)
    print("Saved:", best_weights_path)
    print("Saved:", final_model_path)
    print("Saved:", out_dir / "training_log.csv")
    print("Saved:", out_dir / "training_summary.json")
    print("Saved:", out_dir / "model_summary.txt")


if __name__ == "__main__":
    main()
