# DistortionGuard-IQA v1 Stage B 실제 IQA 미세조정을 실행하는 스크립트.
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

from src.models.distortionguard import (
    build_distortionguard_iqa_v1,
    configure_distortionguard_trainability,
)


IMAGE_SIZE = 384


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


def _parse_dataset_weight_overrides(text: str | None) -> dict[str, float]:
    if text is None or not text.strip():
        return {}
    overrides: dict[str, float] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("--dataset_weight_overrides entries must use name=value format.")
        key, value = item.split("=", 1)
        key = key.strip().lower()
        if not key:
            raise ValueError("--dataset_weight_overrides contains an empty dataset name.")
        weight = float(value)
        if weight <= 0.0:
            raise ValueError("Dataset weights must be positive.")
        overrides[key] = weight
    return overrides


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"1", "true", "yes", "y"})


def _dataset_counts(df: pd.DataFrame, dataset_col: str) -> dict[str, int]:
    if dataset_col not in df.columns:
        return {}
    return {
        str(key): int(value)
        for key, value in df[dataset_col].fillna("missing").value_counts().items()
    }


def _load_manifest(
    csv_path: str | Path,
    target_col: str,
    dataset_col: str,
    dataset_weight_overrides: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(csv_path)
    df = pd.read_csv(path)
    required = {"image_path", target_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    original_count = len(df)
    df = df.copy()
    df["image_path"] = df["image_path"].astype(str)
    if "missing_image" in df.columns:
        preverified_missing = _bool_series(df["missing_image"])
    else:
        preverified_missing = pd.Series(False, index=df.index)
    exists = df["image_path"].map(lambda value: bool(value) and Path(value).is_file())
    missing_image_mask = preverified_missing | ~exists
    df = df.loc[~missing_image_mask].copy()

    targets = pd.to_numeric(df[target_col], errors="coerce")
    missing_target_mask = targets.isna()
    df = df.loc[~missing_target_mask].copy()
    targets = targets.loc[~missing_target_mask].clip(0.0, 1.0).astype("float32")
    df[target_col] = targets

    if dataset_weight_overrides:
        if dataset_col not in df.columns:
            raise ValueError(
                f"--dataset_weight_overrides requires dataset column {dataset_col!r}."
            )
        sample_weights = (
            df[dataset_col]
            .astype(str)
            .str.lower()
            .map(dataset_weight_overrides)
            .fillna(1.0)
            .astype("float32")
        )
    else:
        sample_weights = pd.Series(1.0, index=df.index, dtype="float32")
    df["sample_weight_train"] = sample_weights

    if len(df) == 0:
        raise ValueError(f"{path} has no usable rows after filtering.")
    summary = {
        "path": str(path),
        "columns": list(pd.read_csv(path, nrows=0).columns),
        "original_count": int(original_count),
        "used_count": int(len(df)),
        "dropped_missing_image": int(missing_image_mask.sum()),
        "dropped_missing_target": int(missing_target_mask.sum()),
        "counts_by_dataset": _dataset_counts(df, dataset_col),
        "sample_weight_min": float(sample_weights.min()),
        "sample_weight_max": float(sample_weights.max()),
        "sample_weight_mean": float(sample_weights.mean()),
        "target_min": float(df[target_col].min()),
        "target_max": float(df[target_col].max()),
        "target_mean": float(df[target_col].mean()),
    }
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


def _make_dataset(
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
            df["sample_weight_train"].astype("float32").to_numpy(),
        )
    )
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(df), 8192),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def _map_fn(path: tf.Tensor, target: tf.Tensor, sample_weight: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        image = _decode_resize_with_pad(path)
        return (
            image,
            tf.reshape(tf.cast(target, tf.float32), (1,)),
            tf.reshape(tf.cast(sample_weight, tf.float32), (1,)),
        )

    return (
        dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def _inspect_batch(dataset: tf.data.Dataset) -> dict[str, Any]:
    for images, targets, sample_weights in dataset.take(1):
        image_values = images.numpy()
        target_values = targets.numpy()
        weight_values = sample_weights.numpy()
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
            "sample_weight_min": float(np.min(weight_values)),
            "sample_weight_max": float(np.max(weight_values)),
            "sample_weight_mean": float(np.mean(weight_values)),
        }
    raise RuntimeError("Dataset produced no batches.")


def _manual_huber(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    sample_weights: tf.Tensor | None = None,
    delta: float = 0.1,
) -> tf.Tensor:
    error = y_pred - y_true
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, tf.cast(delta, tf.float32))
    linear = abs_error - quadratic
    loss = 0.5 * tf.square(quadratic) + tf.cast(delta, tf.float32) * linear
    if sample_weights is not None:
        weights = tf.cast(sample_weights, tf.float32)
        weighted_loss = loss * weights
        return tf.reduce_sum(weighted_loss) / tf.maximum(tf.reduce_sum(weights), 1.0e-6)
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
        lambda: tf.constant(0.0, dtype=tf.float32),
    )
    return 1.0 - corr


@tf.function
def _train_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    images: tf.Tensor,
    targets: tf.Tensor,
    sample_weights: tf.Tensor,
    huber_lambda: float,
    plcc_lambda: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        huber = _manual_huber(targets, predictions, sample_weights=sample_weights)
        plcc = _plcc_loss(targets, predictions)
        total = tf.cast(huber_lambda, tf.float32) * huber + tf.cast(plcc_lambda, tf.float32) * plcc
    gradients = tape.gradient(total, model.trainable_variables)
    grad_pairs = [
        (gradient, variable)
        for gradient, variable in zip(gradients, model.trainable_variables)
        if gradient is not None
    ]
    if grad_pairs:
        optimizer.apply_gradients(grad_pairs)
    mae = tf.reduce_mean(tf.abs(predictions - targets))
    rmse = tf.sqrt(tf.reduce_mean(tf.square(predictions - targets)))
    return huber, plcc, total, mae, rmse


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
    max_batches: int,
) -> dict[str, Any]:
    predictions: list[float] = []
    targets: list[float] = []
    batch_count = 0
    for images, batch_targets, _ in dataset.take(max_batches):
        batch_predictions = model(images, training=False).numpy().reshape(-1)
        predictions.extend(float(value) for value in batch_predictions)
        targets.extend(float(value) for value in batch_targets.numpy().reshape(-1))
        batch_count += 1
    if not predictions:
        raise RuntimeError("Validation dataset produced no predictions.")

    pred = np.asarray(predictions, dtype=np.float32)
    target = np.asarray(targets, dtype=np.float32)
    huber = _huber_np(target, pred)
    plcc = _safe_corr(target, pred, method="pearson")
    plcc_loss = 1.0 if plcc is None else 1.0 - plcc
    total_loss = huber_lambda * huber + plcc_lambda * plcc_loss
    pred_std = float(np.std(pred))
    target_std = float(np.std(target))
    std_ratio = None if target_std <= 0.0 else pred_std / target_std
    mae = float(np.mean(np.abs(pred - target)))
    rmse = float(np.sqrt(np.mean(np.square(pred - target))))
    bias = float(np.mean(pred - target))
    return {
        "val_loss": float(total_loss),
        "val_huber_loss": float(huber),
        "val_plcc_loss": float(plcc_loss),
        "val_mae": mae,
        "val_rmse": rmse,
        "val_srcc": _safe_corr(target, pred, method="spearman"),
        "val_plcc": plcc,
        "val_bias": bias,
        "val_pred_mean": float(np.mean(pred)),
        "val_pred_std": pred_std,
        "val_target_mean": float(np.mean(target)),
        "val_target_std": target_std,
        "val_std_ratio": std_ratio,
        "mode_collapse": bool(pred_std < 0.01),
        "sample_count": int(len(pred)),
        "batch_count": int(batch_count),
        "val_mae_100": mae * 100.0,
        "val_rmse_100": rmse * 100.0,
        "val_bias_100": bias * 100.0,
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


def _write_trainable_layers_report(path: Path, trainability: dict[str, Any]) -> None:
    lines = [
        "DistortionGuard-IQA v1 Stage B Trainable Layers",
        "",
        f"freeze_backbone: {trainability.get('freeze_backbone')}",
        f"partial_unfreeze: {trainability.get('partial_unfreeze')}",
        f"unfreeze_top_blocks: {trainability.get('unfreeze_top_blocks')}",
        f"selected_top_blocks: {trainability.get('selected_top_blocks')}",
        f"freeze_batch_norm: {trainability.get('freeze_batch_norm')}",
        f"batch_norm_frozen_count: {trainability.get('batch_norm_frozen_count')}",
        f"batch_norm_trainable_count: {trainability.get('batch_norm_trainable_count')}",
        f"trainable_params: {trainability.get('trainable_params')}",
        f"non_trainable_params: {trainability.get('non_trainable_params')}",
        f"trainable_backbone_param_estimate: {trainability.get('trainable_backbone_param_estimate')}",
        "",
        "trainable_layers:",
    ]
    lines.extend(str(name) for name in trainability.get("trainable_layers", []))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _compact_stageA_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stageA_weights": report.get("stageA_weights"),
        "loaded_layer_count": report.get("loaded_layer_count", 0),
        "skipped_layer_count": report.get("skipped_layer_count", 0),
        "mismatched_layer_count": report.get("mismatched_layer_count", 0),
        "loaded_layers": [row.get("layer") for row in report.get("loaded_layers", [])],
        "skipped_target_layers": [
            row.get("layer") for row in report.get("skipped_layers", [])
        ],
        "skipped_stageA_layers": [
            row.get("layer") for row in report.get("skipped_stageA_layers", [])
        ],
        "mismatched_layers": [
            row.get("layer") for row in report.get("mismatched_layers", [])
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistortionGuard-IQA v1 Stage B on authentic IQA manifests."
    )
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--target_col", default="normalized_mos")
    parser.add_argument("--dataset_col", default="dataset")
    parser.add_argument("--stageA_weights")
    parser.add_argument("--init_weights")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--unfreeze_top_blocks", type=int, default=0)
    parser.add_argument(
        "--freeze_batch_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--head_units", type=int, default=256)
    parser.add_argument("--huber_lambda", type=float, default=1.0)
    parser.add_argument("--plcc_lambda", type=float, default=0.1)
    parser.add_argument("--dataset_weight_overrides")
    parser.add_argument("--max_steps_per_epoch", type=int, default=500)
    parser.add_argument("--max_val_samples", type=int, default=2048)
    parser.add_argument("--trainable_layers_report", action="store_true")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive.")
    if args.unfreeze_top_blocks < 0:
        raise ValueError("--unfreeze_top_blocks must be non-negative.")
    if args.dropout < 0.0 or args.dropout >= 1.0:
        raise ValueError("--dropout must be in [0, 1).")
    if args.head_units <= 0:
        raise ValueError("--head_units must be positive.")
    if args.huber_lambda < 0.0:
        raise ValueError("--huber_lambda must be non-negative.")
    if args.plcc_lambda < 0.0:
        raise ValueError("--plcc_lambda must be non-negative.")
    if args.max_steps_per_epoch <= 0:
        raise ValueError("--max_steps_per_epoch must be positive.")
    if args.max_val_samples <= 0:
        raise ValueError("--max_val_samples must be positive.")
    if args.init_weights and not Path(args.init_weights).is_file():
        raise FileNotFoundError(f"Initial weights file not found: {args.init_weights}")


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    _set_seed(args.seed)
    _configure_tensorflow()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_weight_overrides = _parse_dataset_weight_overrides(args.dataset_weight_overrides)

    train_df, train_summary = _load_manifest(
        csv_path=args.train_csv,
        target_col=args.target_col,
        dataset_col=args.dataset_col,
        dataset_weight_overrides=dataset_weight_overrides,
    )
    val_df, val_summary = _load_manifest(
        csv_path=args.val_csv,
        target_col=args.target_col,
        dataset_col=args.dataset_col,
        dataset_weight_overrides={},
    )

    train_dataset = _make_dataset(
        train_df,
        target_col=args.target_col,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_dataset = _make_dataset(
        val_df,
        target_col=args.target_col,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    model = build_distortionguard_iqa_v1(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        stageA_weights=args.stageA_weights,
        backbone_trainable=(not args.freeze_backbone and args.unfreeze_top_blocks == 0),
        freeze_batch_norm=args.freeze_batch_norm,
        dropout=args.dropout,
        head_units=args.head_units,
    )
    stageA_report = getattr(model, "stageA_weight_load_report", {})
    if args.stageA_weights and stageA_report.get("loaded_layer_count", 0) == 0:
        raise RuntimeError(f"No Stage A layers were loaded from {args.stageA_weights}")
    _write_json(out_dir / "stageA_weight_load_report.json", stageA_report)

    init_weights_loaded = False
    if args.init_weights:
        model.load_weights(args.init_weights)
        init_weights_loaded = True
        print(f"Loaded initial weights from: {args.init_weights}")

    trainability = configure_distortionguard_trainability(
        model=model,
        freeze_backbone=args.freeze_backbone,
        unfreeze_top_blocks=args.unfreeze_top_blocks,
        freeze_batch_norm=args.freeze_batch_norm,
    )
    _write_model_summary(model, out_dir / "model_summary.txt")
    trainable_layers_path = out_dir / "trainable_layers.txt"
    if args.trainable_layers_report:
        _write_trainable_layers_report(trainable_layers_path, trainability)
    print("Stage A weight load report:")
    print(json.dumps(_compact_stageA_report(stageA_report), indent=2, sort_keys=True))
    print("Trainability summary:")
    print(json.dumps(trainability, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "train_csv": train_summary,
                "val_csv": val_summary,
                "batch_inspection": _inspect_batch(train_dataset),
                "dataset_weight_overrides": dataset_weight_overrides,
            },
            indent=2,
            sort_keys=True,
        )
    )

    full_steps_per_epoch = max(1, int(np.ceil(len(train_df) / args.batch_size)))
    steps_per_epoch = min(full_steps_per_epoch, args.max_steps_per_epoch)
    max_val_batches = max(1, int(np.ceil(args.max_val_samples / args.batch_size)))
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights_path = out_dir / "best.weights.h5"
    log_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_iter = iter(train_dataset.repeat())
        huber_metric = tf.keras.metrics.Mean()
        plcc_metric = tf.keras.metrics.Mean()
        total_metric = tf.keras.metrics.Mean()
        mae_metric = tf.keras.metrics.Mean()
        rmse_metric = tf.keras.metrics.Mean()

        for step in range(steps_per_epoch):
            images, targets, sample_weights = next(train_iter)
            huber, plcc, total, mae, rmse = _train_step(
                model=model,
                optimizer=optimizer,
                images=images,
                targets=targets,
                sample_weights=sample_weights,
                huber_lambda=args.huber_lambda,
                plcc_lambda=args.plcc_lambda,
            )
            huber_metric.update_state(huber)
            plcc_metric.update_state(plcc)
            total_metric.update_state(total)
            mae_metric.update_state(mae)
            rmse_metric.update_state(rmse)
            if (step + 1) % 25 == 0 or (step + 1) == steps_per_epoch:
                print(
                    f"epoch {epoch}/{args.epochs} step {step + 1}/{steps_per_epoch} "
                    f"loss={float(total_metric.result().numpy()):.6f} "
                    f"huber={float(huber_metric.result().numpy()):.6f} "
                    f"plcc_loss={float(plcc_metric.result().numpy()):.6f} "
                    f"mae={float(mae_metric.result().numpy()):.4f}"
                )

        val_metrics = _evaluate(
            model=model,
            dataset=val_dataset,
            huber_lambda=args.huber_lambda,
            plcc_lambda=args.plcc_lambda,
            max_batches=max_val_batches,
        )
        row = {
            "epoch": int(epoch),
            "train_loss": float(total_metric.result().numpy()),
            "train_huber_loss": float(huber_metric.result().numpy()),
            "train_plcc_loss": float(plcc_metric.result().numpy()),
            "train_mae": float(mae_metric.result().numpy()),
            "train_rmse": float(rmse_metric.result().numpy()),
            **val_metrics,
        }
        log_rows.append(row)
        print("epoch summary:", json.dumps(row, sort_keys=True))
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = float(val_metrics["val_loss"])
            best_epoch = epoch
            model.save_weights(best_weights_path)
            print(f"Saved best weights: {best_weights_path}")

    final_model_path = out_dir / "final_model.keras"
    model.save(final_model_path)
    _write_training_log(out_dir / "training_log.csv", log_rows)
    final_metrics = log_rows[-1] if log_rows else {}
    summary = {
        "command": _command_for_summary(),
        "args": vars(args),
        "stage": "DistortionGuard-IQA v1 Stage B authentic IQA fine-tuning",
        "not_teacher_student_distillation": True,
        "single_output": "technical_score",
        "input_contract": {
            "shape": [None, IMAGE_SIZE, IMAGE_SIZE, 3],
            "dtype": "float32",
            "pixel_range": "0..255",
            "external_divide_by_255": False,
            "include_preprocessing": True,
        },
        "train_csv": train_summary,
        "val_csv": val_summary,
        "dataset_weight_overrides": dataset_weight_overrides,
        "stageA_weight_load_report": _compact_stageA_report(stageA_report),
        "init_weights": {
            "path": args.init_weights,
            "loaded": init_weights_loaded,
        },
        "trainability": trainability,
        "steps": {
            "full_steps_per_epoch": int(full_steps_per_epoch),
            "steps_per_epoch": int(steps_per_epoch),
            "max_val_batches": int(max_val_batches),
        },
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "final_metrics": final_metrics,
        "loss_decrease": {
            "train_loss": _loss_delta(log_rows, "train_loss"),
            "val_loss": _loss_delta(log_rows, "val_loss"),
            "train_huber_loss": _loss_delta(log_rows, "train_huber_loss"),
        },
        "artifacts": {
            "best_weights": str(best_weights_path),
            "final_model": str(final_model_path),
            "training_log": str(out_dir / "training_log.csv"),
            "training_summary": str(out_dir / "training_summary.json"),
            "model_summary": str(out_dir / "model_summary.txt"),
            "stageA_weight_load_report": str(out_dir / "stageA_weight_load_report.json"),
            "trainable_layers": str(trainable_layers_path) if args.trainable_layers_report else None,
        },
    }
    _write_json(out_dir / "training_summary.json", summary)
    print("Saved:", best_weights_path)
    print("Saved:", final_model_path)
    print("Saved:", out_dir / "training_log.csv")
    print("Saved:", out_dir / "training_summary.json")
    print("Saved:", out_dir / "model_summary.txt")
    print("Saved:", out_dir / "stageA_weight_load_report.json")
    if args.trainable_layers_report:
        print("Saved:", trainable_layers_path)
    print("Final metrics:", json.dumps(final_metrics, sort_keys=True))


if __name__ == "__main__":
    main()
