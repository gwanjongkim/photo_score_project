# TOPIQ-RankCalib v2 Stage 1 스모크 학습 루프를 실행한다.
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

from src.models.topiq_rankcalib import (
    build_topiq_rankcalib_v2,
    list_topiq_rankcalib_layers,
)


IMAGE_SIZE = 384
DATASET_TO_ID = {"spaq": 0, "koniq": 1, "flive": 2}
ID_TO_DATASET = {value: key for key, value in DATASET_TO_ID.items()}


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


def _command_for_summary() -> str:
    command = " ".join([sys.executable, *sys.argv])
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        return f"PYTHONPATH={pythonpath} {command}"
    return command


def _resolve_csv_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _resolve_image_path(value: Any, csv_dir: Path) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = csv_dir / path
    return str(path.resolve())


def _normalise_dataset_name(value: Any) -> str:
    text = str(value).strip().lower()
    if "spaq" in text:
        return "spaq"
    if "koniq" in text:
        return "koniq"
    if "flive" in text:
        return "flive"
    raise ValueError(f"Unsupported dataset value {value!r}; expected SPAQ, KonIQ, or FLIVE.")


def _load_manifest(
    csv_path: str | Path,
    target_col: str,
    dataset_col: str,
    max_rows: int | None = None,
) -> pd.DataFrame:
    path = _resolve_csv_path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    required = ["image_path", dataset_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; found {df.columns.tolist()}")

    if target_col not in df.columns:
        if target_col == "normalized_mos" and "mos" in df.columns:
            df = df.copy()
            df[target_col] = pd.to_numeric(df["mos"], errors="raise") / 100.0
        else:
            raise ValueError(
                f"{path} is missing target column {target_col!r}; "
                f"found {df.columns.tolist()}"
            )

    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows).copy()
    else:
        df = df.copy()

    df["image_path"] = [
        _resolve_image_path(value, path.parent) for value in df["image_path"].astype(str)
    ]
    df[target_col] = pd.to_numeric(df[target_col], errors="raise").astype("float32")
    df["dataset_name"] = df[dataset_col].map(_normalise_dataset_name)
    df["dataset_id"] = df["dataset_name"].map(DATASET_TO_ID).astype("int32")
    df = df.dropna(subset=["image_path", target_col, "dataset_name"]).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"{path} produced no usable rows.")
    return df


def _manifest_summary(
    df: pd.DataFrame,
    csv_path: str | Path,
    target_col: str,
    dataset_col: str,
) -> dict[str, Any]:
    targets = df[target_col].astype("float32")
    return {
        "csv_path": str(_resolve_csv_path(csv_path)),
        "row_count": int(len(df)),
        "target_col": target_col,
        "dataset_col": dataset_col,
        "dataset_counts": {
            str(key): int(value)
            for key, value in df["dataset_name"].value_counts().sort_index().items()
        },
        "target_min": float(targets.min()),
        "target_max": float(targets.max()),
        "target_mean": float(targets.mean()),
        "target_std": float(targets.std(ddof=0)),
    }


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
    paths = df["image_path"].astype(str).tolist()
    targets = df[target_col].astype("float32").tolist()
    dataset_ids = df["dataset_id"].astype("int32").tolist()
    dataset = tf.data.Dataset.from_tensor_slices((paths, targets, dataset_ids))
    if shuffle:
        dataset = dataset.shuffle(
            min(len(df), 10000),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def _map_fn(
        path: tf.Tensor,
        target: tf.Tensor,
        dataset_id: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        image = _decode_resize_with_pad(path)
        return (
            image,
            tf.reshape(tf.cast(target, tf.float32), (1,)),
            tf.cast(dataset_id, tf.int32),
        )

    return (
        dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def _inspect_batch(dataset: tf.data.Dataset) -> dict[str, Any]:
    for images, targets, dataset_ids in dataset.take(1):
        image_values = images.numpy()
        target_values = targets.numpy()
        ids = dataset_ids.numpy().reshape(-1)
        counts = {
            ID_TO_DATASET[index]: int(np.sum(ids == index))
            for index in sorted(ID_TO_DATASET)
        }
        return {
            "image_shape": list(image_values.shape),
            "image_dtype": str(image_values.dtype),
            "image_min": float(np.min(image_values)),
            "image_max": float(np.max(image_values)),
            "image_mean": float(np.mean(image_values)),
            "image_std": float(np.std(image_values)),
            "target_shape": list(target_values.shape),
            "target_dtype": str(target_values.dtype),
            "target_min": float(np.min(target_values)),
            "target_max": float(np.max(target_values)),
            "target_mean": float(np.mean(target_values)),
            "target_std": float(np.std(target_values)),
            "dataset_id_shape": list(dataset_ids.shape),
            "dataset_counts": counts,
        }
    raise RuntimeError("Dataset produced no batches.")


def _selected_specific_predictions(
    outputs: dict[str, tf.Tensor],
    dataset_ids: tf.Tensor,
) -> tf.Tensor:
    ids = tf.reshape(dataset_ids, (-1, 1))
    spaq_predictions = outputs["spaq"]
    koniq_predictions = outputs["koniq"]
    flive_predictions = outputs["flive"]
    return tf.where(
        tf.equal(ids, DATASET_TO_ID["spaq"]),
        spaq_predictions,
        tf.where(
            tf.equal(ids, DATASET_TO_ID["koniq"]),
            koniq_predictions,
            flive_predictions,
        ),
    )


def _calibration_loss(
    outputs: dict[str, tf.Tensor],
    targets: tf.Tensor,
    dataset_ids: tf.Tensor,
    epsilon: float = 1.0e-6,
) -> tf.Tensor:
    terms: list[tf.Tensor] = []
    flat_targets = tf.reshape(targets, (-1,))
    flat_ids = tf.reshape(dataset_ids, (-1,))
    for dataset_name, dataset_id in DATASET_TO_ID.items():
        mask = tf.equal(flat_ids, dataset_id)
        mask_float = tf.cast(mask, tf.float32)
        count = tf.reduce_sum(mask_float)
        if float(count.numpy()) == 0.0:
            continue

        predictions = tf.reshape(outputs[dataset_name], (-1,))
        pred_mean = tf.reduce_sum(predictions * mask_float) / count
        target_mean = tf.reduce_sum(flat_targets * mask_float) / count
        pred_var = tf.reduce_sum(tf.square(predictions - pred_mean) * mask_float) / count
        target_var = tf.reduce_sum(tf.square(flat_targets - target_mean) * mask_float) / count
        pred_std = tf.sqrt(tf.maximum(pred_var, 0.0) + epsilon)
        target_std = tf.sqrt(tf.maximum(target_var, 0.0) + epsilon)
        terms.append(tf.square(pred_mean - target_mean) + tf.square(pred_std - target_std))

    if not terms:
        return tf.constant(0.0, dtype=tf.float32)
    return tf.add_n(terms) / float(len(terms))


def _train_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: tf.keras.losses.Loss,
    images: tf.Tensor,
    targets: tf.Tensor,
    dataset_ids: tf.Tensor,
    calib_lambda: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as tape:
        outputs = model(images, training=True)
        unified_per_sample = loss_fn(targets, outputs["unified"])
        specific_predictions = _selected_specific_predictions(outputs, dataset_ids)
        specific_per_sample = loss_fn(targets, specific_predictions)
        unified_loss = tf.reduce_mean(unified_per_sample)
        specific_loss = tf.reduce_mean(specific_per_sample)
        if calib_lambda > 0.0:
            calib_loss = _calibration_loss(outputs, targets, dataset_ids)
        else:
            calib_loss = tf.constant(0.0, dtype=tf.float32)
        total_loss = unified_loss + specific_loss + tf.cast(calib_lambda, tf.float32) * calib_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    grad_pairs = [
        (gradient, variable)
        for gradient, variable in zip(gradients, model.trainable_variables)
        if gradient is not None
    ]
    optimizer.apply_gradients(grad_pairs)
    return total_loss, unified_loss, specific_loss, calib_loss


def _correlations(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    try:
        from scipy.stats import pearsonr, spearmanr
    except Exception as exc:
        return {"srcc": None, "plcc": None, "correlation_error": str(exc)}

    if len(targets) < 2:
        return {
            "srcc": None,
            "plcc": None,
            "correlation_error": "Need at least two samples for correlation.",
        }
    if float(np.std(targets)) < 1.0e-8 or float(np.std(predictions)) < 1.0e-8:
        return {
            "srcc": None,
            "plcc": None,
            "correlation_error": "Constant target or prediction values.",
        }

    srcc = spearmanr(targets, predictions).correlation
    plcc = pearsonr(targets, predictions).statistic
    return {
        "srcc": None if np.isnan(srcc) else float(srcc),
        "plcc": None if np.isnan(plcc) else float(plcc),
        "correlation_error": None,
    }


def _regression_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    prefix: str,
) -> dict[str, Any]:
    if len(targets) == 0:
        return {f"{prefix}_sample_count": 0}
    pred_std = float(np.std(predictions))
    target_std = float(np.std(targets))
    ratio = None if target_std == 0.0 else pred_std / target_std
    metrics: dict[str, Any] = {
        f"{prefix}_sample_count": int(len(targets)),
        f"{prefix}_mae": float(np.mean(np.abs(predictions - targets))),
        f"{prefix}_rmse": float(np.sqrt(np.mean(np.square(predictions - targets)))),
        f"{prefix}_pred_min": float(np.min(predictions)),
        f"{prefix}_pred_max": float(np.max(predictions)),
        f"{prefix}_pred_mean": float(np.mean(predictions)),
        f"{prefix}_pred_std": pred_std,
        f"{prefix}_target_min": float(np.min(targets)),
        f"{prefix}_target_max": float(np.max(targets)),
        f"{prefix}_target_mean": float(np.mean(targets)),
        f"{prefix}_target_std": target_std,
        f"{prefix}_pred_std_over_target_std": ratio,
        f"{prefix}_mode_collapse": bool(pred_std < 0.01),
    }
    corr = _correlations(targets, predictions)
    metrics[f"{prefix}_srcc"] = corr["srcc"]
    metrics[f"{prefix}_plcc"] = corr["plcc"]
    metrics[f"{prefix}_correlation_error"] = corr["correlation_error"]
    return metrics


def _evaluate(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    loss_fn: tf.keras.losses.Loss,
) -> dict[str, Any]:
    unified_predictions: list[float] = []
    unified_targets: list[float] = []
    dataset_ids_all: list[int] = []
    head_predictions = {name: [] for name in DATASET_TO_ID}
    loss_values: list[float] = []

    for images, targets, dataset_ids in dataset:
        outputs = model(images, training=False)
        batch_unified = outputs["unified"].numpy().reshape(-1).astype("float32")
        batch_targets = targets.numpy().reshape(-1).astype("float32")
        batch_ids = dataset_ids.numpy().reshape(-1).astype("int32")
        per_sample_loss = loss_fn(targets, outputs["unified"]).numpy().reshape(-1)
        loss_values.extend(float(value) for value in per_sample_loss)
        unified_predictions.extend(float(value) for value in batch_unified)
        unified_targets.extend(float(value) for value in batch_targets)
        dataset_ids_all.extend(int(value) for value in batch_ids)
        for dataset_name in DATASET_TO_ID:
            head_predictions[dataset_name].extend(
                float(value)
                for value in outputs[dataset_name].numpy().reshape(-1).astype("float32")
            )

    targets_np = np.asarray(unified_targets, dtype=np.float32)
    unified_np = np.asarray(unified_predictions, dtype=np.float32)
    ids_np = np.asarray(dataset_ids_all, dtype=np.int32)
    metrics = _regression_metrics(targets_np, unified_np, prefix="unified")
    metrics["loss"] = float(np.mean(loss_values)) if loss_values else None
    metrics["dataset_metrics"] = {}
    for dataset_name, dataset_id in DATASET_TO_ID.items():
        mask = ids_np == dataset_id
        head_np = np.asarray(head_predictions[dataset_name], dtype=np.float32)
        subset_metrics = _regression_metrics(
            targets_np[mask],
            head_np[mask],
            prefix=dataset_name,
        )
        metrics["dataset_metrics"][dataset_name] = subset_metrics
    return metrics


def _feature_summary(model: tf.keras.Model) -> dict[str, Any]:
    specs = getattr(model, "topiq_feature_specs", {})
    return {
        level: {
            "layer_name": spec.layer_name,
            "shape": list(spec.shape),
        }
        for level, spec in specs.items()
    }


def _write_model_summary(path: Path, model: tf.keras.Model) -> None:
    with path.open("w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(f"{line}\n"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1 smoke training for TOPIQ-RankCalib v2."
    )
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--target_col", default="normalized_mos")
    parser.add_argument("--dataset_col", default="dataset")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--head_units", type=int, default=256)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_steps_per_epoch", type=int, default=200)
    parser.add_argument("--max_val_samples", type=int, default=1024)
    parser.add_argument("--calib_lambda", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect_batch", action="store_true")
    parser.add_argument("--list_layers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.max_steps_per_epoch == 0:
        raise ValueError("--max_steps_per_epoch must be positive or negative for full epoch.")
    if args.max_val_samples == 0:
        raise ValueError("--max_val_samples must be positive or negative for full validation.")
    if args.calib_lambda < 0.0:
        raise ValueError("--calib_lambda must be non-negative.")

    _set_seed(args.seed)
    _configure_tensorflow()

    if args.list_layers:
        list_topiq_rankcalib_layers(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        return

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = _load_manifest(
        args.train_csv,
        target_col=args.target_col,
        dataset_col=args.dataset_col,
    )
    val_max_rows = args.max_val_samples if args.max_val_samples > 0 else None
    val_df = _load_manifest(
        args.val_csv,
        target_col=args.target_col,
        dataset_col=args.dataset_col,
        max_rows=val_max_rows,
    )
    train_summary = _manifest_summary(
        train_df,
        csv_path=args.train_csv,
        target_col=args.target_col,
        dataset_col=args.dataset_col,
    )
    val_summary = _manifest_summary(
        val_df,
        csv_path=args.val_csv,
        target_col=args.target_col,
        dataset_col=args.dataset_col,
    )
    print("Train CSV summary:")
    print(json.dumps(train_summary, indent=2, sort_keys=True))
    print("Val CSV summary:")
    print(json.dumps(val_summary, indent=2, sort_keys=True))

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

    if args.inspect_batch:
        print("Batch inspection:")
        print(json.dumps(_inspect_batch(train_dataset), indent=2, sort_keys=True))
        return

    full_steps_per_epoch = max(1, int(np.ceil(len(train_df) / args.batch_size)))
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(full_steps_per_epoch, args.max_steps_per_epoch)
    else:
        steps_per_epoch = full_steps_per_epoch

    model = build_topiq_rankcalib_v2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        backbone_trainable=not args.freeze_backbone,
        dropout=args.dropout,
        head_units=args.head_units,
    )
    features = _feature_summary(model)
    model_summary_path = out_dir / "model_summary.txt"
    _write_model_summary(model_summary_path, model)

    print(
        json.dumps(
            {
                "steps_per_epoch": int(steps_per_epoch),
                "full_steps_per_epoch": int(full_steps_per_epoch),
                "epochs": int(args.epochs),
                "freeze_backbone": bool(args.freeze_backbone),
                "calib_lambda": float(args.calib_lambda),
                "feature_layers": features,
                "output_names": list(model.output_names),
            },
            indent=2,
            sort_keys=True,
        )
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_fn = tf.keras.losses.Huber(delta=0.1, reduction="none")
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights_path = out_dir / "best.weights.h5"
    log_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_iter = iter(train_dataset.repeat())
        train_loss_metric = tf.keras.metrics.Mean()
        train_unified_metric = tf.keras.metrics.Mean()
        train_specific_metric = tf.keras.metrics.Mean()
        train_calib_metric = tf.keras.metrics.Mean()
        print_every = max(1, steps_per_epoch // 4)

        for step in range(steps_per_epoch):
            images, targets, dataset_ids = next(train_iter)
            total_loss, unified_loss, specific_loss, calib_loss = _train_step(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                images=images,
                targets=targets,
                dataset_ids=dataset_ids,
                calib_lambda=args.calib_lambda,
            )
            train_loss_metric.update_state(total_loss)
            train_unified_metric.update_state(unified_loss)
            train_specific_metric.update_state(specific_loss)
            train_calib_metric.update_state(calib_loss)
            if (step + 1) % print_every == 0 or (step + 1) == steps_per_epoch:
                print(
                    f"epoch {epoch}/{args.epochs} step {step + 1}/{steps_per_epoch} "
                    f"loss={float(train_loss_metric.result().numpy()):.6f} "
                    f"unified={float(train_unified_metric.result().numpy()):.6f} "
                    f"specific={float(train_specific_metric.result().numpy()):.6f} "
                    f"calib={float(train_calib_metric.result().numpy()):.6f}"
                )

        val_metrics = _evaluate(model=model, dataset=val_dataset, loss_fn=loss_fn)
        row: dict[str, Any] = {
            "epoch": int(epoch),
            "train_loss": float(train_loss_metric.result().numpy()),
            "train_unified_loss": float(train_unified_metric.result().numpy()),
            "train_specific_loss": float(train_specific_metric.result().numpy()),
            "train_calib_loss": float(train_calib_metric.result().numpy()),
            "val_loss": val_metrics["loss"],
            "val_unified_mae": val_metrics["unified_mae"],
            "val_unified_rmse": val_metrics["unified_rmse"],
            "val_unified_srcc": val_metrics["unified_srcc"],
            "val_unified_plcc": val_metrics["unified_plcc"],
            "pred_mean": val_metrics["unified_pred_mean"],
            "pred_std": val_metrics["unified_pred_std"],
            "target_mean": val_metrics["unified_target_mean"],
            "target_std": val_metrics["unified_target_std"],
            "mode_collapse": val_metrics["unified_mode_collapse"],
        }
        for dataset_name, subset_metrics in val_metrics["dataset_metrics"].items():
            row[f"val_{dataset_name}_mae"] = subset_metrics.get(f"{dataset_name}_mae")
            row[f"val_{dataset_name}_srcc"] = subset_metrics.get(f"{dataset_name}_srcc")
            row[f"val_{dataset_name}_sample_count"] = subset_metrics.get(
                f"{dataset_name}_sample_count"
            )

        log_rows.append(row)
        print("epoch summary:", json.dumps(row, sort_keys=True))

        val_loss = row["val_loss"]
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = epoch
            model.save_weights(best_weights_path)
            print(f"Saved best weights: {best_weights_path}")

    final_model_path = out_dir / "final_model.keras"
    training_log_path = out_dir / "training_log.csv"
    training_summary_path = out_dir / "training_summary.json"
    model.save(final_model_path)
    _write_training_log(training_log_path, log_rows)
    final_val_metrics = _evaluate(model=model, dataset=val_dataset, loss_fn=loss_fn)

    summary = {
        "command": _command_for_summary(),
        "args": vars(args),
        "artifacts": {
            "best_weights": str(best_weights_path),
            "final_model": str(final_model_path),
            "model_summary": str(model_summary_path),
            "training_log": str(training_log_path),
            "training_summary": str(training_summary_path),
        },
        "best_epoch": int(best_epoch),
        "best_val_loss": None if best_val_loss == float("inf") else float(best_val_loss),
        "feature_layers": features,
        "final_epoch_metrics": log_rows[-1] if log_rows else {},
        "final_validation_metrics": final_val_metrics,
        "mode_collapse": bool(final_val_metrics.get("unified_mode_collapse", False)),
        "train_csv": train_summary,
        "val_csv": val_summary,
    }
    _write_json(training_summary_path, summary)
    print("Final validation metrics:")
    print(json.dumps(final_val_metrics, indent=2, sort_keys=True))
    print("Saved:", final_model_path)
    print("Saved:", best_weights_path)
    print("Saved:", training_log_path)
    print("Saved:", training_summary_path)
    print("Saved:", model_summary_path)


if __name__ == "__main__":
    main()
