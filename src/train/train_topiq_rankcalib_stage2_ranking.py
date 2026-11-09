# TOPIQ-RankCalib v2 FLIVE-head 순위 미세조정 루프를 실행한다.
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

from src.datasets.topiq_ranking_dataset import make_flive_pairs, make_pair_dataset
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


def _pair_csv_summary(pair_csv: str | Path) -> dict[str, Any]:
    path = _resolve_csv_path(pair_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Pair CSV not found: {path}")
    df = pd.read_csv(path)
    required = ["image_path_a", "image_path_b", "sign"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; found {df.columns.tolist()}")

    signs = pd.to_numeric(df["sign"], errors="raise")
    summary: dict[str, Any] = {
        "pair_csv": str(path),
        "pair_count": int(len(df)),
        "sign_counts": {
            str(key): int(value) for key, value in signs.value_counts().sort_index().items()
        },
    }
    if "gap" in df.columns and len(df):
        gaps = pd.to_numeric(df["gap"], errors="raise").astype("float32")
        summary.update(
            {
                "gap_min": float(gaps.min()),
                "gap_max": float(gaps.max()),
                "gap_mean": float(gaps.mean()),
            }
        )
    return summary


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


def _make_image_target_dataset(
    df: pd.DataFrame,
    target_col: str,
    batch_size: int,
) -> tf.data.Dataset:
    paths = df["image_path"].astype(str).tolist()
    targets = df[target_col].astype("float32").tolist()
    dataset = tf.data.Dataset.from_tensor_slices((paths, targets))

    def _map_fn(path: tf.Tensor, target: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = _decode_resize_with_pad(path)
        return image, tf.reshape(tf.cast(target, tf.float32), (1,))

    return (
        dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def _inspect_regression_batch(dataset: tf.data.Dataset) -> dict[str, Any]:
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
            "target_shape": list(target_values.shape),
            "target_min": float(np.min(target_values)),
            "target_max": float(np.max(target_values)),
            "dataset_counts": counts,
        }
    raise RuntimeError("Regression dataset produced no batches.")


def _inspect_pair_batch(dataset: tf.data.Dataset) -> dict[str, Any]:
    for (images_a, images_b), signs in dataset.take(1):
        values_a = images_a.numpy()
        values_b = images_b.numpy()
        signs_np = signs.numpy().reshape(-1)
        return {
            "image_a_shape": list(values_a.shape),
            "image_b_shape": list(values_b.shape),
            "image_a_min": float(np.min(values_a)),
            "image_a_max": float(np.max(values_a)),
            "image_b_min": float(np.min(values_b)),
            "image_b_max": float(np.max(values_b)),
            "sign_shape": list(signs.shape),
            "sign_values": sorted(float(value) for value in np.unique(signs_np)),
        }
    raise RuntimeError("Pair dataset produced no batches.")


def _selected_specific_predictions(
    outputs: dict[str, tf.Tensor],
    dataset_ids: tf.Tensor,
) -> tf.Tensor:
    ids = tf.reshape(dataset_ids, (-1, 1))
    return tf.where(
        tf.equal(ids, DATASET_TO_ID["spaq"]),
        outputs["spaq"],
        tf.where(
            tf.equal(ids, DATASET_TO_ID["koniq"]),
            outputs["koniq"],
            outputs["flive"],
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


def _flive_head_variables(model: tf.keras.Model) -> list[tf.Variable]:
    variables: list[tf.Variable] = []
    for layer in model.layers:
        if layer.name == "flive" or layer.name.startswith("flive_"):
            variables.extend(layer.trainable_variables)
    if not variables:
        raise RuntimeError("No trainable FLIVE head variables found.")
    return variables


def _add_gradients(
    first: tf.Tensor | None,
    second: tf.Tensor | None,
) -> tf.Tensor | None:
    if first is None:
        return second
    if second is None:
        return first
    return first + second


def _train_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: tf.keras.losses.Loss,
    flive_variables: list[tf.Variable],
    reg_images: tf.Tensor,
    reg_targets: tf.Tensor,
    reg_dataset_ids: tf.Tensor,
    pair_images_a: tf.Tensor,
    pair_images_b: tf.Tensor,
    pair_signs: tf.Tensor,
    ranking_lambda: float,
    tau: float,
    regression_anchor_lambda: float,
    unified_anchor_lambda: float,
    specific_anchor_lambda: float,
    calib_lambda: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    with tf.GradientTape(persistent=True) as tape:
        reg_outputs = model(reg_images, training=True)
        unified_per_sample = loss_fn(reg_targets, reg_outputs["unified"])
        specific_predictions = _selected_specific_predictions(
            reg_outputs,
            reg_dataset_ids,
        )
        specific_per_sample = loss_fn(reg_targets, specific_predictions)
        unified_loss = tf.reduce_mean(unified_per_sample)
        specific_loss = tf.reduce_mean(specific_per_sample)
        total_regression_loss = (
            tf.cast(unified_anchor_lambda, tf.float32) * unified_loss
            + tf.cast(specific_anchor_lambda, tf.float32) * specific_loss
        )
        if calib_lambda > 0.0:
            calib_loss = _calibration_loss(reg_outputs, reg_targets, reg_dataset_ids)
        else:
            calib_loss = tf.constant(0.0, dtype=tf.float32)

        pred_a = model(pair_images_a, training=True)["flive"]
        pred_b = model(pair_images_b, training=True)["flive"]
        signs = tf.reshape(tf.cast(pair_signs, tf.float32), (-1, 1))
        rank_loss = tf.reduce_mean(
            tf.nn.softplus(-signs * (pred_a - pred_b) / tf.cast(tau, tf.float32))
        )
        regression_component = (
            tf.cast(regression_anchor_lambda, tf.float32) * total_regression_loss
            + tf.cast(calib_lambda, tf.float32) * calib_loss
        )
        ranking_component = tf.cast(ranking_lambda, tf.float32) * rank_loss
        total_loss = regression_component + ranking_component

    trainable_variables = model.trainable_variables
    regression_grads = tape.gradient(regression_component, trainable_variables)
    ranking_grads = tape.gradient(ranking_component, flive_variables)
    del tape

    ranking_grad_by_id = {
        id(variable): gradient
        for variable, gradient in zip(flive_variables, ranking_grads)
        if gradient is not None
    }
    grad_pairs = []
    for variable, regression_grad in zip(trainable_variables, regression_grads):
        gradient = _add_gradients(regression_grad, ranking_grad_by_id.get(id(variable)))
        if gradient is not None:
            grad_pairs.append((gradient, variable))
    optimizer.apply_gradients(grad_pairs)
    return (
        total_loss,
        total_regression_loss,
        unified_loss,
        specific_loss,
        rank_loss,
        calib_loss,
    )


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

    errors = predictions - targets
    pred_std = float(np.std(predictions))
    target_std = float(np.std(targets))
    ratio = None if target_std == 0.0 else pred_std / target_std
    metrics: dict[str, Any] = {
        f"{prefix}_sample_count": int(len(targets)),
        f"{prefix}_mae": float(np.mean(np.abs(errors))),
        f"{prefix}_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        f"{prefix}_bias": float(np.mean(errors)),
        f"{prefix}_mae_100": float(np.mean(np.abs(errors)) * 100.0),
        f"{prefix}_rmse_100": float(np.sqrt(np.mean(np.square(errors))) * 100.0),
        f"{prefix}_bias_100": float(np.mean(errors) * 100.0),
        f"{prefix}_pred_min": float(np.min(predictions)),
        f"{prefix}_pred_max": float(np.max(predictions)),
        f"{prefix}_pred_mean": float(np.mean(predictions)),
        f"{prefix}_pred_std": pred_std,
        f"{prefix}_target_min": float(np.min(targets)),
        f"{prefix}_target_max": float(np.max(targets)),
        f"{prefix}_target_mean": float(np.mean(targets)),
        f"{prefix}_target_std": target_std,
        f"{prefix}_std_ratio": ratio,
        f"{prefix}_mode_collapse": bool(pred_std < 0.01),
    }
    corr = _correlations(targets, predictions)
    metrics[f"{prefix}_srcc"] = corr["srcc"]
    metrics[f"{prefix}_plcc"] = corr["plcc"]
    metrics[f"{prefix}_correlation_error"] = corr["correlation_error"]
    return metrics


def _evaluate_mixed(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    loss_fn: tf.keras.losses.Loss,
) -> dict[str, Any]:
    unified_predictions: list[float] = []
    targets: list[float] = []
    dataset_ids_all: list[int] = []
    head_predictions = {name: [] for name in DATASET_TO_ID}
    loss_values: list[float] = []

    for images, batch_targets, dataset_ids in dataset:
        outputs = model(images, training=False)
        batch_targets_np = batch_targets.numpy().reshape(-1).astype("float32")
        batch_ids_np = dataset_ids.numpy().reshape(-1).astype("int32")
        unified_np = outputs["unified"].numpy().reshape(-1).astype("float32")
        per_sample_loss = loss_fn(batch_targets, outputs["unified"]).numpy().reshape(-1)
        loss_values.extend(float(value) for value in per_sample_loss)
        unified_predictions.extend(float(value) for value in unified_np)
        targets.extend(float(value) for value in batch_targets_np)
        dataset_ids_all.extend(int(value) for value in batch_ids_np)
        for dataset_name in DATASET_TO_ID:
            head_predictions[dataset_name].extend(
                float(value)
                for value in outputs[dataset_name].numpy().reshape(-1).astype("float32")
            )

    targets_np = np.asarray(targets, dtype=np.float32)
    unified_np = np.asarray(unified_predictions, dtype=np.float32)
    ids_np = np.asarray(dataset_ids_all, dtype=np.int32)
    metrics = _regression_metrics(targets_np, unified_np, prefix="unified")
    metrics["loss"] = float(np.mean(loss_values)) if loss_values else None
    metrics["dataset_metrics"] = {}
    for dataset_name, dataset_id in DATASET_TO_ID.items():
        mask = ids_np == dataset_id
        head_np = np.asarray(head_predictions[dataset_name], dtype=np.float32)
        metrics["dataset_metrics"][dataset_name] = _regression_metrics(
            targets_np[mask],
            head_np[mask],
            prefix=dataset_name,
        )
    return metrics


def _evaluate_head(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    head_name: str,
    prefix: str,
) -> dict[str, Any]:
    predictions: list[float] = []
    targets: list[float] = []
    for images, batch_targets in dataset:
        outputs = model(images, training=False)
        batch_predictions = outputs[head_name].numpy().reshape(-1).astype("float32")
        predictions.extend(float(value) for value in batch_predictions)
        targets.extend(float(value) for value in batch_targets.numpy().reshape(-1))

    return _regression_metrics(
        np.asarray(targets, dtype=np.float32),
        np.asarray(predictions, dtype=np.float32),
        prefix=prefix,
    )


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


def _safe_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(result):
        return default
    return result


def _secondary_eligible(row: dict[str, Any], baseline_unified_loss: float | None) -> bool:
    mae_100 = row.get("quick_flive_head_mae_100")
    collapse = bool(row.get("quick_flive_head_mode_collapse", True))
    unified_loss = row.get("val_unified_loss")
    mae_ok = mae_100 is not None and float(mae_100) <= 4.5
    collapse_ok = not collapse
    if baseline_unified_loss is None or unified_loss is None:
        unified_stable = False
    else:
        unified_stable = float(unified_loss) <= max(0.05, baseline_unified_loss * 3.0)
    return bool(mae_ok and collapse_ok and unified_stable)


def _candidate_key(
    row: dict[str, Any],
    baseline_unified_loss: float | None,
) -> tuple[int, float]:
    eligible = 1 if _secondary_eligible(row, baseline_unified_loss) else 0
    srcc = _safe_float(row.get("quick_flive_head_srcc"), float("-inf"))
    return (eligible, srcc)


def _row_from_metrics(
    epoch: int,
    phase: str,
    train_metrics: dict[str, Any],
    mixed_metrics: dict[str, Any],
    quick_flive_metrics: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "epoch": int(epoch),
        "phase": phase,
        "train_loss": train_metrics.get("train_loss"),
        "train_regression_loss": train_metrics.get("train_regression_loss"),
        "train_unified_loss": train_metrics.get("train_unified_loss"),
        "train_specific_loss": train_metrics.get("train_specific_loss"),
        "train_rank_loss": train_metrics.get("train_rank_loss"),
        "train_calib_loss": train_metrics.get("train_calib_loss"),
        "val_unified_loss": mixed_metrics.get("loss"),
        "val_unified_mae": mixed_metrics.get("unified_mae"),
        "val_unified_mae_100": mixed_metrics.get("unified_mae_100"),
        "val_unified_rmse": mixed_metrics.get("unified_rmse"),
        "val_unified_srcc": mixed_metrics.get("unified_srcc"),
        "val_unified_plcc": mixed_metrics.get("unified_plcc"),
        "val_unified_bias_100": mixed_metrics.get("unified_bias_100"),
        "val_unified_pred_mean": mixed_metrics.get("unified_pred_mean"),
        "val_unified_pred_std": mixed_metrics.get("unified_pred_std"),
        "val_unified_target_mean": mixed_metrics.get("unified_target_mean"),
        "val_unified_target_std": mixed_metrics.get("unified_target_std"),
        "val_unified_std_ratio": mixed_metrics.get("unified_std_ratio"),
        "val_unified_mode_collapse": mixed_metrics.get("unified_mode_collapse"),
        "quick_flive_head_mae": quick_flive_metrics.get("quick_flive_head_mae"),
        "quick_flive_head_mae_100": quick_flive_metrics.get("quick_flive_head_mae_100"),
        "quick_flive_head_rmse": quick_flive_metrics.get("quick_flive_head_rmse"),
        "quick_flive_head_rmse_100": quick_flive_metrics.get("quick_flive_head_rmse_100"),
        "quick_flive_head_srcc": quick_flive_metrics.get("quick_flive_head_srcc"),
        "quick_flive_head_plcc": quick_flive_metrics.get("quick_flive_head_plcc"),
        "quick_flive_head_bias": quick_flive_metrics.get("quick_flive_head_bias"),
        "quick_flive_head_bias_100": quick_flive_metrics.get("quick_flive_head_bias_100"),
        "quick_flive_head_pred_mean": quick_flive_metrics.get("quick_flive_head_pred_mean"),
        "quick_flive_head_pred_std": quick_flive_metrics.get("quick_flive_head_pred_std"),
        "quick_flive_head_target_mean": quick_flive_metrics.get("quick_flive_head_target_mean"),
        "quick_flive_head_target_std": quick_flive_metrics.get("quick_flive_head_target_std"),
        "quick_flive_head_std_ratio": quick_flive_metrics.get("quick_flive_head_std_ratio"),
        "quick_flive_head_mode_collapse": quick_flive_metrics.get(
            "quick_flive_head_mode_collapse"
        ),
        "quick_flive_head_sample_count": quick_flive_metrics.get(
            "quick_flive_head_sample_count"
        ),
    }
    for dataset_name, dataset_metrics in mixed_metrics.get("dataset_metrics", {}).items():
        row[f"val_{dataset_name}_head_mae_100"] = dataset_metrics.get(
            f"{dataset_name}_mae_100"
        )
        row[f"val_{dataset_name}_head_srcc"] = dataset_metrics.get(f"{dataset_name}_srcc")
        row[f"val_{dataset_name}_head_bias_100"] = dataset_metrics.get(
            f"{dataset_name}_bias_100"
        )
        row[f"val_{dataset_name}_head_std_ratio"] = dataset_metrics.get(
            f"{dataset_name}_std_ratio"
        )
        row[f"val_{dataset_name}_head_sample_count"] = dataset_metrics.get(
            f"{dataset_name}_sample_count"
        )
    return row


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 FLIVE-head ranking fine-tune for TOPIQ-RankCalib v2."
    )
    parser.add_argument("--regression_train_csv", required=True)
    parser.add_argument("--regression_val_csv", required=True)
    parser.add_argument("--pair_csv", required=True)
    parser.add_argument("--flive_pair_source_csv")
    parser.add_argument("--make_pairs", action="store_true")
    parser.add_argument("--target_col", default="normalized_mos")
    parser.add_argument("--dataset_col", default="dataset")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pair_batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--ranking_lambda", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--regression_anchor_lambda", type=float, default=1.0)
    parser.add_argument("--unified_anchor_lambda", type=float, default=1.0)
    parser.add_argument("--specific_anchor_lambda", type=float, default=1.0)
    parser.add_argument("--calib_lambda", type=float, default=0.0)
    parser.add_argument("--min_pair_gap", type=float, default=0.05)
    parser.add_argument("--max_pairs", type=int, default=10000)
    parser.add_argument("--init_weights", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--max_steps_per_epoch", type=int, default=200)
    parser.add_argument("--max_val_samples", type=int, default=1024)
    parser.add_argument("--quick_flive_limit", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect_batch", action="store_true")
    parser.add_argument("--list_layers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.pair_batch_size <= 0:
        raise ValueError("--pair_batch_size must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.tau <= 0.0:
        raise ValueError("--tau must be positive.")
    if args.ranking_lambda < 0.0:
        raise ValueError("--ranking_lambda must be non-negative.")
    if args.regression_anchor_lambda < 0.0:
        raise ValueError("--regression_anchor_lambda must be non-negative.")
    if args.unified_anchor_lambda < 0.0:
        raise ValueError("--unified_anchor_lambda must be non-negative.")
    if args.specific_anchor_lambda < 0.0:
        raise ValueError("--specific_anchor_lambda must be non-negative.")
    if args.calib_lambda < 0.0:
        raise ValueError("--calib_lambda must be non-negative.")
    if args.max_steps_per_epoch == 0:
        raise ValueError("--max_steps_per_epoch must be positive or negative for full epoch.")
    if args.max_val_samples == 0:
        raise ValueError("--max_val_samples must be positive or negative for full validation.")
    if args.quick_flive_limit <= 0:
        raise ValueError("--quick_flive_limit must be positive.")
    if args.make_pairs and not args.flive_pair_source_csv:
        raise ValueError("--flive_pair_source_csv is required with --make_pairs.")

    _set_seed(args.seed)
    _configure_tensorflow()

    if args.list_layers:
        list_topiq_rankcalib_layers(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        return

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_summary = None
    if args.make_pairs:
        pair_summary = make_flive_pairs(
            csv_path=args.flive_pair_source_csv,
            output_csv=args.pair_csv,
            target_col=args.target_col,
            max_pairs=args.max_pairs,
            min_gap=args.min_pair_gap,
            seed=args.seed,
        )
    pair_csv_summary = _pair_csv_summary(args.pair_csv)

    init_weights_path = Path(args.init_weights).expanduser().resolve()
    if not init_weights_path.is_file():
        raise FileNotFoundError(f"Initial weights file not found: {init_weights_path}")

    train_df = _load_manifest(
        args.regression_train_csv,
        target_col=args.target_col,
        dataset_col=args.dataset_col,
    )
    val_max_rows = args.max_val_samples if args.max_val_samples > 0 else None
    val_df = _load_manifest(
        args.regression_val_csv,
        target_col=args.target_col,
        dataset_col=args.dataset_col,
        max_rows=val_max_rows,
    )
    quick_flive_df = _load_manifest(
        args.regression_val_csv,
        target_col=args.target_col,
        dataset_col=args.dataset_col,
    )
    quick_flive_df = (
        quick_flive_df[quick_flive_df["dataset_name"] == "flive"]
        .head(args.quick_flive_limit)
        .reset_index(drop=True)
    )
    if len(quick_flive_df) == 0:
        raise ValueError("No FLIVE rows found in regression validation CSV.")

    train_summary = _manifest_summary(
        train_df,
        csv_path=args.regression_train_csv,
        target_col=args.target_col,
        dataset_col=args.dataset_col,
    )
    val_summary = _manifest_summary(
        val_df,
        csv_path=args.regression_val_csv,
        target_col=args.target_col,
        dataset_col=args.dataset_col,
    )
    quick_flive_summary = _manifest_summary(
        quick_flive_df,
        csv_path=args.regression_val_csv,
        target_col=args.target_col,
        dataset_col=args.dataset_col,
    )
    print("Regression train CSV summary:")
    print(json.dumps(train_summary, indent=2, sort_keys=True))
    print("Regression val CSV summary:")
    print(json.dumps(val_summary, indent=2, sort_keys=True))
    print("Quick FLIVE eval summary:")
    print(json.dumps(quick_flive_summary, indent=2, sort_keys=True))
    print("Pair CSV summary:")
    print(json.dumps(pair_csv_summary, indent=2, sort_keys=True))

    regression_dataset = _make_regression_dataset(
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
    quick_flive_dataset = _make_image_target_dataset(
        quick_flive_df,
        target_col=args.target_col,
        batch_size=args.batch_size,
    )
    pair_dataset = make_pair_dataset(
        args.pair_csv,
        image_size=IMAGE_SIZE,
        batch_size=args.pair_batch_size,
        shuffle=True,
        seed=args.seed,
    )

    if args.inspect_batch:
        print("Regression batch inspection:")
        print(json.dumps(_inspect_regression_batch(regression_dataset), indent=2, sort_keys=True))
        print("Pair batch inspection:")
        print(json.dumps(_inspect_pair_batch(pair_dataset), indent=2, sort_keys=True))
        return

    regression_steps = max(1, int(np.ceil(len(train_df) / args.batch_size)))
    pair_steps = max(1, int(np.ceil(pair_csv_summary["pair_count"] / args.pair_batch_size)))
    full_steps_per_epoch = min(regression_steps, pair_steps)
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(full_steps_per_epoch, args.max_steps_per_epoch)
    else:
        steps_per_epoch = full_steps_per_epoch

    model = build_topiq_rankcalib_v2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        backbone_trainable=not args.freeze_backbone,
    )
    model.load_weights(str(init_weights_path))
    flive_variables = _flive_head_variables(model)
    feature_layers = _feature_summary(model)
    model_summary_path = out_dir / "model_summary.txt"
    _write_model_summary(model_summary_path, model)
    print(
        json.dumps(
            {
                "loaded_init_weights": str(init_weights_path),
                "steps_per_epoch": int(steps_per_epoch),
                "full_steps_per_epoch": int(full_steps_per_epoch),
                "regression_steps": int(regression_steps),
                "pair_steps": int(pair_steps),
                "epochs": int(args.epochs),
                "freeze_backbone": bool(args.freeze_backbone),
                "feature_layers": feature_layers,
                "flive_ranking_variable_count": int(len(flive_variables)),
                "output_names": list(model.output_names),
                "ranking_gradient_scope": "flive head variables only",
            },
            indent=2,
            sort_keys=True,
        )
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_fn = tf.keras.losses.Huber(delta=0.1, reduction="none")
    log_rows: list[dict[str, Any]] = []
    best_weights_path = out_dir / "best.weights.h5"
    best_key = (0, float("-inf"))
    best_epoch = 0
    best_phase = "initial"

    initial_mixed_metrics = _evaluate_mixed(model, val_dataset, loss_fn)
    initial_quick_flive_metrics = _evaluate_head(
        model=model,
        dataset=quick_flive_dataset,
        head_name="flive",
        prefix="quick_flive_head",
    )
    baseline_unified_loss = initial_mixed_metrics.get("loss")
    baseline_unified_loss_float = (
        None if baseline_unified_loss is None else float(baseline_unified_loss)
    )
    initial_row = _row_from_metrics(
        epoch=0,
        phase="initial",
        train_metrics={},
        mixed_metrics=initial_mixed_metrics,
        quick_flive_metrics=initial_quick_flive_metrics,
    )
    initial_row["secondary_eligible"] = _secondary_eligible(
        initial_row,
        baseline_unified_loss_float,
    )
    log_rows.append(initial_row)
    best_key = _candidate_key(initial_row, baseline_unified_loss_float)
    model.save_weights(best_weights_path)
    print("initial summary:", json.dumps(initial_row, sort_keys=True))
    print(f"Saved initial best weights: {best_weights_path}")

    for epoch in range(1, args.epochs + 1):
        regression_iter = iter(regression_dataset.repeat())
        pair_iter = iter(pair_dataset.repeat())
        total_metric = tf.keras.metrics.Mean()
        regression_metric = tf.keras.metrics.Mean()
        unified_metric = tf.keras.metrics.Mean()
        specific_metric = tf.keras.metrics.Mean()
        rank_metric = tf.keras.metrics.Mean()
        calib_metric = tf.keras.metrics.Mean()
        print_every = max(1, steps_per_epoch // 4)

        for step in range(steps_per_epoch):
            reg_images, reg_targets, reg_dataset_ids = next(regression_iter)
            (pair_images_a, pair_images_b), pair_signs = next(pair_iter)
            (
                total_loss,
                total_regression_loss,
                unified_loss,
                specific_loss,
                rank_loss,
                calib_loss,
            ) = _train_step(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                flive_variables=flive_variables,
                reg_images=reg_images,
                reg_targets=reg_targets,
                reg_dataset_ids=reg_dataset_ids,
                pair_images_a=pair_images_a,
                pair_images_b=pair_images_b,
                pair_signs=pair_signs,
                ranking_lambda=args.ranking_lambda,
                tau=args.tau,
                regression_anchor_lambda=args.regression_anchor_lambda,
                unified_anchor_lambda=args.unified_anchor_lambda,
                specific_anchor_lambda=args.specific_anchor_lambda,
                calib_lambda=args.calib_lambda,
            )
            total_metric.update_state(total_loss)
            regression_metric.update_state(total_regression_loss)
            unified_metric.update_state(unified_loss)
            specific_metric.update_state(specific_loss)
            rank_metric.update_state(rank_loss)
            calib_metric.update_state(calib_loss)
            if (step + 1) % print_every == 0 or (step + 1) == steps_per_epoch:
                print(
                    f"epoch {epoch}/{args.epochs} step {step + 1}/{steps_per_epoch} "
                    f"loss={float(total_metric.result().numpy()):.6f} "
                    f"reg={float(regression_metric.result().numpy()):.6f} "
                    f"rank={float(rank_metric.result().numpy()):.6f} "
                    f"unified={float(unified_metric.result().numpy()):.6f} "
                    f"specific={float(specific_metric.result().numpy()):.6f}"
                )

        mixed_metrics = _evaluate_mixed(model, val_dataset, loss_fn)
        quick_flive_metrics = _evaluate_head(
            model=model,
            dataset=quick_flive_dataset,
            head_name="flive",
            prefix="quick_flive_head",
        )
        train_metrics = {
            "train_loss": float(total_metric.result().numpy()),
            "train_regression_loss": float(regression_metric.result().numpy()),
            "train_unified_loss": float(unified_metric.result().numpy()),
            "train_specific_loss": float(specific_metric.result().numpy()),
            "train_rank_loss": float(rank_metric.result().numpy()),
            "train_calib_loss": float(calib_metric.result().numpy()),
        }
        row = _row_from_metrics(
            epoch=epoch,
            phase="train",
            train_metrics=train_metrics,
            mixed_metrics=mixed_metrics,
            quick_flive_metrics=quick_flive_metrics,
        )
        row["secondary_eligible"] = _secondary_eligible(row, baseline_unified_loss_float)
        log_rows.append(row)
        print("epoch summary:", json.dumps(row, sort_keys=True))

        candidate_key = _candidate_key(row, baseline_unified_loss_float)
        if candidate_key > best_key:
            best_key = candidate_key
            best_epoch = epoch
            best_phase = "train"
            model.save_weights(best_weights_path)
            print(f"Saved best weights: {best_weights_path}")

    final_model_path = out_dir / "final_model.keras"
    training_log_path = out_dir / "training_log.csv"
    training_summary_path = out_dir / "training_summary.json"
    model.save(final_model_path)
    _write_training_log(training_log_path, log_rows)
    final_mixed_metrics = _evaluate_mixed(model, val_dataset, loss_fn)
    final_quick_flive_metrics = _evaluate_head(
        model=model,
        dataset=quick_flive_dataset,
        head_name="flive",
        prefix="quick_flive_head",
    )

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
        "best_phase": best_phase,
        "best_key": list(best_key),
        "best_secondary_eligible": bool(best_key[0]),
        "best_selection_note": (
            "Primary key is quick FLIVE-head SRCC. Secondary eligibility requires "
            "FLIVE-head MAE<=4.5 MOS points, no mode collapse, and unified val loss "
            "within the smoke stability guard."
        ),
        "gradient_scope": {
            "ranking_loss": "flive head variables only",
            "regression_loss": "all trainable variables",
        },
        "feature_layers": feature_layers,
        "initial_validation_metrics": initial_mixed_metrics,
        "initial_quick_flive_metrics": initial_quick_flive_metrics,
        "final_validation_metrics": final_mixed_metrics,
        "final_quick_flive_metrics": final_quick_flive_metrics,
        "mode_collapse": bool(
            final_quick_flive_metrics.get("quick_flive_head_mode_collapse", False)
            or final_mixed_metrics.get("unified_mode_collapse", False)
        ),
        "train_csv": train_summary,
        "val_csv": val_summary,
        "quick_flive_csv": quick_flive_summary,
        "pair_csv": pair_csv_summary,
        "pair_generation": pair_summary,
        "training_log_rows": log_rows,
    }
    _write_json(training_summary_path, summary)
    print("Final mixed validation metrics:")
    print(json.dumps(final_mixed_metrics, indent=2, sort_keys=True))
    print("Final quick FLIVE-head metrics:")
    print(json.dumps(final_quick_flive_metrics, indent=2, sort_keys=True))
    print("Saved:", final_model_path)
    print("Saved:", best_weights_path)
    print("Saved:", training_log_path)
    print("Saved:", training_summary_path)
    print("Saved:", model_summary_path)


if __name__ == "__main__":
    main()
