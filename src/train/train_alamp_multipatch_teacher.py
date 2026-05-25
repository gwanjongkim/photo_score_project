# A-LAMP 멀티패치 teacher 기준 모델을 학습한다.
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import yaml

from src.datasets.alamp_external_patch_dataset import (
    PREPROCESSING_MODE,
    label_from_record,
    label_summary,
    load_jsonl_records,
    make_external_patch_dataset,
)
from src.models.alamp_multipatch_teacher import (
    MODEL_DESCRIPTION,
    MODEL_VARIANT,
    REPRODUCTION_CLAIM,
    build_alamp_multipatch_teacher_model,
    get_alamp_multipatch_teacher_custom_objects,
    normalize_patch_projection_mode,
    normalize_unfreeze_from_layer,
    summarize_vgg16_trainability,
)


NOTICE = "A-LAMP Multi-Patch teacher baseline, not full A-LAMP reproduction."
DEFAULT_TRAIN_JSONL = "outputs/alamp_external_patch_full_conversion_20260524/alamp_external_patches_train.jsonl"
DEFAULT_VAL_JSONL = "outputs/alamp_external_patch_full_conversion_20260524/alamp_external_patches_val.jsonl"
DEFAULT_TEST_JSONL = "outputs/alamp_external_patch_full_conversion_20260524/alamp_external_patches_test.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/alamp_multipatch_teacher_ava_20260524/train"


def _read_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return loaded


def _cfg(config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    value = config.get(section, {})
    if isinstance(value, dict) and key in value:
        return value[key]
    return default


def _resolve(value: Any, config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return value if value is not None else _cfg(config, section, key, default)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _compute_balanced_class_weights(
    records: list[dict[str, Any]],
    *,
    label_threshold: float,
) -> dict[int, float]:
    labels = [int(float(label_from_record(record, label_threshold=label_threshold)[0]) >= 0.5) for record in records]
    total = len(labels)
    if total <= 0:
        raise ValueError("Class weights require at least one training record.")
    positive_count = int(sum(labels))
    negative_count = int(total - positive_count)
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError(
            "Class weights require both positive and negative labels in the active training records. "
            f"Got negative={negative_count}, positive={positive_count}."
        )
    num_classes = 2
    return {
        0: float(total / (num_classes * negative_count)),
        1: float(total / (num_classes * positive_count)),
    }


def _setup_tensorflow(seed: int) -> dict[str, Any]:
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.utils.set_random_seed(seed)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            print(f"GPU memory growth setup failed for {gpu}: {exc}", file=sys.stderr)
    return {
        "tensorflow_version": tf.__version__,
        "mixed_precision_policy": str(tf.keras.mixed_precision.global_policy()),
        "visible_gpus": [str(gpu) for gpu in gpus],
    }


def _gpu_memory_info() -> dict[str, Any]:
    devices = tf.config.list_logical_devices("GPU")
    info: dict[str, Any] = {}
    for index, _device in enumerate(devices):
        device_name = f"GPU:{index}"
        try:
            memory_info = tf.config.experimental.get_memory_info(device_name)
        except Exception as exc:
            info[device_name] = {"available": False, "error": str(exc)}
            continue
        info[device_name] = {
            "available": True,
            "current_bytes": int(memory_info.get("current", 0)),
            "peak_bytes": int(memory_info.get("peak", 0)),
        }
    return info


def _print_oom_guidance(exc: BaseException, *, batch_size: int, patch_projection_mode: str) -> None:
    text = str(exc)
    is_resource_exhausted = isinstance(exc, (tf.errors.ResourceExhaustedError, MemoryError))
    if not is_resource_exhausted and "OOM" not in text and "ResourceExhausted" not in text:
        return
    print(
        "Memory exhaustion while training the A-LAMP Multi-Patch teacher. "
        f"Current batch_size={batch_size}, patch_projection_mode={patch_projection_mode}. "
        "For the paper-capacity 4096 variant, retry with batch_size=1, reduce sample limits, "
        "or keep the VGG16 backbone frozen.",
        file=sys.stderr,
    )


def _compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def _history_rows(history: tf.keras.callbacks.History) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_names = list(history.history.keys())
    if not metric_names:
        return rows
    for index in range(len(history.history[metric_names[0]])):
        row: dict[str, Any] = {"epoch": index + 1}
        for metric_name in metric_names:
            row[metric_name] = float(history.history[metric_name][index])
        rows.append(row)
    return rows


def _write_training_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _final_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: float(value)
        for key, value in rows[-1].items()
        if key != "epoch" and isinstance(value, (float, int))
    }


def _best_metric(rows: list[dict[str, Any]], metric_name: str, mode: str) -> dict[str, Any]:
    metric_rows = [row for row in rows if metric_name in row]
    if not metric_rows:
        return {"metric": metric_name, "value": None, "epoch": None, "mode": mode}
    if mode == "min":
        best_row = min(metric_rows, key=lambda row: row[metric_name])
    else:
        best_row = max(metric_rows, key=lambda row: row[metric_name])
    return {
        "metric": metric_name,
        "value": float(best_row[metric_name]),
        "epoch": int(best_row["epoch"]),
        "mode": mode,
    }


def _binary_metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    if y_true.size <= 0:
        return {
            "sample_count": 0,
            "threshold": float(threshold),
            "accuracy_at_0_5": None,
            "balanced_accuracy_at_0_5": None,
            "specificity_at_0_5": None,
            "recall_at_0_5": None,
            "precision_at_0_5": None,
            "f1_at_0_5": None,
            "pred_positive_ratio_at_0_5": None,
            "confusion_matrix_at_0_5": None,
        }

    y_true = y_true.astype(np.int32).reshape(-1)
    y_pred = (y_score.reshape(-1) >= float(threshold)).astype(np.int32)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    accuracy = float((tp + tn) / len(y_true))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else None
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    balanced_accuracy = (
        float((specificity + recall) / 2.0)
        if specificity is not None and recall is not None
        else None
    )
    return {
        "sample_count": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy_at_0_5": accuracy,
        "balanced_accuracy_at_0_5": balanced_accuracy,
        "specificity_at_0_5": specificity,
        "recall_at_0_5": recall,
        "precision_at_0_5": precision,
        "f1_at_0_5": f1,
        "pred_positive_ratio_at_0_5": float(np.mean(y_pred)),
        "actual_positive_ratio": float(np.mean(y_true)),
        "confusion_matrix_at_0_5": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "pred_min": float(np.min(y_score)),
        "pred_max": float(np.max(y_score)),
        "pred_mean": float(np.mean(y_score)),
        "pred_std": float(np.std(y_score)),
    }


def _validation_prediction_metrics(
    model: tf.keras.Model,
    records: list[dict[str, Any]],
    *,
    patch_size: int,
    patch_count: int,
    batch_size: int,
    label_threshold: float,
) -> dict[str, Any]:
    dataset = make_external_patch_dataset(
        records,
        patch_size=patch_size,
        patch_count=patch_count,
        batch_size=batch_size,
        label_threshold=label_threshold,
        training=False,
        repeat=False,
    )
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for patches, batch_labels in dataset:
        batch_scores = model(patches, training=False).numpy().reshape(-1)
        scores.append(batch_scores.astype(np.float32))
        labels.append(batch_labels.numpy().reshape(-1).astype(np.int32))
    if not labels:
        return _binary_metrics_at_threshold(
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.float32),
            threshold=0.5,
        )
    y_true = np.concatenate(labels, axis=0)
    y_score = np.concatenate(scores, axis=0)
    return _binary_metrics_at_threshold(y_true, y_score, threshold=0.5)


def _count_params(weights: list[tf.Variable]) -> int:
    return int(sum(np.prod(weight.shape) for weight in weights))


def _model_summary(model: tf.keras.Model) -> dict[str, Any]:
    input_shapes = {
        tensor.name.split(":")[0]: [int(dim) if dim is not None else None for dim in tensor.shape]
        for tensor in model.inputs
    }
    model_config = getattr(model, "_alamp_multipatch_teacher_config", {})
    return {
        "name": model.name,
        "variant": MODEL_VARIANT,
        "description": MODEL_DESCRIPTION,
        "reproduction_claim": REPRODUCTION_CLAIM,
        "capacity_config": model_config,
        "input_shapes": input_shapes,
        "output_shape": [int(dim) if dim is not None else None for dim in model.output.shape],
        "parameter_count": int(model.count_params()),
        "trainable_parameter_count": _count_params(model.trainable_weights),
        "non_trainable_parameter_count": _count_params(model.non_trainable_weights),
        "vgg16_trainability": summarize_vgg16_trainability(model),
    }


def _command_for_summary() -> str:
    command = " ".join([sys.executable, *sys.argv])
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        return f"PYTHONPATH={pythonpath} {command}"
    return command


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--train_jsonl")
    parser.add_argument("--val_jsonl")
    parser.add_argument("--test_jsonl")
    parser.add_argument("--out_dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_val_samples", type=int)
    parser.add_argument("--max_test_samples", type=int)
    parser.add_argument("--backbone_trainable", type=_parse_bool)
    parser.add_argument("--unfreeze_from_layer")
    parser.add_argument("--use_class_weights", type=_parse_bool)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--patch_count", type=int)
    parser.add_argument("--patch_size", type=int)
    parser.add_argument("--patch_projection_mode", choices=["gap", "flatten_dense"])
    parser.add_argument("--patch_feature_dim", type=int)
    parser.add_argument("--head_units", type=int)
    parser.add_argument("--head_layers", type=int)
    parser.add_argument("--head_dropout", type=float)
    parser.add_argument("--dropout_rate", type=float)
    parser.add_argument("--backbone_weights")
    parser.add_argument("--label_threshold", type=float)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _read_yaml(args.config)

    experiment_name = str(_cfg(config, "experiment", "name", "alamp_multipatch_teacher_ava"))
    train_jsonl = Path(_resolve(args.train_jsonl, config, "dataset", "train_jsonl", DEFAULT_TRAIN_JSONL))
    val_jsonl = Path(_resolve(args.val_jsonl, config, "dataset", "val_jsonl", DEFAULT_VAL_JSONL))
    test_jsonl_value = _resolve(args.test_jsonl, config, "dataset", "test_jsonl", DEFAULT_TEST_JSONL)
    test_jsonl = Path(test_jsonl_value) if test_jsonl_value else None
    out_dir = Path(_resolve(args.out_dir, config, "training", "output_dir", DEFAULT_OUTPUT_DIR))
    epochs = int(_resolve(args.epochs, config, "training", "epochs", 10))
    batch_size = int(_resolve(args.batch_size, config, "training", "batch_size", 4))
    learning_rate = float(_resolve(args.learning_rate, config, "training", "learning_rate", 1.0e-4))
    seed = int(_resolve(args.seed, config, "training", "seed", 42))
    patch_count = int(_resolve(args.patch_count, config, "model", "patch_count", 5))
    patch_size = int(_resolve(args.patch_size, config, "model", "patch_size", 224))
    patch_projection_mode = normalize_patch_projection_mode(
        _resolve(args.patch_projection_mode, config, "model", "patch_projection_mode", "gap")
    )
    patch_feature_dim = int(_resolve(args.patch_feature_dim, config, "model", "patch_feature_dim", 512))
    head_units = int(_resolve(args.head_units, config, "model", "head_units", 256))
    head_layers = int(_resolve(args.head_layers, config, "model", "head_layers", 1))
    head_dropout_value = (
        args.head_dropout
        if args.head_dropout is not None
        else args.dropout_rate
        if args.dropout_rate is not None
        else _cfg(config, "model", "head_dropout", _cfg(config, "model", "dropout_rate", 0.5))
    )
    head_dropout = float(head_dropout_value)
    dropout_rate = head_dropout
    backbone_weights = _resolve(args.backbone_weights, config, "model", "backbone_weights", "imagenet")
    backbone_trainable = _parse_bool(
        _resolve(args.backbone_trainable, config, "model", "backbone_trainable", False)
    )
    unfreeze_from_layer = normalize_unfreeze_from_layer(
        _optional_text(_resolve(args.unfreeze_from_layer, config, "model", "unfreeze_from_layer", None))
    )
    use_class_weights = _parse_bool(
        _resolve(args.use_class_weights, config, "training", "use_class_weights", False)
    )
    label_threshold = float(_resolve(args.label_threshold, config, "dataset", "label_threshold", 5.0))
    max_train_samples = _optional_int(_resolve(args.max_train_samples, config, "training", "max_train_samples", None))
    max_val_samples = _optional_int(_resolve(args.max_val_samples, config, "training", "max_val_samples", None))
    max_test_samples = _optional_int(_resolve(args.max_test_samples, config, "training", "max_test_samples", None))

    if not train_jsonl.is_file():
        raise FileNotFoundError(f"Missing train JSONL: {train_jsonl}")
    if not val_jsonl.is_file():
        raise FileNotFoundError(f"Missing val JSONL: {val_jsonl}")
    if test_jsonl is not None and not test_jsonl.is_file():
        raise FileNotFoundError(f"Missing test JSONL: {test_jsonl}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(NOTICE)

    tf_info = _setup_tensorflow(seed=seed)
    train_records = load_jsonl_records(train_jsonl, max_samples=max_train_samples)
    val_records = load_jsonl_records(val_jsonl, max_samples=max_val_samples)
    test_records = load_jsonl_records(test_jsonl, max_samples=max_test_samples) if test_jsonl is not None else []

    train_label_summary = label_summary(train_records, label_threshold=label_threshold)
    val_label_summary = label_summary(val_records, label_threshold=label_threshold)
    test_label_summary = label_summary(test_records, label_threshold=label_threshold) if test_records else None
    class_weights = (
        _compute_balanced_class_weights(train_records, label_threshold=label_threshold)
        if use_class_weights
        else None
    )

    print(f"Train records loaded: {len(train_records)}")
    print(f"Val records loaded: {len(val_records)}")
    print(f"Train labels: {train_label_summary}")
    print(f"Val labels: {val_label_summary}")
    print(f"Class weights enabled: {use_class_weights}")
    if class_weights is not None:
        print(f"Class weights: {class_weights}")
    print(f"Preprocessing mode: {PREPROCESSING_MODE}")

    steps_per_epoch = math.ceil(len(train_records) / batch_size)
    validation_steps = math.ceil(len(val_records) / batch_size)
    if steps_per_epoch <= 0 or validation_steps <= 0:
        raise ValueError("Training and validation inputs must each provide at least one batch.")

    train_repeat_enabled = True
    val_repeat_enabled = True

    train_dataset = make_external_patch_dataset(
        train_records,
        patch_size=patch_size,
        patch_count=patch_count,
        batch_size=batch_size,
        label_threshold=label_threshold,
        training=True,
        repeat=train_repeat_enabled,
        shuffle_seed=seed,
        class_weights=class_weights,
    )
    val_dataset = make_external_patch_dataset(
        val_records,
        patch_size=patch_size,
        patch_count=patch_count,
        batch_size=batch_size,
        label_threshold=label_threshold,
        training=False,
        repeat=val_repeat_enabled,
        shuffle_seed=seed,
    )

    try:
        model = build_alamp_multipatch_teacher_model(
            patch_count=patch_count,
            patch_size=patch_size,
            backbone_weights=backbone_weights,
            backbone_trainable=backbone_trainable,
            unfreeze_from_layer=unfreeze_from_layer,
            patch_projection_mode=patch_projection_mode,
            patch_feature_dim=patch_feature_dim,
            head_units=head_units,
            head_layers=head_layers,
            head_dropout=head_dropout,
            dropout_rate=dropout_rate,
        )
        _compile_model(model, learning_rate=learning_rate)
    except Exception as exc:
        _print_oom_guidance(exc, batch_size=batch_size, patch_projection_mode=patch_projection_mode)
        raise
    model_info = _model_summary(model)
    model_config = getattr(model, "_alamp_multipatch_teacher_config", {})
    print(f"Model parameter count: {model_info['parameter_count']}")
    print(f"Model trainable parameter count: {model_info['trainable_parameter_count']}")
    print(f"Model non-trainable parameter count: {model_info['non_trainable_parameter_count']}")
    print(f"Patch projection mode: {patch_projection_mode}")
    print(f"Patch feature dim: {model_config.get('patch_feature_dim')}")
    print(f"Head units: {head_units}")
    print(f"Head layers: {head_layers}")
    print(f"Head dropout: {head_dropout}")
    print(f"Backbone trainable: {backbone_trainable}")
    print(f"Unfreeze from layer: {unfreeze_from_layer}")
    print(f"Trainable VGG16 layers: {model_info['vgg16_trainability']['trainable_layers']}")
    gpu_memory_after_model_build = _gpu_memory_info()
    if gpu_memory_after_model_build:
        print(f"GPU memory after model build: {gpu_memory_after_model_build}")

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best.weights.h5"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
    ]

    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=2,
        )
    except Exception as exc:
        _print_oom_guidance(exc, batch_size=batch_size, patch_projection_mode=patch_projection_mode)
        raise
    gpu_memory_after_training = _gpu_memory_info()
    history_rows = _history_rows(history)
    _write_training_history(out_dir / "training_history.csv", history_rows)

    best_weights_path = out_dir / "best.weights.h5"
    if not best_weights_path.is_file():
        raise RuntimeError(f"Expected best weights were not written: {best_weights_path}")

    final_model_path = out_dir / "final_model.keras"
    model.save(final_model_path)

    sample_batch, _ = next(iter(val_dataset.take(1)))
    original_predictions = model(sample_batch, training=False).numpy()
    loaded_model = tf.keras.models.load_model(
        final_model_path,
        compile=False,
        custom_objects=get_alamp_multipatch_teacher_custom_objects(),
    )
    loaded_predictions = loaded_model(sample_batch, training=False).numpy()
    save_load_max_abs_diff = float(np.max(np.abs(original_predictions - loaded_predictions)))
    val_prediction_metrics = _validation_prediction_metrics(
        model,
        val_records,
        patch_size=patch_size,
        patch_count=patch_count,
        batch_size=batch_size,
        label_threshold=label_threshold,
    )

    summary = {
        "status": "training_completed",
        "experiment_name": experiment_name,
        "notice": NOTICE,
        "official_full_alamp_reproduction": False,
        "model_variant": MODEL_VARIANT,
        "model_description": MODEL_DESCRIPTION,
        "reproduction_claim": REPRODUCTION_CLAIM,
        "use_class_weights": bool(use_class_weights),
        "class_weights": class_weights,
        "backbone_trainable": bool(backbone_trainable),
        "unfreeze_from_layer": unfreeze_from_layer,
        "learning_rate": float(learning_rate),
        "patch_projection_mode": patch_projection_mode,
        "patch_feature_dim": int(model_config.get("patch_feature_dim", patch_feature_dim)),
        "patch_feature_dim_requested": int(patch_feature_dim),
        "head_units": int(head_units),
        "head_layers": int(head_layers),
        "head_dropout": float(head_dropout),
        "total_parameter_count": int(model_info["parameter_count"]),
        "trainable_parameter_count": int(model_info["trainable_parameter_count"]),
        "train_count": len(train_records),
        "val_count": len(val_records),
        "steps_per_epoch": int(steps_per_epoch),
        "validation_steps": int(validation_steps),
        "command": _command_for_summary(),
        "config": str(args.config) if args.config else None,
        "train_jsonl": str(train_jsonl),
        "val_jsonl": str(val_jsonl),
        "test_jsonl": str(test_jsonl) if test_jsonl is not None else None,
        "out_dir": str(out_dir),
        "dataset": {
            "patch_box_role": "external A-LAMP adaptive patch selections, not ground-truth labels",
            "label_rule": "mean_score > 5.0 -> 1, else 0",
            "label_threshold": float(label_threshold),
            "preprocessing_mode": PREPROCESSING_MODE,
            "patch_shape": [patch_count, patch_size, patch_size, 3],
            "train_records_loaded": len(train_records),
            "val_records_loaded": len(val_records),
            "test_records_loaded": len(test_records),
            "train_label_summary": train_label_summary,
            "val_label_summary": val_label_summary,
            "test_label_summary": test_label_summary,
        },
        "training": {
            "epochs": int(epochs),
            "train_count": len(train_records),
            "val_count": len(val_records),
            "batch_size": int(batch_size),
            "steps_per_epoch": int(steps_per_epoch),
            "validation_steps": int(validation_steps),
            "train_repeat": bool(train_repeat_enabled),
            "val_repeat": bool(val_repeat_enabled),
            "learning_rate": float(learning_rate),
            "backbone_trainable": bool(backbone_trainable),
            "unfreeze_from_layer": unfreeze_from_layer,
            "patch_projection_mode": patch_projection_mode,
            "patch_feature_dim": int(model_config.get("patch_feature_dim", patch_feature_dim)),
            "patch_feature_dim_requested": int(patch_feature_dim),
            "head_units": int(head_units),
            "head_layers": int(head_layers),
            "head_dropout": float(head_dropout),
            "use_class_weights": bool(use_class_weights),
            "class_weights": class_weights,
            "loss": "binary_crossentropy",
            "metrics": ["accuracy", "auc", "precision", "recall"],
            "monitor_metric": "val_auc",
            "monitor_mode": "max",
            "max_train_samples": max_train_samples,
            "max_val_samples": max_val_samples,
            "max_test_samples": max_test_samples,
            "seed": int(seed),
        },
        "model": model_info,
        "final_metrics": _final_metrics(history_rows),
        "validation_prediction_metrics": val_prediction_metrics,
        "val_balanced_accuracy_at_0_5": val_prediction_metrics["balanced_accuracy_at_0_5"],
        "val_specificity_at_0_5": val_prediction_metrics["specificity_at_0_5"],
        "val_pred_positive_ratio_at_0_5": val_prediction_metrics["pred_positive_ratio_at_0_5"],
        "best_val_auc": _best_metric(history_rows, "val_auc", "max"),
        "best_val_loss": _best_metric(history_rows, "val_loss", "min"),
        "save_load_max_abs_diff": save_load_max_abs_diff,
        "tensorflow_info": tf_info,
        "gpu_memory_after_model_build": gpu_memory_after_model_build,
        "gpu_memory_after_training": gpu_memory_after_training,
        "artifacts": {
            "best_weights": str(best_weights_path),
            "final_model": str(final_model_path),
            "training_history": str(out_dir / "training_history.csv"),
            "train_summary_json": str(out_dir / "train_summary.json"),
        },
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote A-LAMP Multi-Patch teacher baseline artifacts to {out_dir}")


if __name__ == "__main__":
    main()
