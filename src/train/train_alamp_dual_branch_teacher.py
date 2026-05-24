# A-LAMP 듀얼 브랜치 GCN teacher 프로토타입을 학습한다.
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

from src.datasets.alamp_dual_branch_dataset import (
    DEFAULT_MAX_OBJECTS,
    NODE_FEATURE_DIM,
    label_summary,
    make_dual_branch_dataset,
    prepare_dual_branch_split,
    preprocessing_summary,
)
from src.models.alamp_dual_branch_teacher import (
    MODEL_DESCRIPTION,
    MODEL_VARIANT,
    REPRODUCTION_CLAIM,
    build_alamp_dual_branch_teacher_model,
    get_alamp_dual_branch_teacher_custom_objects,
    summarize_vgg16_trainability,
)


NOTICE = "A-LAMP Dual-Branch GCN teacher prototype, not full A-LAMP reproduction."
DEFAULT_TRAIN_JSONL = "outputs/alamp_external_patch_full_conversion_20260524/alamp_external_patches_train.jsonl"
DEFAULT_VAL_JSONL = "outputs/alamp_external_patch_full_conversion_20260524/alamp_external_patches_val.jsonl"
DEFAULT_TRAIN_GRAPH_JSONL = "outputs/alamp_object_graph_subset_20260511/graphs_conf010/train_graphs_4096.jsonl"
DEFAULT_VAL_GRAPH_JSONL = "outputs/alamp_object_graph_subset_20260511/graphs_conf010/val_graphs_4096.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/alamp_dual_branch_teacher_4096_20260524/train"


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
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
    best_row = min(metric_rows, key=lambda row: row[metric_name]) if mode == "min" else max(
        metric_rows,
        key=lambda row: row[metric_name],
    )
    return {
        "metric": metric_name,
        "value": float(best_row[metric_name]),
        "epoch": int(best_row["epoch"]),
        "mode": mode,
    }


def _count_params(weights: list[tf.Variable]) -> int:
    return int(sum(np.prod(weight.shape) for weight in weights))


def _model_summary(model: tf.keras.Model) -> dict[str, Any]:
    input_shapes = {
        tensor.name.split(":")[0]: [int(dim) if dim is not None else None for dim in tensor.shape]
        for tensor in model.inputs
    }
    return {
        "name": model.name,
        "variant": MODEL_VARIANT,
        "description": MODEL_DESCRIPTION,
        "reproduction_claim": REPRODUCTION_CLAIM,
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
    parser.add_argument("--train_graph_jsonl")
    parser.add_argument("--val_graph_jsonl")
    parser.add_argument("--out_dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_val_samples", type=int)
    parser.add_argument("--backbone_trainable", type=_parse_bool)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--patch_count", type=int)
    parser.add_argument("--patch_size", type=int)
    parser.add_argument("--max_objects", type=int)
    parser.add_argument("--patch_feature_units", type=int)
    parser.add_argument("--gcn_units", type=int)
    parser.add_argument("--fusion_units", type=int)
    parser.add_argument("--dropout_rate", type=float)
    parser.add_argument("--backbone_weights")
    parser.add_argument("--label_threshold", type=float)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _read_yaml(args.config)

    train_jsonl = Path(_resolve(args.train_jsonl, config, "dataset", "train_jsonl", DEFAULT_TRAIN_JSONL))
    val_jsonl = Path(_resolve(args.val_jsonl, config, "dataset", "val_jsonl", DEFAULT_VAL_JSONL))
    train_graph_jsonl = Path(
        _resolve(args.train_graph_jsonl, config, "dataset", "train_graph_jsonl", DEFAULT_TRAIN_GRAPH_JSONL)
    )
    val_graph_jsonl = Path(
        _resolve(args.val_graph_jsonl, config, "dataset", "val_graph_jsonl", DEFAULT_VAL_GRAPH_JSONL)
    )
    out_dir = Path(_resolve(args.out_dir, config, "training", "output_dir", DEFAULT_OUTPUT_DIR))
    epochs = int(_resolve(args.epochs, config, "training", "epochs", 3))
    batch_size = int(_resolve(args.batch_size, config, "training", "batch_size", 4))
    learning_rate = float(_resolve(args.learning_rate, config, "training", "learning_rate", 1.0e-4))
    seed = int(_resolve(args.seed, config, "training", "seed", 42))
    patch_count = int(_resolve(args.patch_count, config, "model", "patch_count", 5))
    patch_size = int(_resolve(args.patch_size, config, "model", "patch_size", 224))
    max_objects = int(_resolve(args.max_objects, config, "model", "max_objects", DEFAULT_MAX_OBJECTS))
    patch_feature_units = int(_resolve(args.patch_feature_units, config, "model", "patch_feature_units", 256))
    gcn_units = int(_resolve(args.gcn_units, config, "model", "gcn_units", 64))
    fusion_units = int(_resolve(args.fusion_units, config, "model", "fusion_units", 256))
    dropout_rate = float(_resolve(args.dropout_rate, config, "model", "dropout_rate", 0.4))
    backbone_weights = _resolve(args.backbone_weights, config, "model", "backbone_weights", "imagenet")
    backbone_trainable = _parse_bool(
        _resolve(args.backbone_trainable, config, "model", "backbone_trainable", False)
    )
    label_threshold = float(_resolve(args.label_threshold, config, "dataset", "label_threshold", 5.0))
    max_train_samples = _optional_int(_resolve(args.max_train_samples, config, "training", "max_train_samples", None))
    max_val_samples = _optional_int(_resolve(args.max_val_samples, config, "training", "max_val_samples", None))

    for path, label in (
        (train_jsonl, "train patch JSONL"),
        (val_jsonl, "val patch JSONL"),
        (train_graph_jsonl, "train graph JSONL"),
        (val_graph_jsonl, "val graph JSONL"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(NOTICE)
    tf_info = _setup_tensorflow(seed=seed)

    train_data = prepare_dual_branch_split(
        split="train",
        patch_jsonl=train_jsonl,
        graph_jsonl=train_graph_jsonl,
        max_samples=max_train_samples,
        label_threshold=label_threshold,
        max_objects=max_objects,
    )
    val_data = prepare_dual_branch_split(
        split="val",
        patch_jsonl=val_jsonl,
        graph_jsonl=val_graph_jsonl,
        max_samples=max_val_samples,
        label_threshold=label_threshold,
        max_objects=max_objects,
    )
    steps_per_epoch = math.ceil(train_data.count / batch_size)
    validation_steps = math.ceil(val_data.count / batch_size)
    train_repeat_enabled = True
    val_repeat_enabled = True

    print(f"Train matched examples: {train_data.count}")
    print(f"Val matched examples: {val_data.count}")
    print(f"Train match summary: {train_data.match_summary}")
    print(f"Val match summary: {val_data.match_summary}")

    train_dataset = make_dual_branch_dataset(
        train_data,
        patch_size=patch_size,
        patch_count=patch_count,
        max_objects=max_objects,
        batch_size=batch_size,
        training=True,
        repeat=train_repeat_enabled,
        shuffle_seed=seed,
    )
    val_dataset = make_dual_branch_dataset(
        val_data,
        patch_size=patch_size,
        patch_count=patch_count,
        max_objects=max_objects,
        batch_size=batch_size,
        training=False,
        repeat=val_repeat_enabled,
        shuffle_seed=seed,
    )

    model = build_alamp_dual_branch_teacher_model(
        patch_count=patch_count,
        patch_size=patch_size,
        max_objects=max_objects,
        node_feature_dim=NODE_FEATURE_DIM,
        backbone_weights=backbone_weights,
        backbone_trainable=backbone_trainable,
        patch_feature_units=patch_feature_units,
        gcn_units=gcn_units,
        fusion_units=fusion_units,
        dropout_rate=dropout_rate,
    )
    _compile_model(model, learning_rate=learning_rate)
    model_info = _model_summary(model)
    print(f"Model input shapes: {model_info['input_shapes']}")
    print(f"Model parameter count: {model_info['parameter_count']}")

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

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=2,
    )
    history_rows = _history_rows(history)
    _write_training_history(out_dir / "training_history.csv", history_rows)

    best_weights_path = out_dir / "best.weights.h5"
    if not best_weights_path.is_file():
        raise RuntimeError(f"Expected best weights were not written: {best_weights_path}")

    final_model_path = out_dir / "final_model.keras"
    model.save(final_model_path)

    sample_inputs, _ = next(iter(val_dataset.take(1)))
    original_predictions = model(sample_inputs, training=False).numpy()
    loaded_model = tf.keras.models.load_model(
        final_model_path,
        compile=False,
        custom_objects=get_alamp_dual_branch_teacher_custom_objects(),
    )
    loaded_predictions = loaded_model(sample_inputs, training=False).numpy()
    save_load_max_abs_diff = float(np.max(np.abs(original_predictions - loaded_predictions)))

    summary = {
        "status": "training_completed",
        "notice": NOTICE,
        "official_full_alamp_reproduction": False,
        "model_variant": MODEL_VARIANT,
        "model_description": MODEL_DESCRIPTION,
        "reproduction_claim": REPRODUCTION_CLAIM,
        "command": _command_for_summary(),
        "config": str(args.config) if args.config else None,
        "out_dir": str(out_dir),
        "dataset": {
            "patch_box_role": "external A-LAMP adaptive patch selections, not ground-truth labels",
            "graph_role": "YOLO object graph subset, not full A-LAMP Layout-Aware ground truth",
            "label_rule": "mean_score > 5.0 -> 1, else 0",
            "label_threshold": float(label_threshold),
            "train_jsonl": str(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "train_graph_jsonl": str(train_graph_jsonl),
            "val_graph_jsonl": str(val_graph_jsonl),
            "preprocessing": preprocessing_summary(),
            "train_match_summary": train_data.match_summary,
            "val_match_summary": val_data.match_summary,
            "train_label_summary": label_summary(train_data),
            "val_label_summary": label_summary(val_data),
            "graph_schema": train_data.graph_schema,
        },
        "training": {
            "epochs": int(epochs),
            "train_count": int(train_data.count),
            "val_count": int(val_data.count),
            "batch_size": int(batch_size),
            "steps_per_epoch": int(steps_per_epoch),
            "validation_steps": int(validation_steps),
            "train_repeat": bool(train_repeat_enabled),
            "val_repeat": bool(val_repeat_enabled),
            "learning_rate": float(learning_rate),
            "backbone_trainable": bool(backbone_trainable),
            "loss": "binary_crossentropy",
            "metrics": ["accuracy", "auc", "precision", "recall"],
            "monitor_metric": "val_auc",
            "monitor_mode": "max",
            "max_train_samples": max_train_samples,
            "max_val_samples": max_val_samples,
            "seed": int(seed),
        },
        "model": model_info,
        "final_metrics": _final_metrics(history_rows),
        "best_val_auc": _best_metric(history_rows, "val_auc", "max"),
        "best_val_loss": _best_metric(history_rows, "val_loss", "min"),
        "save_load_max_abs_diff": save_load_max_abs_diff,
        "tensorflow_info": tf_info,
        "artifacts": {
            "best_weights": str(best_weights_path),
            "final_model": str(final_model_path),
            "training_history": str(out_dir / "training_history.csv"),
            "train_summary_json": str(out_dir / "train_summary.json"),
        },
        "comparison_target": {
            "multipatch_full_test_accuracy": 0.7633,
            "multipatch_full_test_f1": 0.8482,
            "multipatch_full_test_roc_auc": 0.7877,
            "multipatch_full_test_average_precision": 0.8955,
        },
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote A-LAMP Dual-Branch GCN teacher artifacts to {out_dir}")


if __name__ == "__main__":
    main()
