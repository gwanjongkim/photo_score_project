# TOPIQ-lite v1 기술 화질 모델을 스모크 학습한다.
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
import tensorflow as tf

from src.datasets.arp_dataset import inspect_csv, make_arp_iqa_dataset
from src.models.topiq_lite import build_topiq_lite, list_efficientnetv2b0_layers


DEFAULT_OUTPUT_DIR = "outputs/topiq_lite_debug"


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


def _print_json(title: str, payload: dict[str, Any]) -> None:
    print(title)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _limit_batches(dataset: tf.data.Dataset, max_batches: int | None) -> tf.data.Dataset:
    if max_batches is None:
        return dataset
    return dataset.take(max_batches)


def _inspect_batch(dataset: tf.data.Dataset) -> dict[str, Any]:
    for images, targets in dataset.take(1):
        image_values = images.numpy()
        target_values = targets.numpy()
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
        }
    raise RuntimeError("Dataset produced no batches.")


def _history_rows(history: tf.keras.callbacks.History) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_names = list(history.history.keys())
    if not metric_names:
        return rows
    for index in range(len(history.history[metric_names[0]])):
        row = {"epoch": index + 1}
        for metric_name in metric_names:
            row[metric_name] = float(history.history[metric_name][index])
        rows.append(row)
    return rows


def _write_training_log(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _final_metrics(history: tf.keras.callbacks.History) -> dict[str, float]:
    return {
        key: float(values[-1])
        for key, values in history.history.items()
        if values
    }


def _correlations_100(
    targets_100: np.ndarray,
    predictions_100: np.ndarray,
) -> dict[str, float | str | None]:
    try:
        from scipy.stats import pearsonr, spearmanr
    except Exception as exc:
        return {
            "srcc": None,
            "plcc": None,
            "correlation_error": str(exc),
        }
    if len(targets_100) < 2:
        return {
            "srcc": None,
            "plcc": None,
            "correlation_error": "Need at least two samples for correlation.",
        }
    srcc = spearmanr(targets_100, predictions_100).correlation
    plcc = pearsonr(targets_100, predictions_100).statistic
    return {
        "srcc": None if np.isnan(srcc) else float(srcc),
        "plcc": None if np.isnan(plcc) else float(plcc),
        "correlation_error": None,
    }


def _prediction_diagnostics(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    limit: int,
) -> dict[str, Any]:
    predictions: list[float] = []
    targets: list[float] = []
    seen = 0
    for images, batch_targets in dataset:
        remaining = limit - seen
        if remaining <= 0:
            break
        images = images[:remaining]
        batch_targets = batch_targets[:remaining]
        batch_predictions = model.predict(images, verbose=0).reshape(-1)
        predictions.extend(float(value) for value in batch_predictions)
        targets.extend(float(value) for value in batch_targets.numpy().reshape(-1))
        seen += len(batch_predictions)

    pred = np.asarray(predictions, dtype=np.float32)
    target = np.asarray(targets, dtype=np.float32)
    if len(pred) == 0:
        return {"sample_count": 0}

    pred_100 = pred * 100.0
    target_100 = target * 100.0
    pred_std = float(np.std(pred))
    target_std = float(np.std(target))
    ratio = None if target_std == 0.0 else pred_std / target_std
    diagnostics: dict[str, Any] = {
        "sample_count": int(len(pred)),
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
        "pred_mean": float(np.mean(pred)),
        "pred_std": pred_std,
        "target_min": float(np.min(target)),
        "target_max": float(np.max(target)),
        "target_mean": float(np.mean(target)),
        "target_std": target_std,
        "pred_std_over_target_std": ratio,
        "mae_100": float(np.mean(np.abs(pred_100 - target_100))),
        "rmse_100": float(np.sqrt(np.mean(np.square(pred_100 - target_100)))),
        "possible_mode_collapse": bool(pred_std < 0.01),
    }
    diagnostics.update(_correlations_100(target_100, pred_100))
    return diagnostics


def _command_for_summary() -> str:
    command = " ".join([sys.executable, *sys.argv])
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        return f"PYTHONPATH={pythonpath} {command}"
    return command


def _feature_summary(model: tf.keras.Model) -> dict[str, Any]:
    specs = getattr(model, "topiq_feature_specs", {})
    return {
        level: {
            "layer_name": spec.layer_name,
            "shape": list(spec.shape),
        }
        for level, spec in specs.items()
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-train TOPIQ-lite v1.")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--target_col", default="normalized_mos")
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--out_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--backbone", default="efficientnetv2b0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list_layers", action="store_true")
    parser.add_argument("--inspect_batch", action="store_true")
    parser.add_argument("--max_train_batches", type=int)
    parser.add_argument("--max_val_batches", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)
    _configure_tensorflow()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_shape = (args.image_size, args.image_size, 3)
    if args.list_layers:
        list_efficientnetv2b0_layers(input_shape=input_shape)
        return

    train_summary = inspect_csv(args.train_csv, target_col=args.target_col)
    val_summary = inspect_csv(args.val_csv, target_col=args.target_col)
    _print_json("Train CSV summary:", train_summary)
    _print_json("Val CSV summary:", val_summary)

    train_dataset = make_arp_iqa_dataset(
        args.train_csv,
        target_col=args.target_col,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_dataset = make_arp_iqa_dataset(
        args.val_csv,
        target_col=args.target_col,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
    )

    if args.inspect_batch:
        _print_json("Batch inspection:", _inspect_batch(train_dataset))
        return

    train_dataset = _limit_batches(train_dataset, args.max_train_batches)
    val_dataset = _limit_batches(val_dataset, args.max_val_batches)

    model = build_topiq_lite(
        input_shape=input_shape,
        backbone_name=args.backbone,
        weights="imagenet",
        dropout_rate=0.3,
        dense_units=256,
        freeze_backbone=args.freeze_backbone,
    )
    features = _feature_summary(model)
    _print_json("Selected TOPIQ-lite feature layers:", features)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.Huber(delta=0.1),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
        jit_compile=False,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(out_dir / "best.weights.h5"),
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    history_rows = _history_rows(history)
    _write_training_log(out_dir / "training_log.csv", history_rows)
    model.save(out_dir / "final_model.keras")
    model.export(out_dir / "saved_model")

    diagnostics = _prediction_diagnostics(
        model=model,
        dataset=val_dataset,
        limit=128,
    )
    _print_json("Prediction diagnostics:", diagnostics)
    if diagnostics.get("possible_mode_collapse"):
        print("WARNING: possible mode collapse.")

    summary = {
        "command": _command_for_summary(),
        "train_csv": train_summary,
        "val_csv": val_summary,
        "feature_layers": features,
        "final_metrics": _final_metrics(history),
        "prediction_diagnostics": diagnostics,
        "artifacts": {
            "best_weights": str(out_dir / "best.weights.h5"),
            "final_model": str(out_dir / "final_model.keras"),
            "saved_model": str(out_dir / "saved_model"),
            "training_log": str(out_dir / "training_log.csv"),
        },
    }
    _write_json(out_dir / "training_summary.json", summary)
    print("Saved:", out_dir / "final_model.keras")
    print("Saved:", out_dir / "best.weights.h5")
    print("Exported:", out_dir / "saved_model")


if __name__ == "__main__":
    main()
