# RGNet 논문 지향 AVA 이진 분류 모델을 평가한다.
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from src.models.rgnet_paper_ava import MODEL_VARIANT, get_rgnet_paper_ava_custom_objects


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


def _setup_tensorflow() -> dict[str, object]:
    tf.keras.mixed_precision.set_global_policy("float32")
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            print("memory growth setup failed:", exc)
    return {
        "visible_gpus": [str(gpu) for gpu in gpus],
        "mixed_precision_policy": str(tf.keras.mixed_precision.global_policy()),
        "tensorflow_version": tf.__version__,
    }


def _resolve_image_paths(frame: pd.DataFrame, image_col: str, image_dir: str | None) -> pd.Series:
    image_root = Path(image_dir) if image_dir else None

    def _resolve_path(value: object) -> str:
        raw = Path(str(value))
        if raw.is_absolute() or raw.exists():
            return str(raw)
        if image_root is not None:
            candidate = image_root / raw.name
            if candidate.exists():
                return str(candidate)
        return str(raw)

    return frame[image_col].map(_resolve_path)


def _load_frame(
    csv_path: str,
    image_col: str,
    score_col: str,
    image_dir: str | None,
    label_threshold: float,
    max_samples: int | None,
) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required = {image_col, score_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")
    frame = frame.dropna(subset=[image_col, score_col]).reset_index(drop=True)
    if max_samples is not None:
        frame = frame.head(int(max_samples)).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{csv_path} produced an empty frame")
    frame = frame.copy()
    frame["_resolved_image_path"] = _resolve_image_paths(frame, image_col, image_dir)
    frame["_label"] = (frame[score_col].astype("float32") >= float(label_threshold)).astype("int32")
    return frame


def _decode_image(path: tf.Tensor, image_size: int) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return tf.image.resize(image, [image_size, image_size])


def _make_dataset(frame: pd.DataFrame, image_size: int, batch_size: int) -> tf.data.Dataset:
    paths = frame["_resolved_image_path"].astype(str).to_numpy()
    labels = frame["_label"].astype("float32").to_numpy().reshape(-1, 1)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return _decode_image(path, image_size=image_size), label

    return dataset.map(_map, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray, curve: str) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    metric = tf.keras.metrics.AUC(curve=curve)
    metric.update_state(y_true.reshape(-1, 1), y_prob.reshape(-1, 1))
    return float(metric.result().numpy())


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float32).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    clipped = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
    bce = -np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1.0 - clipped))
    return {
        "num_samples": int(y_true.size),
        "positive_count": int(np.sum(y_true == 1)),
        "negative_count": int(np.sum(y_true == 0)),
        "prediction_positive_count": int(np.sum(y_pred == 1)),
        "prediction_negative_count": int(np.sum(y_pred == 0)),
        "prediction_prob_min": float(np.min(y_prob)),
        "prediction_prob_max": float(np.max(y_prob)),
        "accuracy": float(np.mean(y_pred == y_true)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": _safe_auc(y_true, y_prob, "ROC"),
        "average_precision": _safe_auc(y_true, y_prob, "PR"),
        "bce_loss": float(bce),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def evaluate_split(
    model: tf.keras.Model,
    split_name: str,
    csv_path: str,
    image_col: str,
    score_col: str,
    image_dir: str | None,
    label_threshold: float,
    image_size: int,
    batch_size: int,
    max_samples: int | None,
    output_dir: Path,
) -> dict[str, object]:
    frame = _load_frame(csv_path, image_col, score_col, image_dir, label_threshold, max_samples)
    dataset = _make_dataset(frame, image_size, batch_size)
    started = time.perf_counter()
    predictions = np.asarray(model.predict(dataset, verbose=1), dtype=np.float32).reshape(-1)[: len(frame)]
    elapsed = time.perf_counter() - started
    labels = frame["_label"].astype("int32").to_numpy()
    metrics = classification_metrics(labels, predictions)
    metrics.update(
        {
            "split": split_name,
            "csv": csv_path,
            "score_col": score_col,
            "image_col": image_col,
            "label_rule": f"label = 1 if {score_col} >= {label_threshold} else 0",
            "elapsed_seconds": float(elapsed),
            "seconds_per_image": float(elapsed / max(1, len(frame))),
        }
    )

    predictions_path = output_dir / f"{split_name}_predictions.csv"
    out_frame = pd.DataFrame(
        {
            "image_path": frame[image_col].astype(str),
            score_col: frame[score_col].astype("float32"),
            "label": labels,
            "prediction_prob": predictions,
            "prediction_label": (predictions >= 0.5).astype(np.int32),
        }
    )
    if score_col != "mean_score":
        out_frame = out_frame.rename(columns={score_col: "mean_score"})
    out_frame.to_csv(predictions_path, index=False)
    metrics["predictions_path"] = str(predictions_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RGNet-paper AVA binary classification outputs.")
    parser.add_argument("--config")
    parser.add_argument("--model_path")
    parser.add_argument("--test_csv")
    parser.add_argument("--val_csv")
    parser.add_argument("--output_dir")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_test_samples", type=int)
    parser.add_argument("--max_val_samples", type=int)
    parser.add_argument("--image_dir")
    parser.add_argument("--image_col")
    parser.add_argument("--score_col")
    parser.add_argument("--label_threshold", type=float)
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--model_name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _read_yaml(args.config)

    model_path = _resolve(args.model_path, config, "evaluation", "model_path", None)
    if model_path is None:
        raise ValueError("--model_path is required")
    test_csv = _resolve(args.test_csv, config, "data", "test_csv", "data/processed/ava/test.csv")
    val_csv = _resolve(args.val_csv, config, "data", "val_csv", "data/processed/ava/val.csv")
    image_dir = _resolve(args.image_dir, config, "data", "image_dir", "data/raw/ava/images")
    image_col = _resolve(args.image_col, config, "data", "image_col", "image_path")
    score_col = _resolve(args.score_col, config, "data", "score_col", "mean_score")
    label_threshold = float(_resolve(args.label_threshold, config, "data", "label_threshold", 5.0))
    image_size = int(_resolve(args.image_size, config, "model", "image_size", 256))
    batch_size = int(_resolve(args.batch_size, config, "evaluation", "batch_size", 16))
    output_dir = Path(_resolve(args.output_dir, config, "evaluation", "output_dir", "outputs/rgnet_paper_ava_classification_20260510/eval/mean"))
    model_name = _resolve(args.model_name, config, "evaluation", "model_name", "rgnet_paper_ava_classification")
    max_test_samples = _resolve(args.max_test_samples, config, "evaluation", "max_test_samples", None)
    max_val_samples = _resolve(args.max_val_samples, config, "evaluation", "max_val_samples", None)

    output_dir.mkdir(parents=True, exist_ok=True)
    tf_info = _setup_tensorflow()
    model = tf.keras.models.load_model(
        str(model_path),
        compile=False,
        safe_mode=False,
        custom_objects=get_rgnet_paper_ava_custom_objects(),
    )

    split_metrics = {
        "test": evaluate_split(
            model,
            "test",
            str(test_csv),
            image_col,
            score_col,
            image_dir,
            label_threshold,
            image_size,
            batch_size,
            max_test_samples,
            output_dir,
        )
    }
    if val_csv:
        split_metrics["val"] = evaluate_split(
            model,
            "val",
            str(val_csv),
            image_col,
            score_col,
            image_dir,
            label_threshold,
            image_size,
            batch_size,
            max_val_samples,
            output_dir,
        )

    summary = {
        "created_at_local": datetime.now().astimezone().isoformat(),
        "model_variant": MODEL_VARIANT,
        "official_reproduction": False,
        "model_name": model_name,
        "model_path": str(model_path),
        "image_size": image_size,
        "batch_size": batch_size,
        "tensorflow": tf_info,
        "splits": split_metrics,
    }
    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"evaluation_summary": str(summary_path), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
