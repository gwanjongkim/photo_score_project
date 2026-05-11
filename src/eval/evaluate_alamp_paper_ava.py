# A-LAMP 논문 지향 AVA 이진 분류 모델을 평가한다.
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
from PIL import Image

from src.datasets.native_size_dataset import prepare_alamp_inputs
from src.models.alamp_paper_ava import (
    MODEL_DESCRIPTION,
    MODEL_VARIANT,
    STYLE_DESCRIPTION,
    get_alamp_paper_ava_custom_objects,
)


LABEL_RULES = {"paper_strict", "project_compatible"}


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


def _label_rule_text(score_col: str, label_threshold: float, label_rule: str) -> str:
    if label_rule == "paper_strict":
        return f"label = 1 if {score_col} > {label_threshold} else 0"
    if label_rule == "project_compatible":
        return f"label = 1 if {score_col} >= {label_threshold} else 0"
    raise ValueError(f"Unsupported label_rule: {label_rule}")


def _labels_from_scores(scores: pd.Series, label_threshold: float, label_rule: str) -> pd.Series:
    if label_rule == "paper_strict":
        return (scores.astype("float32") > float(label_threshold)).astype("int32")
    if label_rule == "project_compatible":
        return (scores.astype("float32") >= float(label_threshold)).astype("int32")
    raise ValueError(f"Unsupported label_rule: {label_rule}")


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


def _verify_decodable_image(path: str) -> None:
    with Image.open(path) as image:
        image.convert("RGB").load()


def _filter_decodable_images(frame: pd.DataFrame, image_col: str) -> pd.DataFrame:
    kept_indices: list[int] = []
    skipped: list[dict[str, str]] = []
    for index, row in frame.iterrows():
        resolved_path = str(row["_resolved_image_path"])
        try:
            _verify_decodable_image(resolved_path)
        except Exception as exc:
            skipped.append(
                {
                    "row_index": str(index),
                    "image_path": str(row.get(image_col, "")),
                    "resolved_image_path": resolved_path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        kept_indices.append(index)
    filtered = frame.loc[kept_indices].reset_index(drop=True)
    filtered.attrs["skipped_images"] = skipped
    return filtered


def _load_frame(
    csv_path: str,
    image_col: str,
    score_col: str,
    image_dir: str | None,
    label_threshold: float,
    label_rule: str,
    max_samples: int | None,
) -> pd.DataFrame:
    if label_rule not in LABEL_RULES:
        raise ValueError(f"Unsupported label_rule: {label_rule}")
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
    frame["_label"] = _labels_from_scores(frame[score_col], label_threshold, label_rule)
    frame = _filter_decodable_images(frame, image_col=image_col)
    if frame.empty:
        raise ValueError(f"{csv_path} produced an empty frame after image validation")
    return frame


def _decode_image(path: tf.Tensor) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image.set_shape([None, None, 3])
    return image


def _layout_features(boxes: tf.Tensor, proposal_scores: tf.Tensor) -> tf.Tensor:
    boxes = tf.cast(boxes, tf.float32)
    proposal_scores = tf.cast(proposal_scores, tf.float32)
    y1, x1, y2, x2 = tf.unstack(boxes, axis=-1)
    height = tf.maximum(y2 - y1, 0.0)
    width = tf.maximum(x2 - x1, 0.0)
    center_y = y1 + height * 0.5
    center_x = x1 + width * 0.5
    area = height * width
    return tf.stack(
        [y1, x1, y2, x2, center_y, center_x, height, width, area, proposal_scores],
        axis=-1,
    )


def _make_dataset(
    frame: pd.DataFrame,
    variant: str,
    patch_size: int,
    global_size: int,
    patch_count: int,
    batch_size: int,
) -> tf.data.Dataset:
    paths = frame["_resolved_image_path"].astype(str).to_numpy()
    labels = frame["_label"].astype("float32").to_numpy().reshape(-1, 1)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map(path: tf.Tensor, label: tf.Tensor) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
        image = _decode_image(path)
        global_view, patches, boxes, proposal_scores = prepare_alamp_inputs(
            image=image,
            global_size=global_size,
            patch_size=patch_size,
            num_patches=patch_count,
        )
        inputs: dict[str, tf.Tensor] = {"patches": patches}
        if variant == "v0_b":
            inputs["global_view"] = global_view
            inputs["layout_features"] = _layout_features(boxes, proposal_scores)
        return inputs, label

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
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
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
        "specificity": float(specificity),
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
    variant: str,
    image_col: str,
    score_col: str,
    image_dir: str | None,
    label_threshold: float,
    label_rule: str,
    patch_size: int,
    global_size: int,
    patch_count: int,
    batch_size: int,
    max_samples: int | None,
    output_dir: Path,
) -> dict[str, object]:
    frame = _load_frame(csv_path, image_col, score_col, image_dir, label_threshold, label_rule, max_samples)
    dataset = _make_dataset(frame, variant, patch_size, global_size, patch_count, batch_size)
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
            "label_rule": label_rule,
            "label_rule_text": _label_rule_text(score_col, label_threshold, label_rule),
            "elapsed_seconds": float(elapsed),
            "seconds_per_image": float(elapsed / max(1, len(frame))),
            "skipped_image_count": int(len(frame.attrs.get("skipped_images", []))),
            "skipped_image_examples": frame.attrs.get("skipped_images", [])[:20],
        }
    )

    predictions_path = output_dir / f"{split_name}_predictions.csv"
    out_frame = pd.DataFrame(
        {
            "image_path": frame[image_col].astype(str),
            "mean_score": frame[score_col].astype("float32"),
            "label": labels,
            "prediction_prob": predictions,
            "prediction_label": (predictions >= 0.5).astype(np.int32),
        }
    )
    out_frame.to_csv(predictions_path, index=False)
    metrics["predictions_path"] = str(predictions_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate A-LAMP-paper-AVA-v0 binary classification outputs.")
    parser.add_argument("--config")
    parser.add_argument("--model_path")
    parser.add_argument("--test_csv")
    parser.add_argument("--val_csv")
    parser.add_argument("--output_dir")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_test_samples", type=int)
    parser.add_argument("--max_val_samples", type=int)
    parser.add_argument("--variant")
    parser.add_argument("--label_rule")
    parser.add_argument("--image_dir")
    parser.add_argument("--image_col")
    parser.add_argument("--score_col")
    parser.add_argument("--label_threshold", type=float)
    parser.add_argument("--patch_size", type=int)
    parser.add_argument("--global_size", type=int)
    parser.add_argument("--patch_count", type=int)
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
    label_rule = str(_resolve(args.label_rule, config, "data", "label_rule", "paper_strict"))
    variant = str(_resolve(args.variant, config, "model", "variant", "v0_a")).lower()
    patch_size = int(_resolve(args.patch_size, config, "model", "patch_size", 224))
    global_size = int(_resolve(args.global_size, config, "model", "global_size", 384))
    patch_count = int(_resolve(args.patch_count, config, "model", "patch_count", 5))
    batch_size = int(_resolve(args.batch_size, config, "evaluation", "batch_size", 4))
    output_dir = Path(_resolve(args.output_dir, config, "evaluation", "output_dir", "outputs/alamp_paper_ava_classification_20260511/mid_eval/v0_a"))
    model_name = _resolve(args.model_name, config, "evaluation", "model_name", "alamp_paper_ava_v0")
    max_test_samples = _resolve(args.max_test_samples, config, "evaluation", "max_test_samples", None)
    max_val_samples = _resolve(args.max_val_samples, config, "evaluation", "max_val_samples", None)

    output_dir.mkdir(parents=True, exist_ok=True)
    tf_info = _setup_tensorflow()
    model = tf.keras.models.load_model(
        str(model_path),
        compile=False,
        safe_mode=False,
        custom_objects=get_alamp_paper_ava_custom_objects(),
    )

    split_metrics = {
        "test": evaluate_split(
            model,
            "test",
            str(test_csv),
            variant,
            image_col,
            score_col,
            image_dir,
            label_threshold,
            label_rule,
            patch_size,
            global_size,
            patch_count,
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
            variant,
            image_col,
            score_col,
            image_dir,
            label_threshold,
            label_rule,
            patch_size,
            global_size,
            patch_count,
            batch_size,
            max_val_samples,
            output_dir,
        )

    summary = {
        "created_at_local": datetime.now().astimezone().isoformat(),
        "model_variant": MODEL_VARIANT,
        "model_variant_full": f"{MODEL_VARIANT}-{variant.replace('_', '-')}",
        "description": MODEL_DESCRIPTION,
        "style_description": STYLE_DESCRIPTION,
        "official_reproduction": False,
        "paper_comparability_note": (
            "A-LAMP-paper-oriented approximation. Exact object/global attribute graph "
            "is not implemented in v0, exact saliency-map pipeline is missing locally, "
            "and official author weights are not used."
        ),
        "model_name": model_name,
        "model_path": str(model_path),
        "variant": variant,
        "label_rule": label_rule,
        "label_rule_text": _label_rule_text(score_col, label_threshold, label_rule),
        "patch_size": patch_size,
        "global_size": global_size,
        "patch_count": patch_count,
        "batch_size": batch_size,
        "tensorflow": tf_info,
        "splits": split_metrics,
    }
    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"evaluation_summary": str(summary_path), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
