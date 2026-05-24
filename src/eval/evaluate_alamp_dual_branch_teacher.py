# A-LAMP 듀얼 브랜치 GCN teacher 프로토타입을 평가한다.
from __future__ import annotations

import argparse
import csv
import json
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
    MODEL_VARIANT,
    build_alamp_dual_branch_teacher_model,
    get_alamp_dual_branch_teacher_custom_objects,
)


DEFAULT_TEST_JSONL = "outputs/alamp_external_patch_full_conversion_20260524/alamp_external_patches_test.jsonl"
DEFAULT_TEST_GRAPH_JSONL = "outputs/alamp_object_graph_subset_20260511/graphs_conf010/test_graphs_4096.jsonl"


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


def _binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    labels = y_true.astype(np.int32)
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=np.float64)
    sorted_scores = y_score[order]
    start = 0
    while start < len(sorted_scores):
        stop = start + 1
        while stop < len(sorted_scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    auc = (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return float(auc)


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    labels = y_true.astype(np.int32)
    positive_count = int(np.sum(labels == 1))
    if positive_count == 0:
        return None
    order = np.argsort(-y_score)
    sorted_labels = labels[order]
    cumulative_positive = np.cumsum(sorted_labels)
    ranks = np.arange(1, len(sorted_labels) + 1)
    precision_at_k = cumulative_positive / ranks
    return float(np.sum(precision_at_k * sorted_labels) / positive_count)


def _metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    y_pred = y_score >= 0.5
    truth = y_true >= 0.5
    tp = int(np.sum(y_pred & truth))
    tn = int(np.sum(~y_pred & ~truth))
    fp = int(np.sum(y_pred & ~truth))
    fn = int(np.sum(~y_pred & truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "sample_count": int(len(y_true)),
        "accuracy": float((tp + tn) / max(len(y_true), 1)),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "roc_auc": _binary_roc_auc(y_true, y_score),
        "average_precision": _average_precision(y_true, y_score),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "prediction_summary": {
            "min": float(np.min(y_score)) if len(y_score) else None,
            "max": float(np.max(y_score)) if len(y_score) else None,
            "mean": float(np.mean(y_score)) if len(y_score) else None,
            "std": float(np.std(y_score)) if len(y_score) else None,
            "positive_prediction_ratio_at_0_5": float(np.mean(y_pred)) if len(y_score) else None,
        },
    }


def _predict(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for inputs, batch_labels in dataset:
        batch_predictions = model(inputs, training=False).numpy().reshape(-1)
        predictions.append(batch_predictions.astype(np.float32))
        labels.append(batch_labels.numpy().reshape(-1).astype(np.float32))
    if not predictions:
        raise RuntimeError("Evaluation dataset produced no batches.")
    return np.concatenate(labels, axis=0), np.concatenate(predictions, axis=0)


def _write_predictions(path: Path, y_true: np.ndarray, y_score: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_index", "y_true", "y_score", "y_pred"])
        writer.writeheader()
        for index, (label, score) in enumerate(zip(y_true, y_score)):
            writer.writerow(
                {
                    "row_index": index,
                    "y_true": float(label),
                    "y_score": float(score),
                    "y_pred": int(score >= 0.5),
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--model_path")
    parser.add_argument("--weights_path")
    parser.add_argument("--test_jsonl")
    parser.add_argument("--test_graph_jsonl")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_test_samples", type=int)
    parser.add_argument("--backbone_trainable", type=_parse_bool)
    parser.add_argument("--patch_count", type=int)
    parser.add_argument("--patch_size", type=int)
    parser.add_argument("--max_objects", type=int)
    parser.add_argument("--patch_feature_units", type=int)
    parser.add_argument("--gcn_units", type=int)
    parser.add_argument("--fusion_units", type=int)
    parser.add_argument("--dropout_rate", type=float)
    parser.add_argument("--backbone_weights")
    parser.add_argument("--label_threshold", type=float)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if bool(args.model_path) == bool(args.weights_path):
        raise ValueError("Provide exactly one of --model_path or --weights_path.")

    config = _read_yaml(args.config)
    test_jsonl = Path(_resolve(args.test_jsonl, config, "dataset", "test_jsonl", DEFAULT_TEST_JSONL))
    test_graph_jsonl = Path(
        _resolve(args.test_graph_jsonl, config, "dataset", "test_graph_jsonl", DEFAULT_TEST_GRAPH_JSONL)
    )
    out_dir = Path(args.out_dir)
    batch_size = int(_resolve(args.batch_size, config, "evaluation", "batch_size", 4))
    max_test_samples = _optional_int(_resolve(args.max_test_samples, config, "evaluation", "max_test_samples", None))
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

    for path, label in ((test_jsonl, "test patch JSONL"), (test_graph_jsonl, "test graph JSONL")):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    test_data = prepare_dual_branch_split(
        split="test",
        patch_jsonl=test_jsonl,
        graph_jsonl=test_graph_jsonl,
        max_samples=max_test_samples,
        label_threshold=label_threshold,
        max_objects=max_objects,
    )
    test_dataset = make_dual_branch_dataset(
        test_data,
        patch_size=patch_size,
        patch_count=patch_count,
        max_objects=max_objects,
        batch_size=batch_size,
        training=False,
        repeat=False,
    )

    if args.model_path:
        model = tf.keras.models.load_model(
            args.model_path,
            compile=False,
            custom_objects=get_alamp_dual_branch_teacher_custom_objects(),
        )
        checkpoint_type = "full_model"
        checkpoint_path = args.model_path
    else:
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
        model.load_weights(args.weights_path)
        checkpoint_type = "weights"
        checkpoint_path = args.weights_path

    y_true, y_score = _predict(model, test_dataset)
    metrics = _metrics(y_true, y_score)
    predictions_csv = out_dir / "predictions.csv"
    _write_predictions(predictions_csv, y_true, y_score)

    summary = {
        "status": "evaluation_completed",
        "model_variant": MODEL_VARIANT,
        "official_full_alamp_reproduction": False,
        "checkpoint_type": checkpoint_type,
        "checkpoint_path": checkpoint_path,
        "test_jsonl": str(test_jsonl),
        "test_graph_jsonl": str(test_graph_jsonl),
        "out_dir": str(out_dir),
        "batch_size": int(batch_size),
        "max_test_samples": max_test_samples,
        "dataset": {
            "patch_box_role": "external A-LAMP adaptive patch selections, not ground-truth labels",
            "graph_role": "YOLO object graph subset, not full A-LAMP Layout-Aware ground truth",
            "label_rule": "mean_score > 5.0 -> 1, else 0",
            "test_match_summary": test_data.match_summary,
            "test_label_summary": label_summary(test_data),
            "preprocessing": preprocessing_summary(),
            "graph_schema": test_data.graph_schema,
        },
        "metrics": metrics,
        "artifacts": {
            "predictions_csv": str(predictions_csv),
            "evaluation_summary_json": str(out_dir / "evaluation_summary.json"),
        },
        "comparison_target": {
            "multipatch_full_test_accuracy": 0.7633,
            "multipatch_full_test_f1": 0.8482,
            "multipatch_full_test_roc_auc": 0.7877,
            "multipatch_full_test_average_precision": 0.8955,
        },
    }
    (out_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Wrote evaluation artifacts to {out_dir}")


if __name__ == "__main__":
    main()
