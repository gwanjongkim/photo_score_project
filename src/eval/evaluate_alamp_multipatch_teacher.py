# A-LAMP 멀티패치 teacher 기준 모델을 AVA 테스트 JSONL로 평가한다.
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.datasets.alamp_external_patch_dataset import (
    PREPROCESSING_MODE,
    label_from_record,
    load_jsonl_records,
    load_patch_tensor,
    resolve_image_path,
)
from src.models.alamp_multipatch_teacher import (
    MODEL_DESCRIPTION,
    MODEL_VARIANT,
    REPRODUCTION_CLAIM,
    build_alamp_multipatch_teacher_model,
    get_alamp_multipatch_teacher_custom_objects,
)


NOTICE = "A-LAMP Multi-Patch teacher baseline, not full A-LAMP reproduction."
DEFAULT_TEST_JSONL = "outputs/alamp_external_patch_full_conversion_20260524/alamp_external_patches_test.jsonl"


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
    raise ValueError(f"Expected boolean value, got {value!r}")


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


def _command_for_summary() -> str:
    command = " ".join([sys.executable, *sys.argv])
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        return f"PYTHONPATH={pythonpath} {command}"
    return command


def _load_model_from_args(
    *,
    model_path: str | None,
    weights_path: str | None,
    config: dict[str, Any],
) -> tuple[tf.keras.Model, str, dict[str, Any]]:
    if model_path and weights_path:
        raise ValueError("Provide only one of --model_path or --weights_path.")
    if not model_path and not weights_path:
        raise ValueError("One of --model_path or --weights_path is required.")

    if model_path:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=get_alamp_multipatch_teacher_custom_objects(),
        )
        return model, "full_model", {"model_path": str(model_path), "weights_path": None}

    patch_count = int(_cfg(config, "model", "patch_count", 5))
    patch_size = int(_cfg(config, "model", "patch_size", 224))
    backbone_trainable = _parse_bool(_cfg(config, "model", "backbone_trainable", False))
    head_units = int(_cfg(config, "model", "head_units", 256))
    dropout_rate = float(_cfg(config, "model", "dropout_rate", 0.5))

    # The weights checkpoint fills the VGG16 backbone, so avoid any ImageNet download during eval.
    model = build_alamp_multipatch_teacher_model(
        patch_count=patch_count,
        patch_size=patch_size,
        backbone_weights=None,
        backbone_trainable=backbone_trainable,
        head_units=head_units,
        dropout_rate=dropout_rate,
    )
    model.load_weights(str(weights_path))
    return model, "weights_only", {"model_path": None, "weights_path": str(weights_path)}


def _record_image_path(record: dict[str, Any]) -> str:
    value = record.get("image_path") or record.get("resolved_image_path")
    return "" if value is None else str(value)


def _record_image_id(record: dict[str, Any], image_path: Path | None) -> str:
    value = record.get("image_id")
    if value is not None:
        return str(value)
    if image_path is not None:
        return image_path.stem
    return ""


def _predict_records(
    *,
    model: tf.keras.Model,
    records: list[dict[str, Any]],
    repo_root: Path,
    patch_size: int,
    patch_count: int,
    batch_size: int,
    label_threshold: float,
    threshold: float,
) -> tuple[list[dict[str, Any]], int, float]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    batch_patches: list[np.ndarray] = []
    batch_rows: list[dict[str, Any]] = []
    start_time = time.time()

    def flush_batch() -> None:
        if not batch_patches:
            return
        scores = model(np.stack(batch_patches, axis=0), training=False).numpy().reshape(-1)
        for row, score in zip(batch_rows, scores):
            y_score = float(score)
            row["y_score"] = y_score
            row["y_pred"] = int(y_score >= threshold)
            rows.append(row)
        batch_patches.clear()
        batch_rows.clear()

    for index, record in enumerate(records, start=1):
        image_path = resolve_image_path(record, repo_root=repo_root)
        if image_path is None:
            skipped += 1
            logging.warning("Skipping record with missing image path: %s", _record_image_path(record))
            continue
        try:
            patches = load_patch_tensor(
                record,
                image_path=image_path,
                patch_size=patch_size,
                patch_count=patch_count,
            )
            label = int(label_from_record(record, label_threshold=label_threshold)[0])
        except Exception as exc:
            skipped += 1
            logging.warning("Skipping %s because patch loading failed: %s", image_path, exc)
            continue

        batch_patches.append(patches)
        batch_rows.append(
            {
                "image_id": _record_image_id(record, image_path),
                "image_path": _record_image_path(record),
                "y_true": label,
                "mean_score": record.get("mean_score", ""),
            }
        )
        if len(batch_patches) >= batch_size:
            flush_batch()
        if index % 1000 == 0:
            print(f"Processed {index} records, evaluated {len(rows)} samples, skipped {skipped}.")

    flush_batch()
    return rows, skipped, time.time() - start_time


def _optional_score(metric_name: str, y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        logging.warning("%s is undefined because only one class is present.", metric_name)
        return None
    if metric_name == "roc_auc":
        return float(roc_auc_score(y_true, y_score))
    if metric_name == "average_precision":
        return float(average_precision_score(y_true, y_score))
    raise ValueError(f"Unsupported optional score: {metric_name}")


def _compute_summary_metrics(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No usable samples were evaluated.")

    y_true = np.asarray([row["y_true"] for row in rows], dtype=np.int32)
    y_score = np.asarray([row["y_score"] for row in rows], dtype=np.float32)
    y_pred = np.asarray([row["y_pred"] for row in rows], dtype=np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    positive_count = int(np.sum(y_true))
    sample_count = int(len(y_true))

    return {
        "sample_count": sample_count,
        "positive_count": positive_count,
        "positive_ratio": float(positive_count / sample_count),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _optional_score("roc_auc", y_true, y_score),
        "average_precision": _optional_score("average_precision", y_true, y_score),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "pred_min": float(np.min(y_score)),
        "pred_max": float(np.max(y_score)),
        "pred_mean": float(np.mean(y_score)),
        "pred_std": float(np.std(y_score)),
    }


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["image_id", "image_path", "y_true", "y_score", "y_pred", "mean_score"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _judgment_text(accuracy: float) -> str:
    if accuracy >= 0.820:
        return (
            "Accuracy is near the paper A-LAMP target, so this is paper-target-adjacent, "
            "but it is still not full A-LAMP without the Layout-Aware subnet."
        )
    if accuracy > 0.80:
        return "Accuracy is above 0.80, so this is a strong teacher candidate."
    if accuracy > 0.75:
        return "Accuracy is above 0.75, so this is a meaningful teacher candidate."
    return "Accuracy is at or below 0.75, so this is not a clearly meaningful teacher candidate yet."


def _format_metric(value: float | None) -> str:
    if value is None:
        return "undefined"
    return f"{value:.6f}"


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    cm = summary["confusion_matrix"]
    checkpoint_value = summary.get("weights_path") or summary.get("model_path")
    report = "\n".join(
        [
            "# A-LAMP Multi-Patch Teacher Test Evaluation",
            "",
            "## 1. Summary",
            "",
            f"- Notice: {NOTICE}",
            f"- Sample count: {summary['sample_count']}",
            f"- Positive ratio: {_format_metric(summary['positive_ratio'])}",
            f"- Accuracy: {_format_metric(summary['accuracy'])}",
            f"- F1 / F-measure: {_format_metric(summary['f1'])}",
            "",
            "## 2. Model Source",
            "",
            f"- Checkpoint type: {summary['checkpoint_type']}",
            f"- Checkpoint: {checkpoint_value}",
            f"- Model variant: {MODEL_VARIANT}",
            f"- Model description: {MODEL_DESCRIPTION}",
            f"- Reproduction claim: {REPRODUCTION_CLAIM}",
            "",
            "## 3. Test Dataset",
            "",
            f"- JSONL: {summary['test_jsonl']}",
            "- Patch inputs: 5 external adaptive patch selections per image, resized to 224x224 RGB.",
            f"- Preprocessing: {summary['preprocessing_mode']}",
            "- Label rule: mean_score > 5.0 -> 1, else 0.",
            "- Patch boxes are external adaptive patch selections, not ground-truth labels.",
            "",
            "## 4. Metrics",
            "",
            f"- Accuracy: {_format_metric(summary['accuracy'])}",
            f"- F1 / F-measure: {_format_metric(summary['f1'])}",
            f"- Precision: {_format_metric(summary['precision'])}",
            f"- Recall: {_format_metric(summary['recall'])}",
            f"- ROC-AUC: {_format_metric(summary['roc_auc'])}",
            f"- Average Precision: {_format_metric(summary['average_precision'])}",
            f"- Prediction min/max/mean/std: {_format_metric(summary['pred_min'])} / {_format_metric(summary['pred_max'])} / {_format_metric(summary['pred_mean'])} / {_format_metric(summary['pred_std'])}",
            "",
            "## 5. Confusion Matrix",
            "",
            "|  | Pred 0 | Pred 1 |",
            "|---|---:|---:|",
            f"| True 0 | {cm['tn']} | {cm['fp']} |",
            f"| True 1 | {cm['fn']} | {cm['tp']} |",
            "",
            "## 6. Comparison Targets",
            "",
            "- Current Mobile A-LAMP v2 full available AVA test: Accuracy ≈ 0.7049, F1 ≈ 0.7647.",
            "- Paper A-LAMP target: Accuracy ≈ 0.825, F-measure ≈ 0.92.",
            "- This model is only a Multi-Patch teacher baseline, not full A-LAMP.",
            "",
            "## 7. Judgment",
            "",
            f"- {_judgment_text(float(summary['accuracy']))}",
            "",
        ]
    )
    path.write_text(report, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--model_path")
    parser.add_argument("--weights_path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_test_samples", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    args = _parse_args()
    config = _read_yaml(args.config)
    test_jsonl = Path(_resolve(args.test_jsonl, config, "dataset", "test_jsonl", DEFAULT_TEST_JSONL))
    out_dir = Path(args.output_dir)
    batch_size = int(args.batch_size)
    threshold = float(args.threshold)
    seed = int(_cfg(config, "training", "seed", 42))
    patch_count = int(_cfg(config, "model", "patch_count", 5))
    patch_size = int(_cfg(config, "model", "patch_size", 224))
    label_threshold = float(_cfg(config, "dataset", "label_threshold", 5.0))

    if batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if not test_jsonl.is_file():
        raise FileNotFoundError(f"Missing test JSONL: {test_jsonl}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(NOTICE)
    tf_info = _setup_tensorflow(seed=seed)
    model, checkpoint_type, checkpoint_paths = _load_model_from_args(
        model_path=args.model_path,
        weights_path=args.weights_path,
        config=config,
    )
    records = load_jsonl_records(test_jsonl, max_samples=args.max_test_samples)
    print(f"Loaded {len(records)} test records from {test_jsonl}")
    rows, skipped_count, elapsed_seconds = _predict_records(
        model=model,
        records=records,
        repo_root=Path("."),
        patch_size=patch_size,
        patch_count=patch_count,
        batch_size=batch_size,
        label_threshold=label_threshold,
        threshold=threshold,
    )
    metrics = _compute_summary_metrics(rows, threshold=threshold)

    summary: dict[str, Any] = {
        **checkpoint_paths,
        "checkpoint_type": checkpoint_type,
        "test_jsonl": str(test_jsonl),
        **metrics,
        "notice": NOTICE,
        "config": str(args.config),
        "model_variant": MODEL_VARIANT,
        "model_description": MODEL_DESCRIPTION,
        "reproduction_claim": REPRODUCTION_CLAIM,
        "preprocessing_mode": PREPROCESSING_MODE,
        "patch_count": int(patch_count),
        "patch_size": int(patch_size),
        "label_rule": "mean_score > 5.0 -> 1, else 0",
        "skipped_count": int(skipped_count),
        "elapsed_seconds": float(elapsed_seconds),
        "seconds_per_sample": float(elapsed_seconds / metrics["sample_count"]),
        "max_test_samples": args.max_test_samples,
        "command": _command_for_summary(),
        "tensorflow_info": tf_info,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_predictions(out_dir / "predictions.csv", rows)
    _write_report(out_dir / "report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote evaluation artifacts to {out_dir}")


if __name__ == "__main__":
    main()
