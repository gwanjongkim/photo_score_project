# RGNet 논문 지향 AADB Keras 모델과 TFLite 출력 패리티를 평가한다.
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-photo-score")

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import mixed_precision

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.rgnet_paper_v1 import (  # noqa: E402
    DenseNetV1UnitPreprocess,
    get_rgnet_paper_v1_custom_objects,
)

DEFAULT_KERAS_MODEL = (
    "outputs/rgnet_paper_v1_ablation_full_candidates_20260510/"
    "full_train/agg_mean_full/final_model.keras"
)
DEFAULT_FP32_TFLITE = "models/aesthetic/rgnet_paper_aadb_fp32.tflite"
DEFAULT_FP16_TFLITE = "models/aesthetic/rgnet_paper_aadb_fp16.tflite"
DEFAULT_OUTPUT_DIR = "outputs/rgnet_paper_aadb_tflite_export_20260514/parity"
DEFAULT_AADB_MANIFESTS = ("data/processed/aadb/val.csv", "data/processed/aadb/test.csv")
FALLBACK_IMAGE_DIRS = ("test_images", "test_samples")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FP32_STRICT_MAX_ABS_DIFF = 1e-4
FP16_MEAN_DIFF_TARGET = 1e-3
FP16_CORRELATION_TARGET = 0.999
FP16_RANKING_TARGET = 0.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Keras-vs-TFLite parity for RGNet-paper AADB.")
    parser.add_argument("--keras_model", default=DEFAULT_KERAS_MODEL)
    parser.add_argument("--fp32_tflite", default=DEFAULT_FP32_TFLITE)
    parser.add_argument("--fp16_tflite", default=DEFAULT_FP16_TFLITE)
    parser.add_argument("--image_dir")
    parser.add_argument("--image_manifest")
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def configure_tensorflow() -> dict[str, object]:
    mixed_precision.set_global_policy("float32")
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:  # pragma: no cover - environment-specific.
            print(f"memory growth setup failed for {gpu}: {exc}", file=sys.stderr)
    return {
        "tensorflow_version": tf.__version__,
        "mixed_precision_policy": str(mixed_precision.global_policy()),
        "visible_gpus": [str(gpu) for gpu in gpus],
    }


def expanded_custom_objects() -> dict[str, object]:
    objects = dict(get_rgnet_paper_v1_custom_objects())
    for name, value in list(objects.items()):
        objects[f"photo_score_project>{name}"] = value
    return objects


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def display_path(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(candidate)


def collect_image_dir_paths(image_dir: Path, max_samples: int) -> list[str]:
    if not image_dir.exists() or not image_dir.is_dir():
        return []
    paths = [
        path
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return [str(path) for path in paths[:max_samples]]


def read_manifest_paths(manifest_path: Path, max_samples: int) -> list[str]:
    if not manifest_path.exists():
        return []
    paths: list[str] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        fields = set(reader.fieldnames)
        path_field = next(
            (field for field in ("image_path", "path", "filepath", "file", "relative_path") if field in fields),
            None,
        )
        if path_field is None:
            return []
        for row in reader:
            raw_path = (row.get(path_field) or "").strip()
            if not raw_path:
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                if path_field == "relative_path":
                    path = REPO_ROOT / "data/raw/aadb/images" / path
                else:
                    path = REPO_ROOT / path
            paths.append(str(path))
            if len(paths) >= max_samples:
                break
    return paths


def choose_image_paths(args: argparse.Namespace, warnings: list[str]) -> tuple[list[str], str]:
    if args.image_manifest:
        manifest = resolve_repo_path(args.image_manifest)
        paths = read_manifest_paths(manifest, args.max_samples)
        if paths:
            return paths, f"manifest:{display_path(manifest)}"
        warnings.append(f"No usable image paths found in requested manifest {display_path(manifest)}.")

    if args.image_dir:
        image_dir = resolve_repo_path(args.image_dir)
        paths = collect_image_dir_paths(image_dir, args.max_samples)
        if paths:
            return paths, f"directory:{display_path(image_dir)}"
        warnings.append(f"No usable images found in requested image_dir {display_path(image_dir)}.")

    for manifest_name in DEFAULT_AADB_MANIFESTS:
        manifest = resolve_repo_path(manifest_name)
        paths = read_manifest_paths(manifest, args.max_samples)
        existing = [path for path in paths if Path(path).exists()]
        if existing:
            return existing[: args.max_samples], f"manifest:{display_path(manifest)}"

    for directory_name in FALLBACK_IMAGE_DIRS:
        image_dir = resolve_repo_path(directory_name)
        paths = collect_image_dir_paths(image_dir, args.max_samples)
        if paths:
            return paths, f"directory:{display_path(image_dir)}"

    raise FileNotFoundError("No usable local parity images found in AADB manifests, test_images, or test_samples.")


def load_rgb_unit(path: str | Path, image_size: int = 256) -> np.ndarray:
    resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), resampling)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_sample_batch(paths: list[str], max_samples: int, warnings: list[str]) -> tuple[list[str], np.ndarray]:
    loaded_paths: list[str] = []
    images: list[np.ndarray] = []
    for path in paths:
        if len(images) >= max_samples:
            break
        try:
            images.append(load_rgb_unit(path))
            loaded_paths.append(path)
        except Exception as exc:
            warnings.append(f"Skipped unreadable image {display_path(path)}: {type(exc).__name__}: {exc}")
    if not images:
        raise ValueError("No images could be decoded for parity evaluation.")
    return loaded_paths, np.stack(images, axis=0).astype(np.float32)


def predict_keras(model: tf.keras.Model, batch: np.ndarray, batch_size: int = 8) -> np.ndarray:
    outputs = []
    for start in range(0, batch.shape[0], batch_size):
        outputs.append(np.asarray(model(batch[start : start + batch_size], training=False), dtype=np.float32).reshape(-1))
    return np.concatenate(outputs, axis=0)


def make_interpreter(path: Path) -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=max(1, min(4, os.cpu_count() or 1)))
    interpreter.allocate_tensors()
    return interpreter


def tflite_predict(interpreter: tf.lite.Interpreter, batch: np.ndarray) -> np.ndarray:
    predictions: list[float] = []
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    expected_shape = [1, int(batch.shape[1]), int(batch.shape[2]), int(batch.shape[3])]
    if [int(value) for value in input_detail["shape"]] != expected_shape:
        interpreter.resize_tensor_input(input_detail["index"], expected_shape, strict=False)
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
    for image in batch:
        interpreter.set_tensor(input_detail["index"], image[None, ...].astype(input_detail["dtype"], copy=False))
        interpreter.invoke()
        predictions.append(float(np.asarray(interpreter.get_tensor(output_detail["index"]), dtype=np.float32).reshape(-1)[0]))
    return np.asarray(predictions, dtype=np.float32)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    return safe_pearson(average_ranks(x), average_ranks(y))


def ranking_agreement(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    total = 0
    agreed = 0
    for left in range(x.size):
        for right in range(left + 1, x.size):
            ref_sign = np.sign(x[left] - x[right])
            other_sign = np.sign(y[left] - y[right])
            if ref_sign == 0 or other_sign == 0:
                continue
            total += 1
            if ref_sign == other_sign:
                agreed += 1
    if total == 0:
        return None
    return float(agreed / total)


def diff_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | None]:
    diff = np.abs(reference - candidate)
    return {
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "pearson_correlation": safe_pearson(reference, candidate),
        "spearman_correlation": safe_spearman(reference, candidate),
        "ranking_agreement": ranking_agreement(reference, candidate),
    }


def fp16_ranking_ok(metrics: dict[str, float | None]) -> bool:
    pearson = metrics["pearson_correlation"]
    spearman = metrics["spearman_correlation"]
    ranking = metrics["ranking_agreement"]
    correlation_ok = pearson is None or pearson >= FP16_CORRELATION_TARGET
    spearman_ok = spearman is None or spearman >= FP16_CORRELATION_TARGET
    ranking_ok = ranking is None or ranking >= FP16_RANKING_TARGET
    return bool(metrics["mean_abs_diff"] <= FP16_MEAN_DIFF_TARGET and correlation_ok and spearman_ok and ranking_ok)


def prediction_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def model_has_unit_preprocess(model: tf.keras.Model) -> bool:
    seen: set[int] = set()

    def visit(layer: tf.keras.layers.Layer) -> bool:
        identity = id(layer)
        if identity in seen:
            return False
        seen.add(identity)
        if isinstance(layer, DenseNetV1UnitPreprocess):
            return True
        return any(visit(child) for child in getattr(layer, "layers", []))

    return visit(model)


def write_predictions_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "keras_pred", "fp32_pred", "fp16_pred", "abs_diff_fp32", "abs_diff_fp16"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown_report(path: Path, summary: dict[str, object]) -> None:
    fp32 = summary["fp32"]
    fp16 = summary["fp16"]
    interpretation = summary["pass_fail_interpretation"]
    lines = [
        "# RGNet Paper AADB TFLite Parity Report",
        "",
        f"- Created: {summary['created_at']}",
        f"- Image source: {summary['image_source']}",
        f"- Sample count: {summary['sample_count']}",
        f"- Built-in DenseNet preprocessing detected: {summary['builtin_densenet_preprocess_detected']}",
        f"- External ImageNet mean/std preprocessing applied: {summary['external_imagenet_preprocessing_applied']}",
        "",
        "## Metrics",
        "",
        "| Model | Max abs diff | Mean abs diff | Pearson | Spearman | Ranking agreement |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| FP32 TFLite | {fp32['max_abs_diff']:.10g} | {fp32['mean_abs_diff']:.10g} | "
            f"{format_optional(fp32['pearson_correlation'])} | {format_optional(fp32['spearman_correlation'])} | "
            f"{format_optional(fp32['ranking_agreement'])} |"
        ),
        (
            f"| FP16 TFLite | {fp16['max_abs_diff']:.10g} | {fp16['mean_abs_diff']:.10g} | "
            f"{format_optional(fp16['pearson_correlation'])} | {format_optional(fp16['spearman_correlation'])} | "
            f"{format_optional(fp16['ranking_agreement'])} |"
        ),
        "",
        "## Interpretation",
        "",
        f"- FP32 strict parity: {interpretation['fp32_message']}",
        f"- FP16 ranking preservation: {interpretation['fp16_message']}",
        f"- Ready for Flutter smoke test: {interpretation['ready_for_flutter_smoke_test']}",
        f"- Ready to replace `rgnet_aadb_gpu.tflite`: {interpretation['ready_to_replace_rgnet_aadb_gpu_tflite']}",
    ]
    if summary["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in summary["warnings"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_optional(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.10g}"


def main() -> None:
    args = parse_args()
    if args.max_samples < 1:
        raise ValueError("--max_samples must be at least 1")

    warnings: list[str] = []
    output_dir = resolve_repo_path(args.output_dir)
    predictions_csv = output_dir / "predictions.csv"
    summary_json = output_dir / "parity_summary.json"
    report_md = output_dir / "parity_report.md"

    keras_model_path = resolve_repo_path(args.keras_model)
    fp32_tflite_path = resolve_repo_path(args.fp32_tflite)
    fp16_tflite_path = resolve_repo_path(args.fp16_tflite)
    for path in (keras_model_path, fp32_tflite_path, fp16_tflite_path):
        if not path.exists():
            raise FileNotFoundError(path)

    tf_info = configure_tensorflow()
    model = tf.keras.models.load_model(
        str(keras_model_path),
        compile=False,
        safe_mode=False,
        custom_objects=expanded_custom_objects(),
    )
    has_preprocess = model_has_unit_preprocess(model)
    image_paths, image_source = choose_image_paths(args, warnings)
    loaded_paths, batch = load_sample_batch(image_paths, args.max_samples, warnings)

    keras_pred = predict_keras(model, batch)
    fp32_pred = tflite_predict(make_interpreter(fp32_tflite_path), batch)
    fp16_pred = tflite_predict(make_interpreter(fp16_tflite_path), batch)
    fp32_metrics = diff_metrics(keras_pred, fp32_pred)
    fp16_metrics = diff_metrics(keras_pred, fp16_pred)

    fp32_pass = bool(fp32_metrics["max_abs_diff"] < FP32_STRICT_MAX_ABS_DIFF)
    fp16_ok = fp16_ranking_ok(fp16_metrics)
    rows = []
    for index, image_path in enumerate(loaded_paths):
        rows.append(
            {
                "image_path": display_path(image_path),
                "keras_pred": float(keras_pred[index]),
                "fp32_pred": float(fp32_pred[index]),
                "fp16_pred": float(fp16_pred[index]),
                "abs_diff_fp32": float(abs(keras_pred[index] - fp32_pred[index])),
                "abs_diff_fp16": float(abs(keras_pred[index] - fp16_pred[index])),
            }
        )

    stats = prediction_stats(keras_pred)
    summary: dict[str, object] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "keras_model": display_path(keras_model_path),
        "fp32_tflite": display_path(fp32_tflite_path),
        "fp16_tflite": display_path(fp16_tflite_path),
        "sample_count": int(len(loaded_paths)),
        "image_source": image_source,
        "keras_pred_mean": stats["mean"],
        "keras_pred_std": stats["std"],
        "keras_pred_min": stats["min"],
        "keras_pred_max": stats["max"],
        "fp32_max_abs_diff": fp32_metrics["max_abs_diff"],
        "fp32_mean_abs_diff": fp32_metrics["mean_abs_diff"],
        "fp16_max_abs_diff": fp16_metrics["max_abs_diff"],
        "fp16_mean_abs_diff": fp16_metrics["mean_abs_diff"],
        "fp32": fp32_metrics,
        "fp16": fp16_metrics,
        "pearson_correlation": {
            "fp32": fp32_metrics["pearson_correlation"],
            "fp16": fp16_metrics["pearson_correlation"],
        },
        "spearman_correlation": {
            "fp32": fp32_metrics["spearman_correlation"],
            "fp16": fp16_metrics["spearman_correlation"],
        },
        "ranking_agreement": {
            "fp32": fp32_metrics["ranking_agreement"],
            "fp16": fp16_metrics["ranking_agreement"],
        },
        "builtin_densenet_preprocess_detected": has_preprocess,
        "external_imagenet_preprocessing_applied": False,
        "tensorflow": tf_info,
        "warnings": warnings,
        "outputs": {
            "predictions_csv": display_path(predictions_csv),
            "summary_json": display_path(summary_json),
            "report_md": display_path(report_md),
        },
        "pass_fail_interpretation": {
            "fp32_strict_target": f"max_abs_diff < {FP32_STRICT_MAX_ABS_DIFF}",
            "fp32_pass": fp32_pass,
            "fp32_message": (
                f"PASS, max_abs_diff={fp32_metrics['max_abs_diff']:.10g}"
                if fp32_pass
                else (
                    f"FAIL, max_abs_diff={fp32_metrics['max_abs_diff']:.10g}; likely causes include "
                    "unsupported conversion approximation, fused op numeric drift, or preprocessing mismatch."
                )
            ),
            "fp16_mean_diff_target": f"mean_abs_diff <= {FP16_MEAN_DIFF_TARGET}",
            "fp16_correlation_target": f"Pearson/Spearman >= {FP16_CORRELATION_TARGET} when computable",
            "fp16_ranking_target": f"ranking_agreement >= {FP16_RANKING_TARGET} when computable",
            "fp16_ranking_preserved": fp16_ok,
            "fp16_message": (
                "PASS for ranking-oriented review; mean difference and ranking/correlation targets are preserved."
                if fp16_ok
                else "FAIL for ranking-oriented review; inspect FP16 differences before app-side smoke."
            ),
            "ready_for_flutter_smoke_test": bool(fp32_pass and fp16_ok),
            "ready_to_replace_rgnet_aadb_gpu_tflite": False,
            "replacement_blockers": [
                "Flutter/app-side smoke test has not been run.",
                "Runtime preprocessing consistency has not been verified in Flutter.",
                "Existing practical mobile model was intentionally not replaced.",
            ],
        },
    }

    write_predictions_csv(predictions_csv, rows)
    write_json(summary_json, summary)
    write_markdown_report(report_md, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
