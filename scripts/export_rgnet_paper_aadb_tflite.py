# RGNet 논문 지향 AADB 모델을 FP32/FP16 TFLite로 내보낸다.
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-photo-score")

import tensorflow as tf
from tensorflow.keras import mixed_precision

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.rgnet_paper_v1 import (  # noqa: E402
    ASPPContextModule,
    DenseNetV1UnitPreprocess,
    RegionScoreAggregation,
    V1RegionSimilarityAdjacency,
    V1ResidualGraphConvolution,
    get_rgnet_paper_v1_custom_objects,
)

DEFAULT_MODEL_PATH = (
    "outputs/rgnet_paper_v1_ablation_full_candidates_20260510/"
    "full_train/agg_mean_full/final_model.keras"
)
DEFAULT_FP32_OUT = "models/aesthetic/rgnet_paper_aadb_fp32.tflite"
DEFAULT_FP16_OUT = "models/aesthetic/rgnet_paper_aadb_fp16.tflite"
DEFAULT_SUMMARY_OUT = "outputs/rgnet_paper_aadb_tflite_export_20260514/export_summary.json"
PROTECTED_MOBILE_TFLITE = (REPO_ROOT / "models/aesthetic/rgnet_aadb_gpu.tflite").resolve()
CUSTOM_LAYER_TYPES = (
    DenseNetV1UnitPreprocess,
    ASPPContextModule,
    V1RegionSimilarityAdjacency,
    V1ResidualGraphConvolution,
    RegionScoreAggregation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RGNet-paper AADB Keras model to FP32 and FP16 TFLite.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--fp32_out", default=DEFAULT_FP32_OUT)
    parser.add_argument("--fp16_out", default=DEFAULT_FP16_OUT)
    parser.add_argument("--summary_out", default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--allow_overwrite", action="store_true")
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


def shape_to_json(shape: Any) -> list[int | None]:
    return [None if dim is None else int(dim) for dim in shape]


def tensor_summary(tensor: Any) -> dict[str, object]:
    return {
        "name": str(getattr(tensor, "name", "")),
        "shape": shape_to_json(getattr(tensor, "shape", [])),
        "dtype": str(getattr(tensor, "dtype", "")),
    }


def iter_layers(layer: tf.keras.layers.Layer) -> list[tf.keras.layers.Layer]:
    seen: set[int] = set()
    collected: list[tf.keras.layers.Layer] = []

    def visit(current: tf.keras.layers.Layer) -> None:
        identity = id(current)
        if identity in seen:
            return
        seen.add(identity)
        collected.append(current)
        for child in getattr(current, "layers", []):
            visit(child)

    visit(layer)
    return collected


def detect_custom_layers(model: tf.keras.Model) -> list[dict[str, str]]:
    layers = []
    for layer in iter_layers(model):
        if isinstance(layer, CUSTOM_LAYER_TYPES):
            layers.append({"name": layer.name, "class_name": layer.__class__.__name__})
    return layers


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def require_safe_output(path: Path, allow_overwrite: bool) -> None:
    if path == PROTECTED_MOBILE_TFLITE:
        raise ValueError(f"Refusing to overwrite protected practical mobile model: {path}")
    if "forWeights" in path.parts:
        raise ValueError(f"Refusing to write under protected forWeights directory: {path}")
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"{path} exists. Pass --allow_overwrite to replace it.")


def convert_model(model: tf.keras.Model, output_path: Path, precision: str) -> dict[str, object]:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    if precision == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    started = datetime.now().astimezone().isoformat()
    tflite_bytes = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_bytes)
    return {
        "status": "success",
        "started_at": started,
        "finished_at": datetime.now().astimezone().isoformat(),
        "path": str(output_path.relative_to(REPO_ROOT)),
        "size_bytes": int(output_path.stat().st_size),
        "size_mib": float(output_path.stat().st_size / (1024 * 1024)),
        "precision": precision,
    }


def write_summary(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_path = resolve_repo_path(args.model_path)
    fp32_out = resolve_repo_path(args.fp32_out)
    fp16_out = resolve_repo_path(args.fp16_out)
    summary_out = resolve_repo_path(args.summary_out)

    warnings: list[str] = []
    summary: dict[str, object] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_model_path": str(model_path.relative_to(REPO_ROOT)),
        "fp32_tflite_path": str(fp32_out.relative_to(REPO_ROOT)),
        "fp16_tflite_path": str(fp16_out.relative_to(REPO_ROOT)),
        "conversion_status": "not_started",
        "warnings": warnings,
        "conversions": {},
    }

    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Keras model not found: {model_path}")
        require_safe_output(fp32_out, args.allow_overwrite)
        require_safe_output(fp16_out, args.allow_overwrite)
        if "forWeights" in summary_out.parts:
            raise ValueError(f"Refusing to write summary under protected forWeights directory: {summary_out}")

        summary.update(configure_tensorflow())
        model = tf.keras.models.load_model(
            str(model_path),
            compile=False,
            safe_mode=False,
            custom_objects=expanded_custom_objects(),
        )
        custom_layers = detect_custom_layers(model)
        if not custom_layers:
            warnings.append("No RGNet paper custom layers were detected after loading.")
        summary.update(
            {
                "model_name": model.name,
                "input_shape": [tensor_summary(tensor) for tensor in model.inputs],
                "output_shape": [tensor_summary(tensor) for tensor in model.outputs],
                "custom_layer_names": custom_layers,
                "conversion_status": "running",
            }
        )

        fp32_result = convert_model(model, fp32_out, "fp32")
        summary["conversions"]["fp32"] = fp32_result
        summary["fp32_file_size_bytes"] = fp32_result["size_bytes"]
        summary["fp32_file_size_mib"] = fp32_result["size_mib"]

        fp16_result = convert_model(model, fp16_out, "fp16")
        summary["conversions"]["fp16"] = fp16_result
        summary["fp16_file_size_bytes"] = fp16_result["size_bytes"]
        summary["fp16_file_size_mib"] = fp16_result["size_mib"]
        summary["conversion_status"] = "success"
        write_summary(summary_out, summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    except Exception as exc:
        summary["conversion_status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        if fp32_out.exists():
            summary["fp32_file_size_bytes"] = int(fp32_out.stat().st_size)
        if fp16_out.exists():
            summary["fp16_file_size_bytes"] = int(fp16_out.stat().st_size)
        write_summary(summary_out, summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
