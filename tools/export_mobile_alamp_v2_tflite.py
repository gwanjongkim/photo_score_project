# Mobile A-LAMP v2 체크포인트를 TFLite로 내보내고 패리티를 검증하는 도구
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-photo-score")

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.mobile_alamp_v2 import MergePatchBatch, RestorePatchBatch  # noqa: E402
from src.train.train_mobile_alamp_v2 import (  # noqa: E402
    PREPROCESSING_MODE,
    _label_from_score,
    _load_full_image,
    _load_jsonl_records,
    _load_patch_crops,
)

DEFAULT_MODEL_PATH = "outputs/mobile_alamp_v2_pretrained_4096/best_val_auc_model.keras"
DEFAULT_VAL_PATCH_JSONL = (
    "outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/"
    "val_patch_boxes_4096_v4.jsonl"
)
DEFAULT_OUTPUT_DIR = "outputs/mobile_alamp_v2_pretrained_4096_tflite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Mobile A-LAMP v2 best checkpoint to FP32/FP16 TFLite and run parity."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--val_patch_jsonl", default=DEFAULT_VAL_PATCH_JSONL)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--patch_count", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--label_threshold", type=float, default=5.0)
    return parser.parse_args()


def resolve_repo_path(path: str | Path) -> Path:
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


def command_for_summary() -> str:
    command = " ".join([sys.executable, *sys.argv])
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        return f"PYTHONPATH={pythonpath} {command}"
    return command


def configure_tensorflow() -> dict[str, Any]:
    mixed_precision.set_global_policy("float32")
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            print(f"memory growth setup failed for {gpu}: {exc}", file=sys.stderr)
    return {
        "tensorflow_version": tf.__version__,
        "mixed_precision_policy": str(mixed_precision.global_policy()),
        "visible_gpus": [str(gpu) for gpu in gpus],
    }


def mobile_alamp_custom_objects() -> dict[str, Any]:
    return {
        "MergePatchBatch": MergePatchBatch,
        "RestorePatchBatch": RestorePatchBatch,
        "MobileALAMPV2>MergePatchBatch": MergePatchBatch,
        "MobileALAMPV2>RestorePatchBatch": RestorePatchBatch,
    }


def shape_to_json(shape: Any) -> list[int | None]:
    values = shape.as_list() if hasattr(shape, "as_list") else list(shape)
    return [None if value is None else int(value) for value in values]


def keras_tensor_summary(tensor: Any) -> dict[str, Any]:
    return {
        "name": str(getattr(tensor, "name", "")),
        "shape": shape_to_json(getattr(tensor, "shape", [])),
        "dtype": str(getattr(tensor, "dtype", "")),
    }


def detail_shape(values: Any) -> list[int]:
    return [int(value) for value in np.asarray(values).tolist()]


def tflite_tensor_summary(detail: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(detail.get("name", "")),
        "index": int(detail["index"]),
        "shape": detail_shape(detail["shape"]),
        "shape_signature": detail_shape(detail.get("shape_signature", detail["shape"])),
        "dtype": np.dtype(detail["dtype"]).name,
        "quantization": [float(detail["quantization"][0]), int(detail["quantization"][1])],
    }


def record_image_path(record: dict[str, Any]) -> Path | None:
    value = record.get("resolved_image_path") or record.get("image_path")
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path if path.exists() else None


def load_parity_examples(
    jsonl_path: Path,
    max_samples: int,
    image_size: int,
    patch_size: int,
    patch_count: int,
    label_threshold: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[str]]:
    records = _load_jsonl_records(jsonl_path)
    examples: list[dict[str, Any]] = []
    full_images: list[np.ndarray] = []
    patches: list[np.ndarray] = []
    warnings: list[str] = []

    for record in records:
        image_path = record_image_path(record)
        if image_path is None:
            warnings.append(
                f"Skipped row {record.get('row_index')} because the image path is missing or unreadable."
            )
            continue

        patch_tensor = _load_patch_crops(
            record,
            image_path=image_path,
            patch_size=patch_size,
            patch_count=patch_count,
        )
        if patch_tensor is None:
            warnings.append(f"Skipped {display_path(image_path)} because patch loading failed.")
            continue

        full_tensor = _load_full_image(image_path, image_size=image_size)
        label = float(_label_from_score(record, label_threshold)[0])
        full_images.append(full_tensor.astype(np.float32))
        patches.append(patch_tensor.astype(np.float32))
        examples.append(
            {
                "sample_index": len(examples),
                "row_index": record.get("row_index"),
                "image_path": display_path(image_path),
                "mean_score": float(record.get("mean_score", 0.0)),
                "label": label,
            }
        )
        if len(examples) >= max_samples:
            break

    if len(examples) < max_samples:
        raise ValueError(
            f"Only loaded {len(examples)} usable examples from {display_path(jsonl_path)}; "
            f"{max_samples} required."
        )

    return {
        "full_image": np.stack(full_images, axis=0).astype(np.float32),
        "patches": np.stack(patches, axis=0).astype(np.float32),
    }, examples, warnings


def predict_keras(model: tf.keras.Model, inputs: dict[str, np.ndarray], batch_size: int) -> np.ndarray:
    predictions: list[np.ndarray] = []
    sample_count = int(inputs["full_image"].shape[0])
    for start in range(0, sample_count, batch_size):
        stop = min(sample_count, start + batch_size)
        batch = {
            "full_image": inputs["full_image"][start:stop],
            "patches": inputs["patches"][start:stop],
        }
        predictions.append(np.asarray(model(batch, training=False), dtype=np.float32).reshape(-1))
    return np.concatenate(predictions, axis=0)


def export_temporary_saved_model(
    model: tf.keras.Model,
    saved_model_dir: Path,
    image_size: int,
    patch_size: int,
    patch_count: int,
) -> None:
    input_signature = [
        {
            "full_image": tf.TensorSpec(
                [None, image_size, image_size, 3],
                tf.float32,
                name="full_image",
            ),
            "patches": tf.TensorSpec(
                [None, patch_count, patch_size, patch_size, 3],
                tf.float32,
                name="patches",
            ),
        }
    ]
    model.export(
        str(saved_model_dir),
        input_signature=input_signature,
        verbose=False,
    )


def convert_saved_model(saved_model_dir: Path, output_path: Path, precision: str) -> dict[str, Any]:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    if precision == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    started_at = datetime.now().astimezone().isoformat()
    tflite_bytes = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_bytes)
    size_bytes = int(output_path.stat().st_size)
    return {
        "precision": precision,
        "path": display_path(output_path),
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(),
        "size_bytes": size_bytes,
        "size_mib": float(size_bytes / (1024 * 1024)),
    }


def map_tflite_inputs(
    input_details: list[dict[str, Any]],
    image_size: int,
    patch_size: int,
    patch_count: int,
) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for detail in input_details:
        name = str(detail.get("name", "")).lower()
        shape = detail_shape(detail.get("shape_signature", detail["shape"]))
        rank = len(shape)
        if "patch" in name or rank == 5:
            mapped["patches"] = detail
        elif "full_image" in name or (
            rank == 4 and shape[-3:] == [image_size, image_size, 3]
        ):
            mapped["full_image"] = detail
        elif rank == 4 and shape[-3:] == [patch_size, patch_size, 3] and patch_count == 1:
            mapped["patches"] = detail

    missing = {"full_image", "patches"} - set(mapped)
    if missing:
        summaries = [tflite_tensor_summary(detail) for detail in input_details]
        raise ValueError(f"Could not map TFLite inputs {sorted(missing)} from details: {summaries}")
    return mapped


def make_interpreter(
    path: Path,
    image_size: int,
    patch_size: int,
    patch_count: int,
) -> tuple[tf.lite.Interpreter, dict[str, dict[str, Any]]]:
    interpreter = tf.lite.Interpreter(
        model_path=str(path),
        num_threads=max(1, min(4, os.cpu_count() or 1)),
    )
    mapped = map_tflite_inputs(
        interpreter.get_input_details(),
        image_size=image_size,
        patch_size=patch_size,
        patch_count=patch_count,
    )
    interpreter.resize_tensor_input(
        mapped["full_image"]["index"],
        [1, image_size, image_size, 3],
        strict=False,
    )
    interpreter.resize_tensor_input(
        mapped["patches"]["index"],
        [1, patch_count, patch_size, patch_size, 3],
        strict=False,
    )
    interpreter.allocate_tensors()
    mapped = map_tflite_inputs(
        interpreter.get_input_details(),
        image_size=image_size,
        patch_size=patch_size,
        patch_count=patch_count,
    )
    return interpreter, mapped


def predict_tflite(
    interpreter: tf.lite.Interpreter,
    mapped_inputs: dict[str, dict[str, Any]],
    inputs: dict[str, np.ndarray],
) -> np.ndarray:
    output_detail = interpreter.get_output_details()[0]
    predictions: list[float] = []
    sample_count = int(inputs["full_image"].shape[0])

    for index in range(sample_count):
        for key, detail in mapped_inputs.items():
            tensor = inputs[key][index : index + 1].astype(detail["dtype"], copy=False)
            interpreter.set_tensor(detail["index"], tensor)
        interpreter.invoke()
        output = np.asarray(interpreter.get_tensor(output_detail["index"]), dtype=np.float32)
        predictions.append(float(output.reshape(-1)[0]))

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


def diff_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | None]:
    diff = np.abs(reference - candidate)
    return {
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "pearson": safe_pearson(reference, candidate),
        "spearman": safe_spearman(reference, candidate),
    }


def prediction_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.10g}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    fp32 = report["metrics"]["fp32"]
    fp16 = report["metrics"]["fp16"]
    lines = [
        "# Mobile A-LAMP v2 TFLite Parity Report",
        "",
        "This export uses the existing Mobile A-LAMP v2 best checkpoint. No retraining was run.",
        "",
        "## Command",
        "",
        "```bash",
        report["command"],
        "```",
        "",
        "## Inputs",
        "",
        f"- Keras model: `{report['keras_model']}`",
        f"- Validation JSONL: `{report['val_patch_jsonl']}`",
        f"- Sample count: `{report['sample_count']}`",
        f"- Preprocessing: `{report['preprocessing_mode']}`",
        "- Pixel convention: `full_image` and `patches` are RGB `float32` pixels in `[0,255]`; MobileNetV3Small preprocessing stays inside the model.",
        "",
        "## Outputs",
        "",
        f"- FP32 TFLite: `{report['tflite_models']['fp32']}`",
        f"- FP16 TFLite: `{report['tflite_models']['fp16']}`",
        f"- Export metadata: `{report['artifacts']['export_metadata_json']}`",
        f"- JSON report: `{report['artifacts']['parity_report_json']}`",
        f"- Markdown report: `{report['artifacts']['parity_report_md']}`",
        "",
        "## Metrics",
        "",
        "| Model | Max abs diff | Mean abs diff | Pearson | Spearman |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| FP32 TFLite | {fp32['max_abs_diff']:.10g} | {fp32['mean_abs_diff']:.10g} | "
            f"{format_optional(fp32['pearson'])} | {format_optional(fp32['spearman'])} |"
        ),
        (
            f"| FP16 TFLite | {fp16['max_abs_diff']:.10g} | {fp16['mean_abs_diff']:.10g} | "
            f"{format_optional(fp16['pearson'])} | {format_optional(fp16['spearman'])} |"
        ),
        "",
        "## Shapes",
        "",
        "```json",
        json.dumps(report["shapes"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Per-Sample Predictions",
        "",
        "| # | row_index | image_path | Keras | FP32 | FP16 | abs FP32 | abs FP16 |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["predictions"]:
        lines.append(
            f"| {row['sample_index']} | {row['row_index']} | `{row['image_path']}` | "
            f"{row['keras_pred']:.10g} | {row['fp32_pred']:.10g} | {row['fp16_pred']:.10g} | "
            f"{row['abs_diff_fp32']:.10g} | {row['abs_diff_fp16']:.10g} |"
        )
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_prediction_rows(
    examples: list[dict[str, Any]],
    keras_pred: np.ndarray,
    fp32_pred: np.ndarray,
    fp16_pred: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        rows.append(
            {
                **example,
                "keras_pred": float(keras_pred[index]),
                "fp32_pred": float(fp32_pred[index]),
                "fp16_pred": float(fp16_pred[index]),
                "abs_diff_fp32": float(abs(keras_pred[index] - fp32_pred[index])),
                "abs_diff_fp16": float(abs(keras_pred[index] - fp16_pred[index])),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.max_samples < 1:
        raise ValueError("--max_samples must be at least 1")

    model_path = resolve_repo_path(args.model_path)
    val_patch_jsonl = resolve_repo_path(args.val_patch_jsonl)
    output_dir = resolve_repo_path(args.output_dir)
    fp32_path = output_dir / "mobile_alamp_v2_fp32.tflite"
    fp16_path = output_dir / "mobile_alamp_v2_fp16.tflite"
    export_metadata_path = output_dir / "export_metadata.json"
    parity_report_json_path = output_dir / "parity_report.json"
    parity_report_md_path = output_dir / "parity_report.md"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing Keras model: {display_path(model_path)}")
    if not val_patch_jsonl.exists():
        raise FileNotFoundError(f"Missing validation JSONL: {display_path(val_patch_jsonl)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    tf_info = configure_tensorflow()
    model = tf.keras.models.load_model(
        str(model_path),
        compile=False,
        safe_mode=False,
        custom_objects=mobile_alamp_custom_objects(),
    )

    with tempfile.TemporaryDirectory(prefix="mobile_alamp_v2_savedmodel_") as temp_dir:
        saved_model_dir = Path(temp_dir) / "saved_model"
        export_temporary_saved_model(
            model=model,
            saved_model_dir=saved_model_dir,
            image_size=args.image_size,
            patch_size=args.patch_size,
            patch_count=args.patch_count,
        )
        fp32_export = convert_saved_model(saved_model_dir, fp32_path, "fp32")
        fp16_export = convert_saved_model(saved_model_dir, fp16_path, "fp16")

    inputs, examples, warnings = load_parity_examples(
        jsonl_path=val_patch_jsonl,
        max_samples=args.max_samples,
        image_size=args.image_size,
        patch_size=args.patch_size,
        patch_count=args.patch_count,
        label_threshold=args.label_threshold,
    )
    keras_pred = predict_keras(model, inputs, batch_size=args.batch_size)

    fp32_interpreter, fp32_inputs = make_interpreter(
        fp32_path,
        image_size=args.image_size,
        patch_size=args.patch_size,
        patch_count=args.patch_count,
    )
    fp16_interpreter, fp16_inputs = make_interpreter(
        fp16_path,
        image_size=args.image_size,
        patch_size=args.patch_size,
        patch_count=args.patch_count,
    )
    fp32_pred = predict_tflite(fp32_interpreter, fp32_inputs, inputs)
    fp16_pred = predict_tflite(fp16_interpreter, fp16_inputs, inputs)

    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "command": command_for_summary(),
        "keras_model": display_path(model_path),
        "val_patch_jsonl": display_path(val_patch_jsonl),
        "output_dir": display_path(output_dir),
        "preprocessing_mode": PREPROCESSING_MODE,
        "sample_count": int(args.max_samples),
        "image_size": int(args.image_size),
        "patch_size": int(args.patch_size),
        "patch_count": int(args.patch_count),
        "label_threshold": float(args.label_threshold),
        "tensorflow": tf_info,
        "conversion_source": "temporary_keras_export_saved_model",
        "keras_model_summary": {
            "name": model.name,
            "parameter_count": int(model.count_params()),
            "inputs": [keras_tensor_summary(tensor) for tensor in model.inputs],
            "outputs": [keras_tensor_summary(tensor) for tensor in model.outputs],
        },
        "exports": {
            "fp32": fp32_export,
            "fp16": fp16_export,
        },
    }

    prediction_rows = build_prediction_rows(examples, keras_pred, fp32_pred, fp16_pred)
    report = {
        "created_at": metadata["created_at"],
        "command": metadata["command"],
        "keras_model": metadata["keras_model"],
        "val_patch_jsonl": metadata["val_patch_jsonl"],
        "sample_count": int(len(examples)),
        "preprocessing_mode": PREPROCESSING_MODE,
        "tflite_models": {
            "fp32": display_path(fp32_path),
            "fp16": display_path(fp16_path),
        },
        "artifacts": {
            "export_metadata_json": display_path(export_metadata_path),
            "parity_report_json": display_path(parity_report_json_path),
            "parity_report_md": display_path(parity_report_md_path),
        },
        "metrics": {
            "fp32": diff_metrics(keras_pred, fp32_pred),
            "fp16": diff_metrics(keras_pred, fp16_pred),
        },
        "prediction_stats": {
            "keras": prediction_stats(keras_pred),
            "fp32": prediction_stats(fp32_pred),
            "fp16": prediction_stats(fp16_pred),
        },
        "shapes": {
            "loaded_inputs": {
                key: list(value.shape)
                for key, value in inputs.items()
            },
            "keras_inputs": metadata["keras_model_summary"]["inputs"],
            "keras_outputs": metadata["keras_model_summary"]["outputs"],
            "fp32_tflite_inputs": [
                tflite_tensor_summary(detail)
                for detail in fp32_interpreter.get_input_details()
            ],
            "fp32_tflite_outputs": [
                tflite_tensor_summary(detail)
                for detail in fp32_interpreter.get_output_details()
            ],
            "fp16_tflite_inputs": [
                tflite_tensor_summary(detail)
                for detail in fp16_interpreter.get_input_details()
            ],
            "fp16_tflite_outputs": [
                tflite_tensor_summary(detail)
                for detail in fp16_interpreter.get_output_details()
            ],
        },
        "predictions": prediction_rows,
        "warnings": warnings,
    }

    write_json(export_metadata_path, metadata)
    write_json(parity_report_json_path, report)
    write_markdown_report(parity_report_md_path, report)
    print(json.dumps(report["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
