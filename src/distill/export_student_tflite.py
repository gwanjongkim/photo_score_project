from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from src.distill.config import resolve_repo_path, validate_training_config, write_json
from src.distill.modeling import build_mobile_student_model
from src.distill.preprocessing import student_preprocessing_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the distilled student to TFLite.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--weights_path")
    parser.add_argument("--keras_model_path")
    parser.add_argument("--saved_model_dir")
    parser.add_argument(
        "--export_source",
        choices=["auto", "rebuild", "keras", "saved_model"],
        default="auto",
        help="Try rebuilt-weights export first, or force a specific source form.",
    )
    parser.add_argument(
        "--conversion_mode",
        choices=["auto", "builtin", "select_tf_ops"],
        default="auto",
        help="Prefer builtin-only TFLite, or force a specific converter op set.",
    )
    parser.add_argument("--output_path")
    parser.add_argument("--metadata_path")
    parser.add_argument("--quantization", choices=["no_quant", "float16"], default="no_quant")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def summarize_exception(exc: Exception, limit: int = 600) -> str:
    message = f"{type(exc).__name__}: {exc}"
    message = " ".join(message.split())
    if len(message) > limit:
        message = f"{message[:limit]}..."
    return message


def build_rebuilt_student_model(resolved_config: dict[str, object]) -> tf.keras.Model:
    data_cfg = dict(resolved_config["data"])
    model_cfg = dict(resolved_config["model"])
    model, _ = build_mobile_student_model(
        input_shape=(int(data_cfg["image_size"]), int(data_cfg["image_size"]), 3),
        alpha=float(model_cfg["alpha"]),
        head_units=int(model_cfg["head_units"]),
        dropout_rate=float(model_cfg["dropout_rate"]),
        backbone_trainable=bool(model_cfg["backbone_trainable"]),
        backbone_weights=None,
    )
    return model


def convert_keras_model(model: tf.keras.Model, mode: str, quantization: str) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if mode == "select_tf_ops":
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
    if quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def convert_saved_model(saved_model_dir: Path, mode: str, quantization: str) -> bytes:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    if mode == "select_tf_ops":
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
    if quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def run_conversion_with_callable(convert_fn, conversion_mode: str) -> tuple[bytes, str, str | None]:
    if conversion_mode == "builtin":
        return convert_fn("builtin"), "builtin", None
    if conversion_mode == "select_tf_ops":
        return convert_fn("select_tf_ops"), "select_tf_ops", None

    builtin_error = None
    try:
        return convert_fn("builtin"), "builtin", None
    except Exception as exc:
        builtin_error = summarize_exception(exc)

    tflite_model = convert_fn("select_tf_ops")
    return tflite_model, "select_tf_ops", builtin_error


def collect_model_io_metadata(model: tf.keras.Model) -> dict[str, object]:
    def tensor_to_json(tensor) -> dict[str, object]:
        return {
            "name": str(tensor.name),
            "shape": list(tensor.shape),
            "dtype": tensor.dtype,
        }

    return {
        "inputs": [tensor_to_json(tensor) for tensor in model.inputs],
        "outputs": [tensor_to_json(tensor) for tensor in model.outputs],
    }


def collect_signature_metadata(saved_model_dir: Path) -> dict[str, object]:
    loaded = tf.saved_model.load(str(saved_model_dir))
    signature = loaded.signatures["serving_default"]
    inputs = {
        key: {
            "shape": list(spec.shape),
            "dtype": spec.dtype.name,
        }
        for key, spec in signature.structured_input_signature[1].items()
    }
    outputs = {
        key: {
            "shape": list(spec.shape),
            "dtype": spec.dtype.name,
        }
        for key, spec in signature.structured_outputs.items()
    }
    return {"inputs": inputs, "outputs": outputs}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    checkpoint_dir = resolve_repo_path(args.checkpoint_dir)
    if checkpoint_dir is None:
        raise ValueError("Could not resolve checkpoint_dir.")
    resolved_config = json.loads((checkpoint_dir / "resolved_config.json").read_text(encoding="utf-8"))
    validate_training_config(resolved_config)
    data_cfg = dict(resolved_config["data"])
    train_cfg = dict(resolved_config["train"])

    weights_path = resolve_repo_path(args.weights_path) if args.weights_path else checkpoint_dir / "best.weights.h5"
    keras_model_path = resolve_repo_path(args.keras_model_path) if args.keras_model_path else checkpoint_dir / "final_model.keras"
    saved_model_dir = resolve_repo_path(args.saved_model_dir) if args.saved_model_dir else checkpoint_dir / "saved_model"
    if weights_path is None or keras_model_path is None or saved_model_dir is None:
        raise ValueError("Could not resolve export source paths.")

    tf.keras.mixed_precision.set_global_policy("float32")

    output_path = resolve_repo_path(args.output_path) if args.output_path else checkpoint_dir / "student_model.tflite"
    metadata_path = resolve_repo_path(args.metadata_path) if args.metadata_path else checkpoint_dir / "student_model.metadata.json"
    if output_path is None or metadata_path is None:
        raise ValueError("Could not resolve output_path or metadata_path.")

    if not args.overwrite:
        for existing in (output_path, metadata_path):
            if existing.exists():
                raise FileExistsError(f"{existing} already exists. Use --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_order = {
        "auto": ["rebuild", "keras", "saved_model"],
        "rebuild": ["rebuild"],
        "keras": ["keras"],
        "saved_model": ["saved_model"],
    }[args.export_source]

    tflite_bytes = None
    used_export_source = None
    used_conversion_mode = None
    builtin_failure = None
    export_model_io = None
    saved_model_signature = None
    attempts: list[dict[str, object]] = []

    for source_name in source_order:
        try:
            if source_name == "rebuild":
                if not weights_path.exists():
                    raise FileNotFoundError(f"Weights file not found: {weights_path}")
                model = build_rebuilt_student_model(resolved_config)
                model.load_weights(weights_path)
                tflite_bytes, used_conversion_mode, builtin_failure = run_conversion_with_callable(
                    lambda mode: convert_keras_model(model, mode, args.quantization),
                    args.conversion_mode,
                )
                export_model_io = collect_model_io_metadata(model)
            elif source_name == "keras":
                if not keras_model_path.exists():
                    raise FileNotFoundError(f"Keras model not found: {keras_model_path}")
                model = tf.keras.models.load_model(keras_model_path, compile=False)
                tflite_bytes, used_conversion_mode, builtin_failure = run_conversion_with_callable(
                    lambda mode: convert_keras_model(model, mode, args.quantization),
                    args.conversion_mode,
                )
                export_model_io = collect_model_io_metadata(model)
            else:
                if not saved_model_dir.exists():
                    raise FileNotFoundError(f"SavedModel directory not found: {saved_model_dir}")
                tflite_bytes, used_conversion_mode, builtin_failure = run_conversion_with_callable(
                    lambda mode: convert_saved_model(saved_model_dir, mode, args.quantization),
                    args.conversion_mode,
                )
                saved_model_signature = collect_signature_metadata(saved_model_dir)

            used_export_source = source_name
            attempts.append(
                {
                    "source": source_name,
                    "status": "success",
                    "used_conversion_mode": used_conversion_mode,
                    "requires_select_tf_ops": used_conversion_mode == "select_tf_ops",
                    "builtin_failure": builtin_failure,
                }
            )
            break
        except Exception as exc:
            attempts.append(
                {
                    "source": source_name,
                    "status": "failed",
                    "error": summarize_exception(exc),
                }
            )
            if args.export_source != "auto":
                raise

    if tflite_bytes is None or used_export_source is None or used_conversion_mode is None:
        raise RuntimeError("All export source attempts failed.")

    output_path.write_bytes(tflite_bytes)

    target_column = str(train_cfg.get("target_column", "teacher_aesthetic_score"))
    manifest_source = data_cfg.get("manifest_path") or resolved_config.get("manifest_csv")
    manifest_path = resolve_repo_path(manifest_source) if manifest_source else None
    score_interpretation = (
        "Predicted aesthetic score in [0, 1], distilled against "
        f"{target_column}. Higher is better."
    )

    metadata = {
        "model_name": str(resolved_config.get("experiment_name", "stage5_student_mobilenetv2")),
        "source_checkpoint_dir": str(checkpoint_dir),
        "source_paths": {
            "weights_path": str(weights_path),
            "keras_model_path": str(keras_model_path),
            "saved_model_dir": str(saved_model_dir),
        },
        "distillation": {
            "target_column": target_column,
            "manifest_path": str(manifest_path) if manifest_path is not None else None,
        },
        "input": student_preprocessing_metadata(int(data_cfg["image_size"])),
        "output": {
            "shape": [1, 1],
            "dtype": "float32",
            "interpretation": score_interpretation,
        },
        "export": {
            "requested_export_source": args.export_source,
            "used_export_source": used_export_source,
            "requested_conversion_mode": args.conversion_mode,
            "used_conversion_mode": used_conversion_mode,
            "requires_select_tf_ops": used_conversion_mode == "select_tf_ops",
            "quantization": args.quantization,
            "tflite_path": str(output_path),
            "tflite_size_bytes": int(output_path.stat().st_size),
            "attempts": attempts,
        },
        "export_model_io": export_model_io,
        "saved_model_signature": saved_model_signature,
        "todo_before_mobile_release": [
            "Run numeric parity checks between Keras and TFLite outputs on a held-out image set.",
            "Benchmark latency and memory on the target Flutter runtime/device tier.",
            "Measure whether float16 quantization preserves ranking quality well enough for A-cut use.",
        ],
    }
    write_json(metadata_path, metadata)

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "source_paths": {
            "weights_path": str(weights_path),
            "keras_model_path": str(keras_model_path),
            "saved_model_dir": str(saved_model_dir),
        },
        "output_path": str(output_path),
        "metadata_path": str(metadata_path),
        "used_export_source": used_export_source,
        "used_conversion_mode": used_conversion_mode,
        "quantization": args.quantization,
        "tflite_size_bytes": int(output_path.stat().st_size),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
