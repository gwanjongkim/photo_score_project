from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from src.export.tflite_presets import build_export_model, get_preset, preset_choices, resolve_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a priority photo-score checkpoint to TFLite.")
    parser.add_argument("--preset", required=True, choices=preset_choices())
    parser.add_argument(
        "--model_path",
        help="Optional checkpoint directory, weights path, .keras path, or SavedModel dir. Defaults to the preset checkpoint.",
    )
    parser.add_argument(
        "--export_source",
        choices=["auto", "rebuild", "saved_model"],
        default="auto",
        help="Prefer rebuilding the model from best.weights.h5 or force SavedModel conversion.",
    )
    parser.add_argument(
        "--conversion_mode",
        choices=["auto", "builtin", "select_tf_ops"],
        default="auto",
        help="Try builtin-only first or force a specific conversion mode.",
    )
    parser.add_argument("--output_dir", default="exports/tflite")
    parser.add_argument("--output_name", help="Optional output filename. Defaults to the preset filename.")
    parser.add_argument("--metadata_path", help="Optional metadata JSON path.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def convert_saved_model(saved_model_dir: Path, mode: str) -> bytes:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    if mode == "select_tf_ops":
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
    return converter.convert()


def convert_keras_model(model: tf.keras.Model, mode: str) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if mode == "select_tf_ops":
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
    return converter.convert()


def summarize_exception(exc: Exception, limit: int = 600) -> str:
    message = f"{type(exc).__name__}: {exc}"
    message = " ".join(message.split())
    if len(message) > limit:
        message = f"{message[:limit]}..."
    return message


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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    preset = get_preset(args.preset)
    checkpoint_path, weights_path, saved_model_dir = resolve_paths(args.preset, args.model_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or preset.default_output_name
    output_path = output_dir / output_name
    metadata_path = Path(args.metadata_path) if args.metadata_path else output_path.with_suffix(".metadata.json")

    if not args.overwrite:
        for existing in (output_path, metadata_path):
            if existing.exists():
                raise FileExistsError(f"{existing} already exists. Use --overwrite to replace it.")

    tf.keras.mixed_precision.set_global_policy("float32")

    tflite_model = None
    used_mode = None
    builtin_error = None
    used_export_source = None
    rebuild_failure = None
    export_model_io = None
    saved_model_signature = None

    if args.export_source in {"auto", "rebuild"}:
        try:
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
            export_model = build_export_model(args.preset)
            export_model.load_weights(weights_path)
            tflite_model, used_mode, builtin_error = run_conversion_with_callable(
                lambda mode: convert_keras_model(export_model, mode),
                args.conversion_mode,
            )
            used_export_source = "rebuild"
            export_model_io = collect_model_io_metadata(export_model)
        except Exception as exc:
            rebuild_failure = summarize_exception(exc)
            if args.export_source == "rebuild":
                raise

    if tflite_model is None:
        tflite_model, used_mode, builtin_error = run_conversion_with_callable(
            lambda mode: convert_saved_model(saved_model_dir, mode),
            args.conversion_mode,
        )
        used_export_source = "saved_model"
        saved_model_signature = collect_signature_metadata(saved_model_dir)

    output_path.write_bytes(tflite_model)

    metadata = {
        **preset.to_metadata(),
        "source": {
            "checkpoint_path": str(checkpoint_path),
            "weights_path": str(weights_path),
            "saved_model_dir": str(saved_model_dir),
        },
        "export": {
            "requested_export_source": args.export_source,
            "used_export_source": used_export_source,
            "rebuild_failure": rebuild_failure,
            "requested_conversion_mode": args.conversion_mode,
            "used_conversion_mode": used_mode,
            "requires_select_tf_ops": used_mode == "select_tf_ops",
            "tflite_path": str(output_path),
            "tflite_size_bytes": output_path.stat().st_size,
            "builtin_failure": builtin_error,
        },
        "saved_model_signature": saved_model_signature,
        "export_model_io": export_model_io,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
