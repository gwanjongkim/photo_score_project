from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.export.tflite_presets import (
    describe_output,
    get_preset,
    load_image_array,
    load_source_model,
    preset_choices,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a real TFLite inference smoke test for a priority mobile model.")
    parser.add_argument("--preset", required=True, choices=preset_choices())
    parser.add_argument("--model_path", required=True, help="Path to a .tflite file.")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--source_checkpoint", help="Optional .keras checkpoint to compare against.")
    parser.add_argument("--output_json", help="Optional path to write the verification result.")
    parser.add_argument("--num_threads", type=int, default=1)
    return parser


def tensor_detail_to_json(detail: dict[str, object]) -> dict[str, object]:
    return {
        "name": str(detail["name"]),
        "shape": [int(dim) for dim in detail["shape"]],
        "shape_signature": [int(dim) for dim in detail["shape_signature"]],
        "dtype": np.dtype(detail["dtype"]).name,
        "index": int(detail["index"]),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    preset = get_preset(args.preset)
    model_path = Path(args.model_path)
    image_path = Path(args.image_path)
    source_checkpoint = Path(args.source_checkpoint or preset.weights_path)

    image_array = load_image_array(image_path, image_size=preset.input_size)
    batched_input = image_array[None, ...].astype(np.float32)

    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=args.num_threads)
    input_details = interpreter.get_input_details()
    if len(input_details) != 1:
        raise ValueError(f"Expected a single-input TFLite model, found {len(input_details)} inputs.")
    interpreter.resize_tensor_input(input_details[0]["index"], batched_input.shape, strict=False)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_tensor = batched_input.astype(input_details[0]["dtype"], copy=False)
    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]["index"])

    source_model = load_source_model(args.preset, source_checkpoint)
    source_output = source_model.predict(batched_input, verbose=0)

    tflite_output_np = np.asarray(tflite_output, dtype=np.float32)
    source_output_np = np.asarray(source_output, dtype=np.float32)
    diff = np.abs(tflite_output_np - source_output_np)

    tflite_summary = describe_output(args.preset, tflite_output_np)
    source_summary = describe_output(args.preset, source_output_np)
    max_abs_diff = float(diff.max())
    mean_abs_diff = float(diff.mean())
    comparison_ok = bool(max_abs_diff < 1e-2)
    plausible = bool(tflite_summary.get("plausible", False))
    verification_passed = bool(plausible and comparison_ok)

    result = {
        "preset": args.preset,
        "model_path": str(model_path),
        "image_path": str(image_path),
        "source_checkpoint": str(source_checkpoint),
        "input_details": [tensor_detail_to_json(detail) for detail in input_details],
        "output_details": [tensor_detail_to_json(detail) for detail in output_details],
        "tflite_output": tflite_summary,
        "source_output": source_summary,
        "comparison": {
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "comparison_ok": comparison_ok,
        },
        "verification_passed": verification_passed,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if not verification_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
