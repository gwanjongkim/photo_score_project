from __future__ import annotations

import argparse
import json

import numpy as np

from src.export.alamp_tflite import (
    DEFAULT_GLOBAL_SIZE,
    DEFAULT_NUM_PATCHES,
    DEFAULT_PATCH_SCALES,
    DEFAULT_PATCH_SIZE,
    load_checkpoint_model,
    load_rebuild_model,
    prepare_sample_inputs,
    resolve_alamp_paths,
    run_source_prediction,
    run_tflite_inference,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the exported A-Lamp TFLite model with a real sample image.")
    parser.add_argument("--checkpoint_dir", default="checkpoints/alamp_aadb_gpu")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--output_json")
    parser.add_argument("--global_size", type=int, default=DEFAULT_GLOBAL_SIZE)
    parser.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--num_patches", type=int, default=DEFAULT_NUM_PATCHES)
    parser.add_argument("--num_threads", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = resolve_alamp_paths(args.checkpoint_dir)
    inputs, debug = prepare_sample_inputs(
        image_path=args.image_path,
        global_size=args.global_size,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
        patch_scales=DEFAULT_PATCH_SCALES,
    )

    rebuild_model = load_rebuild_model(
        weights_path=paths.weights_path,
        global_size=args.global_size,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
    )
    checkpoint_model = load_checkpoint_model(paths.keras_path)

    tflite_result = run_tflite_inference(args.model_path, inputs=inputs, num_threads=args.num_threads)
    rebuild_output = run_source_prediction(rebuild_model, inputs)
    checkpoint_output = run_source_prediction(checkpoint_model, inputs)

    tflite_output = np.asarray(tflite_result["output"], dtype=np.float32)
    rebuild_output = np.asarray(rebuild_output, dtype=np.float32)
    checkpoint_output = np.asarray(checkpoint_output, dtype=np.float32)

    diff_vs_rebuild = np.abs(tflite_output - rebuild_output)
    diff_rebuild_vs_checkpoint = np.abs(rebuild_output - checkpoint_output)

    scalar = float(np.squeeze(tflite_output))
    result = {
        "model_path": str(args.model_path),
        "image_path": str(args.image_path),
        "checkpoint_dir": str(paths.checkpoint_dir),
        "input_details": tflite_result["input_details"],
        "output_details": tflite_result["output_details"],
        "preprocessing": {
            "global_size": args.global_size,
            "patch_size": args.patch_size,
            "num_patches": args.num_patches,
            "patch_scales": list(DEFAULT_PATCH_SCALES),
            "global_view_shape": list(inputs["global_view"].shape),
            "patches_shape": list(inputs["patches"].shape),
        },
        "patch_debug": debug,
        "outputs": {
            "tflite": float(np.squeeze(tflite_output)),
            "rebuild_source": float(np.squeeze(rebuild_output)),
            "checkpoint_model": float(np.squeeze(checkpoint_output)),
        },
        "comparison": {
            "tflite_vs_rebuild_max_abs_diff": float(diff_vs_rebuild.max()),
            "tflite_vs_rebuild_mean_abs_diff": float(diff_vs_rebuild.mean()),
            "rebuild_vs_checkpoint_max_abs_diff": float(diff_rebuild_vs_checkpoint.max()),
            "rebuild_vs_checkpoint_mean_abs_diff": float(diff_rebuild_vs_checkpoint.mean()),
            "tflite_matches_export_source": bool(float(diff_vs_rebuild.max()) < 1e-3),
            "rebuild_close_to_checkpoint_model": bool(float(diff_rebuild_vs_checkpoint.max()) < 5e-2),
        },
        "plausibility": {
            "finite": bool(np.isfinite(scalar)),
            "within_unit_interval": bool(0.0 <= scalar <= 1.0),
        },
        "notes": [
            "The exported TFLite model is validated against the float32 rebuild used for conversion.",
            (
                "The mixed-precision .keras checkpoint is reported separately because the direct checkpoint/SavedModel "
                "conversion path is blocked by LayoutCueAugmentation BroadcastTo on float16 tensors."
            ),
        ],
        "verification_passed": bool(
            np.isfinite(scalar)
            and float(diff_vs_rebuild.max()) < 1e-3
        ),
    }

    if args.output_json:
        write_json(args.output_json, result)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["verification_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
