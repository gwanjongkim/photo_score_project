from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.export.alamp_tflite import (
    DEFAULT_GLOBAL_SIZE,
    DEFAULT_NUM_PATCHES,
    DEFAULT_PATCH_SIZE,
    collect_model_io,
    collect_saved_model_signature,
    convert_keras_model,
    convert_saved_model,
    load_rebuild_model,
    resolve_alamp_paths,
    summarize_exception,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the A-Lamp AADB checkpoint to TFLite.")
    parser.add_argument("--checkpoint_dir", default="checkpoints/alamp_aadb_gpu")
    parser.add_argument("--output_dir", default="exports/tflite")
    parser.add_argument("--output_name", default="alamp_aadb_gpu.tflite")
    parser.add_argument("--metadata_path")
    parser.add_argument("--global_size", type=int, default=DEFAULT_GLOBAL_SIZE)
    parser.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--num_patches", type=int, default=DEFAULT_NUM_PATCHES)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--diagnose_saved_model",
        action="store_true",
        help="Also attempt SavedModel conversion and record the failure or success in metadata.",
    )
    return parser


def ensure_writable(*paths: Path, overwrite: bool) -> None:
    if overwrite:
        return
    for path in paths:
        if path.exists():
            raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")


def main() -> None:
    args = build_parser().parse_args()
    paths = resolve_alamp_paths(args.checkpoint_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    metadata_path = Path(args.metadata_path) if args.metadata_path else output_path.with_suffix(".metadata.json")
    ensure_writable(output_path, metadata_path, overwrite=args.overwrite)

    model = load_rebuild_model(
        weights_path=paths.weights_path,
        global_size=args.global_size,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
    )

    rebuild_attempts: list[dict[str, object]] = []
    tflite_model = None
    for mode in ("builtin", "select_tf_ops"):
        try:
            tflite_model = convert_keras_model(model, mode=mode)
            rebuild_attempts.append({"mode": mode, "status": "success", "size_bytes": len(tflite_model)})
            used_mode = mode
            break
        except Exception as exc:
            rebuild_attempts.append(
                {
                    "mode": mode,
                    "status": "failure",
                    "error": summarize_exception(exc),
                }
            )

    if tflite_model is None:
        raise RuntimeError(f"All rebuild conversion attempts failed: {json.dumps(rebuild_attempts, indent=2)}")

    saved_model_attempts: list[dict[str, object]] = []
    if args.diagnose_saved_model:
        for mode in ("builtin", "select_tf_ops"):
            try:
                saved_bytes = convert_saved_model(paths.saved_model_dir, mode=mode)
                saved_model_attempts.append({"mode": mode, "status": "success", "size_bytes": len(saved_bytes)})
            except Exception as exc:
                saved_model_attempts.append(
                    {
                        "mode": mode,
                        "status": "failure",
                        "error": summarize_exception(exc),
                    }
                )

    output_path.write_bytes(tflite_model)

    metadata = {
        "model": {
            "name": "alamp_aadb_gpu",
            "task": "aesthetic_regression",
            "variant": "A-Lamp",
            "inputs": {
                "global_view": {
                    "shape": [1, args.global_size, args.global_size, 3],
                    "dtype": "float32",
                    "normalization": "pixel_value / 255.0",
                    "layout": "NHWC",
                },
                "patches": {
                    "shape": [1, args.num_patches, args.patch_size, args.patch_size, 3],
                    "dtype": "float32",
                    "normalization": "pixel_value / 255.0",
                    "layout": "NTHWC",
                },
            },
            "output": {
                "shape": [1, 1],
                "dtype": "float32",
                "summary": "Single scalar A-Lamp aesthetic score.",
            },
        },
        "source": {
            "checkpoint_dir": str(paths.checkpoint_dir),
            "weights_path": str(paths.weights_path),
            "keras_path": str(paths.keras_path),
            "saved_model_dir": str(paths.saved_model_dir),
        },
        "export": {
            "used_export_source": "rebuild_from_weights",
            "used_conversion_mode": used_mode,
            "requires_select_tf_ops": used_mode == "select_tf_ops",
            "tflite_path": str(output_path),
            "tflite_size_bytes": output_path.stat().st_size,
            "rebuild_attempts": rebuild_attempts,
            "saved_model_attempts": saved_model_attempts,
        },
        "export_adjustments": {
            "backbone_weights_on_rebuild": None,
            "keras_global_policy": "float32",
            "reason": (
                "Rebuild export uses the repo source model under float32 policy and loads checkpoint weights. "
                "This avoids the SavedModel mixed-precision layout cue BroadcastTo path that fails TFLite conversion."
            ),
        },
        "saved_model_signature": collect_saved_model_signature(paths.saved_model_dir),
        "rebuild_model_io": collect_model_io(model),
    }
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
