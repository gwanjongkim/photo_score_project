from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

from src.datasets.native_size_dataset import prepare_alamp_inputs
from src.models.alamp import LayoutCueAugmentation, WeightedPatchPooling


def _load_image_tensor(path: Path) -> tuple[Image.Image, tf.Tensor]:
    pil = Image.open(path).convert("RGB")
    arr = np.asarray(pil).astype("float32") / 255.0
    return pil, tf.convert_to_tensor(arr, dtype=tf.float32)


def _normalized_boxes_to_pixels(boxes: np.ndarray, width: int, height: int) -> list[list[int]]:
    out = []
    for y1, x1, y2, x2 in boxes.tolist():
        out.append(
            [
                int(round(x1 * width)),
                int(round(y1 * height)),
                int(round(x2 * width)),
                int(round(y2 * height)),
            ]
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run refined A-Lamp-style inference.")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--global_size", type=int, default=384)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--num_patches", type=int, default=5)
    parser.add_argument("--patch_scale", type=float, default=0.35)
    parser.add_argument("--include_debug", action="store_true")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    pil_image, image_tensor = _load_image_tensor(image_path)
    patch_scales = tuple(sorted({max(0.18, args.patch_scale * 0.7), args.patch_scale, min(0.55, args.patch_scale * 1.35)}))
    global_view, patches, boxes, proposal_scores = prepare_alamp_inputs(
        image=image_tensor,
        global_size=args.global_size,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
        patch_scales=patch_scales,
    )

    model = tf.keras.models.load_model(
        args.model,
        compile=False,
        safe_mode=False,
        custom_objects={
            "LayoutCueAugmentation": LayoutCueAugmentation,
            "WeightedPatchPooling": WeightedPatchPooling,
        },
    )

    inputs = {
        "global_view": global_view[None, ...].numpy(),
        "patches": patches[None, ...].numpy(),
    }
    score = float(np.squeeze(model.predict(inputs, verbose=0)))

    result = {
        "image_path": str(image_path),
        "alamp_score": score,
        "global_size": args.global_size,
        "patch_size": args.patch_size,
        "num_patches": args.num_patches,
    }

    if args.include_debug:
        attention_model = tf.keras.Model(
            model.inputs,
            {
                "score": model.output,
                "patch_attention": model.get_layer("patch_attention").output,
            },
        )
        debug_pred = attention_model.predict(inputs, verbose=0)
        patch_attention = np.squeeze(debug_pred["patch_attention"], axis=(0, 2)).astype("float32")
        pixel_boxes = _normalized_boxes_to_pixels(
            boxes.numpy(),
            width=pil_image.width,
            height=pil_image.height,
        )

        result["selected_patch_boxes"] = pixel_boxes
        result["patch_selection_scores"] = proposal_scores.numpy().astype("float32").tolist()
        result["patch_attention_weights"] = patch_attention.tolist()
        result["patch_debug"] = [
            {
                "box_xyxy": box,
                "selection_score": float(selection),
                "attention_weight": float(attn),
            }
            for box, selection, attn in zip(pixel_boxes, result["patch_selection_scores"], result["patch_attention_weights"])
        ]

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
