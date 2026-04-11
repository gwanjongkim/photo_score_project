from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf


def _build_tokens(img: Image.Image, patch_size: int, scale_sizes: tuple[int, ...], patches_per_scale: int):
    patches_all = []
    positions_all = []
    scale_ids_all = []
    mask_all = []

    rgb = np.asarray(img).astype("float32") / 255.0
    height, width = rgb.shape[:2]

    for scale_idx, target_long in enumerate(scale_sizes):
        scale = float(target_long) / max(height, width)
        resized_h = max(patch_size, int(round(height * scale)))
        resized_w = max(patch_size, int(round(width * scale)))

        resized = np.asarray(img.resize((resized_w, resized_h), Image.Resampling.BILINEAR)).astype("float32") / 255.0
        patches = []
        positions = []
        for top in range(0, resized_h - patch_size + 1, patch_size):
            for left in range(0, resized_w - patch_size + 1, patch_size):
                patches.append(resized[top:top + patch_size, left:left + patch_size, :])
                positions.append([top // patch_size, left // patch_size])
        patches = patches[:patches_per_scale]
        positions = positions[:patches_per_scale]

        count = len(patches)
        while len(patches) < patches_per_scale:
            patches.append(np.zeros((patch_size, patch_size, 3), dtype="float32"))
            positions.append([0.0, 0.0])

        patches_all.extend(patches)
        positions_all.extend(positions)
        scale_ids_all.extend([scale_idx] * patches_per_scale)
        mask_all.extend([1.0] * count + [0.0] * (patches_per_scale - count))

    return (
        np.asarray(patches_all, dtype="float32"),
        np.asarray(positions_all, dtype="float32"),
        np.asarray(scale_ids_all, dtype="int32"),
        np.asarray(mask_all, dtype="float32"),
    )


def _parse_scale_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MUSIQ-style inference.")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--scale_sizes", default="224,384,512")
    parser.add_argument("--patches_per_scale", type=int, default=16)
    args = parser.parse_args()

    image_path = Path(args.image_path)
    img = Image.open(image_path).convert("RGB")
    scale_sizes = _parse_scale_sizes(args.scale_sizes)
    patches, positions, scale_ids, token_mask = _build_tokens(
        img=img,
        patch_size=args.patch_size,
        scale_sizes=scale_sizes,
        patches_per_scale=args.patches_per_scale,
    )

    model = tf.keras.models.load_model(args.model, compile=False)
    score = float(
        np.squeeze(
            model.predict(
                {
                    "patches": patches[None, ...],
                    "positions": positions[None, ...],
                    "scale_ids": scale_ids[None, ...],
                    "token_mask": token_mask[None, ...],
                },
                verbose=0,
            )
        )
    )

    result = {
        "image_path": str(image_path),
        "musiq_score": score,
        "patch_size": args.patch_size,
        "scale_sizes": list(scale_sizes),
        "patches_per_scale": args.patches_per_scale,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
