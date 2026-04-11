from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

from src.models.rgnet import GraphConvolution, RegionGraphBuilder, RegionWeightedPooling


def load_image(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(image).astype("float32") / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RGNet-style composition-aware aesthetics inference.")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    model = tf.keras.models.load_model(
        args.model_path,
        compile=False,
        safe_mode=False,
        custom_objects={
            "RegionGraphBuilder": RegionGraphBuilder,
            "GraphConvolution": GraphConvolution,
            "RegionWeightedPooling": RegionWeightedPooling,
        },
    )

    image_path = Path(args.image_path)
    image = load_image(image_path, image_size=args.image_size)
    score = float(np.squeeze(model.predict(image[None, ...], verbose=0)))

    print(
        json.dumps(
            {
                "image_path": str(image_path),
                "rgnet_score": score,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
