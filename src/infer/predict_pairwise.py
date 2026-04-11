from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

from src.models.pairwise_comparator import AbsoluteDifference


def load_image(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(image).astype("float32") / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct pairwise aesthetics comparison.")
    parser.add_argument("--image_a", required=True)
    parser.add_argument("--image_b", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    model = tf.keras.models.load_model(
        args.model_path,
        compile=False,
        safe_mode=False,
        custom_objects={"AbsoluteDifference": AbsoluteDifference},
    )

    image_a = load_image(Path(args.image_a), image_size=args.image_size)
    image_b = load_image(Path(args.image_b), image_size=args.image_size)
    probs = np.squeeze(
        model.predict({"image_a": image_a[None, ...], "image_b": image_b[None, ...]}, verbose=0)
    ).astype("float32")

    print(
        json.dumps(
            {
                "image_a": args.image_a,
                "image_b": args.image_b,
                "probability_a_better": float(probs[0]),
                "probability_similar": float(probs[1]),
                "probability_b_better": float(probs[2]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
