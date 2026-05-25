from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

from src.models.nima_distribution import distribution_mean_score, emd_loss, mean_score_mae


def _resize_center_crop(img: Image.Image, image_size: int) -> Image.Image:
    resize_side = max(256, image_size)
    img = img.resize((resize_side, resize_side), Image.Resampling.BILINEAR)
    left = (resize_side - image_size) // 2
    top = (resize_side - image_size) // 2
    return img.crop((left, top, left + image_size, top + image_size))


def load_image(path: Path, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = _resize_center_crop(img, image_size=image_size)
    arr = np.asarray(img).astype("float32") / 255.0
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NIMA distribution inference.")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    model = tf.keras.models.load_model(
        args.model,
        compile=False,
        custom_objects={"emd_loss": emd_loss, "mean_score_mae": mean_score_mae},
    )

    image_path = Path(args.image_path)
    image = load_image(image_path, image_size=args.image_size)
    distribution = np.squeeze(model.predict(image[None, ...], verbose=0)).astype("float32")
    mean_score = float(distribution_mean_score(tf.convert_to_tensor(distribution[None, :]))[0].numpy())

    result = {
        "image_path": str(image_path),
        "distribution": distribution.tolist(),
        "mean_score": mean_score,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
