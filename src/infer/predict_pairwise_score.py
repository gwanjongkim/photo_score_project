from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf

from src.models.pairwise_comparator import AbsoluteDifference
from src.utils.pairwise_utils import pairwise_probabilities_to_strength, recover_score_from_pairwise_matrix


def load_image(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(image).astype("float32") / 255.0


def compare_pair(model: tf.keras.Model, image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    return np.squeeze(model.predict({"image_a": image_a[None, ...], "image_b": image_b[None, ...]}, verbose=0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover an approximate scalar aesthetics score from pairwise comparisons.")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--reference_csv", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--target_col", default="score")
    parser.add_argument("--max_references", type=int, default=16)
    args = parser.parse_args()

    reference_df = pd.read_csv(args.reference_csv)
    if "image_path" not in reference_df.columns or args.target_col not in reference_df.columns:
        raise ValueError("reference_csv must contain image_path and target_col.")

    reference_df = reference_df[["image_path", args.target_col]].copy().head(args.max_references)
    reference_scores = reference_df[args.target_col].astype("float32").to_numpy()
    reference_images = [load_image(Path(path), image_size=args.image_size) for path in reference_df["image_path"]]

    query_path = Path(args.image_path)
    query_image = load_image(query_path, image_size=args.image_size)
    model = tf.keras.models.load_model(
        args.model_path,
        compile=False,
        safe_mode=False,
        custom_objects={"AbsoluteDifference": AbsoluteDifference},
    )

    query_strengths = []
    for ref_image in reference_images:
        probs = compare_pair(model, query_image, ref_image)
        query_strengths.append(pairwise_probabilities_to_strength(probs))

    ref_matrix = np.ones((len(reference_images), len(reference_images)), dtype=np.float32)
    for i in range(len(reference_images)):
        for j in range(i + 1, len(reference_images)):
            probs = compare_pair(model, reference_images[i], reference_images[j])
            strength = pairwise_probabilities_to_strength(probs)
            ref_matrix[i, j] = strength
            ref_matrix[j, i] = 1.0 / max(strength, 1e-6)

    score, eigen_scores = recover_score_from_pairwise_matrix(
        query_vs_refs=np.asarray(query_strengths, dtype=np.float32),
        reference_scores=reference_scores,
        reference_vs_reference=ref_matrix,
    )

    print(
        json.dumps(
            {
                "image_path": str(query_path),
                "pairwise_recovered_score": float(score),
                "num_references": int(len(reference_images)),
                "query_eigen_weight": float(eigen_scores[0]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
