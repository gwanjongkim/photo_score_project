from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class SimilarityConfig:
    threshold: float = 0.82
    duplicate_threshold: float = 0.88
    penalty_strength: float = 0.12
    thumbnail_size: int = 32
    hash_size: int = 16
    histogram_bins: int = 8


@dataclass
class SimilarityFeatures:
    image_path: str
    grayscale_embedding: np.ndarray
    color_histogram: np.ndarray
    dhash_bits: np.ndarray
    width: int
    height: int


def sanitize_similarity_config(config: SimilarityConfig) -> SimilarityConfig:
    threshold = float(np.clip(config.threshold, 0.0, 1.0))
    duplicate_threshold = float(np.clip(config.duplicate_threshold, threshold, 1.0))
    penalty_strength = float(max(0.0, config.penalty_strength))
    thumbnail_size = max(8, int(config.thumbnail_size))
    hash_size = max(4, int(config.hash_size))
    histogram_bins = max(2, int(config.histogram_bins))
    return SimilarityConfig(
        threshold=threshold,
        duplicate_threshold=duplicate_threshold,
        penalty_strength=penalty_strength,
        thumbnail_size=thumbnail_size,
        hash_size=hash_size,
        histogram_bins=histogram_bins,
    )


def compute_similarity_features(
    image_path: str | Path,
    thumbnail_size: int = 32,
    hash_size: int = 16,
    histogram_bins: int = 8,
) -> SimilarityFeatures:
    image_path = Path(image_path)
    with Image.open(image_path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
        width, height = image.size

        thumb = image.resize((thumbnail_size, thumbnail_size), Image.Resampling.BILINEAR)
        gray = np.asarray(thumb.convert("L"), dtype=np.float32) / 255.0
        embedding = gray.reshape(-1)
        embedding = embedding - float(np.mean(embedding))
        norm = float(np.linalg.norm(embedding))
        if norm > 1e-6:
            embedding = embedding / norm
        else:
            embedding = np.zeros_like(embedding)

        rgb = np.asarray(thumb, dtype=np.float32) / 255.0
        hist = np.histogramdd(
            rgb.reshape(-1, 3),
            bins=histogram_bins,
            range=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        )[0].astype(np.float32)
        hist = hist.reshape(-1)
        hist_sum = float(hist.sum())
        if hist_sum > 0.0:
            hist = hist / hist_sum

        dhash_image = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        dhash_arr = np.asarray(dhash_image, dtype=np.float32)
        dhash_bits = (dhash_arr[:, 1:] >= dhash_arr[:, :-1]).reshape(-1)

    return SimilarityFeatures(
        image_path=str(image_path),
        grayscale_embedding=embedding.astype(np.float32),
        color_histogram=hist.astype(np.float32),
        dhash_bits=dhash_bits.astype(bool),
        width=int(width),
        height=int(height),
    )


def _safe_cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a <= 1e-6 and norm_b <= 1e-6:
        return 1.0
    if norm_a <= 1e-6 or norm_b <= 1e-6:
        return 0.0
    return float(np.clip(np.dot(vector_a, vector_b) / (norm_a * norm_b), -1.0, 1.0))


def compare_similarity_features(features_a: SimilarityFeatures, features_b: SimilarityFeatures) -> dict[str, float]:
    thumbnail_cosine = _safe_cosine_similarity(features_a.grayscale_embedding, features_b.grayscale_embedding)
    thumbnail_similarity = float(np.clip((thumbnail_cosine + 1.0) * 0.5, 0.0, 1.0))
    histogram_similarity = float(
        np.minimum(features_a.color_histogram, features_b.color_histogram).sum()
    )
    hamming_distance = float(np.count_nonzero(features_a.dhash_bits != features_b.dhash_bits))
    dhash_similarity = float(1.0 - hamming_distance / max(1, features_a.dhash_bits.size))

    combined_similarity = float(
        np.clip(
            0.55 * thumbnail_similarity + 0.25 * dhash_similarity + 0.20 * histogram_similarity,
            0.0,
            1.0,
        )
    )
    return {
        "combined": combined_similarity,
        "thumbnail_similarity": thumbnail_similarity,
        "histogram_similarity": histogram_similarity,
        "dhash_similarity": dhash_similarity,
    }
