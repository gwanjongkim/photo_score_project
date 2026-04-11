"""
Pairwise utilities for Siamese aesthetics comparison.

Based on:
- Image Aesthetic Assessment Based on Pairwise Comparison (ICCV 2019)

Faithful parts:
- pairwise supervision over image pairs
- relative-aesthetics probabilities converted into comparison strengths
- optional reference-based scalar score recovery

Approximated parts:
- pair generation uses deterministic score-distance sampling
- score recovery uses a repo-practical reciprocal matrix plus principal eigenvector
- personalization is not implemented in this branch
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_pairwise_frame(
    csv_path: str | Path,
    target_col: str = "score",
    max_pairs: int = 20000,
    similar_margin: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    if "image_path" not in df.columns:
        raise ValueError(f"{csv_path} must contain image_path.")
    if target_col not in df.columns:
        raise ValueError(f"{csv_path} must contain {target_col}.")

    df = df[["image_path", target_col]].copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].notna()].reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Need at least two rows to build pairwise dataset.")

    rng = np.random.default_rng(random_state)
    scores = df[target_col].to_numpy(dtype=np.float32)
    order = np.argsort(scores)
    df = df.iloc[order].reset_index(drop=True)
    scores = df[target_col].to_numpy(dtype=np.float32)

    num_samples = len(df)
    pair_budget = min(max_pairs, max(1, num_samples * 6))
    anchors = rng.integers(0, num_samples, size=pair_budget)
    offsets = rng.integers(1, min(64, num_samples), size=pair_budget)
    directions = rng.choice(np.array([-1, 1]), size=pair_budget)
    partners = np.clip(anchors + offsets * directions, 0, num_samples - 1)

    keep = anchors != partners
    anchors = anchors[keep]
    partners = partners[keep]

    score_a = scores[anchors]
    score_b = scores[partners]
    delta = score_a - score_b

    labels = np.full(len(delta), 1, dtype=np.int32)
    labels[delta > similar_margin] = 0
    labels[delta < -similar_margin] = 2

    out = pd.DataFrame(
        {
            "image_a": df.iloc[anchors]["image_path"].to_numpy(),
            "image_b": df.iloc[partners]["image_path"].to_numpy(),
            "score_a": score_a,
            "score_b": score_b,
            "label": labels,
            "score_delta": delta.astype(np.float32),
        }
    )
    return out.drop_duplicates(subset=["image_a", "image_b"]).reset_index(drop=True)


def pairwise_probabilities_to_strength(probabilities: np.ndarray, eps: float = 1e-4) -> float:
    p_a_better = float(probabilities[0])
    p_similar = float(probabilities[1])
    preference = np.clip(p_a_better + 0.5 * p_similar, eps, 1.0 - eps)
    return float(preference / (1.0 - preference))


def principal_eigenvector_scores(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eig(matrix)
    principal = np.argmax(values.real)
    weights = np.abs(vectors[:, principal].real)
    weights = np.maximum(weights, 1e-8)
    return weights / weights.sum()


def recover_score_from_pairwise_matrix(
    query_vs_refs: np.ndarray,
    reference_scores: np.ndarray,
    reference_vs_reference: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    num_refs = len(reference_scores)
    matrix = np.ones((num_refs + 1, num_refs + 1), dtype=np.float32)

    if reference_vs_reference is not None:
        matrix[1:, 1:] = reference_vs_reference

    for idx, strength in enumerate(query_vs_refs):
        matrix[0, idx + 1] = strength
        matrix[idx + 1, 0] = 1.0 / max(strength, 1e-6)

    eigen_scores = principal_eigenvector_scores(matrix)
    query_weight = eigen_scores[0]
    ref_weights = eigen_scores[1:]
    normalized_refs = ref_weights / np.maximum(ref_weights.sum(), 1e-8)
    recovered = float(np.sum(normalized_refs * reference_scores))

    # Blend by relative eigen strength so the query can move outside the plain reference average.
    strength_ratio = query_weight / np.maximum(ref_weights.mean(), 1e-8)
    scale = np.clip(strength_ratio / (1.0 + strength_ratio), 0.1, 0.9)
    recovered = float(scale * recovered + (1.0 - scale) * float(np.mean(reference_scores)))
    return recovered, eigen_scores
