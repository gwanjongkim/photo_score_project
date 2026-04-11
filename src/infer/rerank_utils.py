from __future__ import annotations

import math
from collections.abc import Sequence


def to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def minmax_normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []

    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum
    if span <= 1e-12:
        return [0.5 for _ in values]
    return [(value - minimum) / span for value in values]


def centered_pool_deltas(values: Sequence[float], weight: float) -> list[dict[str, float]]:
    normalized = minmax_normalize(values)
    if not normalized:
        return []

    mean_normalized = sum(normalized) / len(normalized)
    effective_weight = max(0.0, float(weight))
    return [
        {
            "normalized": float(score),
            "centered": float(score - mean_normalized),
            "delta": float(effective_weight * (score - mean_normalized)),
        }
        for score in normalized
    ]
