from __future__ import annotations

import json
from pathlib import Path

from src.infer.select_best_shots import load_weights, score_component


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEACHER_CONFIG_PATH = PROJECT_ROOT / "configs" / "student_distill_teacher.json"
DEFAULT_AGGREGATION_MODE = "production_aesthetic_component"
DEFAULT_TEACHER_SCORE_KEYS = (
    "aadb_score",
    "nima_mean_score",
    "alamp_score",
    "musiq_score",
    "rgnet_score",
)


def resolve_repo_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_teacher_config(config_path: str | Path | None = None) -> tuple[dict[str, object], Path]:
    resolved_path = resolve_repo_path(config_path or DEFAULT_TEACHER_CONFIG_PATH)
    if resolved_path is None:
        raise ValueError("Teacher config path could not be resolved.")
    with resolved_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    aggregation_mode = str(config.get("aggregation_mode") or DEFAULT_AGGREGATION_MODE).strip()
    if aggregation_mode not in {"production_aesthetic_component", "explicit_weights"}:
        raise ValueError(
            "Teacher config aggregation_mode must be one of: production_aesthetic_component, explicit_weights."
        )
    config["aggregation_mode"] = aggregation_mode

    score_weights = dict(config.get("score_weights") or {})
    if aggregation_mode == "explicit_weights":
        if not score_weights:
            raise ValueError("Teacher config with aggregation_mode=explicit_weights must define non-empty score_weights.")
        config["score_weights"] = {str(key): float(value) for key, value in score_weights.items()}
    else:
        config["score_weights"] = {str(key): float(value) for key, value in score_weights.items()}

    if "required_score_keys" in config:
        config["required_score_keys"] = list(config.get("required_score_keys") or [])
    else:
        config["required_score_keys"] = list(DEFAULT_TEACHER_SCORE_KEYS)
    config["allow_partial_teacher_scores"] = bool(
        config.get("allow_partial_teacher_scores", aggregation_mode == "production_aesthetic_component")
    )
    weights_config = config.get("production_selector_weights_config")
    config["production_selector_weights_config"] = str(weights_config) if weights_config else None
    return config, resolved_path


def normalize_teacher_score(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if -0.05 <= numeric <= 1.05:
        return max(0.0, min(1.0, numeric))
    if 1.0 <= numeric <= 10.5:
        return max(0.0, min(1.0, (numeric - 1.0) / 9.0))
    if 0.0 <= numeric <= 100.0:
        return max(0.0, min(1.0, numeric / 100.0))
    return None


def aggregate_teacher_scores(
    record: dict[str, object],
    score_weights: dict[str, float],
    required_score_keys: list[str] | tuple[str, ...] | None = None,
    allow_partial_teacher_scores: bool = False,
) -> tuple[float | None, dict[str, dict[str, float]], list[str]]:
    required = list(required_score_keys or [])
    contributions: dict[str, dict[str, float]] = {}
    missing_score_keys: list[str] = []
    available: list[tuple[str, float, float, float]] = []

    for score_key, configured_weight in score_weights.items():
        raw_value = record.get(score_key)
        normalized_value = normalize_teacher_score(raw_value)
        if normalized_value is None:
            missing_score_keys.append(score_key)
            continue
        available.append((score_key, float(configured_weight), float(raw_value), float(normalized_value)))

    if not available:
        return None, contributions, missing_score_keys

    if required and not allow_partial_teacher_scores:
        required_missing = [score_key for score_key in required if score_key in missing_score_keys]
        if required_missing:
            return None, contributions, missing_score_keys

    total_weight = sum(weight for _, weight, _, _ in available)
    if total_weight <= 0.0:
        raise ValueError("Teacher score_weights must sum to a positive value.")

    teacher_score = 0.0
    for score_key, configured_weight, raw_value, normalized_value in available:
        effective_weight = configured_weight / total_weight
        weighted_contribution = effective_weight * normalized_value
        teacher_score += weighted_contribution
        contributions[score_key] = {
            "raw_score": raw_value,
            "normalized_score": normalized_value,
            "configured_weight": configured_weight,
            "effective_weight": effective_weight,
            "weighted_contribution": weighted_contribution,
        }

    return float(teacher_score), contributions, missing_score_keys


def compute_production_aesthetic_target(
    record: dict[str, object],
    selector_weights_config: str | Path | None = None,
) -> tuple[float | None, dict[str, dict[str, float]], list[str]]:
    selector_weights_path = resolve_repo_path(selector_weights_config)
    weights = load_weights(str(selector_weights_path) if selector_weights_path else None)
    return score_component(
        record,
        weights=weights["aesthetic"],
        component_name="aesthetic",
    )


def compute_teacher_target(
    record: dict[str, object],
    teacher_config: dict[str, object],
) -> tuple[float | None, dict[str, dict[str, float]], list[str]]:
    aggregation_mode = str(teacher_config.get("aggregation_mode") or DEFAULT_AGGREGATION_MODE)
    required_score_keys = list(teacher_config.get("required_score_keys") or [])
    allow_partial_teacher_scores = bool(teacher_config.get("allow_partial_teacher_scores", False))

    if aggregation_mode == "production_aesthetic_component":
        teacher_score, contributions, missing_score_keys = compute_production_aesthetic_target(
            record=record,
            selector_weights_config=teacher_config.get("production_selector_weights_config"),
        )
    else:
        teacher_score, contributions, missing_score_keys = aggregate_teacher_scores(
            record=record,
            score_weights=dict(teacher_config.get("score_weights") or {}),
            required_score_keys=required_score_keys,
            allow_partial_teacher_scores=allow_partial_teacher_scores,
        )
        return teacher_score, contributions, missing_score_keys

    if teacher_score is None:
        return None, contributions, missing_score_keys
    if required_score_keys and not allow_partial_teacher_scores:
        required_missing = [score_key for score_key in required_score_keys if score_key in missing_score_keys]
        if required_missing:
            return None, contributions, missing_score_keys
    return teacher_score, contributions, missing_score_keys
