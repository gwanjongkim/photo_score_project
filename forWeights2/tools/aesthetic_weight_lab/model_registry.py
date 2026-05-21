# 미적 평가 모델 설정을 실행 가능한 레지스트리 항목으로 변환합니다.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_LABELS = {
    "nima": "NIMA",
    "rgnet": "RGNet",
    "alamp": "A-LAMP",
    "icaa": "ICAA",
}


@dataclass(frozen=True)
class ModelType:
    type_name: str
    runner: str
    default_label: str
    required_fields: tuple[str, ...]


MODEL_TYPES: dict[str, ModelType] = {
    "nima_distribution": ModelType(
        type_name="nima_distribution",
        runner="nima_distribution",
        default_label="NIMA",
        required_fields=("model_path", "input_width", "input_height", "normalization"),
    ),
    "scalar_tflite": ModelType(
        type_name="scalar_tflite",
        runner="scalar_tflite",
        default_label="Scalar TFLite",
        required_fields=("model_path", "input_width", "input_height", "normalization"),
    ),
    "vector_tflite": ModelType(
        type_name="vector_tflite",
        runner="vector_tflite",
        default_label="Vector TFLite",
        required_fields=("model_path", "input_width", "input_height", "normalization", "score_index"),
    ),
    "alamp_signature": ModelType(
        type_name="alamp_signature",
        runner="alamp_signature",
        default_label="A-LAMP",
        required_fields=("model_path", "global_size", "patch_size", "patch_count", "normalization"),
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    enabled: bool
    model_type: str
    runner: str
    display_name: str
    model_path: Path
    score_column: str
    config: dict[str, Any]


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _as_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping.")
    return value


def _model_label(model_id: str, model_config: dict[str, Any], model_type: ModelType) -> str:
    label = model_config.get("display_name") or model_config.get("label")
    if label:
        return str(label)
    return DEFAULT_MODEL_LABELS.get(model_id, model_type.default_label)


def load_model_specs(config: dict[str, Any], repo_root: Path) -> list[ModelSpec]:
    models_config = _as_mapping(config.get("models"), field_name="models")
    specs: list[ModelSpec] = []
    for model_id, raw_model_config in models_config.items():
        model_config = _as_mapping(raw_model_config, field_name=f"models.{model_id}")
        enabled = bool(model_config.get("enabled", True))
        model_type_name = str(model_config.get("type", "")).strip()
        if model_type_name not in MODEL_TYPES:
            known = ", ".join(sorted(MODEL_TYPES))
            raise ValueError(f"models.{model_id}.type must be one of: {known}.")
        model_type = MODEL_TYPES[model_type_name]
        missing = [field for field in model_type.required_fields if field not in model_config]
        if missing:
            raise ValueError(f"models.{model_id} is missing required fields: {', '.join(missing)}.")
        normalized_config = dict(model_config)
        if model_type_name == "vector_tflite":
            try:
                score_index = int(normalized_config["score_index"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"models.{model_id}.score_index must be a non-negative integer.") from exc
            if score_index < 0:
                raise ValueError(f"models.{model_id}.score_index must be a non-negative integer.")
            normalized_config["score_index"] = score_index
        score_column = str(model_config.get("score_column") or f"{model_id}_score")
        specs.append(
            ModelSpec(
                model_id=str(model_id),
                enabled=enabled,
                model_type=model_type_name,
                runner=model_type.runner,
                display_name=_model_label(str(model_id), model_config, model_type),
                model_path=resolve_repo_path(repo_root, model_config["model_path"]),
                score_column=score_column,
                config=normalized_config,
            )
        )
    return specs


def enabled_model_specs(config: dict[str, Any], repo_root: Path) -> list[ModelSpec]:
    return [spec for spec in load_model_specs(config, repo_root) if spec.enabled]


def load_weights(config: dict[str, Any], specs: list[ModelSpec]) -> dict[str, float]:
    raw_weights = _as_mapping(config.get("weights", {}), field_name="weights")
    weights: dict[str, float] = {}
    for spec in specs:
        weights[spec.model_id] = float(raw_weights.get(spec.model_id, 0.0))
    return weights


def normalized_weighted_score(row: dict[str, Any], specs: list[ModelSpec], weights: dict[str, float]) -> float | None:
    weighted_sum = 0.0
    active_weight_sum = 0.0
    for spec in specs:
        value = row.get(spec.score_column)
        if value is None:
            continue
        weight = float(weights.get(spec.model_id, 0.0))
        if weight <= 0.0:
            continue
        unit_score = max(0.0, min(1.0, float(value)))
        weighted_sum += weight * unit_score
        active_weight_sum += weight
    if active_weight_sum <= 0.0:
        return None
    return float(weighted_sum / active_weight_sum)


def build_weight_presets(specs: list[ModelSpec], weights: dict[str, float]) -> dict[str, dict[str, float]]:
    model_ids = [spec.model_id for spec in specs]
    if not model_ids:
        return {}
    equal_weight = 1.0 / len(model_ids)
    presets: dict[str, dict[str, float]] = {
        "equal": {model_id: equal_weight for model_id in model_ids},
        "current_config": {model_id: float(weights.get(model_id, 0.0)) for model_id in model_ids},
    }
    if len(model_ids) == 1:
        presets[f"{model_ids[0]}_heavy"] = {model_ids[0]: 1.0}
        return presets
    for heavy_id in model_ids:
        other_weight = 0.4 / (len(model_ids) - 1)
        presets[f"{heavy_id}_heavy"] = {
            model_id: 0.6 if model_id == heavy_id else other_weight
            for model_id in model_ids
        }
    return presets


def assert_model_files_exist(specs: list[ModelSpec]) -> None:
    missing = [str(spec.model_path) for spec in specs if not spec.model_path.exists()]
    if missing:
        raise FileNotFoundError("Missing configured TFLite model file(s): " + ", ".join(missing))
