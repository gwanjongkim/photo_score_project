from __future__ import annotations

import copy
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_CONFIG = {
    "experiment_name": "stage5_student_mobilenetv2_050",
    "manifest_csv": "data/distill/stage5_aesthetic_labels.csv",
    "output_dir": "checkpoints/stage5_student_baseline",
    "seed": 42,
    "data": {
        "train_splits": ["train"],
        "val_splits": ["val"],
        "test_splits": ["test"],
        "target_col": "teacher_aesthetic_score",
        "image_size": 224,
        "batch_size": 64,
        "shuffle_buffer": 4096,
        "augment_horizontal_flip": False,
        "cache": False,
    },
    "model": {
        "alpha": 0.5,
        "head_units": 128,
        "dropout_rate": 0.2,
        "backbone_trainable": True,
        "backbone_weights": "imagenet",
    },
    "train": {
        "epochs": 20,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "loss": "huber",
        "huber_delta": 0.05,
        "use_ranking_loss": False,
        "ranking_weight": 0.0,
        "ranking_margin": 0.03,
        "early_stopping_patience": 5,
        "reduce_lr_patience": 2,
        "min_learning_rate": 1e-6,
        "checkpoint_monitor": "val_spearman",
        "checkpoint_monitor_mode": "max",
    },
    "eval": {
        "pairwise_margin": 0.03,
        "pairwise_max_pairs": 50000,
        "top_k": 10,
    },
}

REQUIRED_TRAINING_CONFIG_KEYS = {
    "top_level": ["manifest_csv", "output_dir", "data", "model", "train", "eval"],
    "data": ["train_splits", "val_splits", "target_col", "image_size", "batch_size", "shuffle_buffer"],
    "model": ["alpha", "head_units", "dropout_rate", "backbone_trainable"],
    "train": [
        "epochs",
        "learning_rate",
        "weight_decay",
        "loss",
        "huber_delta",
        "use_ranking_loss",
        "ranking_weight",
        "ranking_margin",
        "early_stopping_patience",
        "reduce_lr_patience",
        "min_learning_rate",
        "checkpoint_monitor",
        "checkpoint_monitor_mode",
    ],
    "eval": ["pairwise_margin", "pairwise_max_pairs", "top_k"],
}


def resolve_repo_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def deep_merge(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_training_config(config_path: str | Path) -> tuple[dict[str, object], Path]:
    resolved_path = resolve_repo_path(config_path)
    if resolved_path is None:
        raise ValueError("Training config path could not be resolved.")
    with resolved_path.open("r", encoding="utf-8") as handle:
        override = json.load(handle)
    config = deep_merge(DEFAULT_TRAINING_CONFIG, override)
    validate_training_config(config)
    return config, resolved_path


def validate_training_config(config: dict[str, object]) -> None:
    missing_top_level = [key for key in REQUIRED_TRAINING_CONFIG_KEYS["top_level"] if key not in config]
    if missing_top_level:
        raise ValueError(f"Training config is missing top-level keys: {', '.join(missing_top_level)}")

    for section_name in ("data", "model", "train", "eval"):
        section = config.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"Training config section '{section_name}' must be an object.")
        missing_section_keys = [key for key in REQUIRED_TRAINING_CONFIG_KEYS[section_name] if key not in section]
        if missing_section_keys:
            raise ValueError(
                f"Training config section '{section_name}' is missing keys: {', '.join(missing_section_keys)}"
            )

    if int(config["data"]["image_size"]) <= 0:
        raise ValueError("Training config data.image_size must be > 0.")
    if int(config["data"]["batch_size"]) <= 0:
        raise ValueError("Training config data.batch_size must be > 0.")
    if int(config["train"]["epochs"]) <= 0:
        raise ValueError("Training config train.epochs must be > 0.")
    if float(config["train"]["learning_rate"]) <= 0.0:
        raise ValueError("Training config train.learning_rate must be > 0.")
    if float(config["eval"]["top_k"]) <= 0:
        raise ValueError("Training config eval.top_k must be > 0.")
    monitor_mode = str(config["train"]["checkpoint_monitor_mode"]).strip().lower()
    if monitor_mode not in {"min", "max"}:
        raise ValueError("Training config train.checkpoint_monitor_mode must be 'min' or 'max'.")
    target_col = str(config["data"]["target_col"]).strip()
    if not target_col:
        raise ValueError("Training config data.target_col must be a non-empty string.")


def to_serializable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    return value


def write_json(path: str | Path, payload: dict[str, object]) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(to_serializable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return resolved_path
