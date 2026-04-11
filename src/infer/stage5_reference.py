from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE5_CONFIG_PATH = PROJECT_ROOT / "configs" / "stage5_reference.json"
_PATH_LIKE_KEYS = {
    "aadb_model",
    "koniq_model",
    "flive_image_model",
    "flive_patch_model",
    "nima_model",
    "alamp_model",
    "musiq_model",
    "rgnet_model",
    "pairwise_model",
    "pairwise_reference_csv",
}


def ensure_project_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def repo_relative_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = ensure_project_path(path)
    if candidate is None:
        return None
    try:
        return str(candidate.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(candidate)


def load_stage5_reference_config(config_path: str | Path | None = None) -> tuple[dict[str, object], Path]:
    resolved_path = ensure_project_path(config_path or DEFAULT_STAGE5_CONFIG_PATH)
    if resolved_path is None:
        raise ValueError("Stage5 config path could not be resolved.")
    with resolved_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    bundle_args = dict(config.get("bundle_args", {}))
    for key in _PATH_LIKE_KEYS:
        value = bundle_args.get(key)
        if isinstance(value, str) and value:
            bundle_args[key] = str(ensure_project_path(value))
    config["bundle_args"] = bundle_args
    config["selector_args"] = dict(config.get("selector_args", {}))
    return config, resolved_path


def sanitize_run_label(value: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    compact = compact.strip("._-")
    return compact or "stage5_run"


def default_stage5_output_dir(input_dir: str | Path, run_label: str | None = None) -> Path:
    base_label = run_label or Path(input_dir).name or "stage5_run"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "outputs" / "stage5_runs" / f"{sanitize_run_label(base_label)}_{timestamp}"
