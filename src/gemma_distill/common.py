from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    return (PROJECT_ROOT / candidate).resolve(strict=False)


def repo_relative_path(path_like: str | Path | None) -> str | None:
    path = resolve_repo_path(path_like)
    if path is None:
        return None
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def canonicalize_image_key(path_like: str | Path | None) -> str | None:
    path = resolve_repo_path(path_like)
    if path is None:
        return None
    return str(path)


def read_json(path_like: str | Path) -> Any:
    path = resolve_repo_path(path_like)
    if path is None:
        raise ValueError("Could not resolve JSON path.")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path_like: str | Path, payload: Any) -> Path:
    path = resolve_repo_path(path_like)
    if path is None:
        raise ValueError("Could not resolve output JSON path.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_jsonl(path_like: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    path = resolve_repo_path(path_like)
    if path is None:
        raise ValueError("Could not resolve output JSONL path.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def append_jsonl(path_like: str | Path, payload: dict[str, Any]) -> Path:
    path = resolve_repo_path(path_like)
    if path is None:
        raise ValueError("Could not resolve output JSONL path.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def collapse_whitespace(value: object) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    return re.sub(r"\s+", " ", text).strip() or None


def sentence(text: str) -> str:
    cleaned = (collapse_whitespace(text) or "").strip(" ;,")
    if not cleaned:
        return ""
    return cleaned if cleaned.endswith((".", "!", "?")) else f"{cleaned}."


def maybe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return float(numeric)


def round_float(value: object, digits: int = 6) -> float | None:
    numeric = maybe_float(value)
    if numeric is None:
        return None
    return round(numeric, digits)


def normalize_unit_score(value: object) -> float | None:
    numeric = maybe_float(value)
    if numeric is None:
        return None
    if -0.05 <= numeric <= 1.05:
        return max(0.0, min(1.0, numeric))
    if 1.0 <= numeric <= 10.5:
        return max(0.0, min(1.0, (numeric - 1.0) / 9.0))
    if 0.0 <= numeric <= 100.0:
        return max(0.0, min(1.0, numeric / 100.0))
    return None


def rank_band_from_rank(rank: int | None, total: int | None) -> str:
    if rank is None or total is None or total <= 0:
        return "unknown"
    ratio = float(rank) / float(total)
    if ratio <= 0.20:
        return "top_20_percent"
    if ratio <= 0.50:
        return "upper_mid_30_percent"
    if ratio <= 0.80:
        return "lower_mid_30_percent"
    return "bottom_20_percent"


def slugify(value: object) -> str:
    text = clean_text(value) or "item"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-") or "item"


def ensure_mplconfigdir() -> str:
    path = os.environ.setdefault("MPLCONFIGDIR", "/tmp/photo_score_project_mplconfig")
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

