from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


EXPLANATION_SCHEMA_VERSION = "acut_explanation.v1"
BASELINE_ACUT_EXPLANATION_BACKEND = "baseline_acut"
GEMINI_MULTIMODAL_EXPLANATION_BACKEND = "gemini_multimodal"
VILA_FULL_LOCAL_EXPLANATION_BACKEND = "vila_full_local"

LEGACY_EXPLANATION_SOURCE_ALIASES = {
    "baseline": BASELINE_ACUT_EXPLANATION_BACKEND,
    "multimodal": GEMINI_MULTIMODAL_EXPLANATION_BACKEND,
    # Historical rows used "vila_full" for the Gemini path before the identity split.
    "vila_full": GEMINI_MULTIMODAL_EXPLANATION_BACKEND,
}

HIGH_CAPABILITY_EXPLANATION_BACKENDS = (
    VILA_FULL_LOCAL_EXPLANATION_BACKEND,
    GEMINI_MULTIMODAL_EXPLANATION_BACKEND,
)


@dataclass
class ExplanationRecord:
    explanation_source: str
    short_reason: str | None = None
    detailed_reason: str | None = None
    comparison_reason: str | None = None
    tags: list[str] = field(default_factory=list)
    status: str = "unavailable"
    confidence: float | None = None
    model_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = EXPLANATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "short_reason": _clean_text(self.short_reason),
            "detailed_reason": _clean_text(self.detailed_reason),
            "comparison_reason": _clean_text(self.comparison_reason),
            "tags": _clean_tags(self.tags),
            "explanation_source": _clean_text(self.explanation_source),
            "status": _clean_text(self.status) or "unavailable",
            "confidence": _round_float(self.confidence, digits=4),
            "model_name": _clean_text(self.model_name),
            "metadata": _compact_json_value(self.metadata) if self.metadata else {},
        }
        return payload


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _round_float(value: object, digits: int = 6) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return round(numeric, digits)


def _clean_tags(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        candidate_values = [values]
    elif isinstance(values, (list, tuple, set)):
        candidate_values = list(values)
    else:
        return []

    cleaned: list[str] = []
    seen: set[str] = set()
    for value in candidate_values:
        text = _clean_text(value)
        if text is None:
            continue
        if text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def _compact_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _compact_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_compact_json_value(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return round(value, 6)
    return value


def ensure_explanation_state(row: dict[str, object]) -> None:
    explanations = row.get("explanations")
    if not isinstance(explanations, dict):
        row["explanations"] = {}
    if "active_explanation" not in row:
        row["active_explanation"] = None
    if "active_explanation_backend" not in row:
        row["active_explanation_backend"] = None
    if "explanation_status" not in row:
        row["explanation_status"] = None
    if "explanation_confidence" not in row:
        row["explanation_confidence"] = None
    if "explanation_tags" not in row:
        row["explanation_tags"] = []


def normalize_explanation_source(value: object) -> str | None:
    normalized = _clean_text(value)
    if normalized is None:
        return None
    return LEGACY_EXPLANATION_SOURCE_ALIASES.get(normalized, normalized)


def attach_explanation_record(
    row: dict[str, object],
    explanation: ExplanationRecord | dict[str, object],
) -> dict[str, object]:
    ensure_explanation_state(row)
    payload = explanation.to_dict() if isinstance(explanation, ExplanationRecord) else dict(explanation)
    source = normalize_explanation_source(payload.get("explanation_source"))
    if source is None:
        raise ValueError("Explanation payload must include explanation_source.")
    payload["explanation_source"] = source
    row["explanations"][source] = payload
    return payload


def explanation_record_for_backend(row: dict[str, object], backend_id: str) -> dict[str, object] | None:
    explanations = row.get("explanations")
    if not isinstance(explanations, dict):
        return None
    payload = explanations.get(backend_id)
    if not isinstance(payload, dict):
        return None
    return payload


def available_explanation_backends(row: dict[str, object]) -> list[str]:
    explanations = row.get("explanations")
    if not isinstance(explanations, dict):
        return []
    return sorted(str(key) for key, value in explanations.items() if isinstance(value, dict))


def capture_baseline_acut_explanations(rows: list[dict[str, object]]) -> int:
    generated_count = 0
    for row in rows:
        structured = row.get("acut_explanation_structured")
        record = ExplanationRecord(
            explanation_source=BASELINE_ACUT_EXPLANATION_BACKEND,
            short_reason=_clean_text(row.get("acut_short_reason")),
            detailed_reason=_clean_text(row.get("acut_detailed_reason")),
            comparison_reason=_clean_text(row.get("acut_comparison_reason")),
            tags=_baseline_tags_from_row(row, structured),
            status="generated" if _row_has_reason_text(row) else "unavailable",
            confidence=_baseline_confidence(structured),
            metadata=_baseline_metadata_from_row(row, structured),
        )
        attach_explanation_record(row, record)
        if record.status == "generated":
            generated_count += 1
    return generated_count


def project_explanations_to_legacy_fields(
    rows: list[dict[str, object]],
    *,
    preferred_backend_order: tuple[str, ...] = (
        VILA_FULL_LOCAL_EXPLANATION_BACKEND,
        GEMINI_MULTIMODAL_EXPLANATION_BACKEND,
        BASELINE_ACUT_EXPLANATION_BACKEND,
    ),
) -> None:
    for row in rows:
        ensure_explanation_state(row)
        baseline = explanation_record_for_backend(row, BASELINE_ACUT_EXPLANATION_BACKEND)
        multimodal = _first_available_high_capability_explanation(row)

        active = None
        for backend_id in preferred_backend_order:
            candidate = explanation_record_for_backend(row, backend_id)
            if _explanation_has_text(candidate):
                active = candidate
                break
        if active is None:
            for backend_id in available_explanation_backends(row):
                candidate = explanation_record_for_backend(row, backend_id)
                if _explanation_has_text(candidate):
                    active = candidate
                    break

        row["active_explanation"] = active
        row["active_explanation_backend"] = active.get("explanation_source") if isinstance(active, dict) else None
        row["acut_short_reason"] = active.get("short_reason") if isinstance(active, dict) else None
        row["acut_detailed_reason"] = active.get("detailed_reason") if isinstance(active, dict) else None
        row["acut_comparison_reason"] = active.get("comparison_reason") if isinstance(active, dict) else None
        row["explanation_source"] = active.get("explanation_source") if isinstance(active, dict) else None
        row["explanation_status"] = active.get("status") if isinstance(active, dict) else None
        row["explanation_confidence"] = active.get("confidence") if isinstance(active, dict) else None
        row["explanation_tags"] = active.get("tags") if isinstance(active, dict) else []

        row["baseline_short_reason"] = baseline.get("short_reason") if isinstance(baseline, dict) else None
        row["baseline_detailed_reason"] = baseline.get("detailed_reason") if isinstance(baseline, dict) else None
        row["baseline_comparison_reason"] = baseline.get("comparison_reason") if isinstance(baseline, dict) else None

        row["multimodal_short_reason"] = multimodal.get("short_reason") if isinstance(multimodal, dict) else None
        row["multimodal_detailed_reason"] = multimodal.get("detailed_reason") if isinstance(multimodal, dict) else None
        row["multimodal_comparison_reason"] = multimodal.get("comparison_reason") if isinstance(multimodal, dict) else None


def _first_available_high_capability_explanation(row: dict[str, object]) -> dict[str, object] | None:
    for backend_id in HIGH_CAPABILITY_EXPLANATION_BACKENDS:
        candidate = explanation_record_for_backend(row, backend_id)
        if _explanation_has_text(candidate):
            return candidate
    return None


def _row_has_reason_text(row: dict[str, object]) -> bool:
    return any(
        _clean_text(row.get(field))
        for field in ("acut_short_reason", "acut_detailed_reason", "acut_comparison_reason")
    )


def _explanation_has_text(payload: dict[str, object] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    return any(
        _clean_text(payload.get(field))
        for field in ("short_reason", "detailed_reason", "comparison_reason")
    )


def _baseline_confidence(structured: object) -> float | None:
    if not isinstance(structured, dict):
        return None
    comparison = structured.get("comparison")
    if not isinstance(comparison, dict):
        return None
    score_delta = _round_float(comparison.get("score_delta_vs_reference"), digits=6)
    if score_delta is None:
        return None
    return round(min(1.0, abs(score_delta) / 0.05), 4)


def _baseline_metadata_from_row(row: dict[str, object], structured: object) -> dict[str, object]:
    template_family = None
    compared_to_image_path = None
    if isinstance(structured, dict):
        text = structured.get("text")
        if isinstance(text, dict):
            template_family = _clean_text(text.get("template_family"))
        comparison = structured.get("comparison")
        if isinstance(comparison, dict):
            compared_to_image_path = _clean_text(comparison.get("compared_to_image_path"))
    metadata = {
        "rank": row.get("rank"),
        "template_family": template_family,
        "compared_to_image_path": compared_to_image_path,
        "rejection_reason": _clean_text(row.get("acut_rejection_reason")),
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _baseline_tags_from_row(row: dict[str, object], structured: object) -> list[str]:
    tags: list[str] = []
    if isinstance(structured, dict):
        selector = structured.get("selector")
        if isinstance(selector, dict):
            dominant_component = _clean_text(selector.get("dominant_component"))
            if dominant_component is not None:
                tags.append(dominant_component)
        comparison = structured.get("comparison")
        if isinstance(comparison, dict):
            tags.extend(str(value) for value in comparison.get("key_advantages") or [])
            tags.extend(str(value) for value in comparison.get("key_disadvantages") or [])
        vila = structured.get("vila")
        if isinstance(vila, dict):
            tags.extend(str(value) for value in vila.get("top_strengths") or [])
            tags.extend(str(value) for value in vila.get("top_weaknesses") or [])
    selector_notes = row.get("selector_notes")
    if isinstance(selector_notes, list) and selector_notes:
        tags.append("selector_reasoning")
    return _clean_tags(tags)
