from __future__ import annotations

import json
from typing import Any


def _to_string_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return []
        if (trimmed.startswith('[') and trimmed.endswith(']')) or \
           (trimmed.startswith('{') and trimmed.endswith('}')):
            try:
                decoded = json.loads(trimmed)
                return _to_string_list(decoded)
            except json.JSONDecodeError:
                pass
        return [item.strip() for item in trimmed.split(',') if item.strip()]
    return []


def extract_composition_tags(
    row: dict[str, Any],
    explanation_structured: dict[str, Any] | None,
) -> list[str]:
    tags = set()

    def add_tag_candidates(source: Any):
        for tag in _to_string_list(source):
            tags.add(tag)

    add_tag_candidates(row.get('composition_tags'))
    add_tag_candidates(row.get('compositionTags'))
    add_tag_candidates(row.get('composition_tag'))
    add_tag_candidates(row.get('compositionTag'))

    if explanation_structured:
        add_tag_candidates(explanation_structured.get('composition_tags'))
        signals = explanation_structured.get('signals', {})
        if signals:
            add_tag_candidates(signals.get('top_strengths'))
            add_tag_candidates(signals.get('top_weaknesses'))
            add_tag_candidates(signals.get('strengths'))
            add_tag_candidates(signals.get('weaknesses'))
        vila = explanation_structured.get('vila', {})
        if vila:
            add_tag_candidates(vila.get('top_strengths'))
            add_tag_candidates(vila.get('top_weaknesses'))

    if not tags:
        reason_text = ' '.join([
            row.get('acut_short_reason', '') or '',
            row.get('acut_detailed_reason', '') or '',
        ]).lower()
        fallback_map = {
            'composition': 'composition',
            'subject clarity': 'subject_clarity',
            'background cleanliness': 'background_cleanliness',
            'lighting': 'lighting',
            'technical quality': 'technical_quality',
            'aesthetic score': 'aesthetic_score',
            'overall image appeal': 'overall_image_appeal',
        }
        for keyword, tag in fallback_map.items():
            if keyword in reason_text:
                tags.add(tag)

    return list(tags)[:10]
