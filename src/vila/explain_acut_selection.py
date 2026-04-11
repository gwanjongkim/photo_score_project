from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from src.vila.explain_selection import build_explanation_signals
from src.vila.prompt_sets import DEFAULT_PROMPT_PRESET, get_prompt_preset


DEFAULT_REASON_REFERENCE_MODE = "nearest_competitor"
DEFAULT_REASON_DETAIL_LEVEL = "standard"
REASON_REFERENCE_MODES = ("nearest_higher", "nearest_competitor", "top1")
REASON_DETAIL_LEVELS = ("short", "standard", "full")


MODEL_LABELS = {
    "aadb_score": "AADB",
    "nima_mean_score": "NIMA",
    "alamp_score": "ALAMP",
    "rgnet_score": "RGNet",
    "pairwise_recovered_score": "pairwise recovered score",
    "koniq_score": "KonIQ",
    "flive_image_score": "FLIVE image",
    "flive_patch_mean": "FLIVE patch mean",
    "flive_patch_min": "FLIVE patch min",
    "musiq_score": "MUSIQ",
}

FEATURE_LABELS = {
    "aesthetic_component": "aesthetic score",
    "technical_component": "technical quality",
    "good_image": "overall image appeal",
    "good_composition": "composition",
    "good_lighting": "lighting",
    "clear_subject": "subject clarity",
    "clean_background": "background cleanliness",
}

PROMPT_DELTA_KEYS = (
    "good_composition",
    "good_lighting",
    "clear_subject",
    "clean_background",
)

DELTA_FEATURE_ORDER = (
    "technical_component",
    "aesthetic_component",
    "good_composition",
    "clean_background",
    "clear_subject",
    "good_lighting",
    "good_image",
)


def _prompt_pair_map(prompt_preset: str = DEFAULT_PROMPT_PRESET) -> dict[str, object]:
    return {pair.key: pair for pair in get_prompt_preset(prompt_preset)}


PROMPT_PAIR_MAP = _prompt_pair_map()


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _to_float(value: object) -> float | None:
    if _is_missing(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _clip_unit_score(value: object) -> float | None:
    numeric = _to_float(value)
    if numeric is None:
        return None
    return max(0.0, min(1.0, numeric))


def _row_vila_score(row: Mapping[str, object]) -> float | None:
    return _to_float(row.get("vila_score_raw")) if row.get("vila_score_raw") is not None else _to_float(row.get("vila_score"))


def _round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _display_image_name(path_value: object) -> str:
    if path_value is None:
        return "<unknown>"
    return Path(str(path_value)).name


def _join_phrases(phrases: list[str]) -> str:
    phrases = [phrase for phrase in phrases if phrase]
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return f"{', '.join(phrases[:-1])}, and {phrases[-1]}"


def _unique_labels(labels: list[str]) -> list[str]:
    unique = []
    seen = set()
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        unique.append(label)
    return unique


def _be_verb(labels: list[str], singular: str = "is", plural: str = "are") -> str:
    return singular if len(labels) == 1 else plural


def _sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    return text if text.endswith(".") else f"{text}."


def _deterministic_index(seed: str, size: int) -> int:
    if size <= 0:
        return 0
    total = 0
    for index, char in enumerate(seed):
        total += (index + 1) * ord(char)
    return total % size


def _choose_variant(options: list[str], *seed_parts: object) -> str:
    options = [option for option in options if option]
    if not options:
        return ""
    seed = "|".join("" if part is None else str(part) for part in seed_parts)
    return options[_deterministic_index(seed, len(options))]


def _filtered_reason_labels(labels: list[str]) -> list[str]:
    return _unique_labels([label for label in labels if label and label != "overall image appeal"])


def _selector_backbone_phrase(dominant_component: str | None) -> str:
    if dominant_component == "technical":
        return "technical consistency"
    if dominant_component == "aesthetic":
        return "aesthetic control"
    if dominant_component == "balanced":
        return "a balanced selector profile"
    return "the combined selector score"


def _reason_margin_delta(
    row: Mapping[str, object],
    selected: bool,
    cutline_row: Mapping[str, object] | None,
    comparison: Mapping[str, object] | None,
) -> float | None:
    if cutline_row is not None and cutline_row is not row:
        delta = _delta(row.get("final_score"), cutline_row.get("final_score"))
        if delta is not None:
            return delta
    if comparison is None:
        return None
    return _to_float(comparison.get("score_delta_vs_reference"))


def _reason_margin_bucket(delta: float | None) -> str:
    if delta is None:
        return "unknown"
    magnitude = abs(delta)
    if magnitude < 0.012:
        return "tight"
    if magnitude < 0.03:
        return "moderate"
    return "clear"


def _reason_family(
    selected: bool,
    dominant_component: str | None,
    comparison: Mapping[str, object] | None,
    vila_strengths: list[str],
    vila_weaknesses: list[str],
    margin_delta: float | None,
) -> str:
    advantages = _filtered_reason_labels(
        [str(label) for label in (comparison or {}).get("key_advantages") or []]
    )
    disadvantages = _filtered_reason_labels(
        [str(label) for label in (comparison or {}).get("key_disadvantages") or []]
    )
    positives = advantages[:2] or vila_strengths[:2]
    negatives = disadvantages[:2] or vila_weaknesses[:2]
    leading_positive = positives[0] if positives else None
    tradeoff = bool(positives and negatives)
    margin_bucket = _reason_margin_bucket(margin_delta)

    if selected:
        if margin_bucket == "tight":
            return "weak_selected"
        if leading_positive == "composition" and margin_bucket != "tight":
            return "strong_composition"
        if leading_positive == "technical quality" or (
            dominant_component == "technical" and margin_bucket == "clear"
        ):
            return "strong_technical"
        if dominant_component == "balanced" or (
            "technical quality" in advantages and "composition" in advantages
        ):
            return "balanced_win"
        if tradeoff:
            return "tradeoff_selected"
        return "selected_general"

    if tradeoff:
        return "tradeoff_rejected"
    return "rejected_general"


def _reason_phrases(
    comparison: Mapping[str, object] | None,
    dominant_component: str | None,
    vila_strengths: list[str],
    vila_weaknesses: list[str],
) -> tuple[list[str], list[str], str, str]:
    advantages = _filtered_reason_labels(
        [str(label) for label in (comparison or {}).get("key_advantages") or []]
    )
    disadvantages = _filtered_reason_labels(
        [str(label) for label in (comparison or {}).get("key_disadvantages") or []]
    )
    positives = advantages[:2] or _filtered_reason_labels(vila_strengths[:2])
    negatives = disadvantages[:2] or _filtered_reason_labels(vila_weaknesses[:2])
    positive_phrase = _join_phrases(positives) if positives else _selector_backbone_phrase(dominant_component)
    negative_phrase = _join_phrases(negatives) if negatives else "weaker supporting cues"
    return positives, negatives, positive_phrase, negative_phrase


def _support_phrase_without(labels: list[str], excluded_label: str, fallback: str) -> str:
    remaining = [label for label in labels if label != excluded_label]
    if remaining:
        return _join_phrases(remaining)
    return fallback


def _short_reason(
    row: Mapping[str, object],
    selected: bool,
    comparison: Mapping[str, object] | None,
    dominant_component: str | None,
    vila_strengths: list[str],
    vila_weaknesses: list[str],
    cutline_row: Mapping[str, object] | None,
) -> tuple[str, str]:
    positives, negatives, positive_phrase, negative_phrase = _reason_phrases(
        comparison=comparison,
        dominant_component=dominant_component,
        vila_strengths=vila_strengths,
        vila_weaknesses=vila_weaknesses,
    )
    margin_delta = _reason_margin_delta(row=row, selected=selected, cutline_row=cutline_row, comparison=comparison)
    family = _reason_family(
        selected=selected,
        dominant_component=dominant_component,
        comparison=comparison,
        vila_strengths=vila_strengths,
        vila_weaknesses=vila_weaknesses,
        margin_delta=margin_delta,
    )
    vila_phrase = _join_phrases(_filtered_reason_labels(vila_strengths[:2]))
    backbone_phrase = _selector_backbone_phrase(dominant_component)
    technical_support = _support_phrase_without(positives, "technical quality", positive_phrase)
    composition_support = _support_phrase_without(positives, "composition", positive_phrase)

    if family == "strong_technical":
        options = [
            f"Selected on technical consistency, with {technical_support} keeping it ahead of nearby alternatives",
            f"The selector held this frame on technical quality; {technical_support} created the clearest edge",
            (
                f"The selector favored this image on technical consistency, while VILA reinforced the choice through {vila_phrase}"
                if vila_phrase
                else ""
            ),
        ]
    elif family == "strong_composition":
        options = [
            f"Chosen because composition and related cues held up better than nearby alternatives",
            f"Composition carried this frame into the A-cut, helped by {composition_support}",
            f"This image stayed above the cut because composition gave it the clearest edge",
        ]
    elif family == "balanced_win":
        options = [
            f"It stayed in because {positive_phrase} held together without a major drop elsewhere",
            f"A balanced score profile kept this frame above the cut, led by {positive_phrase}",
            f"The pick came from balance rather than one spike; {positive_phrase} kept it ahead",
        ]
    elif family == "weak_selected":
        options = [
            f"It stayed in on a narrow margin because {positive_phrase} barely outweighed {negative_phrase}",
            f"This frame cleared the cut by a small edge, mostly through {positive_phrase}",
            f"The A-cut kept it despite a thin margin, with {positive_phrase} just covering for {negative_phrase}",
        ]
    elif family == "tradeoff_selected":
        options = [
            f"This frame stayed above the cut because {positive_phrase} outweighed {negative_phrase}",
            f"Chosen on balance: {positive_phrase} beat nearby alternatives even with {negative_phrase} softer",
            f"It remained selected because {positive_phrase} offset weaker {negative_phrase}",
        ]
    elif family == "tradeoff_rejected":
        options = [
            f"It missed the A-cut mainly because {negative_phrase} trailed the selected set, even though {positive_phrase} remained acceptable",
            f"This frame fell below the cut when {negative_phrase} outweighed {positive_phrase} in nearby comparisons",
            f"Rejected after the tradeoff broke the wrong way: {positive_phrase} {_be_verb(positives, 'was', 'were')} not enough to offset {negative_phrase}",
        ]
    elif family == "rejected_general":
        options = [
            f"It missed the A-cut because {negative_phrase} lagged behind the selected set",
            f"The frame fell short once {negative_phrase} gave higher-ranked images the edge",
            f"Rejected because {backbone_phrase} could not recover weaker {negative_phrase} against nearby options",
        ]
    else:
        options = [
            f"Chosen because {positive_phrase} held up better than nearby alternatives",
            f"This frame stayed above the cut on {positive_phrase}",
            f"Selected because {backbone_phrase} stayed ahead of nearby alternatives",
        ]

    return _sentence(_choose_variant(options, row.get("image_path"), row.get("rank"), family, "short")), family


def _evidence_sentence(
    row: Mapping[str, object],
    selected: bool,
    comparison: Mapping[str, object] | None,
    dominant_component: str | None,
    vila_strengths: list[str],
    vila_weaknesses: list[str],
    family: str,
) -> str:
    positives, negatives, positive_phrase, negative_phrase = _reason_phrases(
        comparison=comparison,
        dominant_component=dominant_component,
        vila_strengths=vila_strengths,
        vila_weaknesses=vila_weaknesses,
    )
    vila_phrase = _join_phrases(_filtered_reason_labels(vila_strengths[:2]))
    backbone_phrase = _selector_backbone_phrase(dominant_component)
    technical_support = _support_phrase_without(positives, "technical quality", positive_phrase)
    composition_support = _support_phrase_without(positives, "composition", positive_phrase)

    if family == "strong_technical":
        options = [
            f"Technical quality was the selector's clearest lever here, and {technical_support} created most of the separation",
            f"The decisive edge came from the technical side, especially {technical_support}",
            (
                f"The technical backbone stayed stronger, and VILA agreed on {vila_phrase}"
                if vila_phrase
                else f"The technical backbone stayed stronger through {technical_support}"
            ),
        ]
    elif family == "strong_composition":
        options = [
            f"Composition supplied the cleanest advantage, with {composition_support} separating it from nearby frames",
            f"The comparison was mostly won on composition-side evidence rather than raw score spread alone",
            (
                f"Prompt-side evidence kept leaning toward composition, especially through {vila_phrase}"
                if vila_phrase
                else f"Composition stayed more reliable than competing frames here"
            ),
        ]
    elif family == "balanced_win":
        options = [
            f"No single component dominated, so the win came from a steadier mix of {positive_phrase}",
            f"The selector did not rely on one spike; it held together through {positive_phrase}",
            f"The edge was distributed across the score stack rather than coming from one rescue signal",
        ]
    elif family == "weak_selected":
        options = [
            f"The margin was slim, and {positive_phrase} only just covered for weaker {negative_phrase}",
            f"This was a keepable but fragile pick because the positives were narrow rather than overwhelming",
            f"It survived the cut line by holding a small edge on {positive_phrase} while {negative_phrase} stayed softer",
        ]
    elif family == "tradeoff_selected":
        options = [
            f"The deciding tradeoff favored {positive_phrase}; {negative_phrase} was weaker, but not enough to overturn the selector lead",
            f"The strongest positives survived the weaker {negative_phrase} in the nearby comparison set",
            f"This stayed selectable because the gains in {positive_phrase} outweighed the softer {negative_phrase}",
        ]
    elif family == "tradeoff_rejected":
        options = [
            f"The surviving positives on {positive_phrase} were real, but they did not cover for weaker {negative_phrase}",
            f"The tradeoff broke against this frame once {negative_phrase} fell behind the cut-line alternatives",
            f"It remained close in places, yet the loss concentrated on {negative_phrase}",
        ]
    elif family == "rejected_general":
        options = [
            f"The selector never found enough support beyond {negative_phrase} to close the gap",
            f"The comparison deficit stayed concentrated in {negative_phrase}, which the higher-ranked set handled better",
            f"Nearby competitors kept more complete support than this frame on the deciding cues",
        ]
    else:
        options = [
            f"The selector edge was carried by {positive_phrase}",
            (
                f"VILA backed the choice with {vila_phrase} while the selector kept the backbone score ahead"
                if selected and vila_phrase
                else ""
            ),
            f"The combined evidence stayed stronger through {positive_phrase} than through any single rescue factor",
        ]

    return _sentence(_choose_variant(options, row.get("image_path"), row.get("rank"), family, "evidence"))


def _prompt_scores_for_row(
    row: Mapping[str, object],
    vila_row: Mapping[str, object] | None = None,
) -> dict[str, float]:
    nested = row.get("vila_prompt_scores")
    if isinstance(nested, Mapping):
        out = {}
        for key, value in nested.items():
            numeric = _to_float(value)
            if numeric is not None:
                out[str(key)] = numeric
        if out:
            return out

    source_rows = [row]
    if vila_row is not None:
        source_rows.append(vila_row)

    out: dict[str, float] = {}
    for source in source_rows:
        for key in PROMPT_PAIR_MAP:
            for candidate_key in (key, f"prompt_{key}"):
                numeric = _to_float(source.get(candidate_key))
                if numeric is not None:
                    out[key] = numeric
                    break
    return out


def _signals_for_row(
    row: Mapping[str, object],
    prompt_scores: Mapping[str, float],
) -> dict[str, object] | None:
    existing = row.get("vila_explanation_signals")
    if isinstance(existing, Mapping):
        return dict(existing)
    if not prompt_scores:
        return None
    return build_explanation_signals(
        prompt_scores=prompt_scores,
        vila_score=_clip_unit_score(_row_vila_score(row)),
        prompt_preset=DEFAULT_PROMPT_PRESET,
    )


def _top_prompt_labels(
    prompt_scores: Mapping[str, float],
    signals: Mapping[str, object] | None,
    prefer_strengths: bool,
    limit: int = 2,
) -> list[str]:
    if not prompt_scores:
        return []

    buckets = "strengths" if prefer_strengths else "weaknesses"
    if isinstance(signals, Mapping):
        items = signals.get(buckets) or []
        labels = _unique_labels(
            [str(item["label"]) for item in items[:limit] if isinstance(item, Mapping) and item.get("label")]
        )
        if labels:
            return labels

    ordered = sorted(
        (
            (key, value)
            for key, value in prompt_scores.items()
            if key in FEATURE_LABELS and _to_float(value) is not None
        ),
        key=lambda item: (item[1], item[0]),
        reverse=prefer_strengths,
    )
    fallback = []
    for key, value in ordered:
        if prefer_strengths and value < 0.48:
            continue
        if not prefer_strengths and value > 0.52:
            continue
        fallback.append(FEATURE_LABELS[key])
        if len(fallback) >= limit:
            break
    if fallback:
        return _unique_labels(fallback)
    if ordered:
        return [FEATURE_LABELS[ordered[0][0]]]
    return []


def _top_model_contributions(row: Mapping[str, object], limit: int = 3) -> list[dict[str, object]]:
    payload = row.get("per_model_contributions")
    if not isinstance(payload, Mapping):
        return []

    entries = []
    for component_name in ("aesthetic", "technical"):
        component_payload = payload.get(component_name)
        if not isinstance(component_payload, Mapping):
            continue
        for model_key, detail in component_payload.items():
            if not isinstance(detail, Mapping):
                continue
            contribution = _to_float(detail.get("weighted_contribution"))
            if contribution is None:
                continue
            entries.append(
                {
                    "component": component_name,
                    "model_key": str(model_key),
                    "model_label": MODEL_LABELS.get(str(model_key), str(model_key)),
                    "weighted_contribution": contribution,
                    "normalized_score": _to_float(detail.get("normalized_score")),
                    "raw_score": _to_float(detail.get("raw_score")),
                }
            )

    entries.sort(key=lambda item: (-item["weighted_contribution"], item["model_label"]))
    trimmed = []
    for item in entries[:limit]:
        trimmed.append(
            {
                "component": item["component"],
                "model_key": item["model_key"],
                "model_label": item["model_label"],
                "weighted_contribution": _round_float(item["weighted_contribution"]),
                "normalized_score": _round_float(item["normalized_score"]),
                "raw_score": _round_float(item["raw_score"]),
            }
        )
    return trimmed


def _component_summary(row: Mapping[str, object], include_model_contributions: bool) -> tuple[str, str | None]:
    aesthetic = _to_float(row.get("aesthetic_component"))
    technical = _to_float(row.get("technical_component"))
    if aesthetic is None and technical is None:
        sentence = "The selector had no usable aesthetic or technical component scores to lean on"
        return _sentence(sentence), None
    if aesthetic is None:
        sentence = f"The selector fell back to technical-only evidence ({technical:.3f})"
        dominant = "technical"
    elif technical is None:
        sentence = f"The selector fell back to aesthetic-only evidence ({aesthetic:.3f})"
        dominant = "aesthetic"
    else:
        delta = technical - aesthetic
        if abs(delta) < 0.03:
            sentence = (
                f"The selector stayed balanced between aesthetic ({aesthetic:.3f}) "
                f"and technical ({technical:.3f}) components"
            )
            dominant = "balanced"
        elif delta > 0:
            sentence = f"The selector leaned on technical quality ({technical:.3f} vs {aesthetic:.3f} aesthetic)"
            dominant = "technical"
        else:
            sentence = f"The selector leaned on aesthetic score ({aesthetic:.3f} vs {technical:.3f} technical)"
            dominant = "aesthetic"

    if include_model_contributions:
        top_models = _top_model_contributions(row)
        if top_models:
            labels = [item["model_label"] for item in top_models]
            sentence += f", with the largest weighted contributions from {_join_phrases(labels)}"
    return _sentence(sentence), dominant


def _mechanics_sentence(row: Mapping[str, object]) -> str:
    clauses = []
    pairwise_delta = _to_float(row.get("pairwise_rerank_delta"))
    if row.get("pairwise_rerank_applied") and pairwise_delta is not None and abs(pairwise_delta) >= 0.002:
        direction = "added" if pairwise_delta > 0 else "trimmed"
        clauses.append(f"pairwise reranking {direction} {abs(pairwise_delta):.3f}")

    vila_delta = _to_float(row.get("vila_rerank_delta"))
    if row.get("vila_rerank_applied") and vila_delta is not None and abs(vila_delta) >= 0.002:
        direction = "added" if vila_delta > 0 else "trimmed"
        clauses.append(f"VILA reranking {direction} {abs(vila_delta):.3f}")

    diversity_penalty = _to_float(row.get("diversity_penalty"))
    if diversity_penalty is not None and diversity_penalty > 0.0:
        clauses.append(f"diversity control subtracted {diversity_penalty:.3f}")

    if not clauses:
        return ""
    return _sentence(f"In the finalized ranking stack, {_join_phrases(clauses)}")


def _feature_threshold(feature_key: str) -> float:
    if feature_key in {"aesthetic_component", "technical_component"}:
        return 0.015
    if feature_key == "good_image":
        return 0.03
    return 0.025


def _comparison_feature_entries(
    row: Mapping[str, object],
    reference_row: Mapping[str, object] | None,
    prompt_scores: Mapping[str, float],
    reference_prompt_scores: Mapping[str, float],
) -> tuple[list[dict[str, object]], dict[str, float | None], dict[str, float | None]]:
    component_deltas: dict[str, float | None] = {}
    prompt_deltas: dict[str, float | None] = {}
    entries = []
    if reference_row is None:
        return entries, component_deltas, prompt_deltas

    for feature_key in ("aesthetic_component", "technical_component"):
        delta = _delta(row.get(feature_key), reference_row.get(feature_key))
        component_deltas[feature_key] = delta
        if delta is not None:
            entries.append(
                {
                    "feature_key": feature_key,
                    "label": FEATURE_LABELS[feature_key],
                    "delta": delta,
                }
            )

    for feature_key in (*PROMPT_DELTA_KEYS, "good_image"):
        delta = _delta(prompt_scores.get(feature_key), reference_prompt_scores.get(feature_key))
        if feature_key in PROMPT_DELTA_KEYS:
            prompt_deltas[feature_key] = delta
        if delta is not None:
            entries.append(
                {
                    "feature_key": feature_key,
                    "label": FEATURE_LABELS[feature_key],
                    "delta": delta,
                }
            )

    entries.sort(
        key=lambda item: (
            -abs(float(item["delta"])),
            DELTA_FEATURE_ORDER.index(item["feature_key"]) if item["feature_key"] in DELTA_FEATURE_ORDER else 10**6,
            item["label"],
        )
    )
    return entries, component_deltas, prompt_deltas


def _labels_for_direction(
    entries: list[dict[str, object]],
    positive: bool,
    limit: int = 3,
) -> list[str]:
    labels = []
    seen = set()
    for entry in entries:
        delta = float(entry["delta"])
        if positive and delta <= 0:
            continue
        if (not positive) and delta >= 0:
            continue
        if abs(delta) < _feature_threshold(str(entry["feature_key"])):
            continue
        label = str(entry["label"])
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
        if len(labels) >= limit:
            return labels

    for entry in entries:
        delta = float(entry["delta"])
        if positive and delta <= 0:
            continue
        if (not positive) and delta >= 0:
            continue
        label = str(entry["label"])
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
        if len(labels) >= limit:
            break
    return labels


def _delta(current_value: object, reference_value: object) -> float | None:
    current = _to_float(current_value)
    reference = _to_float(reference_value)
    if current is None or reference is None:
        return None
    return current - reference


def _score_delta_phrase(delta: float | None) -> str:
    if delta is None:
        return "not directly comparable"
    magnitude = abs(delta)
    if magnitude < 0.008:
        return f"roughly tied ({delta:+.3f})"
    if delta > 0:
        if magnitude >= 0.05:
            return f"clearly higher by {magnitude:.3f}"
        if magnitude >= 0.02:
            return f"higher by {magnitude:.3f}"
        return f"slightly higher by {magnitude:.3f}"
    if magnitude >= 0.05:
        return f"clearly lower by {magnitude:.3f}"
    if magnitude >= 0.02:
        return f"lower by {magnitude:.3f}"
    return f"slightly lower by {magnitude:.3f}"


def _comparison_payload(
    row: Mapping[str, object],
    reference_row: Mapping[str, object] | None,
    prompt_scores: Mapping[str, float],
    reference_prompt_scores: Mapping[str, float],
) -> dict[str, object] | None:
    if reference_row is None:
        return None

    feature_entries, component_deltas, prompt_deltas = _comparison_feature_entries(
        row=row,
        reference_row=reference_row,
        prompt_scores=prompt_scores,
        reference_prompt_scores=reference_prompt_scores,
    )
    key_advantages = _labels_for_direction(feature_entries, positive=True)
    key_disadvantages = _labels_for_direction(feature_entries, positive=False)

    return {
        "compared_to_image_path": reference_row.get("image_path"),
        "score_delta_vs_reference": _round_float(_delta(row.get("final_score"), reference_row.get("final_score"))),
        "vila_delta_vs_reference": _round_float(_delta(_row_vila_score(row), _row_vila_score(reference_row))),
        "component_deltas": {
            key: _round_float(value)
            for key, value in component_deltas.items()
        },
        "prompt_deltas": {
            key: _round_float(value)
            for key, value in prompt_deltas.items()
        },
        "key_advantages": key_advantages,
        "key_disadvantages": key_disadvantages,
    }


def _comparison_sentence(
    row: Mapping[str, object],
    selected: bool,
    reference_row: Mapping[str, object] | None,
    comparison: Mapping[str, object] | None,
) -> str:
    if reference_row is None or comparison is None:
        return ""

    reference_name = _display_image_name(reference_row.get("image_path"))
    score_delta = _to_float(comparison.get("score_delta_vs_reference"))
    advantages = _unique_labels([str(label) for label in comparison.get("key_advantages") or []])
    disadvantages = _unique_labels([str(label) for label in comparison.get("key_disadvantages") or []])

    if selected:
        options = []
        if score_delta is not None and score_delta < 0 and advantages:
            options.append(
                f"Against {reference_name}, the final score was {_score_delta_phrase(score_delta)}, but { _join_phrases(advantages[:2]) } stayed stronger"
            )
        if advantages:
            options.append(
                f"Relative to {reference_name}, the score edge was {_score_delta_phrase(score_delta)}, driven mostly by {_join_phrases(advantages[:2])}"
            )
            options.append(
                f"{reference_name} stayed closest in rank, yet {_join_phrases(advantages[:2])} kept this frame ahead"
            )
        else:
            options.append(f"Compared with {reference_name}, the final score was {_score_delta_phrase(score_delta)}")
        if disadvantages:
            options = [
                f"{option}, although {_join_phrases(disadvantages[:1])} {_be_verb(disadvantages[:1], 'was', 'were')} weaker"
                for option in options
            ]
        return _sentence(_choose_variant(options, row.get("image_path"), row.get("rank"), "comparison"))

    options = []
    if disadvantages:
        options.append(
            f"Against {reference_name}, the final score was {_score_delta_phrase(score_delta)} and the main deficits were {_join_phrases(disadvantages[:2])}"
        )
        options.append(
            f"{reference_name} stayed ahead because {_join_phrases(disadvantages[:2])} were weaker here"
        )
    else:
        options.append(f"Compared with {reference_name}, the final score was {_score_delta_phrase(score_delta)}")
    if advantages:
        options = [f"{option}, although {_join_phrases(advantages[:1])} remained competitive" for option in options]
    return _sentence(_choose_variant(options, row.get("image_path"), row.get("rank"), "comparison"))


def _secondary_comparison_sentence(
    row: Mapping[str, object],
    selected: bool,
    neighbor_row: Mapping[str, object] | None,
    comparison: Mapping[str, object] | None,
) -> str:
    if neighbor_row is None or comparison is None:
        return ""

    neighbor_name = _display_image_name(neighbor_row.get("image_path"))
    advantages = _unique_labels([str(label) for label in comparison.get("key_advantages") or []])
    disadvantages = _unique_labels([str(label) for label in comparison.get("key_disadvantages") or []])
    vila_delta = _to_float(comparison.get("vila_delta_vs_reference"))

    if selected:
        if advantages:
            edge_noun = "edge was" if len(advantages[:2]) == 1 else "edges were"
            options = [
                f"Against {neighbor_name}, the clearest {edge_noun} {_join_phrases(advantages[:2])}",
                f"The nearby comparison with {neighbor_name} tilted on {_join_phrases(advantages[:2])}",
            ]
        else:
            options = [
                f"Against {neighbor_name}, the selector edge was narrow",
                f"The nearby comparison with {neighbor_name} stayed close overall",
            ]
        if vila_delta is not None and abs(vila_delta) >= 0.03:
            direction = "higher" if vila_delta > 0 else "lower"
            options = [f"{option}, and the VILA score was {direction} by {abs(vila_delta):.3f}" for option in options]
        return _sentence(_choose_variant(options, row.get("image_path"), row.get("rank"), "secondary_comparison"))

    if disadvantages:
        area_noun = "area was" if len(disadvantages[:2]) == 1 else "areas were"
        options = [
            f"Against {neighbor_name}, the weaker {area_noun} {_join_phrases(disadvantages[:2])}",
            f"The nearby comparison with {neighbor_name} was lost on {_join_phrases(disadvantages[:2])}",
        ]
    else:
        options = [
            f"Against {neighbor_name}, the margin stayed narrow",
            f"The nearby comparison with {neighbor_name} remained close overall",
        ]
    if vila_delta is not None and abs(vila_delta) >= 0.03:
        direction = "higher" if vila_delta > 0 else "lower"
        options = [f"{option}, while the VILA score was {direction} by {abs(vila_delta):.3f}" for option in options]
    return _sentence(_choose_variant(options, row.get("image_path"), row.get("rank"), "secondary_comparison"))


def _reason_summary_sentence(
    row: Mapping[str, object],
    selected: bool,
    top_k: int,
    cutline_row: Mapping[str, object] | None,
) -> str:
    rank = int(_to_float(row.get("rank")) or 0)
    final_score = _to_float(row.get("final_score"))
    if final_score is None:
        if selected:
            return _sentence(f"Ranked #{rank} inside the selected set without a usable final score")
        return _sentence(f"Ranked #{rank} outside the selected set without a usable final score")

    if cutline_row is None or cutline_row is row:
        if selected:
            return _sentence(f"Ranked #{rank} in the top-{top_k} with a final score of {final_score:.3f}")
        return _sentence(f"Ranked #{rank} outside the top-{top_k} with a final score of {final_score:.3f}")

    delta_to_cutline = _delta(row.get("final_score"), cutline_row.get("final_score"))
    if delta_to_cutline is None:
        if selected:
            return _sentence(f"Ranked #{rank} in the top-{top_k} with a final score of {final_score:.3f}")
        return _sentence(f"Ranked #{rank} outside the top-{top_k} with a final score of {final_score:.3f}")

    if selected:
        return _sentence(
            f"Ranked #{rank} with a final score of {final_score:.3f}, staying {abs(delta_to_cutline):.3f} above the cut line"
        )
    return _sentence(
        f"Ranked #{rank} with a final score of {final_score:.3f}, trailing the cut line by {abs(delta_to_cutline):.3f}"
    )


def _vila_sentence(
    row: Mapping[str, object],
    selected: bool,
    prompt_scores: Mapping[str, float],
    signals: Mapping[str, object] | None,
    family: str,
) -> str:
    if not prompt_scores:
        return ""

    strengths = _top_prompt_labels(prompt_scores=prompt_scores, signals=signals, prefer_strengths=True)
    weaknesses = _top_prompt_labels(prompt_scores=prompt_scores, signals=signals, prefer_strengths=False)
    vila_score = _clip_unit_score(prompt_scores.get("good_image"))
    strength_phrase = _join_phrases(_filtered_reason_labels(strengths[:2]))
    weakness_phrase = _join_phrases(_filtered_reason_labels(weaknesses[:2]))

    if selected:
        if strength_phrase and weakness_phrase:
            options = [
                f"VILA mostly backed the decision through {strength_phrase}, with {weakness_phrase} still softer",
                f"Prompt-side evidence agreed on {strength_phrase}, even though {weakness_phrase} remained weaker",
                f"On the VILA side, {strength_phrase} supported the pick more than {weakness_phrase} hurt it",
            ]
            return _sentence(_choose_variant(options, row.get("image_path"), row.get("rank"), family, "vila"))
        if strength_phrase:
            options = [
                f"VILA reinforced the pick through {strength_phrase}",
                f"Prompt-side evidence supported the selection on {strength_phrase}",
                f"VILA added support mainly on {strength_phrase}",
            ]
            return _sentence(_choose_variant(options, row.get("image_path"), row.get("rank"), family, "vila"))
        if vila_score is not None:
            options = [
                "VILA support stayed limited despite the selected ranking",
                "The prompt-side signal was present but not especially strong",
                "VILA remained secondary to the selector evidence here",
            ]
            return _sentence(_choose_variant(options, row.get("image_path"), row.get("rank"), family, "vila"))
        return ""

    if weakness_phrase and strength_phrase:
        options = [
            f"Prompt-side evidence also slipped on {weakness_phrase}, although {strength_phrase} was not absent",
            f"VILA did not rescue the frame: {weakness_phrase} stayed behind despite {strength_phrase}",
            f"The prompt signals were mixed, but {weakness_phrase} remained the larger issue",
        ]
        return _sentence(_choose_variant(options, row.get("image_path"), row.get("rank"), family, "vila"))
    if weakness_phrase:
        options = [
            f"VILA also trailed on {weakness_phrase}",
            f"Prompt-side evidence stayed weakest on {weakness_phrase}",
            f"The VILA signal remained soft on {weakness_phrase}",
        ]
        return _sentence(_choose_variant(options, row.get("image_path"), row.get("rank"), family, "vila"))
    if vila_score is not None:
        options = [
            "VILA support was not strong enough to offset the selector deficit",
            "The prompt-side signal never became strong enough to recover the selector gap",
            "VILA remained too weak to change the rejection outcome",
        ]
        return _sentence(_choose_variant(options, row.get("image_path"), row.get("rank"), family, "vila"))
    return ""


def _reference_row_for_mode(
    row: Mapping[str, object],
    rows: list[dict[str, object]],
    index: int,
    top_k: int,
    reference_mode: str,
) -> dict[str, object] | None:
    higher = rows[index - 1] if index > 0 else None
    lower = rows[index + 1] if index + 1 < len(rows) else None
    rank = int(_to_float(row.get("rank")) or (index + 1))

    if reference_mode == "nearest_higher":
        return higher or lower
    if reference_mode == "top1":
        if index == 0:
            return lower
        return rows[0]

    if higher is not None and rank > top_k and int(_to_float(higher.get("rank")) or 0) == top_k:
        return higher
    if lower is not None and rank == top_k:
        return lower

    candidates = [candidate for candidate in (higher, lower) if candidate is not None]
    if not candidates:
        return None

    def candidate_key(candidate: Mapping[str, object]) -> tuple[float, int, str]:
        delta = _delta(row.get("final_score"), candidate.get("final_score"))
        return (
            abs(delta) if delta is not None else float("inf"),
            abs(int(_to_float(candidate.get("rank")) or 0) - rank),
            str(candidate.get("image_path")),
        )

    return min(candidates, key=candidate_key)


def synthesize_acut_selection_explanations(
    rows: list[dict[str, object]],
    top_k: int,
    reference_mode: str = DEFAULT_REASON_REFERENCE_MODE,
    detail_level: str = DEFAULT_REASON_DETAIL_LEVEL,
    include_model_contributions: bool = False,
    include_vila_signals: bool = True,
    include_comparison: bool = True,
    vila_rows_by_path: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    if reference_mode not in REASON_REFERENCE_MODES:
        raise ValueError(
            f"Unknown reason reference mode '{reference_mode}'. Expected one of: {', '.join(REASON_REFERENCE_MODES)}"
        )
    if detail_level not in REASON_DETAIL_LEVELS:
        raise ValueError(
            f"Unknown reason detail level '{detail_level}'. Expected one of: {', '.join(REASON_DETAIL_LEVELS)}"
        )

    detail_limits = {
        "short": 2,
        "standard": 3,
        "full": 4,
    }

    for index, row in enumerate(rows):
        selected = int(_to_float(row.get("rank")) or (index + 1)) <= top_k
        image_path = row.get("image_path")
        vila_row = None
        if vila_rows_by_path is not None and image_path is not None:
            vila_row = vila_rows_by_path.get(str(image_path))

        prompt_scores = _prompt_scores_for_row(row=row, vila_row=vila_row)
        signals = _signals_for_row(row=row, prompt_scores=prompt_scores)
        vila_strengths = _top_prompt_labels(prompt_scores=prompt_scores, signals=signals, prefer_strengths=True)
        vila_weaknesses = _top_prompt_labels(prompt_scores=prompt_scores, signals=signals, prefer_strengths=False)

        cutline_row = None
        if rows:
            if selected and len(rows) > top_k:
                cutline_row = rows[top_k]
            elif (not selected) and top_k > 0 and len(rows) >= top_k:
                cutline_row = rows[top_k - 1]

        higher_neighbor = rows[index - 1] if index > 0 else None
        lower_neighbor = rows[index + 1] if index + 1 < len(rows) else None
        primary_reference = _reference_row_for_mode(
            row=row,
            rows=rows,
            index=index,
            top_k=top_k,
            reference_mode=reference_mode,
        )

        primary_prompt_scores = _prompt_scores_for_row(primary_reference, vila_row=None) if primary_reference else {}
        higher_prompt_scores = _prompt_scores_for_row(higher_neighbor, vila_row=None) if higher_neighbor else {}
        lower_prompt_scores = _prompt_scores_for_row(lower_neighbor, vila_row=None) if lower_neighbor else {}

        primary_comparison = (
            _comparison_payload(
                row=row,
                reference_row=primary_reference,
                prompt_scores=prompt_scores,
                reference_prompt_scores=primary_prompt_scores,
            )
            if include_comparison
            else None
        )
        higher_comparison = (
            _comparison_payload(
                row=row,
                reference_row=higher_neighbor,
                prompt_scores=prompt_scores,
                reference_prompt_scores=higher_prompt_scores,
            )
            if include_comparison and higher_neighbor is not None
            else None
        )
        lower_comparison = (
            _comparison_payload(
                row=row,
                reference_row=lower_neighbor,
                prompt_scores=prompt_scores,
                reference_prompt_scores=lower_prompt_scores,
            )
            if include_comparison and lower_neighbor is not None
            else None
        )

        selector_sentence, dominant_component = _component_summary(
            row=row,
            include_model_contributions=include_model_contributions,
        )
        summary_sentence = _reason_summary_sentence(
            row=row,
            selected=selected,
            top_k=top_k,
            cutline_row=cutline_row,
        )
        short_reason, reason_family = _short_reason(
            row=row,
            selected=selected,
            comparison=primary_comparison if include_comparison else None,
            dominant_component=dominant_component,
            vila_strengths=vila_strengths if include_vila_signals else [],
            vila_weaknesses=vila_weaknesses if include_vila_signals else [],
            cutline_row=cutline_row,
        )
        evidence_sentence = _evidence_sentence(
            row=row,
            selected=selected,
            comparison=primary_comparison if include_comparison else None,
            dominant_component=dominant_component,
            vila_strengths=vila_strengths if include_vila_signals else [],
            vila_weaknesses=vila_weaknesses if include_vila_signals else [],
            family=reason_family,
        )
        vila_sentence = _vila_sentence(
            row=row,
            selected=selected,
            prompt_scores=prompt_scores if include_vila_signals else {},
            signals=signals if include_vila_signals else None,
            family=reason_family,
        )
        comparison_sentence = _comparison_sentence(
            row=row,
            selected=selected,
            reference_row=primary_reference,
            comparison=primary_comparison,
        )

        secondary_sentence = ""
        if include_comparison and detail_level == "full":
            secondary_target = higher_neighbor if primary_reference is not higher_neighbor else lower_neighbor
            secondary_payload = higher_comparison if secondary_target is higher_neighbor else lower_comparison
            secondary_sentence = _secondary_comparison_sentence(
                row=row,
                selected=selected,
                neighbor_row=secondary_target,
                comparison=secondary_payload,
            )

        mechanics_sentence = _mechanics_sentence(row) if detail_level == "full" else ""

        detailed_sentences = [summary_sentence, evidence_sentence]
        if reason_family in {"strong_technical", "balanced_win"}:
            detailed_sentences.append(selector_sentence)
            if include_comparison and comparison_sentence:
                detailed_sentences.append(comparison_sentence)
            if include_vila_signals and vila_sentence:
                detailed_sentences.append(vila_sentence)
        elif reason_family in {"strong_composition", "tradeoff_selected", "tradeoff_rejected", "weak_selected"}:
            if include_comparison and comparison_sentence:
                detailed_sentences.append(comparison_sentence)
            if include_vila_signals and vila_sentence:
                detailed_sentences.append(vila_sentence)
            detailed_sentences.append(selector_sentence)
        else:
            if include_vila_signals and vila_sentence:
                detailed_sentences.append(vila_sentence)
            if include_comparison and comparison_sentence:
                detailed_sentences.append(comparison_sentence)
            detailed_sentences.append(selector_sentence)
        if secondary_sentence:
            detailed_sentences.append(secondary_sentence)
        if mechanics_sentence:
            detailed_sentences.append(mechanics_sentence)

        detailed_reason = " ".join(
            sentence
            for sentence in detailed_sentences[: detail_limits[detail_level]]
            if sentence
        )

        comparison_reason_parts = []
        if include_comparison and comparison_sentence:
            comparison_reason_parts.append(comparison_sentence)
        if include_comparison and detail_level == "full" and secondary_sentence:
            comparison_reason_parts.append(secondary_sentence)
        comparison_reason = " ".join(part for part in comparison_reason_parts if part) or None

        rejection_reason = short_reason if not selected else None

        structured = {
            "status": "selected" if selected else "rejected",
            "rank": int(_to_float(row.get("rank")) or (index + 1)),
            "top_k": int(top_k),
            "reference_mode": reference_mode,
            "detail_level": detail_level,
            "selector": {
                "final_score": _round_float(_to_float(row.get("final_score"))),
                "base_score": _round_float(_to_float(row.get("base_score"))),
                "final_score_before_vila": _round_float(_to_float(row.get("final_score_before_vila"))),
                "final_score_after_rerank": _round_float(_to_float(row.get("final_score_after_rerank"))),
                "aesthetic_component": _round_float(_to_float(row.get("aesthetic_component"))),
                "technical_component": _round_float(_to_float(row.get("technical_component"))),
                "dominant_component": dominant_component,
                "top_model_contributions": _top_model_contributions(row) if include_model_contributions else [],
            },
            "comparison": {
                "higher_neighbor_image_path": higher_neighbor.get("image_path") if higher_neighbor is not None else None,
                "lower_neighbor_image_path": lower_neighbor.get("image_path") if lower_neighbor is not None else None,
                "compared_to_image_path": primary_comparison.get("compared_to_image_path") if primary_comparison else None,
                "score_delta_vs_reference": primary_comparison.get("score_delta_vs_reference") if primary_comparison else None,
                "vila_delta_vs_reference": primary_comparison.get("vila_delta_vs_reference") if primary_comparison else None,
                "key_advantages": primary_comparison.get("key_advantages") if primary_comparison else [],
                "key_disadvantages": primary_comparison.get("key_disadvantages") if primary_comparison else [],
                "component_deltas": primary_comparison.get("component_deltas") if primary_comparison else {},
                "prompt_deltas": primary_comparison.get("prompt_deltas") if primary_comparison else {},
            }
            if include_comparison
            else None,
            "vila": {
                "available": bool(prompt_scores or _row_vila_score(row) is not None),
                "vila_score_raw": _round_float(_row_vila_score(row)),
                "vila_score_normalized_in_pool": _round_float(_to_float(row.get("vila_score_normalized_in_pool"))),
                "top_strengths": vila_strengths,
                "top_weaknesses": vila_weaknesses,
                "signals": signals,
            }
            if include_vila_signals
            else {
                "available": bool(prompt_scores or _row_vila_score(row) is not None),
                "vila_score_raw": _round_float(_row_vila_score(row)),
                "vila_score_normalized_in_pool": _round_float(_to_float(row.get("vila_score_normalized_in_pool"))),
            },
            "mechanics": {
                "pairwise_rerank_applied": bool(row.get("pairwise_rerank_applied")),
                "pairwise_rerank_delta": _round_float(_to_float(row.get("pairwise_rerank_delta"))),
                "vila_rerank_applied": bool(row.get("vila_rerank_applied")),
                "vila_rerank_delta": _round_float(_to_float(row.get("vila_rerank_delta"))),
                "diversity_penalty": _round_float(_to_float(row.get("diversity_penalty"))),
            },
            "text": {
                "template_family": reason_family,
                "short_reason": short_reason,
                "detailed_reason": detailed_reason,
                "comparison_reason": comparison_reason,
                "rejection_reason": rejection_reason,
            },
        }

        row["acut_short_reason"] = short_reason
        row["acut_detailed_reason"] = detailed_reason
        row["acut_comparison_reason"] = comparison_reason
        row["acut_rejection_reason"] = rejection_reason
        row["acut_explanation_structured"] = structured
