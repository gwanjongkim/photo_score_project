from __future__ import annotations

import math
from collections.abc import Mapping

from src.vila.prompt_sets import DEFAULT_PROMPT_PRESET, PromptPair, get_prompt_preset, prompt_score_column


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _known_pairs(prompt_preset: str = DEFAULT_PROMPT_PRESET) -> tuple[PromptPair, ...]:
    return get_prompt_preset(prompt_preset)


def extract_prompt_scores(
    source: Mapping[str, object] | None,
    prompt_preset: str = DEFAULT_PROMPT_PRESET,
) -> dict[str, float]:
    if source is None:
        return {}

    nested = source.get("per_prompt_scores") if isinstance(source, Mapping) else None
    if isinstance(nested, Mapping):
        out = {}
        for pair in _known_pairs(prompt_preset):
            value = nested.get(pair.key)
            if not _is_missing(value):
                out[pair.key] = float(value)
        if out:
            return out

    out = {}
    for pair in _known_pairs(prompt_preset):
        for key in (pair.key, prompt_score_column(pair)):
            value = source.get(key)
            if _is_missing(value):
                continue
            out[pair.key] = float(value)
            break
    return out


def _score_band(vila_score: float | None) -> str:
    if vila_score is None:
        return "unknown"
    if vila_score >= 0.67:
        return "strong"
    if vila_score >= 0.5:
        return "mixed"
    return "weak"


def build_explanation_signals(
    prompt_scores: Mapping[str, float],
    vila_score: float | None = None,
    prompt_preset: str = DEFAULT_PROMPT_PRESET,
) -> dict[str, object]:
    pairs = _known_pairs(prompt_preset)
    scored_pairs = []
    for pair in pairs:
        score = prompt_scores.get(pair.key)
        if score is None:
            continue
        scored_pairs.append(
            {
                "key": pair.key,
                "label": pair.display_name,
                "score": float(score),
                "positive_reason": pair.positive_reason,
                "negative_reason": pair.negative_reason,
            }
        )

    scored_pairs.sort(key=lambda item: item["score"], reverse=True)
    strengths = [item for item in scored_pairs if item["score"] >= 0.58][:3]
    weaknesses = [item for item in sorted(scored_pairs, key=lambda item: item["score"]) if item["score"] <= 0.42][:2]
    neutral = [item for item in scored_pairs if item not in strengths and item not in weaknesses]

    return {
        "score_band": _score_band(vila_score),
        "num_prompt_pairs": len(scored_pairs),
        "top_strength": strengths[0]["key"] if strengths else (scored_pairs[0]["key"] if scored_pairs else None),
        "top_weakness": weaknesses[0]["key"] if weaknesses else (scored_pairs[-1]["key"] if scored_pairs else None),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "neutral": neutral,
    }


def _join_phrases(phrases: list[str]) -> str:
    phrases = [phrase for phrase in phrases if phrase]
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return f"{', '.join(phrases[:-1])}, and {phrases[-1]}"


def build_selection_explanation(
    prompt_scores: Mapping[str, float],
    vila_score: float | None = None,
    selected: bool | None = None,
    prompt_preset: str = DEFAULT_PROMPT_PRESET,
) -> str:
    signals = build_explanation_signals(
        prompt_scores=prompt_scores,
        vila_score=vila_score,
        prompt_preset=prompt_preset,
    )
    if selected is None:
        selected = vila_score is not None and vila_score >= 0.5

    if selected:
        phrases = [item["positive_reason"] for item in signals["strengths"][:3]]
        if not phrases and signals["neutral"]:
            phrases = [signals["neutral"][0]["positive_reason"]]
        if not phrases:
            band = signals["score_band"]
            if band == "strong":
                phrases = ["the overall visual-language signal is strong"]
            elif band == "mixed":
                phrases = ["the overall visual-language signal is balanced"]
            else:
                phrases = ["it remains competitive within the current candidate pool"]
        return f"Selected because {_join_phrases(phrases)}."

    phrases = [item["negative_reason"] for item in signals["weaknesses"][:2]]
    if not phrases and signals["top_weakness"] is not None:
        fallback_scores = {item["key"]: item["negative_reason"] for item in signals["neutral"]}
        if signals["top_weakness"] in fallback_scores:
            phrases = [fallback_scores[signals["top_weakness"]]]
    if not phrases:
        band = signals["score_band"]
        if band == "weak":
            phrases = ["the overall visual-language signal is weak"]
        else:
            phrases = ["it trails stronger alternatives on the prompt-based aesthetic cues"]
    return f"Rejected mainly due to {_join_phrases(phrases)}."


def explain_prompt_scores(
    prompt_scores: Mapping[str, float],
    vila_score: float | None = None,
    selected: bool | None = None,
    prompt_preset: str = DEFAULT_PROMPT_PRESET,
) -> dict[str, object]:
    signals = build_explanation_signals(
        prompt_scores=prompt_scores,
        vila_score=vila_score,
        prompt_preset=prompt_preset,
    )
    text = build_selection_explanation(
        prompt_scores=prompt_scores,
        vila_score=vila_score,
        selected=selected,
        prompt_preset=prompt_preset,
    )
    return {
        "text": text,
        "signals": signals,
    }
