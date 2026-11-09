from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.gemma_distill.common import (
    clean_text,
    collapse_whitespace,
    normalize_unit_score,
    rank_band_from_rank,
    read_json,
    round_float,
    sentence,
)


DEFAULT_LINE_PREFIXES = ("Strength:", "Weakness:", "Verdict:")
STUDENT_VERDICT_FORBIDDEN_RE = re.compile(
    r"\bselection_state\b|\bscore\b|\bthreshold\b|\bweighted_aesthetic_score\b|\brank(?:ing|ed)?\b|top\s*-?\s*k|top\s*\d+\s*percent|nima_signal|rgnet_signal|alamp_signal|nima_unit_score|rgnet_score|alamp_score",
    flags=re.IGNORECASE,
)
VISUAL_PHRASE_REPLACEMENTS = (
    (r"\btechnical quality\b", "detail clarity"),
    (r"\baesthetic score\b", "overall appeal"),
    (r"\bsubject clarity\b", "subject emphasis"),
    (r"\bbackground cleanliness\b", "cleaner background"),
    (r"\boverall image quality\b", "overall polish"),
    (r"\boverall image appeal\b", "overall appeal"),
)


def load_teacher_weight_config(path_like: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(path_like)
    if not path.is_absolute():
        from src.gemma_distill.common import resolve_repo_path

        resolved = resolve_repo_path(path)
        if resolved is None:
            raise ValueError("Could not resolve teacher weight config path.")
        path = resolved
    config = read_json(path)
    weights = dict(config.get("score_weights") or {})
    required = {"nima_unit_score", "rgnet_score", "alamp_score"}
    missing = sorted(required - set(weights))
    if missing:
        raise ValueError(f"Teacher weight config is missing keys: {', '.join(missing)}")
    total_weight = sum(float(value) for value in weights.values())
    if total_weight <= 0.0:
        raise ValueError("Teacher weights must sum to a positive value.")
    config["score_weights"] = {str(key): float(value) / float(total_weight) for key, value in weights.items()}
    return config, path


def compute_three_model_teacher_score(row: dict[str, Any], weight_config: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    nima_unit = normalize_unit_score(row.get("nima_unit_score"))
    rgnet_score = normalize_unit_score(row.get("rgnet_score"))
    alamp_score = normalize_unit_score(row.get("alamp_score"))
    components = {
        "nima_unit_score": nima_unit,
        "rgnet_score": rgnet_score,
        "alamp_score": alamp_score,
    }
    if any(value is None for value in components.values()):
        return None, {"components": components, "missing_components": sorted(key for key, value in components.items() if value is None)}

    weights = dict(weight_config["score_weights"])
    weighted_score = (
        float(weights["nima_unit_score"]) * float(nima_unit)
        + float(weights["rgnet_score"]) * float(rgnet_score)
        + float(weights["alamp_score"]) * float(alamp_score)
    )
    contributions = {
        key: {
            "normalized_score": round_float(value),
            "configured_weight": round_float(weights[key]),
            "weighted_contribution": round_float(float(weights[key]) * float(value)),
        }
        for key, value in components.items()
    }
    return round_float(weighted_score), contributions


def build_rank_summary(rank: int | None, total_rows: int | None) -> dict[str, Any]:
    return {
        "rank": int(rank) if rank is not None else None,
        "total_rows": int(total_rows) if total_rows is not None else None,
        "rank_band": rank_band_from_rank(rank=rank, total=total_rows),
    }


def build_student_prompt_text(row: dict[str, Any]) -> str:
    selection_state = "selected" if bool(row.get("selected")) else "not selected"
    return "\n".join(
        [
            "Use the image first. The internal selection context controls the verdict only.",
            f"nima_signal: {float(row['nima_unit_score']):.6f}",
            f"rgnet_signal: {float(row['rgnet_score']):.6f}",
            f"alamp_signal: {float(row['alamp_score']):.6f}",
            f"selection_state: {selection_state}",
            "Write only visible evidence for why the frame made or missed the A-cut.",
            "The Verdict line must match selection_state exactly.",
            'If selection_state is selected, write "Verdict: selected because ...".',
            'If selection_state is not selected, write "Verdict: not selected because ...".',
            "Use the numeric signals only as silent guidance for what to emphasize visually.",
            "Do not mention scores, numbers, thresholds, ranking, percent bands, model names, or hidden context.",
            "Return exactly 3 lines and nothing else:",
            "Strength: ...",
            "Weakness: ...",
            "Verdict: selected because ... / Verdict: not selected because ...",
        ]
    )


def build_training_messages(row: dict[str, Any], teacher_comment_3line: str) -> list[dict[str, Any]]:
    system_text = (
        "You are the Gemma 4 E4B A-cut student. "
        "Return exactly three lines with the prefixes Strength:, Weakness:, and Verdict:. "
        "Use visible evidence only. "
        "The Verdict line must follow selection_state exactly. "
        "Do not mention scores, numbers, thresholds, ranking, percent bands, model names, bullets, numbering, or extra text."
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": build_student_prompt_text(row)},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": teacher_comment_3line}],
        },
    ]


def build_vila_teacher_prompt(row: dict[str, Any]) -> str:
    selected_label = "selected" if bool(row.get("selected")) else "rejected"
    verdict_prefix = "selected because" if selected_label == "selected" else "not selected because"
    return f"""
You are the VILA-full teacher for A-cut aesthetic selection.
This is not captioning.
Use the image first and the numeric context only as selection guidance.
The numeric context uses exactly three aesthetic inputs: NIMA unit, RGNet, and A-Lamp, plus their weighted aesthetic score.
Do not mention technical score, final mixed score, AADB composition, captions, file names, other images, or raw numbers.

Return exactly 3 lines and nothing else:
Strength: ...
Weakness: ...
Verdict: {verdict_prefix} ...

Rules:
- Strength line: one short visible strength grounded in the image
- Weakness line: one short visible weakness grounded in the image
- Verdict line: one short A-cut judgment that starts with "{verdict_prefix}"
- Use visible evidence phrases like clean framing, clearer subject emphasis, limited separation, busier background, flatter light, softer detail, cleaner background
- Keep each line concise
- Do not add bullets, numbering, quotes, or extra explanation

Selection context:
- selected_label: {selected_label}
- rank_band_label: {row.get("rank_band", "unknown")}
- weighted_aesthetic_score: {float(row['weighted_aesthetic_score']):.6f}
- nima_unit_score: {float(row['nima_unit_score']):.6f}
- rgnet_score: {float(row['rgnet_score']):.6f}
- alamp_score: {float(row['alamp_score']):.6f}
""".strip()


def _extract_status_and_tail(short_reason: object, selected: bool) -> tuple[str, str, str | None]:
    text = collapse_whitespace(short_reason) or ""
    match = re.match(r"^(selected|rejected)\s*:\s*(.+)$", text, flags=re.IGNORECASE)
    if match:
        return ("selected" if selected else "rejected"), match.group(2).strip(), match.group(1).lower()
    return ("selected" if selected else "rejected"), text, None


def _split_strength_and_weakness(short_reason: object, selected: bool) -> tuple[str | None, str | None]:
    output_status, tail, source_status = _extract_status_and_tail(short_reason, selected=selected)
    negative_tradeoff = re.match(
        r"^(?:rejected after the tradeoff broke the wrong way|this frame fell below the cut when)\s+(.+?)\s+(?:were|was|outweighed|outweigh)\s+(.+?)(?:\s+in nearby comparisons)?[.]?$",
        tail,
        flags=re.IGNORECASE,
    )
    if negative_tradeoff:
        weakness_phrase = negative_tradeoff.group(1).strip(" .;")
        strength_phrase = negative_tradeoff.group(2).strip(" .;")
        if re.search(r"\bnot enough to offset\b", tail, flags=re.IGNORECASE):
            strength_phrase = negative_tradeoff.group(1).strip(" .;")
            weakness_phrase = negative_tradeoff.group(2).strip(" .;")
        return strength_phrase, weakness_phrase

    missed_cut = re.match(
        r"^it missed the a-cut mainly because\s+(.+?),\s*even though\s+(.+?)\s+remained\s+acceptable[.]?$",
        tail,
        flags=re.IGNORECASE,
    )
    if missed_cut:
        weakness_phrase = missed_cut.group(1).strip(" .;")
        strength_phrase = missed_cut.group(2).strip(" .;")
        return strength_phrase, weakness_phrase

    rejected_offset = re.match(
        r"^rejected after the tradeoff broke the wrong way:\s+(.+?)\s+(?:were|was) not enough to offset\s+(.+?)[.]?$",
        tail,
        flags=re.IGNORECASE,
    )
    if rejected_offset:
        return rejected_offset.group(1).strip(" .;"), rejected_offset.group(2).strip(" .;")

    parts = [part.strip(" .;") for part in re.split(r"[,;]", tail) if clean_text(part)]
    if len(parts) >= 2:
        return parts[0], ", ".join(parts[1:])
    if len(parts) == 1:
        if selected and source_status == "rejected":
            return None, parts[0]
        if not selected and source_status == "selected":
            return parts[0], None
        if output_status == "selected":
            return parts[0], None
        return None, parts[0]
    return None, None


def _detail_sentences(detailed_reason: object) -> list[str]:
    text = collapse_whitespace(detailed_reason) or ""
    if not text:
        return []
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]


def _looks_like_file_name(text: str) -> bool:
    return bool(re.search(r"\b[\w-]+\.(?:jpg|jpeg|png|webp|heic)\b", text, flags=re.IGNORECASE))


def _sanitize_visual_phrase(value: object, *, fallback: str) -> str:
    text = collapse_whitespace(value) or ""
    if not text:
        return fallback
    text = re.sub(r"\bRanked\s*#?\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwith a final score of\s*[-+]?\d+(?:\.\d+)?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\btrailing the cut line by\s*[-+]?\d+(?:\.\d+)?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthe final score was roughly tied\s*\([^)]*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwhile the VILA score was [^.]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bnearby:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAgainst\s+[\w.-]+\s*,?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[A-Za-z0-9_-]+\.(?:jpg|jpeg|png|webp|heic)\b", "", text, flags=re.IGNORECASE)
    for pattern, replacement in VISUAL_PHRASE_REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:this frame|the frame|it)\s+fell below the cut when\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:it\s+)?missed the a-cut mainly because\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\brejected after the tradeoff broke the wrong way:\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:selected because|not selected because)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthe final outcome stays (?:selected|rejected) because\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthose strengths remain competitive\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\band the main deficits were\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthe main deficits were\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthe weaker areas were\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bstayed ahead because\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\btrailed the stronger alternatives\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\btrailed the selected set\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bin nearby comparisons\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\beven though\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\balthough\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bremained acceptable\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bremained competitive\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwere weaker here\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwas weaker here\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthe nearby comparison [^.]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bprompt-side evidence\b", "the visual read", text, flags=re.IGNORECASE)
    text = re.sub(r"\bprompt signals\b", "the visual read", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcut-line alternatives\b", "stronger alternatives", text, flags=re.IGNORECASE)
    text = re.sub(r"\bselected set\b", "stronger alternatives", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*because\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bscore\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .;,:-")
    return text or fallback


def _comparison_reason_usable(text: str | None) -> bool:
    if not text:
        return False
    disallowed_patterns = (
        r"\bfinal score\b",
        r"\bcut line\b",
        r"\bVILA score\b",
        r"\bAgainst\b",
        r"\btechnical quality\b",
        r"\b\d+\.\d+\b",
        r"\b\d+\b",
    )
    if _looks_like_file_name(text):
        return False
    return not any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in disallowed_patterns)


def _infer_strength_from_detail(detailed_reason: object, selected: bool) -> str:
    strength, weakness = _split_strength_and_weakness(detailed_reason, selected=selected)
    if strength:
        return strength
    text = collapse_whitespace(detailed_reason) or ""
    match = re.search(r"such as ([^.]+?) help", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(" .")
    sentences = _detail_sentences(detailed_reason)
    if sentences:
        return sentences[0].rstrip(".")
    return "competitive visual evidence"


def _infer_weakness_from_detail(detailed_reason: object, *, selected: bool) -> str:
    text = collapse_whitespace(detailed_reason) or ""
    match = re.search(r"because (.+?)(?:\.|$)", text, flags=re.IGNORECASE)
    if match:
        clause = re.sub(
            r"^(those strengths|the final outcome stays selected|the final outcome stays rejected)\s+",
            "",
            match.group(1).strip(),
            flags=re.IGNORECASE,
        )
        clause = re.sub(r"^because\s+", "", clause, flags=re.IGNORECASE)
        clause = clause.strip(" .")
        lowered_clause = clause.lower()
        if lowered_clause in {
            "remain competitive",
            "those strengths remain competitive",
            "those strengths stayed competitive",
            "those strengths remain strong",
        }:
            clause = ""
        if clause:
            return clause
    sentences = _detail_sentences(detailed_reason)
    if len(sentences) >= 2:
        return sentences[1].rstrip(".")
    if sentences:
        return sentences[-1].rstrip(".")
    return "weaker secondary cues" if selected else "weaker A-cut impact"


def format_teacher_comment_3line(
    *,
    short_reason: object,
    detailed_reason: object,
    comparison_reason: object,
    selected: bool,
) -> str:
    selected_label, _, source_status = _extract_status_and_tail(short_reason, selected=selected)
    strength_phrase, weakness_phrase = _split_strength_and_weakness(short_reason, selected=selected)
    if not strength_phrase:
        strength_phrase = _infer_strength_from_detail(detailed_reason, selected=selected)
    if not weakness_phrase:
        weakness_phrase = "weaker secondary cues" if selected_label == "selected" else _infer_weakness_from_detail(
            detailed_reason,
            selected=selected,
        )
    strength_phrase = _sanitize_visual_phrase(strength_phrase, fallback="clear visual strengths")
    weakness_phrase = _sanitize_visual_phrase(weakness_phrase, fallback="weaker secondary cues")

    verdict_prefix = "selected because" if selected_label == "selected" else "not selected because"
    if selected_label == "selected":
        if weakness_phrase.strip(" ."):
            verdict_text = f"{verdict_prefix} {strength_phrase.strip(' .')} carries the frame despite {weakness_phrase.strip(' .')}"
        else:
            verdict_text = f"{verdict_prefix} {strength_phrase.strip(' .')} carries the frame"
    else:
        verdict_text = f"{verdict_prefix} {weakness_phrase.strip(' .')} undercuts {strength_phrase.strip(' .')}"

    lines = [
        f"Strength: {sentence(strength_phrase)}",
        f"Weakness: {sentence(weakness_phrase)}",
        f"Verdict: {sentence(verdict_text)}",
    ]
    return "\n".join(lines)


def validate_teacher_comment_3line(text: object) -> dict[str, Any]:
    raw = clean_text(text)
    lines = [] if raw is None else [line.strip() for line in str(raw).splitlines() if line.strip()]
    prefixes = [line.split(" ", 1)[0] for line in lines]
    prefix_targets = list(DEFAULT_LINE_PREFIXES)
    prefix_ok = len(lines) == 3 and all(lines[index].startswith(prefix_targets[index]) for index in range(min(3, len(lines))))
    non_empty_line_values = [
        re.sub(r"^[A-Za-z]+:\s*", "", line).strip()
        for line in lines[:3]
    ]
    role_separation_ok = prefix_ok and len({value.lower() for value in non_empty_line_values if value}) == 3
    return {
        "line_count": len(lines),
        "prefix_ok": prefix_ok,
        "role_separation_ok": role_separation_ok,
        "is_valid": prefix_ok and role_separation_ok,
        "lines": lines,
    }


def repair_student_comment_3line(text: object, *, selected: bool) -> str:
    raw = clean_text(text)
    if raw is None:
        return ""
    lines = [line.strip() for line in str(raw).splitlines() if line.strip()]
    if len(lines) < 2:
        return str(raw).strip()

    def extract_value(line: str, prefix: str, fallback: str) -> str:
        if line.startswith(prefix):
            value = line[len(prefix) :].strip()
        else:
            value = line.strip()
        return _sanitize_visual_phrase(value, fallback=fallback)

    strength_phrase = extract_value(lines[0], "Strength:", "clear visual strengths")
    weakness_fallback = "weaker secondary cues" if selected else "weaker A-cut impact"
    weakness_phrase = extract_value(lines[1], "Weakness:", weakness_fallback)

    verdict_line = lines[2] if len(lines) >= 3 else ""
    label_match = re.match(r"^Verdict:\s*(selected|not selected)\s+because\s*(.+)$", verdict_line, flags=re.IGNORECASE)
    expected_label = "selected" if selected else "not selected"
    verdict_reason = ""
    if label_match:
        verdict_reason = label_match.group(2).strip(" .")

    verdict_reason_lower = verdict_reason.lower()
    tautological_reasons = {
        "selected",
        "not selected",
        "selection_state",
        f"selection_state: {expected_label}",
        f"selection state: {expected_label}",
    }
    verdict_bad = (
        not verdict_line.startswith("Verdict:")
        or label_match is None
        or label_match.group(1).lower() != expected_label
        or not verdict_reason
        or verdict_reason_lower in tautological_reasons
        or STUDENT_VERDICT_FORBIDDEN_RE.search(verdict_line) is not None
    )
    if not verdict_bad:
        return "\n".join(lines[:3])

    verdict_prefix = "selected because" if selected else "not selected because"
    if selected:
        verdict_text = f"{verdict_prefix} {strength_phrase.strip(' .')} carries the frame despite {weakness_phrase.strip(' .')}"
    else:
        verdict_text = f"{verdict_prefix} {weakness_phrase.strip(' .')} undercuts {strength_phrase.strip(' .')}"

    return "\n".join(
        [
            f"Strength: {sentence(strength_phrase)}",
            f"Weakness: {sentence(weakness_phrase)}",
            f"Verdict: {sentence(verdict_text)}",
        ]
    )
