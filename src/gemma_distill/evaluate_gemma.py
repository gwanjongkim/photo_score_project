from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path
from typing import Any

from src.gemma_distill.common import read_json, repo_relative_path, resolve_repo_path, write_json
REQUIRED_JSON_FIELDS = {"comment_type", "short_reason", "detailed_reason", "comparison_reason"}
ALLOWED_COMMENT_TYPES = {"selected_explanation", "near_miss_feedback", "rejection_reason"}

FORBIDDEN_OUTPUT_RE = re.compile(
    r"\b(score|scores|threshold|rank|ranked|ranking|top[- ]?k|top\s+k|model|"
    r"nima|rgnet|a[- ]?lamp|alamp|weighted(?:_|\s+)score|weighted_aesthetic_score)\b|"
    r"\bbecause the score is high\b|\bbecause it passed the threshold\b",
    flags=re.IGNORECASE,
)

POSITIVE_ASPECT_RE = re.compile(
    r"\b(good|well|strong|clear|effective|appealing|compelling|engaging|pleasant|"
    r"interesting|excellent|successful|balanced|sharp|vibrant|rich|works|well[- ]executed)\b",
    flags=re.IGNORECASE,
)

LACKING_OR_IMPROVEMENT_RE = re.compile(
    r"\b(but|however|although|lacks?|lacking|weak|weaker|unclear|poor|distracting|"
    r"blur|soft|flat|overexposed|underexposed|cluttered|held back|issue|problem|"
    r"improv(?:e|ing|ement)|could|should|needs?|would benefit|would help|removing|"
    r"adjusting|simplifying|strengthen)\b",
    flags=re.IGNORECASE,
)

FORCED_PRAISE_RE = re.compile(
    r"\b(?:while|although|though|even though)\b[^.]{0,100}\b(?:good|great|excellent|"
    r"strong|interesting|appealing|nice|compelling|pleasant|well[- ]executed|works well)\b|"
    r"\b(?:has|have|shows|contains|offers)\b[^.]{0,80}\b(?:a good aspect|good|great|"
    r"excellent|strong points?|interesting|appealing|nice|positive)\b",
    flags=re.IGNORECASE,
)

NEGATED_PRAISE_CONTEXT_RE = re.compile(
    r"\b(?:not|no|without|doesn'?t|do not|lacks?|lacking|fails? to|cannot|can't|"
    r"isn'?t|is not)\b[^.]{0,80}\b(?:good|great|excellent|strong|interesting|"
    r"appealing|nice|compelling|pleasant|well[- ]executed|works well)\b",
    flags=re.IGNORECASE,
)

FAKE_SELECTED_WEAKNESS_RE = re.compile(
    r"\b(but|however|although|though|weakness|lacks?|lacking|weak|poor|flaw|issue|"
    r"problem|held back|could be improved|would benefit|needs improvement)\b",
    flags=re.IGNORECASE,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Gemma 4 A-cut student prediction file.")
    parser.add_argument("--teacher_manifest", required=True)
    parser.add_argument("--predictions_jsonl")
    parser.add_argument("--output_dir", required=True)
    return parser


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_text(text: object) -> str:
    value = "" if text is None else str(text)
    value = value.replace("\r\n", "\n").strip()
    return re.sub(r"\s+", " ", value).strip()


def token_f1(reference: str, prediction: str) -> float:
    ref_tokens = normalize_text(reference).lower().split()
    pred_tokens = normalize_text(prediction).lower().split()
    if not ref_tokens or not pred_tokens:
        return 0.0
    ref_counts: dict[str, int] = {}
    pred_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    overlap = sum(min(ref_counts[token], pred_counts.get(token, 0)) for token in ref_counts)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def canonical_json_text(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return ""
    return json.dumps({key: payload.get(key) for key in sorted(REQUIRED_JSON_FIELDS)}, ensure_ascii=False, sort_keys=True)


def extract_json_payload(text: object) -> tuple[dict[str, Any] | None, str | None]:
    raw = "" if text is None else str(text).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"{", raw):
            try:
                parsed, _ = decoder.raw_decode(raw[match.start() :])
                break
            except json.JSONDecodeError:
                continue
        else:
            return None, "json_decode_error"
    if not isinstance(parsed, dict):
        return None, "json_root_not_object"
    return parsed, None


def target_payload_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("teacher_comment_json"):
        parsed, _ = extract_json_payload(row.get("teacher_comment_json"))
        if parsed is not None:
            return parsed
    messages = row.get("messages") or []
    if messages:
        try:
            parsed, _ = extract_json_payload(messages[-1]["content"][0]["text"])
            if parsed is not None:
                return parsed
        except (KeyError, IndexError, TypeError):
            pass
    payload = {key: row.get(key) for key in REQUIRED_JSON_FIELDS}
    return payload if any(value is not None for value in payload.values()) else None


def has_forced_praise(text: str) -> bool:
    for match in FORCED_PRAISE_RE.finditer(text):
        if NEGATED_PRAISE_CONTEXT_RE.search(match.group(0)):
            continue
        return True
    return False


def validate_teacher_json(payload: dict[str, Any] | None, selection_state: object = None) -> dict[str, Any]:
    flags: list[str] = []
    if not isinstance(payload, dict):
        return {"is_valid": False, "flags": ["json_invalid"]}
    missing = sorted(REQUIRED_JSON_FIELDS - set(payload))
    if missing:
        flags.append("missing_required_fields")
    if payload.get("comment_type") not in ALLOWED_COMMENT_TYPES:
        flags.append("invalid_comment_type")
    if payload.get("comparison_reason") is not None and not isinstance(payload.get("comparison_reason"), str):
        flags.append("invalid_comparison_reason_type")
    combined = " ".join(str(payload.get(key) or "") for key in REQUIRED_JSON_FIELDS)
    if FORBIDDEN_OUTPUT_RE.search(combined):
        flags.append("forbidden_leakage")

    comment_type = payload.get("comment_type")
    if comment_type == "selected_explanation" or str(selection_state) == "selected":
        if FAKE_SELECTED_WEAKNESS_RE.search(combined):
            flags.append("fake_weakness_in_selected")
    if comment_type == "near_miss_feedback" or str(selection_state) == "near_miss":
        if not POSITIVE_ASPECT_RE.search(combined):
            flags.append("near_miss_missing_positive_aspect")
        if not LACKING_OR_IMPROVEMENT_RE.search(combined):
            flags.append("near_miss_missing_lacking_or_improvement")
    if comment_type == "rejection_reason" or str(selection_state) == "not_selected":
        if has_forced_praise(combined):
            flags.append("forced_praise")
    return {"is_valid": not flags, "flags": flags}


def evaluate_predictions(teacher_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions_by_id = {str(row.get("id")): row for row in prediction_rows}
    examples: list[dict[str, Any]] = []
    valid_json = 0
    required_fields_present = 0
    comment_type_valid = 0
    exact_comment_type = 0
    flag_counts: dict[str, int] = {}
    seq_ratios: list[float] = []
    token_f1_scores: list[float] = []

    for teacher_row in teacher_rows:
        row_id = str(teacher_row.get("id"))
        prediction = predictions_by_id.get(row_id)
        if prediction is None:
            continue
        teacher_payload = target_payload_from_row(teacher_row)
        predicted_text = str(prediction.get("prediction") or prediction.get("text") or "")
        predicted_payload = prediction.get("prediction_json")
        if not isinstance(predicted_payload, dict):
            predicted_payload, _ = extract_json_payload(predicted_text)
        validation = validate_teacher_json(predicted_payload, teacher_row.get("selection_state"))
        valid_json += int(isinstance(predicted_payload, dict))
        required_fields_present += int(isinstance(predicted_payload, dict) and not (REQUIRED_JSON_FIELDS - set(predicted_payload)))
        comment_type_valid += int(isinstance(predicted_payload, dict) and predicted_payload.get("comment_type") in ALLOWED_COMMENT_TYPES)
        exact_comment_type += int(
            isinstance(teacher_payload, dict)
            and isinstance(predicted_payload, dict)
            and teacher_payload.get("comment_type") == predicted_payload.get("comment_type")
        )
        for flag in validation["flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        teacher_text = canonical_json_text(teacher_payload)
        predicted_canonical = canonical_json_text(predicted_payload)
        seq_ratio = difflib.SequenceMatcher(a=normalize_text(teacher_text), b=normalize_text(predicted_text)).ratio()
        seq_ratios.append(seq_ratio)
        token_f1_scores.append(token_f1(teacher_text, predicted_canonical or predicted_text))
        examples.append(
            {
                "id": row_id,
                "split": teacher_row.get("split"),
                "image_path": teacher_row.get("image_path"),
                "teacher_json": teacher_payload,
                "predicted_json": predicted_payload,
                "raw_prediction": predicted_text,
                "sequence_ratio": round(seq_ratio, 6),
                "token_f1": round(token_f1_scores[-1], 6),
                "valid_json": validation["is_valid"],
                "validation_flags": validation["flags"],
            }
        )

    matched = len(examples)
    examples.sort(key=lambda row: (row["valid_json"], row["sequence_ratio"], row["token_f1"]))
    return {
        "matched_predictions": matched,
        "valid_json_rate": round(valid_json / matched, 6) if matched else 0.0,
        "required_fields_rate": round(required_fields_present / matched, 6) if matched else 0.0,
        "comment_type_valid_rate": round(comment_type_valid / matched, 6) if matched else 0.0,
        "exact_comment_type_rate": round(exact_comment_type / matched, 6) if matched else 0.0,
        "validation_flag_counts": flag_counts,
        "forbidden_leakage_count": flag_counts.get("forbidden_leakage", 0),
        "fake_weakness_in_selected_count": flag_counts.get("fake_weakness_in_selected", 0),
        "forced_praise_count": flag_counts.get("forced_praise", 0),
        "near_miss_structure_fail_count": flag_counts.get("near_miss_missing_positive_aspect", 0)
        + flag_counts.get("near_miss_missing_lacking_or_improvement", 0),
        "mean_sequence_ratio": round(sum(seq_ratios) / matched, 6) if matched else 0.0,
        "mean_token_f1": round(sum(token_f1_scores) / matched, 6) if matched else 0.0,
        "problem_examples": examples[:5],
        "sample_outputs": examples[:3] + list(reversed(examples[-3:])),
    }


def main() -> None:
    args = build_parser().parse_args()
    teacher_manifest_path = resolve_repo_path(args.teacher_manifest)
    output_dir = resolve_repo_path(args.output_dir)
    if teacher_manifest_path is None or output_dir is None:
        raise ValueError("Could not resolve teacher_manifest or output_dir.")
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_rows = load_jsonl(teacher_manifest_path)
    teacher_valid = [validate_teacher_json(target_payload_from_row(row), row.get("selection_state")) for row in teacher_rows]
    teacher_target_format = {
        "rows": len(teacher_rows),
        "json_schema_valid_rate": round(sum(1 for item in teacher_valid if item["is_valid"]) / len(teacher_rows), 6)
        if teacher_rows
        else 0.0,
        "flag_counts": {
            flag: sum(1 for item in teacher_valid if flag in item["flags"])
            for flag in sorted({flag for item in teacher_valid for flag in item["flags"]})
        },
    }

    summary: dict[str, Any] = {
        "teacher_manifest": repo_relative_path(teacher_manifest_path),
        "teacher_target_format": teacher_target_format,
    }
    if not args.predictions_jsonl:
        summary["status"] = "blocked_missing_predictions"
        summary["blocked_reason"] = "No student predictions_jsonl was provided."
        write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    predictions_path = resolve_repo_path(args.predictions_jsonl)
    if predictions_path is None or not predictions_path.exists():
        summary["status"] = "blocked_missing_predictions"
        summary["blocked_reason"] = f"Predictions file not found: {args.predictions_jsonl}"
        write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    prediction_rows = load_jsonl(predictions_path)
    summary["status"] = "ok"
    summary["predictions_jsonl"] = repo_relative_path(predictions_path)
    summary["student_metrics"] = evaluate_predictions(teacher_rows, prediction_rows)
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
