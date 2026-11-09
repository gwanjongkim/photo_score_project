from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.gemma_distill.common import append_jsonl, repo_relative_path, resolve_repo_path, write_json


DEFAULT_GOLD_MODEL = "gemini-2.5-pro"
DEFAULT_LARGE_MODEL = "gemini-2.5-flash"

COMMENT_TYPE_BY_STATE = {
    "selected": "selected_explanation",
    "near_miss": "near_miss_feedback",
    "not_selected": "rejection_reason",
}

FORBIDDEN_OUTPUT_RE = re.compile(
    r"\b(score|scores|threshold|rank|ranked|ranking|top[- ]?k|top\s+k|model|"
    r"nima|rgnet|a[- ]?lamp|alamp|weighted(?:_|\s+)score|weighted_aesthetic_score)\b|"
    r"\bbecause the score is high\b|\bbecause it passed the threshold\b",
    flags=re.IGNORECASE,
)

VISIBLE_EVIDENCE_TERMS = {
    "background",
    "balance",
    "balanced",
    "blur",
    "color",
    "colour",
    "composition",
    "contrast",
    "detail",
    "exposure",
    "face",
    "framing",
    "light",
    "lighting",
    "mood",
    "separation",
    "shadow",
    "sharp",
    "soft",
    "subject",
    "texture",
    "tone",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate strict 3-part Gemini API teacher comments for Gemma distillation.")
    parser.add_argument("--manifest", required=True, help="Candidate manifest JSONL/CSV.")
    parser.add_argument("--output_jsonl", required=True, help="Incremental Gemini teacher JSONL output.")
    parser.add_argument("--summary_json", default=None, help="Optional generation summary JSON path.")
    parser.add_argument("--model", default=DEFAULT_GOLD_MODEL)
    parser.add_argument("--transport", choices=["auto", "rest", "sdk"], default="auto")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--start_offset", type=int, default=0)
    parser.add_argument("--sleep_seconds", type=float, default=1.0)
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--initial_backoff", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_output_tokens", type=int, default=256)
    parser.add_argument("--api_key_env", default="GEMINI_API_KEY")
    parser.add_argument("--prepare_batch_requests", action="store_true", help="Write batch-style request JSONL instead of calling the API.")
    parser.add_argument("--batch_requests_jsonl", default=None)
    return parser


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".csv":
        import csv

        with path.open("r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported manifest format: {path}")


def read_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row_is_complete(row):
                continue
            row_id = row.get("candidate_id") or row.get("id")
            if row_id:
                seen.add(str(row_id))
    return seen


def row_is_complete(row: dict[str, Any]) -> bool:
    if row.get("gemini_api_error") or row.get("gemini_parse_error"):
        return False
    if not row.get("comment_type") or not clean_field(row.get("short_reason")) or not clean_field(row.get("detailed_reason")):
        return False
    if row.get("is_valid_pre_audit") is False:
        return False
    return True


def resolved_image_path(row: dict[str, Any]) -> Path:
    value = row.get("image_path") or row.get("image") or row.get("source_image_path")
    path = resolve_repo_path(str(value)) if value else None
    if path is None:
        raise ValueError("row is missing image_path")
    return path


def candidate_id(row: dict[str, Any]) -> str:
    value = row.get("candidate_id") or row.get("teacher_subset_id") or row.get("id")
    if value:
        return str(value)
    return f"{row.get('dataset_source', row.get('dataset', 'unknown'))}::{Path(str(row.get('image_path', 'image'))).name}"


def selection_state(row: dict[str, Any]) -> str:
    state = str(row.get("selection_state") or "").strip().lower()
    if state in {"selected", "near_miss", "not_selected"}:
        return state
    band = str(row.get("selection_band") or row.get("selection_bucket") or "").strip().lower()
    if band == "selected_high":
        return "selected"
    if band == "middle_near_miss":
        return "near_miss"
    return "not_selected"


def aesthetic_signal(row: dict[str, Any]) -> str:
    signal = str(row.get("aesthetic_signal") or "").strip().lower()
    if signal in {"strong", "medium", "weak"}:
        return signal
    state = selection_state(row)
    if state == "selected":
        return "strong"
    if state == "near_miss":
        return "medium"
    return "weak"


def expected_verdict_prefix(state: str) -> str:
    if state == "selected":
        return "selected because"
    return "not selected because"


def build_prompt(row: dict[str, Any]) -> str:
    state = selection_state(row)
    if state == "selected":
        comment_type = COMMENT_TYPE_BY_STATE[state]
        explanation_prompt = """
Explain why the image is excellent and received a high aesthetic/quality judgment.
Focus on visible strengths such as composition, subject emphasis, lighting, color, mood, clarity, framing, and visual balance.
If technical quality is also strong, mention why it contributes.
Do not invent a flaw just to fill a weakness line.
Acceptable selected_explanation style: "The image is strong because the subject is clear, the framing is balanced, and the light supports the mood."
Do not write selected_explanation style like: "It is strong, but it could be better if..."
"""
    elif state == "near_miss":
        comment_type = COMMENT_TYPE_BY_STATE[state]
        explanation_prompt = """
Mention what works well, then explain what kept it from being selected.
Include a practical improvement direction.
This should feel like: "this part is good, but this part is lacking; improving this would help."
Acceptable near_miss_feedback style: "The subject and color work well, but the frame lacks separation; simplifying the background would make it stronger."
"""
    else:
        comment_type = COMMENT_TYPE_BY_STATE[state]
        explanation_prompt = """
Explain why the image is weak or low-ranked.
Focus on visible reasons such as unclear subject, weak composition, poor separation, distracting background, blur, exposure issue, flat lighting, weak color, or lack of visual focus.
Keep the tone evaluative, not insulting. Do not invent a strength just to praise the image.
Do not start with concessions such as "while the concept is interesting", "although it has a good aspect", or "the image has strong points".
Acceptable rejection_reason style: "The subject is unclear, the background is distracting, and the flat light gives the frame little visual focus."
"""

    return f"""
You are creating teacher data for a Gemma 4 E4B A-cut photo judgment student.
Judge the image as an A-cut candidate, not as a captioning task.
Use visible evidence only.

The image's internal selection state is "{state}".
Based on this, the comment style must be "{comment_type}".

{explanation_prompt}

Return exactly one JSON object and nothing else:
{{
  "comment_type": "{comment_type}",
  "short_reason": "...",
  "detailed_reason": "...",
  "comparison_reason": null
}}

Rules:
- short_reason: one concise sentence summarizing the judgment.
- detailed_reason: 2-4 sentences following the style for "{comment_type}".
- comparison_reason: leave as null unless you have context about why this image was chosen over a near-duplicate.
- Use all four keys exactly as shown.
- Do not use markdown.
- Do not include extra keys.
- Do not wrap the JSON in a code block.
- Do not mention scores, thresholds, ranks, top-k, model names, NIMA, RGNet, A-Lamp, weighted score, or numeric scoring logic.
- Do not say "because the score is high" or "because it passed the threshold".
- Do not mention hidden guidance, datasets, file names, prompts, JSON, or training.
- Ground every field in what is visible in the image.
""".strip()


def build_repair_prompt(row: dict[str, Any]) -> str:
    state = selection_state(row)
    comment_type = COMMENT_TYPE_BY_STATE[state]
    style_rule = {
        "selected": (
            "Explain only visible strengths that make the image strong. "
            "Do not invent a flaw or improvement."
        ),
        "near_miss": (
            "Mention one visible positive aspect, one visible lacking aspect, "
            "and one improvement direction."
        ),
        "not_selected": (
            "Explain visible weaknesses only. Do not force praise or use concessions "
            "such as 'while the concept is interesting' or 'it has a good aspect'."
        ),
    }[state]
    return f"""
Return valid JSON only. No markdown, no prose before or after the object.
The image selection_state is "{state}", so comment_type must be "{comment_type}".
{style_rule}

Required exact schema:
{{
  "comment_type": "{comment_type}",
  "short_reason": "one sentence grounded in visible image evidence",
  "detailed_reason": "two or three concise sentences grounded in visible image evidence",
  "comparison_reason": null
}}

Forbidden words/topics: score, threshold, rank, top-k, NIMA, RGNet, A-Lamp, weighted score, model name, dataset, file name, prompt, training.
""".strip()


def teacher_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "comment_type": {
                "type": "string",
                "enum": ["selected_explanation", "near_miss_feedback", "rejection_reason"],
            },
            "short_reason": {"type": "string"},
            "detailed_reason": {"type": "string"},
            "comparison_reason": {"type": ["string", "null"]},
        },
        "required": ["comment_type", "short_reason", "detailed_reason", "comparison_reason"],
        "additionalProperties": False,
    }


def mime_type_for(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed and guessed.startswith("image/"):
        return guessed
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "image/jpeg"


def build_rest_body(
    row: dict[str, Any],
    image_path: Path,
    *,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    prompt: str | None = None,
) -> dict[str, Any]:
    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type_for(image_path),
                            "data": image_data,
                        }
                    },
                    {"text": prompt or build_prompt(row)},
                ],
            }
        ],
        "generationConfig": {
            "temperature": float(temperature),
            "topP": float(top_p),
            "maxOutputTokens": int(max_output_tokens),
            "responseMimeType": "application/json",
            "responseJsonSchema": teacher_schema(),
        },
    }


def extract_response_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise ValueError(f"Gemini response has no candidates: {payload}")
    parts = (((candidates[0] or {}).get("content") or {}).get("parts") or [])
    texts = [str(part.get("text")) for part in parts if part.get("text") is not None]
    if not texts:
        raise ValueError(f"Gemini response has no text parts: {payload}")
    return "\n".join(texts).strip()


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    raw = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        if "{" not in raw or "}" not in raw:
            return None, "no_json_object_found"
        return None, f"json_decode_error:{exc.msg}:line{exc.lineno}:col{exc.colno}"
    if not isinstance(parsed, dict):
        return None, "json_root_not_object"
    return parsed, None


def clean_field(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^\s*(Strength|Weakness|Verdict)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" \t\n\"'")
    return text


def sentence(value: str) -> str:
    text = clean_field(value).strip(" .;,:")
    if not text:
        return ""
    return text if text.endswith((".", "!", "?")) else f"{text}."


def three_line_comment(strength: str, weakness: str, verdict: str) -> str:
    return "\n".join(
        [
            f"Strength: {sentence(strength)}",
            f"Weakness: {sentence(weakness)}",
            f"Verdict: {sentence(verdict)}",
        ]
    )


def validate_structured_output(payload: dict[str, Any], state: str) -> dict[str, Any]:
    comment_type = payload.get("comment_type")
    short_reason = clean_field(payload.get("short_reason"))
    detailed_reason = clean_field(payload.get("detailed_reason"))
    comparison_value = payload.get("comparison_reason")
    comparison_reason = None if comparison_value is None else clean_field(comparison_value)

    expected_comment_type = COMMENT_TYPE_BY_STATE[state]
    allowed_keys = {"comment_type", "short_reason", "detailed_reason", "comparison_reason"}

    flags = {
        "extra_keys": bool(set(payload) - allowed_keys),
        "missing_comment_type": not bool(comment_type),
        "missing_short_reason": not bool(short_reason),
        "missing_detailed_reason": not bool(detailed_reason),
        "missing_comparison_reason": "comparison_reason" not in payload,
        "comment_type_mismatch": comment_type != expected_comment_type,
        "invalid_comparison_reason_type": comparison_value is not None and not isinstance(comparison_value, str),
    }

    combined = f"{short_reason} {detailed_reason} {comparison_reason or ''}"
    if FORBIDDEN_OUTPUT_RE.search(combined):
        flags["forbidden_term"] = True

    return {
        "comment_type": comment_type,
        "short_reason": short_reason,
        "detailed_reason": detailed_reason,
        "comparison_reason": comparison_reason,
        "validation_flags": {key: value for key, value in flags.items() if value},
        "is_valid_pre_audit": not any(flags.values()),
    }



def call_rest(
    *,
    api_key: str,
    model: str,
    row: dict[str, Any],
    image_path: Path,
    timeout: float,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    prompt: str | None = None,
) -> dict[str, Any]:
    body = build_rest_body(
        row,
        image_path,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        prompt=prompt,
    )
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_sdk(
    *,
    api_key: str,
    model: str,
    row: dict[str, Any],
    image_path: Path,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    prompt: str | None = None,
) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    image_part = types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime_type_for(image_path))
    config_kwargs = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_output_tokens": int(max_output_tokens),
        "response_mime_type": "application/json",
    }
    try:
        config = types.GenerateContentConfig(**config_kwargs, response_json_schema=teacher_schema())
    except TypeError:
        config = types.GenerateContentConfig(**config_kwargs, response_schema=teacher_schema())
    response = client.models.generate_content(
        model=model,
        contents=[image_part, prompt or build_prompt(row)],
        config=config,
    )
    return {"candidates": [{"content": {"parts": [{"text": response.text}]}}]}


class GeminiRequestError(RuntimeError):
    def __init__(self, message: str, api_attempts: int) -> None:
        super().__init__(message)
        self.api_attempts = api_attempts


def call_with_retry(
    args: argparse.Namespace,
    api_key: str,
    row: dict[str, Any],
    image_path: Path,
    *,
    prompt: str | None = None,
) -> tuple[dict[str, Any], int]:
    transport = args.transport
    last_error: str | None = None
    api_attempts = 0
    for attempt in range(int(args.retries) + 1):
        try:
            if transport == "sdk":
                api_attempts += 1
                return call_sdk(
                    api_key=api_key,
                    model=args.model,
                    row=row,
                    image_path=image_path,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_output_tokens=args.max_output_tokens,
                    prompt=prompt,
                ), api_attempts
            if transport == "auto":
                try:
                    api_attempts += 1
                    return call_sdk(
                        api_key=api_key,
                        model=args.model,
                        row=row,
                        image_path=image_path,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_output_tokens=args.max_output_tokens,
                        prompt=prompt,
                    ), api_attempts
                except ModuleNotFoundError:
                    api_attempts -= 1
                    transport = "rest"
            api_attempts += 1
            return call_rest(
                api_key=api_key,
                model=args.model,
                row=row,
                image_path=image_path,
                timeout=args.request_timeout,
                temperature=args.temperature,
                top_p=args.top_p,
                max_output_tokens=args.max_output_tokens,
                prompt=prompt,
            ), api_attempts
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTPError:{exc.code}:{body[:500]}"
            retryable = exc.code in {429, 500, 502, 503, 504}
        except urllib.error.URLError as exc:
            last_error = f"URLError:{exc.reason}"
            retryable = True
        except Exception as exc:
            last_error = f"{exc.__class__.__name__}:{exc}"
            retryable = attempt == 0 and args.transport == "auto"
            if args.transport == "auto":
                transport = "rest"
        if not retryable or attempt >= int(args.retries):
            break
        sleep_for = float(args.initial_backoff) * (2 ** attempt) + random.uniform(0.0, 0.5)
        time.sleep(sleep_for)
    raise GeminiRequestError(last_error or "unknown Gemini API error", api_attempts)


def build_result_row(
    *,
    row: dict[str, Any],
    model: str,
    raw_text: str | None,
    parsed: dict[str, Any] | None,
    parse_error: str | None,
    api_error: str | None,
) -> dict[str, Any]:
    state = selection_state(row)
    payload: dict[str, Any] = {
        **row,
        "candidate_id": candidate_id(row),
        "selection_state": state,
        "aesthetic_signal": aesthetic_signal(row),
        "teacher_source_mode": "gemini_api",
        "teacher_model_name": model,
        "gemini_raw_text": raw_text,
        "gemini_parse_error": parse_error,
        "gemini_api_error": api_error,
    }
    if parsed is not None:
        validation = validate_structured_output(parsed, state)
        payload.update(
            {
                "comment_type": validation["comment_type"],
                "short_reason": validation["short_reason"],
                "detailed_reason": validation["detailed_reason"],
                "comparison_reason": validation["comparison_reason"],
                "generation_validation_flags": validation["validation_flags"],
                "is_valid_pre_audit": validation["is_valid_pre_audit"],
            }
        )
    else:
        payload.update(
            {
                "comment_type": None,
                "short_reason": None,
                "detailed_reason": None,
                "comparison_reason": None,
                "generation_validation_flags": {"parse_error": parse_error or api_error or "unknown_error"},
                "is_valid_pre_audit": False,
            }
        )
    return payload


def prepare_batch_requests(args: argparse.Namespace, rows: list[dict[str, Any]], existing_ids: set[str]) -> dict[str, Any]:
    output_path = resolve_repo_path(args.batch_requests_jsonl or Path(args.output_jsonl).with_suffix(".batch_requests.jsonl"))
    if output_path is None:
        raise ValueError("Could not resolve batch request output path.")
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            row_id = candidate_id(row)
            if row_id in existing_ids:
                continue
            image_path = resolved_image_path(row)
            if not image_path.exists():
                continue
            body = build_rest_body(
                row,
                image_path,
                temperature=args.temperature,
                top_p=args.top_p,
                max_output_tokens=args.max_output_tokens,
            )
            handle.write(json.dumps({"key": row_id, "request": body}, ensure_ascii=False) + "\n")
            count += 1
    return {
        "status": "batch_requests_prepared",
        "batch_requests_jsonl": repo_relative_path(output_path),
        "rows_prepared": count,
        "note": "This prepares request payloads only; it does not submit a Gemini Batch API job.",
    }


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = resolve_repo_path(args.manifest)
    output_path = resolve_repo_path(args.output_jsonl)
    summary_path = resolve_repo_path(args.summary_json) if args.summary_json else None
    if manifest_path is None or output_path is None:
        raise ValueError("Could not resolve manifest or output path.")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = read_manifest(manifest_path)
    if int(args.start_offset) > 0:
        rows = rows[int(args.start_offset) :]
    if args.max_rows is not None:
        rows = rows[: int(args.max_rows)]
    existing_ids = read_existing_ids(output_path)
    pending = [row for row in rows if candidate_id(row) not in existing_ids]

    if args.prepare_batch_requests:
        summary = prepare_batch_requests(args, pending, existing_ids)
        if summary_path:
            write_json(summary_path, summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    api_key = os.environ.get(str(args.api_key_env))
    if not api_key:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch(exist_ok=True)
        summary = {
            "status": "api_key_missing",
            "api_key_env": str(args.api_key_env),
            "manifest": repo_relative_path(manifest_path),
            "output_jsonl": repo_relative_path(output_path),
            "model": args.model,
            "rows_requested_this_run": len(rows),
            "rows_pending_this_run": len(pending),
            "rows_generated_this_run": 0,
        }
        if summary_path:
            write_json(summary_path, summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated = 0
    errors = 0
    invalid_json_rows = 0
    api_requests_this_run = 0
    for index, row in enumerate(pending, start=1):
        raw_text = None
        parsed = None
        parse_error = None
        api_error = None
        request_attempts = 0
        json_repair_attempted = False
        initial_parse_error = None
        try:
            image_path = resolved_image_path(row)
            if not image_path.exists():
                raise FileNotFoundError(f"image not found: {image_path}")
            prompts = [build_prompt(row), build_repair_prompt(row)]
            for prompt_index, prompt in enumerate(prompts):
                if prompt_index == 1:
                    json_repair_attempted = True
                try:
                    response_payload, attempts = call_with_retry(args, api_key, row, image_path, prompt=prompt)
                    request_attempts += attempts
                    api_requests_this_run += attempts
                except GeminiRequestError as exc:
                    request_attempts += exc.api_attempts
                    api_requests_this_run += exc.api_attempts
                    raise
                raw_text = extract_response_text(response_payload)
                parsed, parse_error = parse_json_object(raw_text)
                if parse_error and prompt_index == 0:
                    initial_parse_error = parse_error
                    continue
                break
        except Exception as exc:
            api_error = f"{exc.__class__.__name__}: {exc}"
            errors += 1

        result = build_result_row(
            row=row,
            model=args.model,
            raw_text=raw_text,
            parsed=parsed,
            parse_error=parse_error,
            api_error=api_error,
        )
        if result.get("gemini_parse_error"):
            invalid_json_rows += 1
        result["gemini_request_attempts"] = request_attempts
        result["gemini_json_repair_attempted"] = json_repair_attempted
        result["gemini_initial_parse_error"] = initial_parse_error
        append_jsonl(output_path, result)
        generated += 1
        print(
            json.dumps(
                {
                    "generated": generated,
                    "pending": len(pending),
                    "candidate_id": result["candidate_id"],
                    "ok": not bool(api_error or parse_error),
                    "api_error": api_error,
                    "parse_error": parse_error,
                    "request_attempts": request_attempts,
                    "json_repair_attempted": json_repair_attempted,
                },
                ensure_ascii=False,
            )
        )
        if index < len(pending) and args.sleep_seconds > 0:
            time.sleep(float(args.sleep_seconds))

    summary = {
        "status": "completed",
        "manifest": repo_relative_path(manifest_path),
        "output_jsonl": repo_relative_path(output_path),
        "model": args.model,
        "transport": args.transport,
        "rows_requested_this_run": len(rows),
        "rows_previously_existing": len(existing_ids),
        "rows_pending_this_run": len(pending),
        "rows_generated_this_run": generated,
        "error_rows_this_run": errors,
        "invalid_json_rows_this_run": invalid_json_rows,
        "api_requests_this_run": api_requests_this_run,
    }
    if summary_path:
        write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
