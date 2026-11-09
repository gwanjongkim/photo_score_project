from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gemma_distill.common import repo_relative_path, resolve_repo_path, write_json, write_jsonl
from src.gemma_distill import generate_gemini_teacher as gemini_teacher


PRIMARY_FORMULA = "weighted_aesthetic_score = 0.10 * nima_unit_score + 0.40 * rgnet_score + 0.50 * alamp_score"
COMPARISON_FORMULA = "balanced_comparison_score = 0.15 * nima_unit_score + 0.45 * rgnet_score + 0.40 * alamp_score"

REQUIRED_SCORED_COLUMNS = {
    "image_path",
    "dataset_source",
    "nima_unit_score",
    "rgnet_score",
    "alamp_score",
    "weighted_aesthetic_score",
}

FORBIDDEN_OUTPUT_RE = re.compile(
    r"\b(score|scores|threshold|rank|ranked|ranking|top[- ]?k|top\s+k|model|"
    r"nima|rgnet|a[- ]?lamp|alamp|weighted(?:_|\s+)score|weighted_aesthetic_score)\b|"
    r"\bbecause the score is high\b|\bbecause it passed the threshold\b",
    flags=re.IGNORECASE,
)

NUMERIC_SCORING_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:/|out of|percent|%)\b|\b(?:score|rating)\s*(?:of|is|was)?\s*\d",
    flags=re.IGNORECASE,
)

GENERIC_RE = re.compile(
    r"\b(nice photo|good image|bad image|overall appeal|aesthetic appeal|visual strength|"
    r"clear visual strengths|usable strengths|weaker secondary cues|stronger alternatives|"
    r"selected frame|high quality|low quality|looks good|looks weak)\b",
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

STAGE_SPECS = [
    ("stage_a_300", 300),
    ("stage_b_1000", 1000),
    ("stage_c_3000", 3000),
    ("stage_d_10000", 10000),
]

HIGHSPEC_OUTPUT_DIR = "outputs/gemini_api_teacher_aesthetic3_highspec_100_batch_20260427"
HIGHSPEC_COMPARE_OUTPUT_DIR = "outputs/gemini_api_teacher_aesthetic3_highspec_batch_compare_20260427"
HIGHSPEC_CANDIDATE_JSONL = "highspec_100_candidate_manifest.jsonl"
HIGHSPEC_BATCH_REQUESTS_JSONL = "gemini_batch_requests_highspec_100_20260427.jsonl"
HIGHSPEC_GENERATION_JSONL = "gemini_teacher_generation_batch_100_20260427.jsonl"
HIGHSPEC_GENERATION_SUMMARY_JSON = "gemini_teacher_generation_batch_100_20260427_summary.json"

COMPLETED_BATCH_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "BATCH_STATE_SUCCEEDED",
    "BATCH_STATE_FAILED",
    "BATCH_STATE_CANCELLED",
    "BATCH_STATE_EXPIRED",
}

VISIBLE_ANCHOR_RE = re.compile(
    r"\b("
    r"person|people|portrait|face|eyes|hand|hands|skin|hair|child|man|woman|crowd|"
    r"dog|cat|bird|horse|animal|flower|tree|leaf|leaves|grass|forest|mountain|hill|"
    r"sky|cloud|water|ocean|sea|river|lake|wave|beach|snow|rock|sand|sun|sunset|"
    r"building|window|door|wall|street|road|sidewalk|bridge|sign|lettering|text|"
    r"car|vehicle|bike|bicycle|motorcycle|bus|train|boat|ship|airplane|wheel|"
    r"table|chair|plate|food|cup|glass|bottle|bag|clothing|shirt|hat|dress|"
    r"lamp|shadow|reflection|railing|fence|roof|floor|tile|brick|metal|wood|fabric|"
    r"foreground|background|left|right|center|corner|edge|frame|horizon"
    r")\b",
    flags=re.IGNORECASE,
)

GENERIC_PHRASE_RE = re.compile(
    r"\b("
    r"strong composition|clear subject|good lighting|nice lighting|balanced composition|"
    r"visual interest|aesthetic appeal|overall appeal|strong visual|clear visual|"
    r"high quality|low quality|nice photo|good image|bad image|looks good|looks weak|"
    r"well composed|well-composed|pleasing image|appealing image|effective framing|"
    r"interesting mood|good color|nice color|visual strength|usable strengths"
    r")\b",
    flags=re.IGNORECASE,
)


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def round_float(value: object, digits: int = 6) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return round(numeric, digits)


def stable_id(*parts: object) -> str:
    text = "__".join(str(part) for part in parts)
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:110]
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{clean}__{digest}"


def write_jsonl_local(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported file format: {path}")


def ensure_output_dir(path_like: str | Path) -> Path:
    output_dir = repo_path(path_like)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def inspect_and_repair_scored_pool(scored_path: Path, output_dir: Path) -> pd.DataFrame:
    if not scored_path.exists():
        raise FileNotFoundError(f"Scored pool not found: {scored_path}")
    frame = pd.read_csv(scored_path)
    missing = sorted(REQUIRED_SCORED_COLUMNS - set(frame.columns))
    repair_actions: list[str] = []
    if "weighted_aesthetic_score" in missing:
        score_cols = {"nima_unit_score", "rgnet_score", "alamp_score"}
        if score_cols <= set(frame.columns):
            frame["weighted_aesthetic_score"] = (
                0.10 * pd.to_numeric(frame["nima_unit_score"], errors="coerce")
                + 0.40 * pd.to_numeric(frame["rgnet_score"], errors="coerce")
                + 0.50 * pd.to_numeric(frame["alamp_score"], errors="coerce")
            ).round(6)
            missing.remove("weighted_aesthetic_score")
            repair_actions.append("computed weighted_aesthetic_score from existing three score columns")
    if "balanced_comparison_score" not in frame.columns:
        score_cols = {"nima_unit_score", "rgnet_score", "alamp_score"}
        if score_cols <= set(frame.columns):
            frame["balanced_comparison_score"] = (
                0.15 * pd.to_numeric(frame["nima_unit_score"], errors="coerce")
                + 0.45 * pd.to_numeric(frame["rgnet_score"], errors="coerce")
                + 0.40 * pd.to_numeric(frame["alamp_score"], errors="coerce")
            ).round(6)
            repair_actions.append("computed balanced_comparison_score from existing three score columns")
    if missing:
        raise ValueError(f"Scored pool is missing required columns: {missing}")

    for column in ("nima_unit_score", "rgnet_score", "alamp_score", "weighted_aesthetic_score", "balanced_comparison_score"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["image_path", "dataset_source", "nima_unit_score", "rgnet_score", "alamp_score", "weighted_aesthetic_score"]).copy()
    if "balanced_comparison_score" not in frame.columns:
        frame["balanced_comparison_score"] = np.nan
    frame["image_path"] = frame["image_path"].astype(str)
    frame["dataset_source"] = frame["dataset_source"].astype(str).str.lower()
    frame = frame.drop_duplicates("image_path", keep="first").reset_index(drop=True)

    summary = {
        "scored_pool": repo_relative_path(scored_path),
        "rows": int(len(frame)),
        "required_columns_present": sorted(REQUIRED_SCORED_COLUMNS),
        "has_balanced_comparison_score": "balanced_comparison_score" in frame.columns,
        "repair_actions": repair_actions,
        "regenerated_scoring": False,
        "primary_formula": PRIMARY_FORMULA,
        "comparison_formula": COMPARISON_FORMULA,
        "dataset_counts": {str(key): int(value) for key, value in frame["dataset_source"].value_counts().to_dict().items()},
    }
    write_json(output_dir / "scored_pool_inspection.json", summary)
    return frame


def assign_selection_bands(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.sort_values(["weighted_aesthetic_score", "image_path"], ascending=[False, True]).reset_index(drop=True).copy()
    total = len(working)
    working["primary_rank"] = np.arange(1, total + 1)
    working["primary_rank_percentile"] = (working["primary_rank"] - 1) / max(1, total)
    working["selection_band"] = working["primary_rank_percentile"].map(
        lambda pct: "selected_high" if pct < 0.30 else ("middle_near_miss" if pct < 0.70 else "rejected_low")
    )
    working["selection_state"] = working["selection_band"].map(
        {
            "selected_high": "selected",
            "middle_near_miss": "near_miss",
            "rejected_low": "not_selected",
        }
    )
    working["aesthetic_signal"] = working["selection_band"].map(
        {
            "selected_high": "strong",
            "middle_near_miss": "medium",
            "rejected_low": "weak",
        }
    )
    return working


def band_counts_for_target(target: int) -> dict[str, int]:
    counts = {
        "selected_high": int(round(target * 0.30)),
        "middle_near_miss": int(round(target * 0.40)),
    }
    counts["rejected_low"] = target - counts["selected_high"] - counts["middle_near_miss"]
    return counts


def source_balanced_sample(frame: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    if target <= 0 or frame.empty:
        return frame.head(0)
    desired_ava = int(round(target * 2 / 3))
    source_targets = {"ava": desired_ava, "aadb": target - desired_ava}
    samples: list[pd.DataFrame] = []
    used_indices: set[int] = set()
    for source, source_target in source_targets.items():
        source_frame = frame[frame["dataset_source"] == source]
        take = min(source_target, len(source_frame))
        if take > 0:
            part = source_frame.sample(n=take, random_state=seed + len(samples))
            samples.append(part)
            used_indices.update(int(index) for index in part.index)
    remaining = target - sum(len(part) for part in samples)
    if remaining > 0:
        leftover = frame[~frame.index.isin(used_indices)]
        take = min(remaining, len(leftover))
        if take > 0:
            samples.append(leftover.sample(n=take, random_state=seed + 31))
    if not samples:
        return frame.head(0)
    return pd.concat(samples).sample(frac=1.0, random_state=seed + 101)


def build_candidate_rows(frame: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    band_targets = band_counts_for_target(target)
    parts: list[pd.DataFrame] = []
    for index, (band, band_target) in enumerate(band_targets.items()):
        band_frame = frame[frame["selection_band"] == band]
        part = source_balanced_sample(band_frame, band_target, seed + index * 17)
        parts.append(part)
    sampled = pd.concat(parts) if parts else frame.head(0)
    if len(sampled) < target:
        leftover = frame[~frame.index.isin(sampled.index)]
        take = min(target - len(sampled), len(leftover))
        if take > 0:
            sampled = pd.concat([sampled, leftover.sample(n=take, random_state=seed + 999)])
    sampled = sampled.sample(frac=1.0, random_state=seed + target).reset_index(drop=True)
    rows = sampled.copy()
    rows["image_file_name"] = rows["image_path"].map(lambda value: Path(str(value)).name)
    rows["candidate_id"] = [
        stable_id("gemini", target, idx, row["dataset_source"], row["selection_band"], Path(str(row["image_path"])).name)
        for idx, row in rows.iterrows()
    ]
    keep_columns = [
        "candidate_id",
        "image_path",
        "image_file_name",
        "dataset_source",
        "selection_band",
        "selection_state",
        "aesthetic_signal",
        "nima_unit_score",
        "rgnet_score",
        "alamp_score",
        "weighted_aesthetic_score",
        "balanced_comparison_score",
        "primary_rank",
        "primary_rank_percentile",
    ]
    optional = [column for column in ("source_split", "ava_mean_score", "aadb_score") if column in rows.columns]
    return rows[keep_columns + optional]


def build_candidates(args: argparse.Namespace) -> None:
    output_dir = ensure_output_dir(args.output_dir)
    scored_path = repo_path(args.scored_pool)
    frame = inspect_and_repair_scored_pool(scored_path, output_dir)
    frame = assign_selection_bands(frame)

    summary: dict[str, Any] = {
        "output_dir": repo_relative_path(output_dir),
        "scored_pool": repo_relative_path(scored_path),
        "scored_rows_available": int(len(frame)),
        "primary_formula": PRIMARY_FORMULA,
        "comparison_formula": COMPARISON_FORMULA,
        "selection_band_rule": "weighted_aesthetic_score top 30% selected_high, middle 40% middle_near_miss, bottom 30% rejected_low",
        "test_vila_policy": "kept separate; not sampled into Gemini training candidates",
        "stages": {},
    }
    for stage_name, target in STAGE_SPECS:
        if len(frame) < target:
            summary["stages"][stage_name] = {
                "requested_rows": target,
                "created_rows": 0,
                "status": "not_created_insufficient_scored_rows",
                "required_scored_rows": target,
                "available_scored_rows": int(len(frame)),
            }
            continue
        stage = build_candidate_rows(frame, target, int(args.seed) + target)
        csv_path = output_dir / f"{stage_name}_candidate_manifest.csv"
        jsonl_path = output_dir / f"{stage_name}_candidate_manifest.jsonl"
        stage.to_csv(csv_path, index=False)
        rows = stage.replace({np.nan: None}).to_dict(orient="records")
        write_jsonl_local(jsonl_path, rows)
        summary["stages"][stage_name] = {
            "requested_rows": target,
            "created_rows": int(len(stage)),
            "csv": repo_relative_path(csv_path),
            "jsonl": repo_relative_path(jsonl_path),
            "selection_band_counts": {str(key): int(value) for key, value in stage["selection_band"].value_counts().to_dict().items()},
            "selection_state_counts": {str(key): int(value) for key, value in stage["selection_state"].value_counts().to_dict().items()},
            "dataset_counts": {str(key): int(value) for key, value in stage["dataset_source"].value_counts().to_dict().items()},
        }
    write_json(output_dir / "candidate_manifests_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_highspec_candidates(args: argparse.Namespace) -> None:
    output_dir = ensure_output_dir(args.output_dir)
    scored_path = repo_path(args.scored_pool)
    target = int(args.target)
    if target > 100:
        raise ValueError("This command is intentionally capped at 100 rows for the high-specificity batch experiment.")
    frame = inspect_and_repair_scored_pool(scored_path, output_dir)
    frame = assign_selection_bands(frame)
    frame = frame[frame["dataset_source"].astype(str).str.lower() != "test_vila"].copy()
    if len(frame) < target:
        raise ValueError(f"Need at least {target} non-test_vila scored rows; found {len(frame)}")

    manifest = build_candidate_rows(frame, target, int(args.seed) + target + 7000)
    manifest_prefix = str(getattr(args, "manifest_prefix", "") or f"highspec_{target}")
    csv_path = output_dir / f"{manifest_prefix}_candidate_manifest.csv"
    jsonl_path = output_dir / f"{manifest_prefix}_candidate_manifest.jsonl"
    manifest.to_csv(csv_path, index=False)
    rows = manifest.replace({np.nan: None}).to_dict(orient="records")
    write_jsonl_local(jsonl_path, rows)

    summary = {
        "status": "highspec_candidate_manifest_created",
        "output_dir": repo_relative_path(output_dir),
        "scored_pool": repo_relative_path(scored_path),
        "requested_rows": target,
        "created_rows": int(len(manifest)),
        "jsonl": repo_relative_path(jsonl_path),
        "csv": repo_relative_path(csv_path),
        "selection_band_counts": {str(key): int(value) for key, value in manifest["selection_band"].value_counts().to_dict().items()},
        "selection_state_counts": {str(key): int(value) for key, value in manifest["selection_state"].value_counts().to_dict().items()},
        "dataset_counts": {str(key): int(value) for key, value in manifest["dataset_source"].value_counts().to_dict().items()},
        "test_vila_policy": "excluded from the high-specificity batch candidate manifest",
    }
    write_json(output_dir / f"{manifest_prefix}_candidate_manifest_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def highspec_prompt(row: dict[str, Any]) -> str:
    state = gemini_teacher.selection_state(row)
    comment_type = gemini_teacher.COMMENT_TYPE_BY_STATE[state]
    if state == "selected":
        style_rule = (
            "Explain why the image is strong. Do not invent a weakness or improvement. "
            "Use 3-4 concise sentences in detailed_reason, all tied to concrete visible cues."
        )
    elif state == "near_miss":
        style_rule = (
            "Mention one concrete visible strength, one concrete visible limitation, and a practical improvement direction. "
            "The limitation must be visible in the image, not a generic statement."
        )
    else:
        style_rule = (
            "Explain why the image is weak using visible evidence only. Do not force praise, concessions, or compliments."
        )

    return f"""
You are creating high-specificity teacher data for a Gemma 4 E4B A-cut photo judgment student.
Judge the image as an A-cut candidate, not as a captioning task.
Use the image itself first. The internal selection state controls only the comment style.

Internal selection_state: "{state}"
Required comment_type: "{comment_type}"

High-specificity requirements:
- Every reason must be grounded in visible evidence from the image.
- Include at least two concrete visible anchors, such as a subject, object, surface, background element, color relationship, spatial placement, or visible flaw.
- Avoid generic-only wording like "strong composition", "clear subject", "good lighting", or "visual interest" unless you immediately tie it to a concrete visible element.
- Name what is visible instead of writing generic aesthetic language.
- Do not identify uncertain objects as facts; use cautious wording if the object is ambiguous.

Style rule:
{style_rule}

Return exactly one JSON object and nothing else:
{{
  "comment_type": "{comment_type}",
  "short_reason": "...",
  "detailed_reason": "...",
  "comparison_reason": null
}}

Field rules:
- short_reason: one concise sentence with at least one concrete visible anchor.
- detailed_reason: two to four concise sentences with concrete visible evidence.
- comparison_reason: null unless the image itself clearly supports a comparison.
- Use all four keys exactly as shown.
- Do not use markdown or code fences.
- Do not include extra keys.
- Do not mention scores, thresholds, ranks, top-k, model names, NIMA, RGNet, A-Lamp, weighted score, or numeric scoring logic.
- Do not mention hidden guidance, datasets, file names, prompts, JSON, or training.
""".strip()


def build_batch_request_body(
    row: dict[str, Any],
    image_path: Path,
    *,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    response_mime_type: str,
    thinking_budget: int | None = None,
    include_response_schema: bool = False,
) -> dict[str, Any]:
    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    generation_config: dict[str, Any] = {
        "temperature": float(temperature),
        "topP": float(top_p),
        "maxOutputTokens": int(max_output_tokens),
        "responseMimeType": str(response_mime_type),
    }
    if thinking_budget is not None:
        generation_config["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}
    if include_response_schema:
        generation_config["responseJsonSchema"] = gemini_teacher.teacher_schema()
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": gemini_teacher.mime_type_for(image_path),
                            "data": image_data,
                        }
                    },
                    {"text": highspec_prompt(row)},
                ],
            }
        ],
        "generationConfig": generation_config,
    }


def prepare_batch(args: argparse.Namespace) -> None:
    output_dir = ensure_output_dir(args.output_dir)
    manifest_path = repo_path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Candidate manifest not found: {manifest_path}")
    requests_path = repo_path(args.batch_requests_jsonl) if args.batch_requests_jsonl else output_dir / HIGHSPEC_BATCH_REQUESTS_JSONL
    rows = read_jsonl(manifest_path)
    if args.max_rows is not None:
        rows = rows[: int(args.max_rows)]

    prepared = 0
    skipped_missing_image = 0
    requests_path.parent.mkdir(parents=True, exist_ok=True)
    with requests_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            row_id = gemini_teacher.candidate_id(row)
            image_path = gemini_teacher.resolved_image_path(row)
            if not image_path.exists():
                skipped_missing_image += 1
                continue
            body = build_batch_request_body(
                row,
                image_path,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_output_tokens=int(args.max_output_tokens),
                response_mime_type=str(args.response_mime_type),
                thinking_budget=args.thinking_budget,
                include_response_schema=bool(args.include_response_schema),
            )
            handle.write(json.dumps({"key": row_id, "request": body}, ensure_ascii=False) + "\n")
            prepared += 1

    summary = {
        "status": "batch_requests_prepared",
        "manifest": repo_relative_path(manifest_path),
        "model": args.model,
        "batch_requests_jsonl": repo_relative_path(requests_path),
        "rows_loaded": len(rows),
        "requests_prepared": prepared,
        "skipped_missing_image": skipped_missing_image,
        "generation_config": {
            "maxOutputTokens": int(args.max_output_tokens),
            "temperature": float(args.temperature),
            "topP": float(args.top_p),
            "responseMimeType": str(args.response_mime_type),
            "thinkingBudget": None if args.thinking_budget is None else int(args.thinking_budget),
            "responseJsonSchema_included": bool(args.include_response_schema),
        },
        "submission_policy": "request file only; no API call was made by prepare-batch",
    }
    write_json(output_dir / "batch_request_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def api_key_from_env(env_name: str) -> str:
    api_key = os.environ.get(env_name)
    if not api_key:
        raise RuntimeError(f"{env_name} is not set")
    return api_key


def read_http_json(url: str, *, api_key: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_http_json(url: str, body: dict[str, Any], *, api_key: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def upload_batch_file(api_key: str, request_path: Path, display_name: str, timeout: float) -> dict[str, Any]:
    content = request_path.read_bytes()
    start_request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/upload/v1beta/files",
        data=json.dumps({"file": {"display_name": display_name}}).encode("utf-8"),
        headers={
            "x-goog-api-key": api_key,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(content)),
            "X-Goog-Upload-Header-Content-Type": "application/jsonl",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(start_request, timeout=timeout) as response:
        upload_url = response.headers.get("x-goog-upload-url")
    if not upload_url:
        raise RuntimeError("File upload start response did not include x-goog-upload-url")

    upload_request = urllib.request.Request(
        upload_url,
        data=content,
        headers={
            "Content-Length": str(len(content)),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        },
        method="POST",
    )
    with urllib.request.urlopen(upload_request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_batch_name(payload: dict[str, Any]) -> str | None:
    value = payload.get("name") or payload.get("batch_name") or payload.get("batchJobName")
    if value:
        return str(value)
    batch = payload.get("batch")
    if isinstance(batch, dict) and batch.get("name"):
        return str(batch["name"])
    return None


def extract_batch_state(payload: dict[str, Any]) -> str:
    for path in (("metadata", "state"), ("state",), ("batch", "state")):
        value: Any = payload
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if value:
            return str(value)
    if payload.get("done") is False:
        return "JOB_STATE_RUNNING"
    return "UNKNOWN"


def extract_batch_dest_file(payload: dict[str, Any]) -> str | None:
    candidates = [
        ("dest", "fileName"),
        ("dest", "file_name"),
        ("response", "responsesFile"),
        ("response", "responses_file"),
        ("metadata", "dest", "fileName"),
        ("metadata", "dest", "file_name"),
        ("metadata", "outputConfig", "fileName"),
        ("metadata", "output_config", "file_name"),
        ("metadata", "response", "responsesFile"),
    ]
    for path in candidates:
        value: Any = payload
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if value:
            return str(value)
    return None


def summarize_batch_status(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    response = payload.get("response") if isinstance(payload.get("response"), dict) else {}
    batch_stats = metadata.get("batchStats") if isinstance(metadata.get("batchStats"), dict) else {}
    return {
        "name": extract_batch_name(payload),
        "state": extract_batch_state(payload),
        "request_count": metadata.get("requestCount") or metadata.get("request_count") or metadata.get("totalRequestCount") or batch_stats.get("requestCount"),
        "success_count": metadata.get("successCount") or metadata.get("success_count") or response.get("successCount") or batch_stats.get("successfulRequestCount") or batch_stats.get("successCount"),
        "failed_count": metadata.get("failedCount") or metadata.get("failed_count") or response.get("failedCount") or batch_stats.get("failedRequestCount") or batch_stats.get("failureCount"),
        "output_file": extract_batch_dest_file(payload),
    }


def submit_batch(args: argparse.Namespace) -> None:
    output_dir = ensure_output_dir(args.output_dir)
    requests_path = repo_path(args.batch_requests_jsonl)
    if not requests_path.exists():
        raise FileNotFoundError(f"Batch request JSONL not found: {requests_path}")
    request_count = sum(1 for line in requests_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if request_count != int(args.expected_requests):
        raise ValueError(f"Refusing to submit {request_count} requests; expected exactly {args.expected_requests}")

    api_key = api_key_from_env(args.api_key_env)
    display_name = args.display_name or f"gemini-teacher-batch-{request_count}-{int(time.time())}"
    uploaded = upload_batch_file(api_key, requests_path, display_name, float(args.timeout))
    uploaded_file = uploaded.get("file") if isinstance(uploaded.get("file"), dict) else {}
    file_name = uploaded_file.get("name")
    if not file_name:
        raise RuntimeError(f"Upload response did not include file.name: {uploaded}")

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{args.model}:batchGenerateContent"
    created = post_http_json(
        endpoint,
        {
            "batch": {
                "display_name": display_name,
                "input_config": {
                    "file_name": file_name,
                },
            }
        },
        api_key=api_key,
        timeout=float(args.timeout),
    )
    metadata = {
        "status": "batch_submitted",
        "submitted_at_unix": time.time(),
        "model": args.model,
        "request_count": request_count,
        "batch_requests_jsonl": repo_relative_path(requests_path),
        "uploaded_file": uploaded_file,
        "batch_job": created,
        "batch_job_name": extract_batch_name(created),
        "batch_state": extract_batch_state(created),
    }
    metadata_path = output_dir / "batch_job.json"
    write_json(metadata_path, metadata)
    print(json.dumps({**summarize_batch_status(created), "metadata_path": repo_relative_path(metadata_path)}, indent=2, ensure_ascii=False))


def load_batch_job_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Batch job metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def poll_batch(args: argparse.Namespace) -> None:
    output_dir = ensure_output_dir(args.output_dir)
    metadata_path = repo_path(args.batch_job_json)
    metadata = load_batch_job_metadata(metadata_path)
    batch_name = args.batch_name or metadata.get("batch_job_name") or extract_batch_name(metadata.get("batch_job", {}))
    if not batch_name:
        raise RuntimeError(f"Could not determine batch job name from {metadata_path}")
    api_key = api_key_from_env(args.api_key_env)

    last_status: dict[str, Any] = {}
    while True:
        status = read_http_json(
            f"https://generativelanguage.googleapis.com/v1beta/{batch_name}",
            api_key=api_key,
            timeout=float(args.timeout),
        )
        last_status = status
        summary = summarize_batch_status(status)
        write_json(output_dir / "batch_status.json", status)
        metadata["latest_status"] = status
        metadata["batch_state"] = summary["state"]
        metadata["batch_job_name"] = batch_name
        write_json(metadata_path, metadata)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if not args.wait or summary["state"] in COMPLETED_BATCH_STATES:
            break
        time.sleep(float(args.wait_seconds))

    if args.wait and extract_batch_state(last_status) not in COMPLETED_BATCH_STATES:
        raise RuntimeError("Polling ended before the batch reached a terminal state")


def fetch_batch(args: argparse.Namespace) -> None:
    output_dir = ensure_output_dir(args.output_dir)
    metadata_path = repo_path(args.batch_job_json)
    metadata = load_batch_job_metadata(metadata_path)
    api_key = api_key_from_env(args.api_key_env)
    batch_name = args.batch_name or metadata.get("batch_job_name") or extract_batch_name(metadata.get("batch_job", {}))
    if not batch_name:
        raise RuntimeError(f"Could not determine batch job name from {metadata_path}")

    status = read_http_json(
        f"https://generativelanguage.googleapis.com/v1beta/{batch_name}",
        api_key=api_key,
        timeout=float(args.timeout),
    )
    state = extract_batch_state(status)
    write_json(output_dir / "batch_status.json", status)
    if state not in {"JOB_STATE_SUCCEEDED", "BATCH_STATE_SUCCEEDED"}:
        summary = {"status": "not_fetched", "batch_state": state, "batch_job_name": batch_name}
        write_json(output_dir / "batch_fetch_summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    raw_path = repo_path(args.raw_results_jsonl) if args.raw_results_jsonl else output_dir / "batch_raw_results.jsonl"
    output_file = extract_batch_dest_file(status)
    inlined = None
    response = status.get("response") if isinstance(status.get("response"), dict) else {}
    if isinstance(response, dict):
        inlined = response.get("inlinedResponses") or response.get("inlined_responses")
    if not output_file and inlined:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w", encoding="utf-8") as handle:
            for item in inlined:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        summary = {
            "status": "fetched_inline_results",
            "batch_state": state,
            "batch_job_name": batch_name,
            "raw_results_jsonl": repo_relative_path(raw_path),
            "rows": len(inlined),
        }
        write_json(output_dir / "batch_fetch_summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    if not output_file:
        raise RuntimeError(f"Batch succeeded but no output file was found in status: {status}")

    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/download/v1beta/{output_file}:download?alt=media",
        headers={"x-goog-api-key": api_key},
        method="GET",
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=float(args.timeout)) as response_obj:
        raw_path.write_bytes(response_obj.read())
    summary = {
        "status": "fetched_file_results",
        "batch_state": state,
        "batch_job_name": batch_name,
        "output_file": output_file,
        "raw_results_jsonl": repo_relative_path(raw_path),
        "bytes": raw_path.stat().st_size,
    }
    write_json(output_dir / "batch_fetch_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def batch_result_key(row: dict[str, Any]) -> str | None:
    for key in ("key", "request_key", "requestKey"):
        if row.get(key):
            return str(row[key])
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in ("key", "request_key", "requestKey"):
            if metadata.get(key):
                return str(metadata[key])
    return None


def batch_response_payload(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if row.get("error"):
        return None, json.dumps(row["error"], ensure_ascii=False)
    response = row.get("response") or row.get("generateContentResponse")
    if isinstance(response, dict):
        if response.get("error"):
            return None, json.dumps(response["error"], ensure_ascii=False)
        body = response.get("body") or response.get("generateContentResponse")
        if isinstance(body, dict):
            return body, None
        return response, None
    return None, "missing_response"


def response_candidates(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        return [candidate for candidate in candidates if isinstance(candidate, dict)]
    return []


def response_finish_reason(payload: dict[str, Any] | None) -> str | None:
    candidates = response_candidates(payload)
    if candidates:
        reason = candidates[0].get("finishReason") or candidates[0].get("finish_reason")
        return str(reason) if reason else None
    return None


def response_usage_metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("usageMetadata"), dict):
        return payload["usageMetadata"]
    return {}


def extract_batch_response_text(payload: dict[str, Any] | None) -> tuple[str | None, str | None]:
    candidates = response_candidates(payload)
    finish_reason = response_finish_reason(payload) or "unknown"
    if not candidates:
        return None, f"no_candidates:finishReason={finish_reason}"
    content = candidates[0].get("content")
    if not isinstance(content, dict):
        return None, f"missing_candidate_content:finishReason={finish_reason}"
    parts = content.get("parts")
    if not isinstance(parts, list):
        return None, f"no_text_parts:finishReason={finish_reason}"
    texts = [str(part.get("text")) for part in parts if isinstance(part, dict) and part.get("text") is not None]
    if not texts:
        return None, f"no_text_parts:finishReason={finish_reason}"
    return "\n".join(texts).strip(), None


def strip_markdown_json_fence(text: str) -> str:
    raw = text.strip()
    full = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    if full:
        return full.group(1).strip()
    partial = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    if partial:
        return partial.group(1).strip()
    return raw


def parse_teacher_json_text(text: str | None, *, finish_reason: str | None = None) -> tuple[dict[str, Any] | None, str | None]:
    if text is None:
        return None, f"missing_text:finishReason={finish_reason or 'unknown'}"
    raw = strip_markdown_json_fence(text)
    decoder = json.JSONDecoder()
    first_error = "no_json_object_found"
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, str):
            return parse_teacher_json_text(parsed, finish_reason=finish_reason)
        if isinstance(parsed, dict):
            return parsed, None
        return None, "json_root_not_object"
    except json.JSONDecodeError as exc:
        first_error = f"json_decode_error:{exc.msg}:line{exc.lineno}:col{exc.colno}"

    for match in re.finditer(r"\{", raw):
        start = match.start()
        try:
            parsed_obj, _ = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed_obj, str):
            parsed_obj, nested_error = parse_teacher_json_text(parsed_obj, finish_reason=finish_reason)
            if parsed_obj is not None:
                return parsed_obj, None
            return None, nested_error
        if isinstance(parsed_obj, dict):
            return parsed_obj, None

    if finish_reason == "MAX_TOKENS":
        return None, f"truncated_json:finishReason=MAX_TOKENS:text_len={len(raw)}"
    if "{" in raw and "}" not in raw:
        return None, "no_complete_json_object_found"
    return None, first_error


def parse_batch(args: argparse.Namespace) -> None:
    output_dir = ensure_output_dir(args.output_dir)

    raw_path_str = args.raw_results_jsonl
    if not raw_path_str:
        raw_path_str = str(output_dir / "batch_raw_results.jsonl")
    raw_path = repo_path(raw_path_str)
    if not raw_path.exists():
        raise FileNotFoundError(f"Batch raw results JSONL not found: {raw_path}")

    manifest_path_str = args.manifest
    if not manifest_path_str:
        summary_path = output_dir / "batch_request_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Cannot auto-detect manifest; batch_request_summary.json not in {output_dir}")
        summary = json.loads(summary_path.read_text("utf-8"))
        manifest_path_str = summary.get("manifest")
        if not manifest_path_str:
            raise ValueError(f"batch_request_summary.json in {output_dir} does not contain a 'manifest' field")
    manifest_path = repo_path(manifest_path_str)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Candidate manifest not found: {manifest_path}")

    manifest_rows = read_jsonl(manifest_path)
    by_id = {gemini_teacher.candidate_id(row): row for row in manifest_rows}
    generated_path = repo_path(args.output_jsonl) if args.output_jsonl else output_dir / "gemini_teacher_generation.jsonl"
    generated_rows: list[dict[str, Any]] = []
    parse_errors = 0
    api_errors = 0
    missing_mapping = 0
    finish_reason_counts: dict[str, int] = {}

    raw_rows = read_jsonl(raw_path)
    for index, raw_row in enumerate(raw_rows):
        row_key = batch_result_key(raw_row)
        source_row = by_id.get(row_key or "")
        if source_row is None and index < len(manifest_rows):
            source_row = manifest_rows[index]
            missing_mapping += 1
        if source_row is None:
            missing_mapping += 1
            source_row = {"candidate_id": row_key or f"unmapped_batch_row_{index}", "selection_state": "not_selected"}

        response_payload, api_error = batch_response_payload(raw_row)
        finish_reason = response_finish_reason(response_payload)
        if finish_reason:
            finish_reason_counts[finish_reason] = finish_reason_counts.get(finish_reason, 0) + 1
        usage_metadata = response_usage_metadata(response_payload)
        raw_text = None
        parsed = None
        parse_error = None
        if api_error:
            api_errors += 1
        else:
            raw_text, text_error = extract_batch_response_text(response_payload)
            if text_error:
                parse_error = text_error
            else:
                parsed, parse_error = parse_teacher_json_text(raw_text, finish_reason=finish_reason)
        if parse_error:
            parse_errors += 1
        result = gemini_teacher.build_result_row(
            row=source_row,
            model=args.model,
            raw_text=raw_text,
            parsed=parsed,
            parse_error=parse_error,
            api_error=api_error,
        )
        result["teacher_source_mode"] = "gemini_batch_api"
        result["batch_result_key"] = row_key
        result["gemini_finish_reason"] = finish_reason
        result["gemini_usage_metadata"] = usage_metadata
        generated_rows.append(result)

    write_jsonl_local(generated_path, generated_rows)
    summary = {
        "status": "batch_results_parsed",
        "raw_results_jsonl": repo_relative_path(raw_path),
        "manifest": repo_relative_path(manifest_path),
        "output_jsonl": repo_relative_path(generated_path),
        "model": args.model,
        "raw_rows": len(raw_rows),
        "generated_rows": len(generated_rows),
        "api_error_rows": api_errors,
        "parse_error_rows": parse_errors,
        "missing_or_order_fallback_mappings": missing_mapping,
        "finish_reason_counts": finish_reason_counts,
    }
    summary_path = output_dir / "parsed_batch_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def clean_field(value: object) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"^\s*(Strength|Weakness|Verdict)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text.replace("\r\n", "\n").replace("\r", "\n")).strip(" \t\n\"'")
    return text


def sentence(value: str) -> str:
    text = clean_field(value).strip(" .;,:")
    if not text:
        return ""
    return text if text.endswith((".", "!", "?")) else f"{text}."


def expected_verdict_prefix(selection_state: str) -> str:
    return "selected because" if selection_state == "selected" else "not selected because"


def to_three_line(strength: str, weakness: str, verdict: str) -> str:
    return "\n".join(
        [
            f"Strength: {sentence(strength)}",
            f"Weakness: {sentence(weakness)}",
            f"Verdict: {sentence(verdict)}",
        ]
    )


def has_forced_praise(text: str) -> bool:
    for match in FORCED_PRAISE_RE.finditer(text):
        if NEGATED_PRAISE_CONTEXT_RE.search(match.group(0)):
            continue
        return True
    return False


def specificity_metrics(short_reason: str, detailed_reason: str, comparison_reason: str | None) -> dict[str, Any]:
    combined = f"{short_reason} {detailed_reason} {comparison_reason or ''}".strip()
    anchors = {match.group(0).lower() for match in VISIBLE_ANCHOR_RE.finditer(combined)}
    generic_phrases = [match.group(0).lower() for match in GENERIC_PHRASE_RE.finditer(combined)]
    generic_only = len(anchors) < 2 or (len(generic_phrases) >= 2 and len(anchors) < 3)
    return {
        "visible_anchor_count": len(anchors),
        "generic_phrase_count": len(generic_phrases),
        "specificity_pass": not generic_only,
        "generic_only_wording": generic_only,
    }


def audit_one(row: dict[str, Any]) -> dict[str, Any]:
    state = str(row.get("selection_state") or "").strip().lower()
    if state not in {"selected", "near_miss", "not_selected"}:
        state = "selected" if str(row.get("selection_band")) == "selected_high" else "not_selected"

    comment_type = row.get("comment_type")
    short_reason = clean_field(row.get("short_reason"))
    detailed_reason = clean_field(row.get("detailed_reason"))
    comparison_reason = clean_field(row.get("comparison_reason"))

    expected_comment_type = {
        "selected": "selected_explanation",
        "near_miss": "near_miss_feedback",
        "not_selected": "rejection_reason",
    }[state]

    flags = {
        "api_error": bool(row.get("gemini_api_error")),
        "json_invalid": bool(row.get("gemini_parse_error")),
        "missing_comment_type": not bool(comment_type),
        "missing_short_reason": not bool(short_reason),
        "missing_detailed_reason": not bool(detailed_reason),
        "comment_type_mismatch": comment_type != expected_comment_type,
    }

    combined = f"{short_reason} {detailed_reason} {comparison_reason or ''}"
    if FORBIDDEN_OUTPUT_RE.search(combined):
        flags["forbidden_term"] = True
    if NUMERIC_SCORING_RE.search(combined):
        flags["numeric_scoring_leakage"] = True
    if GENERIC_RE.search(combined):
        flags["generic_phrase"] = True

    specificity = specificity_metrics(short_reason, detailed_reason, comparison_reason)
    if specificity["generic_only_wording"]:
        flags["generic_only_wording"] = True

    if comment_type == "selected_explanation":
        if FAKE_SELECTED_WEAKNESS_RE.search(combined):
            flags["fake_weakness_in_selected"] = True
    elif comment_type == "near_miss_feedback":
        if not POSITIVE_ASPECT_RE.search(combined):
            flags["near_miss_missing_positive_aspect"] = True
        if not LACKING_OR_IMPROVEMENT_RE.search(combined):
            flags["near_miss_missing_lacking_or_improvement"] = True
    elif comment_type == "rejection_reason":
        if has_forced_praise(combined):
            flags["forced_praise"] = True

    active_flags = [key for key, value in flags.items() if value]
    return {
        "candidate_id": row.get("candidate_id") or row.get("id"),
        "image_path": row.get("image_path"),
        "image_file_name": row.get("image_file_name") or Path(str(row.get("image_path", ""))).name,
        "dataset_source": row.get("dataset_source") or row.get("dataset"),
        "selection_band": row.get("selection_band") or row.get("selection_bucket"),
        "selection_state": state,
        "teacher_model_name": row.get("teacher_model_name"),
        "comment_type": comment_type,
        "short_reason": short_reason,
        "detailed_reason": detailed_reason,
        "comparison_reason": comparison_reason,
        "visible_anchor_count": specificity["visible_anchor_count"],
        "generic_phrase_count": specificity["generic_phrase_count"],
        "specificity_pass": specificity["specificity_pass"],
        "audit_flags": active_flags,
        "is_clean": not active_flags,
    }


def audit(args: argparse.Namespace) -> None:
    output_dir = ensure_output_dir(args.output_dir)
    generated_path = repo_path(args.generated_jsonl)
    rows = load_rows(generated_path) if generated_path.exists() else []
    audit_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for row in rows:
        audit_payload = audit_one(row)
        merged = {**row, **audit_payload}
        audit_rows.append(audit_payload)
        if audit_payload["is_clean"]:
            clean_rows.append(merged)
        else:
            rejected_rows.append(merged)

    audit_csv = output_dir / "teacher_audit_rows.csv"
    pd.DataFrame(audit_rows).to_csv(audit_csv, index=False)
    specificity_csv = output_dir / "teacher_specificity_rows.csv"
    specificity_columns = [
        "candidate_id",
        "image_file_name",
        "dataset_source",
        "selection_band",
        "selection_state",
        "comment_type",
        "visible_anchor_count",
        "generic_phrase_count",
        "specificity_pass",
        "audit_flags",
    ]
    pd.DataFrame(audit_rows).reindex(columns=specificity_columns).to_csv(specificity_csv, index=False)
    clean_jsonl = output_dir / "clean_teacher_dataset.jsonl"
    clean_csv = output_dir / "clean_teacher_dataset.csv"
    rejected_csv = output_dir / "rejected_teacher_rows.csv"
    write_jsonl_local(clean_jsonl, clean_rows)
    pd.DataFrame(clean_rows).to_csv(clean_csv, index=False)
    pd.DataFrame(rejected_rows).to_csv(rejected_csv, index=False)

    flag_counts: dict[str, int] = {}
    for row in audit_rows:
        for flag in row["audit_flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    specificity_pass_rows = sum(1 for row in audit_rows if row.get("specificity_pass"))
    specificity_summary = {
        "generated_jsonl": repo_relative_path(generated_path),
        "rows_audited": len(rows),
        "specificity_pass_rows": specificity_pass_rows,
        "specificity_fail_rows": len(rows) - specificity_pass_rows,
        "specificity_pass_rate": round(specificity_pass_rows / len(rows), 6) if rows else 0.0,
        "visible_anchor_count_average": round(
            sum(int(row.get("visible_anchor_count") or 0) for row in audit_rows) / len(audit_rows),
            6,
        )
        if audit_rows
        else 0.0,
        "generic_phrase_count_average": round(
            sum(int(row.get("generic_phrase_count") or 0) for row in audit_rows) / len(audit_rows),
            6,
        )
        if audit_rows
        else 0.0,
        "specificity_rows_csv": repo_relative_path(specificity_csv),
    }
    summary = {
        "generated_jsonl": repo_relative_path(generated_path),
        "rows_audited": len(rows),
        "clean_rows": len(clean_rows),
        "rejected_rows": len(rejected_rows),
        "clean_rate": round(len(clean_rows) / len(rows), 6) if rows else 0.0,
        "specificity_pass_rows": specificity_pass_rows,
        "specificity_pass_rate": specificity_summary["specificity_pass_rate"],
        "flag_counts": flag_counts,
        "clean_dataset_jsonl": repo_relative_path(clean_jsonl),
        "clean_dataset_csv": repo_relative_path(clean_csv),
        "rejected_teacher_rows_csv": repo_relative_path(rejected_csv),
        "audit_rows_csv": repo_relative_path(audit_csv),
        "specificity_rows_csv": repo_relative_path(specificity_csv),
    }
    write_json(output_dir / "teacher_audit_summary.json", summary)
    write_json(output_dir / "teacher_specificity_audit.json", specificity_summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_student_prompt(row: dict[str, Any]) -> str:
    state = str(row.get("selection_state") or "not_selected")
    signal = str(row.get("aesthetic_signal") or ("strong" if state == "selected" else "medium" if state == "near_miss" else "weak"))
    verdict_rule = "selected" if state == "selected" else "not selected"
    return "\n".join(
        [
            "Use the image first. The internal guidance controls only the verdict polarity.",
            f"Internal selection state: {state}.",
            f"Internal aesthetic signal: {signal}.",
            "Write visible evidence for why this frame made or missed the A-cut.",
            f"The Verdict line must start with \"Verdict: {verdict_rule} because\".",
            "Do not mention internal guidance, scores, thresholds, ranks, model names, numbers, bullets, or extra text.",
            "Return exactly 3 lines:",
            "Strength: ...",
            "Weakness: ...",
            "Verdict: selected because ... / Verdict: not selected because ...",
        ]
    )


def manifest_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    state = str(row.get("selection_state") or "not_selected")
    selected = state == "selected"
    
    # Prepare the target text for Gemma
    gemma_target = {
        "comment_type": row.get("comment_type"),
        "short_reason": row.get("short_reason"),
        "detailed_reason": row.get("detailed_reason"),
        "comparison_reason": row.get("comparison_reason"),
    }
    gemma_target_text = json.dumps(gemma_target, ensure_ascii=False)

    system_text = (
        "You are the Gemma 4 E4B A-cut student. Return a JSON object with the fields: "
        "comment_type, short_reason, detailed_reason, and comparison_reason. "
        "Use visible evidence only. Do not mention internal guidance, "
        "scores, numbers, thresholds, rankings, percent bands, model names, bullets, or extra text."
    )
    
    # Build the student prompt
    student_prompt = f"""
Use the image first. The internal guidance controls only the comment style.
Internal selection state: {state}.
Write visible evidence for why this frame made or missed the A-cut.
The comment_type must be appropriate for the selection_state.
Return a single JSON object.
"""

    return {
        "id": str(row.get("candidate_id") or row.get("id")),
        "image": str(row["image_path"]),
        "image_path": str(row["image_path"]),
        "split": split,
        "dataset": str(row.get("dataset_source") or row.get("dataset")),
        "selected": selected,
        "selection_band": row.get("selection_band"),
        "selection_state": state,
        "aesthetic_signal": row.get("aesthetic_signal"),
        "teacher_comment_json": gemma_target_text,
        "teacher_source_mode": "gemini_api",
        "teacher_model_name": row.get("teacher_model_name"),
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": student_prompt.strip()},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": gemma_target_text}]},
        ],
    }


def stratified_split(clean_rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in clean_rows:
        key = (
            str(row.get("selection_band")),
            str(row.get("selection_state")),
            str(row.get("dataset_source") or row.get("dataset")),
            str(row.get("comment_type")),
        )
        groups.setdefault(key, []).append(row)
    
    splits = {"train": [], "val": [], "test": []}
    for key in sorted(groups):
        group = list(groups[key])
        order = rng.permutation(len(group)).tolist() if group else []
        group = [group[index] for index in order]
        n = len(group)
        
        # Ensure at least one sample per split if group is large enough
        if n >= 3:
            val_n = 1
            test_n = 1
            train_n = n - 2
        elif n == 2:
            val_n = 1
            test_n = 0
            train_n = 1
        elif n == 1:
            val_n = 0
            test_n = 0
            train_n = 1
        else:
            train_n, val_n, test_n = 0, 0, 0

        # Proportional split for larger groups
        if n > 10:
            train_n = int(math.floor(n * 0.80))
            val_n = int(math.floor(n * 0.10))
            test_n = n - train_n - val_n

        splits["train"].extend(group[:train_n])
        splits["val"].extend(group[train_n : train_n + val_n])
        splits["test"].extend(group[train_n + val_n :])

    for split_rows in splits.values():
        split_rows.sort(key=lambda row: str(row.get("candidate_id") or row.get("id")))
    return splits


def is_test_vila_row(row: dict[str, Any]) -> bool:
    for key in ("dataset_source", "dataset", "source_split", "split"):
        if str(row.get(key) or "").strip().lower() == "test_vila":
            return True
    return False


def build_manifests(args: argparse.Namespace) -> None:
    output_dir = ensure_output_dir(args.output_dir)
    clean_path = repo_path(args.clean_jsonl)
    if getattr(args, "require_quality", False):
        audit_summary_path = repo_path(args.audit_summary_json)
        if not audit_summary_path.exists() and not Path(args.audit_summary_json).is_absolute():
            audit_summary_path = output_dir / args.audit_summary_json
        if not audit_summary_path.exists():
            raise FileNotFoundError(f"Audit summary required for quality gate: {audit_summary_path}")
        audit_summary = json.loads(audit_summary_path.read_text(encoding="utf-8"))
        clean_rows_count = int(audit_summary.get("clean_rows") or 0)
        specificity_pass_rate = float(audit_summary.get("specificity_pass_rate") or 0.0)
        min_clean_rows = int(args.min_clean_rows)
        min_specificity_pass_rate = float(args.min_specificity_pass_rate)
        if clean_rows_count < min_clean_rows or specificity_pass_rate < min_specificity_pass_rate:
            summary = {
                "status": "not_built_quality_gate_failed",
                "clean_rows": clean_rows_count,
                "min_clean_rows": min_clean_rows,
                "specificity_pass_rate": specificity_pass_rate,
                "min_specificity_pass_rate": min_specificity_pass_rate,
                "blocker": "clean rows or specificity pass rate did not meet the manifest build gate",
            }
            write_json(output_dir / "dataset_split_summary.json", summary)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return
    all_clean_rows = read_jsonl(clean_path) if clean_path.exists() else []
    clean_rows = [row for row in all_clean_rows if not is_test_vila_row(row)]
    splits = stratified_split(clean_rows, int(args.seed))
    all_manifest_rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        manifest_rows = [manifest_row(row, split_name) for row in splits[split_name]]
        split_counts[split_name] = len(manifest_rows)
        all_manifest_rows.extend(manifest_rows)
        write_jsonl_local(output_dir / f"{split_name}.jsonl", manifest_rows)
        pd.DataFrame(manifest_rows).to_csv(output_dir / f"{split_name}.csv", index=False)
    write_jsonl_local(output_dir / "dataset_all.jsonl", all_manifest_rows)
    pd.DataFrame(all_manifest_rows).to_csv(output_dir / "dataset_all.csv", index=False)

    test_vila_source = repo_path(args.test_vila_source) if args.test_vila_source else None
    test_vila_rows = 0
    test_vila_output = output_dir / "test_vila.jsonl"
    if test_vila_source and test_vila_source.exists():
        shutil.copyfile(test_vila_source, test_vila_output)
        test_vila_rows = len(read_jsonl(test_vila_output))
    else:
        write_jsonl_local(test_vila_output, [])
    summary = {
        "clean_teacher_dataset": repo_relative_path(clean_path),
        "split_rule": "train 80%, val 10%, test 10%, stratified by selection_band, selection_state, dataset_source, and comment_type",
        "train_rows": split_counts["train"],
        "val_rows": split_counts["val"],
        "test_rows": split_counts["test"],
        "dataset_all_rows": len(all_manifest_rows),
        "clean_rows_loaded": len(all_clean_rows),
        "test_vila_rows_excluded_from_train_val_test": len(all_clean_rows) - len(clean_rows),
        "test_vila_rows": test_vila_rows,
        "test_vila_policy": "external sanity/evaluation set only; not included in train/val/test/dataset_all",
        "outputs": {
            "train": repo_relative_path(output_dir / "train.jsonl"),
            "val": repo_relative_path(output_dir / "val.jsonl"),
            "test": repo_relative_path(output_dir / "test.jsonl"),
            "test_vila": repo_relative_path(test_vila_output),
            "dataset_all": repo_relative_path(output_dir / "dataset_all.jsonl"),
        },
    }
    write_json(output_dir / "dataset_split_summary.json", summary)
    write_json(output_dir / "gemma_manifest_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def image_uri(path: object) -> str:
    try:
        return Path(str(path)).resolve(strict=False).as_uri()
    except ValueError:
        return str(path)


def parse_audit_flags(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        try:
            parsed = json.loads(text)
        except Exception:
            return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item)]
    return [str(parsed)] if parsed else []


def thumbnail_name(row: dict[str, Any]) -> str:
    image_path = str(row.get("image_path") or row.get("image") or "")
    row_id = str(row.get("candidate_id") or row.get("id") or image_path)
    stem = Path(image_path).stem or "image"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")[:80] or "image"
    digest = hashlib.sha1(f"{row_id}::{image_path}".encode("utf-8")).hexdigest()[:12]
    return f"{safe_stem}__{digest}.jpg"


def create_thumbnail(row: dict[str, Any], thumbs_dir: Path, max_side: int = 320) -> tuple[str, str | None]:
    image_value = row.get("image_path") or row.get("image") or row.get("source_image_path")
    if not image_value:
        return "", "missing_image_path"
    image_path = repo_path(str(image_value))
    if not image_path.exists():
        return "", f"image_not_found:{image_path}"
    thumb_file = thumbnail_name(row)
    thumb_path = thumbs_dir / thumb_file
    try:
        from PIL import Image, ImageOps

        thumbs_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            elif image.mode == "L":
                image = image.convert("RGB")
            image.save(thumb_path, "JPEG", quality=88, optimize=True)
    except Exception as exc:
        return "", f"thumbnail_error:{exc.__class__.__name__}:{exc}"
    return f"thumbs/{thumb_file}", None


def build_review(args: argparse.Namespace) -> None:
    output_dir = ensure_output_dir(args.output_dir)
    generated_rows = read_jsonl(repo_path(args.generated_jsonl)) if args.generated_jsonl else []
    audit_rows_by_id = {
        str(row.get("candidate_id")): row
        for row in load_rows(repo_path(args.audit_rows_csv))
    } if args.audit_rows_csv and repo_path(args.audit_rows_csv).exists() else {}
    generated_ids = set()
    rows = []
    for row in generated_rows:
        row_id = str(row.get("candidate_id") or row.get("id"))
        generated_ids.add(row_id)
        audit_row = audit_rows_by_id.get(row_id, {})
        rows.append({**row, **audit_row})
    for row_id, audit_row in audit_rows_by_id.items():
        if row_id not in generated_ids:
            rows.append(audit_row)
    thumbs_dir = output_dir / "thumbs"
    css = """
body{margin:0;background:#f7f7f4;color:#1f2328;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{padding:24px 32px;background:#1f2933;color:white;border-bottom:4px solid #d08b34}
h1{margin:0;font-size:26px;letter-spacing:0}.note{max-width:1020px;line-height:1.45;color:#e6edf3}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:16px;padding:22px 28px}
.card{background:white;border:1px solid #d8dee4;border-radius:8px;overflow:hidden;box-shadow:0 6px 18px #0000000c}
.card img{width:100%;height:220px;object-fit:contain;background:#eceff3}
.body{padding:12px 14px}.file{font-size:13px;word-break:break-all;color:#57606a}
.meta{font-size:13px;color:#3b434b;margin-top:4px}
.flag{display:inline-block;margin:2px 4px 2px 0;padding:2px 7px;border-radius:999px;background:#9a3412;color:white;font-size:12px}
.clean{background:#2f6f4e}.field{margin-top:8px;line-height:1.35}.label{font-weight:700;color:#24292f}.error{color:#9a3412}
"""
    cards = []
    for row in rows:
        flags = parse_audit_flags(row.get("audit_flags"))
        thumb_src, thumb_error = create_thumbnail(row, thumbs_dir)
        if thumb_error:
            flags = flags + [thumb_error]
        flag_html = "".join(f"<span class='flag'>{html.escape(str(flag))}</span>" for flag in flags) or "<span class='flag clean'>clean</span>"
        rejected_reason = ", ".join(flags)
        api_error = str(row.get("gemini_api_error") or "")
        parse_error = str(row.get("gemini_parse_error") or "")
        if api_error:
            rejected_reason = f"{rejected_reason}, api_error={api_error}" if rejected_reason else f"api_error={api_error}"
        if parse_error:
            rejected_reason = f"{rejected_reason}, parse_error={parse_error}" if rejected_reason else f"parse_error={parse_error}"
        rejected_html = (
            f"<div class='field error'><span class='label'>Rejection Reason:</span> {html.escape(rejected_reason)}</div>"
            if rejected_reason
            else ""
        )
        img_html = (
            f"<img src='{html.escape(thumb_src)}' loading='lazy' alt='{html.escape(str(row.get('image_file_name') or Path(str(row.get('image_path', ''))).name))}'>"
            if thumb_src
            else "<div class='field error'>Thumbnail unavailable</div>"
        )
        cards.append(
            "<article class='card'>"
            f"{img_html}"
            "<div class='body'>"
            f"<div class='file'>{html.escape(str(row.get('image_file_name') or Path(str(row.get('image_path', ''))).name))}</div>"
            f"<div class='meta'>{html.escape(str(row.get('dataset_source') or row.get('dataset')))} | {html.escape(str(row.get('selection_band')))} | {html.escape(str(row.get('selection_state')))}</div>"
            f"<div class='field'><span class='label'>Comment Type:</span> {html.escape(str(row.get('comment_type') or ''))}</div>"
            f"<div class='field'><span class='label'>Short Reason:</span> {html.escape(str(row.get('short_reason') or ''))}</div>"
            f"<div class='field'><span class='label'>Detailed Reason:</span> {html.escape(str(row.get('detailed_reason') or ''))}</div>"
            f"<div class='field'><span class='label'>Comparison Reason:</span> {html.escape(str(row.get('comparison_reason') or ''))}</div>"
            f"<div class='field'><span class='label'>Visible Anchors:</span> {html.escape(str(row.get('visible_anchor_count') or 0))} | <span class='label'>Generic Phrases:</span> {html.escape(str(row.get('generic_phrase_count') or 0))}</div>"
            f"<div class='field'>{flag_html}</div>"
            f"{rejected_html}"
            "</div></article>"
        )
    review_path = output_dir / "gemini_teacher_review.html"
    body = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Gemini Teacher Review</title>",
        f"<style>{css}</style></head><body>",
        "<header><h1>Gemini API Aesthetic3 Teacher Review</h1>",
        "<p class='note'>Review Gemini teacher comments before using them for Gemma 4 E4B distillation.</p></header>",
        f"<div class='grid'>{''.join(cards) if cards else '<p>No Gemini teacher rows generated yet.</p>'}</div>",
        "</body></html>",
    ]
    review_path.write_text("\n".join(body), encoding="utf-8")
    summary = {
        "review_html": repo_relative_path(review_path),
        "rows": len(rows),
        "thumbs_dir": repo_relative_path(thumbs_dir),
        "thumbnail_count": len(list(thumbs_dir.glob("*.jpg"))) if thumbs_dir.exists() else 0,
    }
    write_json(output_dir / "review_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and audit Gemini API teacher data for Gemma 4 E4B.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    candidates = subparsers.add_parser("build_candidates")
    candidates.add_argument("--scored_pool", default="outputs/gemma4_e4b_aesthetic3_expanded_20260427/scored_pool.csv")
    candidates.add_argument("--output_dir", required=True)
    candidates.add_argument("--seed", type=int, default=42)
    candidates.set_defaults(func=build_candidates)

    highspec = subparsers.add_parser("build-highspec-100")
    highspec.add_argument("--scored_pool", default="outputs/gemma4_e4b_aesthetic3_expanded_20260427/scored_pool.csv")
    highspec.add_argument("--output_dir", default=HIGHSPEC_OUTPUT_DIR)
    highspec.add_argument("--target", type=int, default=100)
    highspec.add_argument("--manifest_prefix", default="highspec_100")
    highspec.add_argument("--seed", type=int, default=42)
    highspec.set_defaults(func=build_highspec_candidates)

    highspec_underscore = subparsers.add_parser("build_highspec_100")
    highspec_underscore.add_argument("--scored_pool", default="outputs/gemma4_e4b_aesthetic3_expanded_20260427/scored_pool.csv")
    highspec_underscore.add_argument("--output_dir", default=HIGHSPEC_OUTPUT_DIR)
    highspec_underscore.add_argument("--target", type=int, default=100)
    highspec_underscore.add_argument("--manifest_prefix", default="highspec_100")
    highspec_underscore.add_argument("--seed", type=int, default=42)
    highspec_underscore.set_defaults(func=build_highspec_candidates)

    prepare = subparsers.add_parser("prepare-batch")
    prepare.add_argument("--manifest", default=f"{HIGHSPEC_OUTPUT_DIR}/{HIGHSPEC_CANDIDATE_JSONL}")
    prepare.add_argument("--output_dir", default=HIGHSPEC_OUTPUT_DIR)
    prepare.add_argument("--batch_requests_jsonl", default=None)
    prepare.add_argument("--model", default="gemini-2.5-pro")
    prepare.add_argument("--max_rows", type=int, default=100)
    prepare.add_argument("--max_output_tokens", type=int, default=1024)
    prepare.add_argument("--temperature", type=float, default=0.1)
    prepare.add_argument("--top_p", type=float, default=0.9)
    prepare.add_argument("--response_mime_type", default="application/json")
    prepare.add_argument("--thinking_budget", type=int, default=None)
    prepare.add_argument("--include_response_schema", action="store_true")
    prepare.set_defaults(func=prepare_batch)

    submit = subparsers.add_parser("submit-batch")
    submit.add_argument("--batch_requests_jsonl", default=f"{HIGHSPEC_OUTPUT_DIR}/{HIGHSPEC_BATCH_REQUESTS_JSONL}")
    submit.add_argument("--output_dir", default=HIGHSPEC_OUTPUT_DIR)
    submit.add_argument("--model", default="gemini-2.5-pro")
    submit.add_argument("--display_name", default=None)
    submit.add_argument("--api_key_env", default="GEMINI_API_KEY")
    submit.add_argument("--timeout", type=float, default=180.0)
    submit.add_argument("--expected_requests", type=int, default=100)
    submit.set_defaults(func=submit_batch)

    poll = subparsers.add_parser("poll-batch")
    poll.add_argument("--batch_job_json", default=f"{HIGHSPEC_OUTPUT_DIR}/batch_job.json")
    poll.add_argument("--output_dir", default=HIGHSPEC_OUTPUT_DIR)
    poll.add_argument("--batch_name", default=None)
    poll.add_argument("--api_key_env", default="GEMINI_API_KEY")
    poll.add_argument("--timeout", type=float, default=120.0)
    poll.add_argument("--wait", action="store_true")
    poll.add_argument("--wait_seconds", type=float, default=30.0)
    poll.set_defaults(func=poll_batch)

    fetch = subparsers.add_parser("fetch-batch")
    fetch.add_argument("--batch_job_json", default=f"{HIGHSPEC_OUTPUT_DIR}/batch_job.json")
    fetch.add_argument("--output_dir", default=HIGHSPEC_OUTPUT_DIR)
    fetch.add_argument("--batch_name", default=None)
    fetch.add_argument("--raw_results_jsonl", default=None)
    fetch.add_argument("--api_key_env", default="GEMINI_API_KEY")
    fetch.add_argument("--timeout", type=float, default=180.0)
    fetch.set_defaults(func=fetch_batch)

    parse = subparsers.add_parser("parse-batch")
    parse.add_argument("--raw_results_jsonl", default=None)
    parse.add_argument("--manifest", default=None)
    parse.add_argument("--output_jsonl", default=None)
    parse.add_argument("--output_dir", required=True)
    parse.add_argument("--model", default="gemini-2.5-pro")
    parse.set_defaults(func=parse_batch)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--generated_jsonl", required=True)
    audit_parser.add_argument("--output_dir", required=True)
    audit_parser.set_defaults(func=audit)

    manifests = subparsers.add_parser("build_manifests")
    manifests.add_argument("--clean_jsonl", required=True)
    manifests.add_argument("--output_dir", required=True)
    manifests.add_argument("--seed", type=int, default=42)
    manifests.add_argument("--test_vila_source", default="outputs/gemma4_e4b_aesthetic3_expanded_20260427/test_vila.jsonl")
    manifests.add_argument("--require_quality", action="store_true")
    manifests.add_argument("--audit_summary_json", default="teacher_audit_summary.json")
    manifests.add_argument("--min_clean_rows", type=int, default=90)
    manifests.add_argument("--min_specificity_pass_rate", type=float, default=0.85)
    manifests.set_defaults(func=build_manifests)

    manifests_alias = subparsers.add_parser("build-manifests")
    manifests_alias.add_argument("--clean_jsonl", required=True)
    manifests_alias.add_argument("--output_dir", required=True)
    manifests_alias.add_argument("--seed", type=int, default=42)
    manifests_alias.add_argument("--test_vila_source", default="outputs/gemma4_e4b_aesthetic3_expanded_20260427/test_vila.jsonl")
    manifests_alias.add_argument("--require_quality", action="store_true")
    manifests_alias.add_argument("--audit_summary_json", default="teacher_audit_summary.json")
    manifests_alias.add_argument("--min_clean_rows", type=int, default=90)
    manifests_alias.add_argument("--min_specificity_pass_rate", type=float, default=0.85)
    manifests_alias.set_defaults(func=build_manifests)

    review = subparsers.add_parser("build_review")
    review.add_argument("--generated_jsonl", required=True)
    review.add_argument("--audit_rows_csv", required=False)
    review.add_argument("--output_dir", required=True)
    review.set_defaults(func=build_review)

    review_alias = subparsers.add_parser("build-review")
    review_alias.add_argument("--generated_jsonl", required=True)
    review_alias.add_argument("--audit_rows_csv", required=False)
    review_alias.add_argument("--output_dir", required=True)
    review_alias.set_defaults(func=build_review)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

