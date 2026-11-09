from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from src.gemma_distill.common import repo_relative_path, resolve_repo_path, write_json


NUMERIC_RUBRIC_PATTERNS = {
    "score": re.compile(r"\bscore\b", re.IGNORECASE),
    "threshold": re.compile(r"\bthreshold\b", re.IGNORECASE),
    "weighted_aesthetic_score": re.compile(r"\bweighted_aesthetic_score\b", re.IGNORECASE),
    "rank": re.compile(r"\brank(?:ed|ing|_band)?\b|higher-ranked|lower-ranked", re.IGNORECASE),
    "top_k_style": re.compile(r"\btop[- ]?k\b|top_\d+_percent|top \d+ percent", re.IGNORECASE),
}

GENERIC_BASELINE_PATTERNS = {
    "overall_appeal": re.compile(r"\boverall appeal\b", re.IGNORECASE),
    "weaker_secondary_cues": re.compile(r"\bweaker secondary cues\b", re.IGNORECASE),
    "clear_visual_strengths": re.compile(r"\bclear visual strengths\b", re.IGNORECASE),
    "usable_strengths": re.compile(r"\busable strengths\b", re.IGNORECASE),
    "aesthetic_appeal": re.compile(r"\baesthetic appeal\b", re.IGNORECASE),
    "kept_it_ahead": re.compile(r"\bkept it ahead\b", re.IGNORECASE),
    "selected_frame": re.compile(r"\bselected frame\b", re.IGNORECASE),
    "stronger_alternatives": re.compile(r"\bstronger alternatives\b", re.IGNORECASE),
    "higher_ranked_frame": re.compile(r"\bhigher-ranked frame\b", re.IGNORECASE),
}

GROUNDING_FAIL_PATTERNS = {
    "weaker_secondary_cues": re.compile(r"\bweaker secondary cues\b", re.IGNORECASE),
    "clear_visual_strengths": re.compile(r"\bclear visual strengths\b", re.IGNORECASE),
    "usable_strengths": re.compile(r"\busable strengths\b", re.IGNORECASE),
    "higher_ranked_frame": re.compile(r"\bhigher-ranked frame\b", re.IGNORECASE),
    "selected_frame": re.compile(r"\bselected frame\b", re.IGNORECASE),
    "kept_it_ahead": re.compile(r"\bkept it ahead\b", re.IGNORECASE),
    "the_frame_fell_short": re.compile(r"\bthe frame fell short once\b", re.IGNORECASE),
    "weaker_weak": re.compile(r"\bweaker weak\b", re.IGNORECASE),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the current Gemma teacher rows for leakage and grounding issues.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported manifest format: {path}")


def truthy_flags(text: str, patterns: dict[str, re.Pattern[str]]) -> list[str]:
    return [name for name, pattern in patterns.items() if pattern.search(text)]


def build_visible_verdict(selected: bool, strength_line: str, weakness_line: str) -> str:
    strength_text = re.sub(r"^Strength:\s*", "", str(strength_line)).strip(" .")
    weakness_text = re.sub(r"^Weakness:\s*", "", str(weakness_line)).strip(" .")
    if selected:
        if weakness_text:
            return f"Verdict: selected because {strength_text} carries the frame despite {weakness_text}."
        return f"Verdict: selected because {strength_text} carries the frame."
    if weakness_text:
        return f"Verdict: not selected because {weakness_text} undercuts {strength_text}."
    return f"Verdict: not selected because {strength_text} does not carry the frame."


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = resolve_repo_path(args.manifest)
    output_dir = resolve_repo_path(args.output_dir)
    if manifest_path is None or output_dir is None:
        raise ValueError("Could not resolve manifest or output_dir.")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(manifest_path)
    row_payloads: list[dict[str, Any]] = []
    summary_counts = {
        "rows": len(rows),
        "numeric_rubric_row_count": 0,
        "generic_baseline_row_count": 0,
        "grounding_fail_row_count": 0,
        "verdict_rewritable_without_numeric_row_count": 0,
    }
    literal_counts = {key: 0 for key in NUMERIC_RUBRIC_PATTERNS}

    for row in rows:
        teacher_text = str(row.get("teacher_comment_3line") or "")
        numeric_hits = truthy_flags(teacher_text, NUMERIC_RUBRIC_PATTERNS)
        generic_hits = truthy_flags(teacher_text, GENERIC_BASELINE_PATTERNS)
        grounding_hits = truthy_flags(teacher_text, GROUNDING_FAIL_PATTERNS)
        selected = str(row.get("selected")).lower() in {"true", "1", "yes"}
        proposed_verdict = build_visible_verdict(
            selected=selected,
            strength_line=str(row.get("teacher_comment_strength_line") or ""),
            weakness_line=str(row.get("teacher_comment_weakness_line") or ""),
        )
        verdict_line = str(row.get("teacher_comment_verdict_line") or "")
        rewritable = bool(proposed_verdict and proposed_verdict != verdict_line)

        for hit in numeric_hits:
            literal_counts[hit] += 1
        summary_counts["numeric_rubric_row_count"] += int(bool(numeric_hits))
        summary_counts["generic_baseline_row_count"] += int(bool(generic_hits))
        summary_counts["grounding_fail_row_count"] += int(bool(grounding_hits))
        summary_counts["verdict_rewritable_without_numeric_row_count"] += int(rewritable)

        row_payloads.append(
            {
                "id": row.get("id"),
                "split": row.get("split"),
                "image_path": row.get("image_path"),
                "selected": selected,
                "numeric_rubric_hits": numeric_hits,
                "generic_baseline_hits": generic_hits,
                "grounding_fail_hits": grounding_hits,
                "teacher_comment_strength_line": row.get("teacher_comment_strength_line"),
                "teacher_comment_weakness_line": row.get("teacher_comment_weakness_line"),
                "teacher_comment_verdict_line": verdict_line,
                "proposed_visible_verdict_line": proposed_verdict,
                "verdict_rewritable_without_numeric": rewritable,
                "teacher_comment_3line": teacher_text,
            }
        )

    row_csv_path = output_dir / "teacher_audit_rows.csv"
    with row_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "split",
                "image_path",
                "selected",
                "numeric_rubric_hits",
                "generic_baseline_hits",
                "grounding_fail_hits",
                "teacher_comment_strength_line",
                "teacher_comment_weakness_line",
                "teacher_comment_verdict_line",
                "proposed_visible_verdict_line",
                "verdict_rewritable_without_numeric",
                "teacher_comment_3line",
            ],
        )
        writer.writeheader()
        for payload in row_payloads:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, list)
                        else value
                    )
                    for key, value in payload.items()
                }
            )

    summary = {
        "manifest": repo_relative_path(manifest_path),
        "row_csv": repo_relative_path(row_csv_path),
        "summary_counts": summary_counts,
        "literal_numeric_rubric_counts": literal_counts,
        "sample_problem_rows": [
            {
                "id": payload["id"],
                "grounding_fail_hits": payload["grounding_fail_hits"],
                "generic_baseline_hits": payload["generic_baseline_hits"],
                "teacher_comment_3line": payload["teacher_comment_3line"],
            }
            for payload in row_payloads
            if payload["grounding_fail_hits"] or payload["numeric_rubric_hits"]
        ][:10],
        "notes": [
            "generic_baseline_row_count uses broad abstract/template phrase matching such as overall appeal, weaker secondary cues, and higher-ranked frame.",
            "grounding_fail_row_count uses a stricter subset focused on visibly ungrounded placeholders or malformed verdict wording.",
            "proposed_visible_verdict_line is a deterministic A-cut rewrite tied only to the current Strength and Weakness lines.",
        ],
    }
    write_json(output_dir / "teacher_audit_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
