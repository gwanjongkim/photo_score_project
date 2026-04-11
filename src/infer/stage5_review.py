from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCT_RANKING_FIELDS = [
    "rank",
    "base_rank",
    "image_path",
    "image_name",
    "pre_pairwise_score",
    "base_score",
    "final_score",
    "pairwise_rerank_applied",
    "pairwise_rerank_delta",
    "aesthetic_component",
    "technical_component",
    "diversity_penalty",
    "similarity_to_higher_ranked",
    "most_similar_higher_ranked_image",
    "baseline_final_score",
    "aadb_score",
    "koniq_score",
    "flive_image_score",
    "flive_patch_mean",
    "flive_patch_min",
    "nima_mean_score",
    "alamp_score",
    "musiq_score",
    "rgnet_score",
    "pairwise_recovered_score",
    "reason_selected",
    "selector_notes",
    "missing_models",
    "warnings",
]
TOP_K_REVIEW_FIELDS = [
    "rank",
    "image_name",
    "image_path",
    "final_score",
    "base_score",
    "pre_pairwise_score",
    "pairwise_rerank_applied",
    "pairwise_rerank_delta",
    "aesthetic_component",
    "technical_component",
    "diversity_penalty",
    "pairwise_recovered_score",
    "similarity_to_higher_ranked",
    "most_similar_higher_ranked_image",
    "reason_selected",
    "selector_notes",
    "warnings",
]
RUN_COMPARISON_FIELDS = [
    "run_label",
    "pipeline_name",
    "pipeline_version",
    "input_dir",
    "output_dir",
    "num_images",
    "top1_image",
    "top3_images",
    "top5_images",
    "top_k_signature",
    "num_duplicate_groups",
    "duplicate_member_count",
    "penalized_images",
    "top_k_penalized_images",
    "pairwise_adjusted_images",
    "top_k_pairwise_adjusted_images",
    "max_diversity_penalty",
]


def format_float(value: float | int | None, precision: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{precision}f}"


def safe_float(value) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_maybe_json(value):
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def join_values(values: list[object], separator: str = " | ") -> str:
    return separator.join(str(value) for value in values if value not in ("", None))


def display_path(path: str | Path | None) -> str:
    if path is None:
        return ""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    try:
        return str(candidate.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(candidate)


def canonicalize_ranked_row(row: dict[str, object]) -> dict[str, object]:
    data = dict(row)
    for key in [
        "missing_models",
        "selector_notes",
        "similarity_components_to_higher_ranked",
        "per_model_scores",
    ]:
        data[key] = parse_maybe_json(data.get(key))

    if not isinstance(data.get("missing_models"), list):
        data["missing_models"] = []
    if not isinstance(data.get("selector_notes"), list):
        data["selector_notes"] = []
    if not isinstance(data.get("per_model_scores"), dict):
        data["per_model_scores"] = {}

    data["rank"] = safe_int(data.get("rank"))
    data["base_rank"] = safe_int(data.get("base_rank"))
    data["pairwise_rerank_applied"] = bool(
        data.get("pairwise_rerank_applied")
        if not isinstance(data.get("pairwise_rerank_applied"), str)
        else data.get("pairwise_rerank_applied", "").strip().lower() == "true"
    )
    for key in [
        "pre_pairwise_score",
        "base_score",
        "final_score",
        "pairwise_rerank_delta",
        "aesthetic_component",
        "technical_component",
        "diversity_penalty",
        "similarity_to_higher_ranked",
    ]:
        data[key] = safe_float(data.get(key))
    return data


def load_ranked_rows(output_dir: str | Path) -> list[dict[str, object]]:
    output_path = Path(output_dir)
    ranked_jsonl = output_path / "ranked_results.jsonl"
    ranked_csv = output_path / "ranked_results.csv"
    rows: list[dict[str, object]] = []

    if ranked_jsonl.exists():
        with ranked_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(canonicalize_ranked_row(json.loads(line)))
    elif ranked_csv.exists():
        with ranked_csv.open("r", encoding="utf-8", newline="") as f:
            rows = [canonicalize_ranked_row(row) for row in csv.DictReader(f)]
    else:
        raise FileNotFoundError(f"Could not find ranked_results.jsonl or ranked_results.csv in {output_path}")

    return sorted(rows, key=lambda row: (row.get("rank") is None, row.get("rank") or 10**9))


def load_duplicate_report(output_dir: str | Path) -> dict[str, object] | None:
    report_path = Path(output_dir) / "duplicate_groups.json"
    if not report_path.exists():
        return None
    with report_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_run_manifest(output_dir: str | Path) -> dict[str, object] | None:
    manifest_path = Path(output_dir) / "run_manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_product_row(row: dict[str, object]) -> dict[str, object]:
    per_model_scores = dict(row.get("per_model_scores", {}) or {})
    selector_notes = list(row.get("selector_notes", []) or [])
    missing_models = list(row.get("missing_models", []) or [])
    pairwise_rerank_delta = safe_float(row.get("pairwise_rerank_delta"))
    pairwise_rerank_applied = bool(row.get("pairwise_rerank_applied"))

    if not pairwise_rerank_applied:
        pairwise_rerank_applied = any(
            "Top-pool rerank applied with pairwise recovered score." in note
            for note in selector_notes
        )

    base_score = safe_float(row.get("base_score"))
    pre_pairwise_score = safe_float(row.get("pre_pairwise_score"))
    if pre_pairwise_score is None and pairwise_rerank_delta is not None and base_score is not None:
        pre_pairwise_score = float(base_score - pairwise_rerank_delta)
    if pre_pairwise_score is None and not pairwise_rerank_applied:
        pre_pairwise_score = base_score

    warnings = []
    if missing_models:
        warnings.append("missing_models=" + ",".join(str(item) for item in missing_models))
    if row.get("final_score") is None:
        warnings.append("missing_final_score")

    return {
        "rank": row.get("rank"),
        "base_rank": row.get("base_rank"),
        "image_path": row.get("image_path"),
        "image_name": Path(str(row.get("image_path", ""))).name,
        "pre_pairwise_score": pre_pairwise_score,
        "base_score": base_score,
        "final_score": safe_float(row.get("final_score")),
        "pairwise_rerank_applied": pairwise_rerank_applied,
        "pairwise_rerank_delta": pairwise_rerank_delta,
        "aesthetic_component": safe_float(row.get("aesthetic_component")),
        "technical_component": safe_float(row.get("technical_component")),
        "diversity_penalty": safe_float(row.get("diversity_penalty")),
        "similarity_to_higher_ranked": safe_float(row.get("similarity_to_higher_ranked")),
        "most_similar_higher_ranked_image": row.get("most_similar_higher_ranked_image"),
        "baseline_final_score": safe_float(per_model_scores.get("baseline_final_score")),
        "aadb_score": safe_float(per_model_scores.get("aadb_score")),
        "koniq_score": safe_float(per_model_scores.get("koniq_score")),
        "flive_image_score": safe_float(per_model_scores.get("flive_image_score")),
        "flive_patch_mean": safe_float(per_model_scores.get("flive_patch_mean")),
        "flive_patch_min": safe_float(per_model_scores.get("flive_patch_min")),
        "nima_mean_score": safe_float(per_model_scores.get("nima_mean_score")),
        "alamp_score": safe_float(per_model_scores.get("alamp_score")),
        "musiq_score": safe_float(per_model_scores.get("musiq_score")),
        "rgnet_score": safe_float(per_model_scores.get("rgnet_score")),
        "pairwise_recovered_score": safe_float(per_model_scores.get("pairwise_recovered_score")),
        "reason_selected": row.get("reason_selected"),
        "selector_notes": join_values(selector_notes),
        "missing_models": ",".join(str(item) for item in missing_models),
        "warnings": " | ".join(warnings),
    }


def build_run_summary(
    output_dir: str | Path,
    rows: list[dict[str, object]],
    duplicate_report: dict[str, object] | None = None,
    manifest: dict[str, object] | None = None,
    top_k: int = 5,
) -> dict[str, object]:
    output_path = Path(output_dir)
    product_rows = [flatten_product_row(row) for row in rows]
    top_rows = product_rows[:top_k]
    duplicate_groups = list((duplicate_report or {}).get("duplicate_groups", []))
    penalties = [row.get("diversity_penalty") or 0.0 for row in product_rows]

    return {
        "run_label": (manifest or {}).get("run_label") or output_path.name,
        "pipeline_name": (manifest or {}).get("pipeline_name"),
        "pipeline_version": (manifest or {}).get("pipeline_version"),
        "input_dir": (manifest or {}).get("input_dir") or "",
        "output_dir": display_path(output_path),
        "num_images": len(product_rows),
        "top1_image": top_rows[0]["image_name"] if top_rows else "",
        "top3_images": join_values([row["image_name"] for row in top_rows[:3]]),
        "top5_images": join_values([row["image_name"] for row in top_rows[:5]]),
        "top_k_signature": " > ".join(row["image_name"] for row in top_rows),
        "num_duplicate_groups": len(duplicate_groups),
        "duplicate_member_count": sum(int(group.get("group_size", 0)) for group in duplicate_groups),
        "penalized_images": sum(1 for row in product_rows if (row.get("diversity_penalty") or 0.0) > 0.0),
        "top_k_penalized_images": sum(1 for row in top_rows if (row.get("diversity_penalty") or 0.0) > 0.0),
        "pairwise_adjusted_images": sum(
            1
            for row in product_rows
            if row.get("pairwise_rerank_applied") or abs(row.get("pairwise_rerank_delta") or 0.0) > 0.0
        ),
        "top_k_pairwise_adjusted_images": sum(
            1
            for row in top_rows
            if row.get("pairwise_rerank_applied") or abs(row.get("pairwise_rerank_delta") or 0.0) > 0.0
        ),
        "max_diversity_penalty": max(penalties) if penalties else 0.0,
    }


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_jsonl_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_review_markdown(
    summary: dict[str, object],
    top_rows: list[dict[str, object]],
    duplicate_report: dict[str, object] | None,
    top_k: int,
) -> str:
    lines = [
        "# Stage5 Ranking Review",
        "",
        f"- Run: `{summary.get('run_label')}`",
        f"- Input dir: `{summary.get('input_dir')}`",
        f"- Output dir: `{summary.get('output_dir')}`",
        f"- Images scored: `{summary.get('num_images')}`",
        f"- Top-{top_k} signature: `{summary.get('top_k_signature')}`",
        f"- Duplicate groups: `{summary.get('num_duplicate_groups')}`",
        "",
        f"## Top-{top_k}",
        "",
        "| rank | image | final | base | pre-pairwise | pairwise delta | diversity | pairwise score | similar-to-higher |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in top_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("rank") or ""),
                    row.get("image_name") or "",
                    format_float(row.get("final_score")),
                    format_float(row.get("base_score")),
                    format_float(row.get("pre_pairwise_score")),
                    format_float(row.get("pairwise_rerank_delta")),
                    format_float(row.get("diversity_penalty")),
                    format_float(row.get("pairwise_recovered_score")),
                    format_float(row.get("similarity_to_higher_ranked")),
                ]
            )
            + " |"
        )

    duplicate_groups = list((duplicate_report or {}).get("duplicate_groups", []))
    if duplicate_groups:
        lines.extend(
            [
                "",
                "## Duplicate Groups",
                "",
                "| group | leader | size | max pair similarity | members |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for group in duplicate_groups:
            member_names = ", ".join(Path(str(member.get("image_path", ""))).name for member in group.get("members", []))
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(group.get("group_id", "")),
                        Path(str(group.get("leader_image", ""))).name,
                        str(group.get("group_size", "")),
                        format_float(group.get("max_pair_similarity")),
                        member_names,
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Ranking Checklist",
            "",
            "- Best-shot plausibility: does rank 1 still feel like the obvious hero shot on manual review?",
            f"- Top-{top_k} conviction: do the first few swaps still feel worse than the current order?",
            "- Duplicate suppression: did any near-duplicates survive too high, or did diversity bury a clearly better frame?",
            "- Product readiness: would `product_ranking.csv` be safe to hand to downstream code as the canonical order?",
            "",
        ]
    )
    return "\n".join(lines)


def write_stage5_review_artifacts(
    output_dir: str | Path,
    top_k: int | None = None,
    json_indent: int = 2,
) -> dict[str, object]:
    output_path = Path(output_dir)
    rows = load_ranked_rows(output_path)
    manifest = load_run_manifest(output_path)
    effective_top_k = int(top_k or (manifest or {}).get("top_k") or 5)
    duplicate_report = load_duplicate_report(output_path)
    product_rows = [flatten_product_row(row) for row in rows]
    summary = build_run_summary(
        output_dir=output_path,
        rows=rows,
        duplicate_report=duplicate_report,
        manifest=manifest,
        top_k=effective_top_k,
    )

    product_csv = output_path / "product_ranking.csv"
    product_jsonl = output_path / "product_ranking.jsonl"
    review_csv = output_path / "review_topk.csv"
    review_md = output_path / "review_topk.md"
    ranking_summary_json = output_path / "ranking_summary.json"

    write_csv_rows(product_csv, PRODUCT_RANKING_FIELDS, product_rows)
    write_jsonl_rows(product_jsonl, product_rows)
    write_csv_rows(review_csv, TOP_K_REVIEW_FIELDS, product_rows[:effective_top_k])
    review_md.write_text(
        build_review_markdown(summary, product_rows[:effective_top_k], duplicate_report, top_k=effective_top_k),
        encoding="utf-8",
    )
    ranking_summary_json.write_text(
        json.dumps(summary, indent=json_indent, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "top_k": effective_top_k,
        "product_ranking_csv": str(product_csv),
        "product_ranking_jsonl": str(product_jsonl),
        "review_topk_csv": str(review_csv),
        "review_topk_md": str(review_md),
        "ranking_summary_json": str(ranking_summary_json),
    }
