from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


APP_RESULTS_SCHEMA_VERSION = "acut_product_app.v1"

APP_RESULT_FIELDS = [
    "rank",
    "image_path",
    "image_file_name",
    "selected",
    "status",
    "base_score",
    "aesthetic_score",
    "aesthetic_score_contributed",
    "aesthetic_models_used",
    "aesthetic_backend",
    "final_score_after_rerank",
    "vila_score_raw",
    "vila_score_normalized_in_pool",
    "acut_short_reason",
    "acut_detailed_reason",
    "acut_comparison_reason",
]

APP_DEBUG_FIELDS = [
    "top_model_contributions",
    "explanation_structured",
    "vila_prompt_details",
]

TOP_K_SUMMARY_ITEM_FIELDS = [
    "rank",
    "image_path",
    "image_file_name",
    "final_score_after_rerank",
    "aesthetic_score",
    "aesthetic_models_used",
    "acut_short_reason",
]

REVIEW_SHEET_FIELDS = [
    "case",
    "rank",
    "image_path",
    "status",
    "base_score",
    "aesthetic_score",
    "aesthetic_score_contributed",
    "aesthetic_models_used",
    "aesthetic_backend",
    "final_score_after_rerank",
    "vila_score_raw",
    "acut_short_reason",
    "acut_detailed_reason",
    "acut_comparison_reason",
    "acut_rejection_reason",
]

SUMMARY_REQUIRED_FIELDS = [
    "schema_version",
    "generated_at",
    "ranking_stage",
    "score_semantics",
    "diversity_enabled",
    "final_ordering_uses_diversity",
    "final_score_matches_final_ranking",
    "top_k",
    "selected_count",
    "rejected_count",
    "pipeline_config",
]

AESTHETIC_SCORE_LABELS = {
    "aadb_score": "composition_aadb",
    "nima_mean_score": "nima_ava",
    "alamp_score": "alamp_aadb",
    "musiq_score": "musiq_aadb",
    "rgnet_score": "rgnet_aadb",
    "pairwise_recovered_score": "pairwise_aadb",
}


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _round_float(value: object, digits: int = 6) -> float | None:
    numeric = _to_float(value)
    if numeric is None:
        return None
    return round(numeric, digits)


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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


def _top_model_contributions(row: dict[str, object], limit: int = 3) -> list[dict[str, object]]:
    contributions = []
    per_model_contributions = row.get("per_model_contributions") or {}
    if not isinstance(per_model_contributions, dict):
        return contributions

    for component_name, component_map in per_model_contributions.items():
        if not isinstance(component_map, dict):
            continue
        for model_name, payload in component_map.items():
            if not isinstance(payload, dict):
                continue
            contributions.append(
                {
                    "model": str(model_name),
                    "component": str(component_name),
                    "weighted_contribution": _round_float(payload.get("weighted_contribution")),
                    "normalized_score": _round_float(payload.get("normalized_score")),
                    "raw_score": _round_float(payload.get("raw_score")),
                }
            )

    contributions.sort(
        key=lambda item: (
            item.get("weighted_contribution") is not None,
            float(item.get("weighted_contribution") or float("-inf")),
            item.get("model") or "",
        ),
        reverse=True,
    )
    return contributions[:limit]


def _aesthetic_models_used(row: dict[str, object]) -> list[str]:
    per_model_scores = row.get("per_model_scores") or {}
    if not isinstance(per_model_scores, dict):
        return []
    return [
        model_label
        for score_key, model_label in AESTHETIC_SCORE_LABELS.items()
        if per_model_scores.get(score_key) is not None
    ]


def _vila_prompt_details(row: dict[str, object]) -> dict[str, object] | None:
    prompt_scores = row.get("vila_prompt_scores")
    explanation_signals = row.get("vila_explanation_signals")
    vila_explanation = _clean_text(row.get("vila_explanation"))
    if prompt_scores is None and explanation_signals is None and vila_explanation is None:
        return None
    return {
        "prompt_scores": _compact_json_value(prompt_scores),
        "explanation_signals": _compact_json_value(explanation_signals),
        "vila_explanation": vila_explanation,
    }


def _selected_for_rank(rank: object, top_k: int) -> bool:
    if rank is None:
        return False
    try:
        return int(rank) <= int(top_k)
    except (TypeError, ValueError):
        return False


def _generated_at_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _file_name_from_path(value: object) -> str | None:
    cleaned = _clean_text(value)
    if cleaned is None:
        return None
    normalized = cleaned.replace("\\", "/")
    return normalized.rsplit("/", maxsplit=1)[-1]


def build_contract_metadata(
    *,
    pipeline_config: dict[str, object],
    generated_at: str | None = None,
) -> dict[str, object]:
    diversity_enabled = bool(pipeline_config.get("enable_diversity", False))
    if diversity_enabled:
        ranking_stage = "post_diversity"
        score_semantics = (
            "rank, selected, and status reflect the canonical final post-diversity order. "
            "final_score_after_rerank remains the pre-diversity score after pairwise and VILA rerank."
        )
    else:
        ranking_stage = "post_rerank"
        score_semantics = (
            "rank, selected, status, and final_score_after_rerank all reflect the same final post-rerank order "
            "because diversity reranking is disabled."
        )

    return {
        "schema_version": APP_RESULTS_SCHEMA_VERSION,
        "generated_at": generated_at or _generated_at_utc(),
        "ranking_stage": ranking_stage,
        "score_semantics": score_semantics,
        "diversity_enabled": diversity_enabled,
        "final_ordering_uses_diversity": diversity_enabled,
        "final_score_matches_final_ranking": not diversity_enabled,
    }


def build_app_result_row(
    row: dict[str, object],
    *,
    top_k: int,
    export_debug_reasoning: bool,
    aesthetic_backend: object = None,
) -> dict[str, object]:
    rank = row.get("rank")
    selected = _selected_for_rank(rank, top_k=top_k)
    aesthetic_score = _round_float(row.get("aesthetic_component"))
    aesthetic_models_used = _aesthetic_models_used(row)
    app_row = {
        "rank": None if rank is None else int(rank),
        "image_path": row.get("image_path"),
        "image_file_name": _file_name_from_path(row.get("image_path")),
        "selected": selected,
        "status": "selected" if selected else "rejected",
        "base_score": _round_float(row.get("base_score")),
        "aesthetic_score": aesthetic_score,
        "aesthetic_score_contributed": aesthetic_score is not None,
        "aesthetic_models_used": aesthetic_models_used,
        "aesthetic_backend": _clean_text(aesthetic_backend) if aesthetic_models_used else None,
        "final_score_after_rerank": _round_float(
            row.get("final_score_after_rerank")
            if row.get("final_score_after_rerank") is not None
            else row.get("final_score")
        ),
        "vila_score_raw": _round_float(row.get("vila_score_raw")),
        "vila_score_normalized_in_pool": _round_float(row.get("vila_score_normalized_in_pool")),
        "acut_short_reason": _clean_text(row.get("acut_short_reason")),
        "acut_detailed_reason": _clean_text(row.get("acut_detailed_reason")),
        "acut_comparison_reason": _clean_text(row.get("acut_comparison_reason")),
    }
    if export_debug_reasoning:
        app_row["top_model_contributions"] = _top_model_contributions(row)
        app_row["explanation_structured"] = _compact_json_value(row.get("acut_explanation_structured"))
        app_row["vila_prompt_details"] = _vila_prompt_details(row)
    return app_row


def build_app_result_rows(
    rows: list[dict[str, object]],
    *,
    top_k: int,
    export_debug_reasoning: bool,
    pipeline_config: dict[str, object],
) -> list[dict[str, object]]:
    aesthetic_backend = pipeline_config.get("aesthetic_backend")
    return [
        build_app_result_row(
            row,
            top_k=top_k,
            export_debug_reasoning=export_debug_reasoning,
            aesthetic_backend=aesthetic_backend,
        )
        for row in rows
    ]


def build_top_k_summary(
    app_rows: list[dict[str, object]],
    *,
    pipeline_config: dict[str, object],
    top_k: int,
    contract_metadata: dict[str, object],
) -> dict[str, object]:
    selected_rows = [row for row in app_rows if row.get("selected")]
    rejected_rows = [row for row in app_rows if not row.get("selected")]
    summary_rows = []
    for row in selected_rows[:top_k]:
        summary_rows.append({field: row.get(field) for field in TOP_K_SUMMARY_ITEM_FIELDS})
    return {
        **contract_metadata,
        "top_k": summary_rows,
        "selected_count": len(selected_rows),
        "rejected_count": len(rejected_rows),
        "pipeline_config": pipeline_config,
    }


def build_review_sheet_rows(
    rows: list[dict[str, object]],
    *,
    top_k: int,
    near_cut_rejected_count: int,
    pipeline_config: dict[str, object],
) -> list[dict[str, object]]:
    review_rows = []
    aesthetic_backend = _clean_text(pipeline_config.get("aesthetic_backend"))
    selected_rows = [row for row in rows if _selected_for_rank(row.get("rank"), top_k=top_k)]
    rejected_rows = [row for row in rows if not _selected_for_rank(row.get("rank"), top_k=top_k)]

    for row in selected_rows[:top_k]:
        aesthetic_score = _round_float(row.get("aesthetic_component"))
        aesthetic_models_used = _aesthetic_models_used(row)
        review_rows.append(
            {
                "case": "selected_top",
                "rank": row.get("rank"),
                "image_path": row.get("image_path"),
                "status": "selected",
                "base_score": _round_float(row.get("base_score")),
                "aesthetic_score": aesthetic_score,
                "aesthetic_score_contributed": aesthetic_score is not None,
                "aesthetic_models_used": aesthetic_models_used,
                "aesthetic_backend": aesthetic_backend if aesthetic_models_used else None,
                "final_score_after_rerank": _round_float(
                    row.get("final_score_after_rerank")
                    if row.get("final_score_after_rerank") is not None
                    else row.get("final_score")
                ),
                "vila_score_raw": _round_float(row.get("vila_score_raw")),
                "acut_short_reason": _clean_text(row.get("acut_short_reason")),
                "acut_detailed_reason": _clean_text(row.get("acut_detailed_reason")),
                "acut_comparison_reason": _clean_text(row.get("acut_comparison_reason")),
                "acut_rejection_reason": _clean_text(row.get("acut_rejection_reason")),
            }
        )

    for row in rejected_rows[: max(0, int(near_cut_rejected_count))]:
        aesthetic_score = _round_float(row.get("aesthetic_component"))
        aesthetic_models_used = _aesthetic_models_used(row)
        review_rows.append(
            {
                "case": "near_cut_rejected",
                "rank": row.get("rank"),
                "image_path": row.get("image_path"),
                "status": "rejected",
                "base_score": _round_float(row.get("base_score")),
                "aesthetic_score": aesthetic_score,
                "aesthetic_score_contributed": aesthetic_score is not None,
                "aesthetic_models_used": aesthetic_models_used,
                "aesthetic_backend": aesthetic_backend if aesthetic_models_used else None,
                "final_score_after_rerank": _round_float(
                    row.get("final_score_after_rerank")
                    if row.get("final_score_after_rerank") is not None
                    else row.get("final_score")
                ),
                "vila_score_raw": _round_float(row.get("vila_score_raw")),
                "acut_short_reason": _clean_text(row.get("acut_short_reason")),
                "acut_detailed_reason": _clean_text(row.get("acut_detailed_reason")),
                "acut_comparison_reason": _clean_text(row.get("acut_comparison_reason")),
                "acut_rejection_reason": _clean_text(row.get("acut_rejection_reason")),
            }
        )

    return review_rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = {}
            for field in fieldnames:
                value = row.get(field)
                if isinstance(value, (dict, list)):
                    serialized[field] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                else:
                    serialized[field] = value
            writer.writerow(serialized)


def write_product_exports(
    rows: list[dict[str, object]],
    *,
    output_dir: Path,
    top_k: int,
    pipeline_config: dict[str, object],
    export_debug_reasoning: bool,
    near_cut_rejected_count: int,
    json_indent: int = 2,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_metadata = build_contract_metadata(pipeline_config=pipeline_config)
    app_rows = build_app_result_rows(
        rows,
        top_k=top_k,
        export_debug_reasoning=export_debug_reasoning,
        pipeline_config=pipeline_config,
    )
    top_k_summary = build_top_k_summary(
        app_rows,
        pipeline_config=pipeline_config,
        top_k=top_k,
        contract_metadata=contract_metadata,
    )
    review_rows = build_review_sheet_rows(
        rows,
        top_k=top_k,
        near_cut_rejected_count=near_cut_rejected_count,
        pipeline_config=pipeline_config,
    )

    app_results_json = output_dir / "app_results.json"
    app_results_csv = output_dir / "app_results.csv"
    top_k_summary_json = output_dir / "top_k_summary.json"
    review_sheet_csv = output_dir / "review_sheet.csv"

    with app_results_json.open("w", encoding="utf-8") as f:
        json.dump(app_rows, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")

    _write_csv(
        app_results_csv,
        APP_RESULT_FIELDS + (APP_DEBUG_FIELDS if export_debug_reasoning else []),
        app_rows,
    )

    with top_k_summary_json.open("w", encoding="utf-8") as f:
        json.dump(top_k_summary, f, indent=json_indent, ensure_ascii=False)
        f.write("\n")

    _write_csv(review_sheet_csv, REVIEW_SHEET_FIELDS, review_rows)

    return {
        "app_results_json": str(app_results_json),
        "app_results_csv": str(app_results_csv),
        "top_k_summary_json": str(top_k_summary_json),
        "review_sheet_csv": str(review_sheet_csv),
        **contract_metadata,
        "selected_count": top_k_summary["selected_count"],
        "rejected_count": top_k_summary["rejected_count"],
    }
