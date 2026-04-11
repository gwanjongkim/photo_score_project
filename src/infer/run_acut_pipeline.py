from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys

from src.infer.acut_product_export import write_product_exports
from src.infer.select_best_shots import (
    build_diversity_config,
    build_parser as build_selector_parser,
    load_or_compute_scores,
    load_weights,
    materialize_top_k,
    merge_vila_scores,
    populate_vila_explanations,
    rank_records,
    write_duplicate_groups_report,
    write_ranked_outputs,
)
from src.infer.stage5_reference import (
    DEFAULT_STAGE5_CONFIG_PATH,
    load_stage5_reference_config,
    repo_relative_path,
)
from src.infer.generate_multimodal_explanations import generate_multimodal_explanations
from src.vila.explain_acut_selection import (
    DEFAULT_REASON_REFERENCE_MODE,
    synthesize_acut_selection_explanations,
)


DEFAULT_VILA_RERANK_WEIGHT = 0.10
DEFAULT_REASON_DETAIL_LEVEL = "full"
DEFAULT_REVIEW_NEAR_CUT_REJECTED_COUNT = 3
DEFAULT_AESTHETIC_BACKEND = "stage5_reference_tensorflow_bundle"
DEFAULT_AESTHETIC_MODEL_FIELDS = (
    "aadb_model",
    "koniq_model",
    "flive_image_model",
    "flive_patch_model",
    "nima_model",
    "alamp_model",
    "musiq_model",
    "rgnet_model",
)
AESTHETIC_FAMILY_MODEL_FIELDS = (
    "aadb_model",
    "nima_model",
    "alamp_model",
    "musiq_model",
    "rgnet_model",
)
PAIRWISE_MODEL_FIELDS = (
    "pairwise_model",
    "pairwise_reference_csv",
)
AESTHETIC_SCORE_FIELDS = (
    "aadb_score",
    "nima_mean_score",
    "alamp_score",
    "musiq_score",
    "rgnet_score",
    "pairwise_recovered_score",
)
SCORE_FAILURE_REPORT_FILENAME = "score_failures.json"


def configure_runtime_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        stream=sys.stderr,
        force=True,
    )


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def build_parser() -> argparse.ArgumentParser:
    parser = build_selector_parser()
    parser.description = "Run the productized A-cut pipeline and emit stable app-facing exports."
    parser.set_defaults(
        enable_vila_rerank=None,
        enable_vila_explanations=None,
        enable_acut_reasoning=None,
        enable_multimodal_explanations=None,
        reason_reference_mode=DEFAULT_REASON_REFERENCE_MODE,
        reason_detail_level=DEFAULT_REASON_DETAIL_LEVEL,
    )
    parser.add_argument("--disable_vila_rerank", action="store_true")
    parser.add_argument("--disable_vila_explanations", action="store_true")
    parser.add_argument("--disable_acut_reasoning", action="store_true")
    parser.add_argument("--disable_multimodal_explanations", action="store_true")
    parser.add_argument(
        "--disable_aesthetic_scoring",
        action="store_true",
        default=_env_flag("ACUT_DISABLE_AESTHETIC_SCORING", default=False),
        help="Do not auto-resolve the repo's server-side aesthetic/quality checkpoint bundle.",
    )
    parser.add_argument(
        "--aesthetic_config",
        default=os.environ.get("ACUT_AESTHETIC_CONFIG", str(DEFAULT_STAGE5_CONFIG_PATH)),
        help="Stage5-style config used to auto-resolve server-side aesthetic/quality model paths.",
    )
    parser.add_argument(
        "--gemini_model_name",
        default=os.environ.get("GEMINI_MODEL_NAME"),
        help="Optional Gemini model override used only when multimodal explanations are enabled.",
    )
    parser.add_argument(
        "--enable_gemini",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Explicitly enable or disable Gemini multimodal calls. "
            "If omitted, ENABLE_GEMINI env is used when present."
        ),
    )
    parser.add_argument("--export_debug_reasoning", action="store_true")
    parser.add_argument(
        "--review_near_cut_rejected_count",
        type=int,
        default=DEFAULT_REVIEW_NEAR_CUT_REJECTED_COUNT,
        help="How many near-cut rejected rows to include in review_sheet.csv.",
    )
    return parser


def _resolve_vila_default(
    explicit_enabled: bool | None,
    *,
    disabled: bool,
    vila_scores_csv: str | None,
) -> bool:
    if disabled:
        return False
    if explicit_enabled is not None:
        return bool(explicit_enabled)
    return bool(vila_scores_csv)


def _resolve_reasoning_default(
    explicit_enabled: bool | None,
    *,
    disabled: bool,
) -> bool:
    if disabled:
        return False
    if explicit_enabled is not None:
        return bool(explicit_enabled)
    return True


def _relative_model_paths(model_paths: dict[str, str]) -> dict[str, str]:
    return {
        field_name: repo_relative_path(model_path) or model_path
        for field_name, model_path in sorted(model_paths.items())
    }


def _build_disabled_aesthetic_summary(reason: str) -> dict[str, object]:
    return {
        "enabled": False,
        "backend": None,
        "config_path": None,
        "auto_resolved": False,
        "model_paths": {},
        "aesthetic_model_paths": {},
        "pairwise_model_paths": {},
        "missing_model_paths": {},
        "warnings": [],
        "disabled_reason": reason,
    }


def _has_score_value(value: object) -> bool:
    if value is None:
        return False
    try:
        return bool(value == value)
    except Exception:
        return True


def configure_aesthetic_scoring(args: argparse.Namespace) -> dict[str, object]:
    if args.disable_aesthetic_scoring:
        return _build_disabled_aesthetic_summary("disabled_by_cli")
    if args.scores_jsonl or args.scores_csv:
        return _build_disabled_aesthetic_summary("precomputed_scores_supplied")
    if not args.input_dir:
        return _build_disabled_aesthetic_summary("no_input_dir")

    summary: dict[str, object] = {
        "enabled": False,
        "backend": DEFAULT_AESTHETIC_BACKEND,
        "config_path": None,
        "auto_resolved": False,
        "model_paths": {},
        "aesthetic_model_paths": {},
        "pairwise_model_paths": {},
        "missing_model_paths": {},
        "warnings": [],
        "disabled_reason": None,
    }

    try:
        config, config_path = load_stage5_reference_config(args.aesthetic_config)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        summary["warnings"] = [f"Could not load aesthetic config: {exc.__class__.__name__}: {exc}"]
        summary["disabled_reason"] = "config_unavailable"
        return summary

    summary["config_path"] = repo_relative_path(config_path)
    bundle_args = dict(config.get("bundle_args", {}))
    auto_resolved_fields: list[str] = []
    resolved_paths: dict[str, str] = {}
    missing_paths: dict[str, str] = {}

    fields_to_resolve = list(DEFAULT_AESTHETIC_MODEL_FIELDS)
    if args.enable_pairwise_rerank or args.pairwise_model or args.pairwise_reference_csv:
        fields_to_resolve.extend(PAIRWISE_MODEL_FIELDS)

    for field_name in fields_to_resolve:
        current_value = getattr(args, field_name, None)
        if current_value:
            resolved_paths[field_name] = str(current_value)
            continue

        candidate_value = bundle_args.get(field_name)
        if not isinstance(candidate_value, str) or not candidate_value.strip():
            continue

        candidate_path = Path(candidate_value)
        if candidate_path.exists():
            setattr(args, field_name, str(candidate_path))
            resolved_paths[field_name] = str(candidate_path)
            auto_resolved_fields.append(field_name)
        else:
            missing_paths[field_name] = str(candidate_path)

    aesthetic_paths = {
        field_name: model_path
        for field_name, model_path in resolved_paths.items()
        if field_name in AESTHETIC_FAMILY_MODEL_FIELDS
    }
    pairwise_paths = {
        field_name: model_path
        for field_name, model_path in resolved_paths.items()
        if field_name in PAIRWISE_MODEL_FIELDS
    }

    summary["enabled"] = bool(aesthetic_paths)
    summary["auto_resolved"] = bool(auto_resolved_fields)
    summary["model_paths"] = _relative_model_paths(resolved_paths)
    summary["aesthetic_model_paths"] = _relative_model_paths(aesthetic_paths)
    summary["pairwise_model_paths"] = _relative_model_paths(pairwise_paths)
    summary["missing_model_paths"] = _relative_model_paths(missing_paths)
    if not aesthetic_paths:
        summary["disabled_reason"] = "no_aesthetic_models_available"
    if missing_paths:
        summary["warnings"] = [
            f"Configured model path is missing for {field_name}: {repo_relative_path(model_path) or model_path}"
            for field_name, model_path in sorted(missing_paths.items())
        ]
    args._auto_resolved_model_fields = auto_resolved_fields
    return summary


def build_pipeline_config(
    args: argparse.Namespace,
    *,
    source_jsonl: Path | None,
    source_csv: Path | None,
    score_failures_json: Path | None,
    num_input_images: int,
    num_scored_images: int,
    num_skipped_images: int,
    num_recoverable_stage_failures: int,
    enable_vila_rerank: bool,
    enable_vila_explanations: bool,
    enable_acut_reasoning: bool,
    effective_vila_rerank_weight: float,
    weights: dict[str, object],
    aesthetic_scoring: dict[str, object],
) -> dict[str, object]:
    component_mix = weights.get("component_mix", {})
    aesthetic_weight = None
    if isinstance(component_mix, dict) and component_mix.get("aesthetic") is not None:
        aesthetic_weight = round(float(component_mix["aesthetic"]), 6)
    return {
        "input_dir": args.input_dir,
        "scores_jsonl": str(source_jsonl) if source_jsonl else None,
        "scores_csv": str(source_csv) if source_csv else None,
        "score_failures_json": str(score_failures_json) if score_failures_json else None,
        "num_input_images": int(num_input_images),
        "num_scored_images": int(num_scored_images),
        "num_skipped_images": int(num_skipped_images),
        "num_recoverable_stage_failures": int(num_recoverable_stage_failures),
        "aesthetic_scores_csv": str(source_csv) if source_csv and aesthetic_scoring.get("enabled") else None,
        "aesthetic_enabled": bool(aesthetic_scoring.get("enabled")),
        "aesthetic_backend": aesthetic_scoring.get("backend"),
        "aesthetic_config": aesthetic_scoring.get("config_path"),
        "aesthetic_weight": aesthetic_weight,
        "aesthetic_models": sorted((aesthetic_scoring.get("aesthetic_model_paths") or {}).keys()),
        "aesthetic_model_paths": aesthetic_scoring.get("aesthetic_model_paths") or {},
        "aesthetic_pairwise_model_paths": aesthetic_scoring.get("pairwise_model_paths") or {},
        "aesthetic_missing_model_paths": aesthetic_scoring.get("missing_model_paths") or {},
        "aesthetic_auto_resolved": bool(aesthetic_scoring.get("auto_resolved")),
        "aesthetic_disabled_reason": aesthetic_scoring.get("disabled_reason"),
        "aesthetic_warnings": aesthetic_scoring.get("warnings") or [],
        "vila_scores_csv": str(args.vila_scores_csv) if args.vila_scores_csv else None,
        "top_k": int(args.top_k),
        "enable_pairwise_rerank": bool(args.enable_pairwise_rerank),
        "enable_vila_rerank": enable_vila_rerank,
        "vila_rerank_weight": round(float(effective_vila_rerank_weight), 6),
        "enable_vila_explanations": enable_vila_explanations,
        "enable_acut_reasoning": enable_acut_reasoning,
        "reason_reference_mode": args.reason_reference_mode,
        "reason_detail_level": args.reason_detail_level,
        "enable_diversity": bool(args.enable_diversity),
        "rerank_pool_size": args.rerank_pool_size,
        "weights_config": args.weights_config,
        "export_debug_reasoning": bool(args.export_debug_reasoning),
    }


def write_score_failure_report(
    output_dir: Path,
    *,
    failures: list[dict[str, object]],
    num_input_images: int,
    num_scored_images: int,
    num_skipped_images: int,
    num_recoverable_stage_failures: int,
    json_indent: int,
) -> Path | None:
    if not failures and num_recoverable_stage_failures <= 0:
        return None
    report_path = output_dir / SCORE_FAILURE_REPORT_FILENAME
    payload = {
        "num_input_images": int(num_input_images),
        "num_scored_images": int(num_scored_images),
        "num_skipped_images": int(num_skipped_images),
        "num_recoverable_stage_failures": int(num_recoverable_stage_failures),
        "skipped_images": failures,
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=json_indent, ensure_ascii=False)
        f.write("\n")
    return report_path


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.copy_top_k and args.symlink_top_k:
        raise ValueError("Choose only one of --copy_top_k or --symlink_top_k.")

    weights = load_weights(args.weights_config)
    aesthetic_scoring = configure_aesthetic_scoring(args)
    records, source_jsonl, source_csv = load_or_compute_scores(args, output_dir)
    score_failures = list(getattr(args, "_score_failures", []) or [])
    num_input_images = int(getattr(args, "_score_num_input_images", len(records)))
    num_scored_images = int(getattr(args, "_score_num_scored_images", len(records)))
    num_skipped_images = int(getattr(args, "_score_num_skipped_images", len(score_failures)))
    num_recoverable_stage_failures = int(getattr(args, "_score_num_recoverable_stage_failures", 0))
    score_failures_json = write_score_failure_report(
        output_dir,
        failures=score_failures,
        num_input_images=num_input_images,
        num_scored_images=num_scored_images,
        num_skipped_images=num_skipped_images,
        num_recoverable_stage_failures=num_recoverable_stage_failures,
        json_indent=args.json_indent,
    )
    if num_input_images > 0 and not records:
        first_failure = score_failures[0] if score_failures else {}
        failed_stage = first_failure.get("stage")
        failed_path = first_failure.get("image_path")
        failed_error = first_failure.get("error")
        raise ValueError(
            "No usable images could be scored from input_dir. "
            f"input_images={num_input_images}, skipped_images={num_skipped_images}. "
            f"First failure image={failed_path!r}, stage={failed_stage!r}, error={failed_error!r}"
        )
    model_load_warnings = getattr(args, "_model_load_warnings", [])
    if model_load_warnings:
        existing_warnings = list(aesthetic_scoring.get("warnings") or [])
        aesthetic_scoring["warnings"] = [
            *existing_warnings,
            *[
                (
                    f"Skipped auto-resolved model {warning.get('field')} "
                    f"({repo_relative_path(warning.get('path')) or warning.get('path')}): {warning.get('error')}"
                )
                for warning in model_load_warnings
            ],
        ]
    aesthetic_scores_present = any(
        _has_score_value(record.get(score_field))
        for record in records
        for score_field in AESTHETIC_SCORE_FIELDS
    )
    if aesthetic_scores_present and aesthetic_scoring.get("disabled_reason") == "precomputed_scores_supplied":
        aesthetic_scoring["enabled"] = True
        aesthetic_scoring["backend"] = "precomputed_scores"
        aesthetic_scoring["disabled_reason"] = None
    if aesthetic_scoring.get("enabled") and not aesthetic_scores_present:
        aesthetic_scoring["enabled"] = False
        aesthetic_scoring["disabled_reason"] = "no_aesthetic_scores_generated"
    records, vila_merge_summary = merge_vila_scores(records, args.vila_scores_csv)

    enable_vila_rerank = _resolve_vila_default(
        args.enable_vila_rerank,
        disabled=args.disable_vila_rerank,
        vila_scores_csv=args.vila_scores_csv,
    )
    enable_vila_explanations = _resolve_vila_default(
        args.enable_vila_explanations,
        disabled=args.disable_vila_explanations,
        vila_scores_csv=args.vila_scores_csv,
    )
    enable_acut_reasoning = _resolve_reasoning_default(
        args.enable_acut_reasoning,
        disabled=args.disable_acut_reasoning,
    )
    enable_multimodal_explanations = _resolve_reasoning_default(
        args.enable_multimodal_explanations,
        disabled=args.disable_multimodal_explanations,
    )
    effective_vila_rerank_weight = float(
        args.vila_rerank_weight
        if args.vila_rerank_weight is not None
        else weights["vila_rerank"].get("weight", DEFAULT_VILA_RERANK_WEIGHT)
    )

    diversity_config = build_diversity_config(args) if args.enable_diversity else None
    ranked_rows, duplicate_groups_report = rank_records(
        records,
        weights=weights,
        enable_pairwise_rerank=args.enable_pairwise_rerank,
        enable_vila_rerank=enable_vila_rerank,
        vila_rerank_weight=effective_vila_rerank_weight,
        rerank_pool_size=args.rerank_pool_size,
        diversity_config=diversity_config,
    )

    populate_vila_explanations(
        ranked_rows,
        top_k=args.top_k,
        enable_vila_explanations=enable_vila_explanations,
    )
    # Always synthesize baseline explanations first.
    if enable_acut_reasoning:
        synthesize_acut_selection_explanations(
            ranked_rows,
            top_k=args.top_k,
            reference_mode=args.reason_reference_mode,
            detail_level=args.reason_detail_level,
            include_model_contributions=args.reason_include_model_contributions,
            include_vila_signals=args.reason_include_vila_signals,
            include_comparison=args.reason_include_comparison,
        )

    # If enabled, attempt to overwrite with richer multimodal explanations.
    if enable_multimodal_explanations:
        generate_multimodal_explanations(
            rows=ranked_rows,
            top_k=args.top_k,
            gemini_model_name=args.gemini_model_name,
            enable_gemini=args.enable_gemini,
        )

    ranked_jsonl, ranked_csv, top_k_list = write_ranked_outputs(
        ranked_rows,
        output_dir=output_dir,
        top_k=args.top_k,
    )
    duplicate_groups_json = write_duplicate_groups_report(duplicate_groups_report, output_dir=output_dir)
    top_k_dir = materialize_top_k(
        ranked_rows,
        output_dir=output_dir,
        top_k=args.top_k,
        copy_top_k=args.copy_top_k,
        symlink_top_k=args.symlink_top_k,
    )

    pipeline_config = build_pipeline_config(
        args,
        source_jsonl=source_jsonl,
        source_csv=source_csv,
        score_failures_json=score_failures_json,
        num_input_images=num_input_images,
        num_scored_images=num_scored_images,
        num_skipped_images=num_skipped_images,
        num_recoverable_stage_failures=num_recoverable_stage_failures,
        enable_vila_rerank=enable_vila_rerank,
        enable_vila_explanations=enable_vila_explanations,
        enable_acut_reasoning=enable_acut_reasoning,
        effective_vila_rerank_weight=effective_vila_rerank_weight,
        weights=weights,
        aesthetic_scoring=aesthetic_scoring,
    )
    product_exports = write_product_exports(
        ranked_rows,
        output_dir=output_dir,
        top_k=args.top_k,
        pipeline_config=pipeline_config,
        export_debug_reasoning=args.export_debug_reasoning,
        near_cut_rejected_count=args.review_near_cut_rejected_count,
        json_indent=args.json_indent,
    )

    return {
        "num_images": len(ranked_rows),
        "num_input_images": int(num_input_images),
        "num_scored_images": int(num_scored_images),
        "num_skipped_images": int(num_skipped_images),
        "num_recoverable_stage_failures": int(num_recoverable_stage_failures),
        "top_k": int(args.top_k),
        "scores_jsonl": str(source_jsonl) if source_jsonl else None,
        "scores_csv": str(source_csv) if source_csv else None,
        "score_failures_json": str(score_failures_json) if score_failures_json else None,
        "vila_scores_csv": str(args.vila_scores_csv) if args.vila_scores_csv else None,
        "vila_merge_summary": vila_merge_summary,
        "aesthetic_scoring": aesthetic_scoring,
        "pipeline_config": pipeline_config,
        "ranked_jsonl": str(ranked_jsonl),
        "ranked_csv": str(ranked_csv),
        "duplicate_groups_json": str(duplicate_groups_json) if duplicate_groups_json else None,
        "top_k_list": str(top_k_list),
        "top_k_dir": str(top_k_dir) if top_k_dir else None,
        **product_exports,
    }


def main() -> None:
    configure_runtime_logging()
    args = build_parser().parse_args()
    summary = run_pipeline(args)
    print(json.dumps(summary, indent=args.json_indent, ensure_ascii=False))


if __name__ == "__main__":
    main()
