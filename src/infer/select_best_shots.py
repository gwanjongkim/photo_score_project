from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.infer.diversity_utils import (
    SimilarityConfig,
    compare_similarity_features,
    compute_similarity_features,
    sanitize_similarity_config,
)
from src.infer.predict_quality_bundle import add_bundle_arguments
from src.infer.rerank_utils import centered_pool_deltas
from src.infer.score_image_folder import score_folder_results, write_score_outputs
from src.vila.explain_acut_selection import (
    DEFAULT_REASON_DETAIL_LEVEL,
    DEFAULT_REASON_REFERENCE_MODE,
    REASON_DETAIL_LEVELS,
    REASON_REFERENCE_MODES,
    synthesize_acut_selection_explanations,
)
from src.vila.explain_selection import explain_prompt_scores, extract_prompt_scores


DEFAULT_SELECTOR_WEIGHTS = {
    "component_mix": {
        "aesthetic": 0.6,
        "technical": 0.4,
    },
    "aesthetic": {
        "aadb_score": 0.28,
        "nima_mean_score": 0.24,
        "alamp_score": 0.22,
        "rgnet_score": 0.18,
        "pairwise_recovered_score": 0.08,
    },
    "technical": {
        "koniq_score": 0.32,
        "flive_image_score": 0.24,
        "flive_patch_mean": 0.22,
        "flive_patch_min": 0.12,
        "musiq_score": 0.10,
    },
    "pairwise_rerank": {
        "enabled": False,
        "weight": 0.15,
        "pool_size": 10,
    },
    "vila_rerank": {
        "enabled": False,
        "weight": 0.10,
        "pool_size": 10,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank a folder of candidate images into a practical first-pass A-cut.")
    parser.add_argument("--input_dir")
    parser.add_argument("--scores_csv")
    parser.add_argument("--scores_jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--copy_top_k", action="store_true")
    parser.add_argument("--symlink_top_k", action="store_true")
    parser.add_argument("--weights_config")
    parser.add_argument("--enable_pairwise_rerank", action="store_true")
    parser.add_argument("--rerank_pool_size", type=int)
    parser.add_argument("--vila_scores_csv")
    parser.add_argument("--enable_vila_rerank", action="store_true")
    parser.add_argument("--vila_rerank_weight", type=float)
    parser.add_argument("--enable_vila_explanations", action="store_true")
    parser.add_argument("--enable_acut_reasoning", action="store_true")
    parser.add_argument(
        "--reason_reference_mode",
        choices=REASON_REFERENCE_MODES,
        default=DEFAULT_REASON_REFERENCE_MODE,
    )
    parser.add_argument(
        "--reason_detail_level",
        choices=REASON_DETAIL_LEVELS,
        default=DEFAULT_REASON_DETAIL_LEVEL,
    )
    parser.add_argument(
        "--reason_include_model_contributions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--reason_include_vila_signals",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reason_include_comparison",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable_diversity",
        action="store_true",
        help="Apply lightweight similarity-aware reranking to reduce near-duplicate picks in the final A-cut.",
    )
    parser.add_argument(
        "--diversity_threshold",
        type=float,
        default=0.82,
        help="Combined similarity threshold where diversity penalties begin.",
    )
    parser.add_argument(
        "--hard_duplicate_threshold",
        type=float,
        default=0.88,
        help="Combined similarity threshold treated as a near-duplicate for full penalty and duplicate grouping.",
    )
    parser.add_argument(
        "--diversity_penalty_strength",
        type=float,
        default=0.12,
        help="Maximum absolute score penalty applied to near-duplicate candidates.",
    )
    parser.add_argument("--score_recursive", action="store_true")
    parser.add_argument("--score_extensions", default=".jpg,.jpeg,.png,.webp,.bmp")
    return add_bundle_arguments(parser)


def load_weights(weights_config: str | None) -> dict[str, object]:
    if not weights_config:
        return json.loads(json.dumps(DEFAULT_SELECTOR_WEIGHTS))
    with Path(weights_config).open("r", encoding="utf-8") as f:
        override = json.load(f)
    weights = json.loads(json.dumps(DEFAULT_SELECTOR_WEIGHTS))
    merge_nested_dict(weights, override)
    return weights


def merge_nested_dict(base: dict[str, object], override: dict[str, object]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_nested_dict(base[key], value)
        else:
            base[key] = value


def load_records_from_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_records_from_csv(path: Path) -> list[dict[str, object]]:
    df = pd.read_csv(path)
    return df.replace({np.nan: None}).to_dict(orient="records")


def is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def normalize_unit_score(value) -> float | None:
    if is_missing(value):
        return None
    value = float(value)
    if -0.05 <= value <= 1.05:
        return float(np.clip(value, 0.0, 1.0))
    if 1.0 <= value <= 10.5:
        return float(np.clip((value - 1.0) / 9.0, 0.0, 1.0))
    if 0.0 <= value <= 100.0:
        return float(np.clip(value / 100.0, 0.0, 1.0))
    return None


def canonicalize_image_key(path_value: object) -> str | None:
    if is_missing(path_value):
        return None
    text = str(path_value).strip()
    if not text:
        return None
    try:
        return str(Path(text).expanduser().resolve(strict=False))
    except OSError:
        return text


def maybe_parse_json_like(value: object) -> object:
    if is_missing(value):
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def merge_vila_scores(
    records: list[dict[str, object]],
    vila_scores_csv: str | None,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    if not vila_scores_csv:
        return records, None

    vila_df = pd.read_csv(vila_scores_csv)
    required_columns = {"image_path", "vila_score"}
    missing_columns = sorted(required_columns - set(vila_df.columns))
    if missing_columns:
        raise ValueError(
            f"VILA scores CSV must contain {', '.join(sorted(required_columns))}. Missing: {', '.join(missing_columns)}"
        )

    raw_rows = len(vila_df)
    vila_df["__image_key"] = vila_df["image_path"].apply(canonicalize_image_key)
    vila_df = vila_df.dropna(subset=["__image_key"]).copy()
    vila_df["__sort_score"] = pd.to_numeric(vila_df["vila_score"], errors="coerce")
    vila_df = vila_df.sort_values("__sort_score", ascending=False, na_position="last")
    deduped_df = vila_df.drop_duplicates("__image_key", keep="first").copy()

    prompt_columns = sorted(
        column
        for column in deduped_df.columns
        if column.startswith("prompt_") and column != "prompt_preset"
    )
    vila_map: dict[str, dict[str, object]] = {}
    for row in deduped_df.to_dict(orient="records"):
        payload = {}
        for column in ["vila_score", "vila_explanation", "explanation_signals", *prompt_columns]:
            if column not in row or is_missing(row.get(column)):
                continue
            payload[column] = row[column]

        if "vila_score" in payload:
            score_value = pd.to_numeric(payload["vila_score"], errors="coerce")
            if pd.isna(score_value):
                payload.pop("vila_score", None)
            else:
                payload["vila_score"] = float(score_value)
        if "explanation_signals" in payload:
            payload["explanation_signals"] = maybe_parse_json_like(payload["explanation_signals"])
        vila_map[str(row["__image_key"])] = payload

    merged_records = []
    matched_records = 0
    for record in records:
        merged = dict(record)
        image_key = canonicalize_image_key(record.get("image_path"))
        vila_payload = vila_map.get(image_key)
        if vila_payload is not None:
            merged.update(vila_payload)
            matched_records += 1
        merged_records.append(merged)

    merge_summary = {
        "vila_scores_csv": str(vila_scores_csv),
        "source_rows": int(raw_rows),
        "usable_rows": int(len(vila_map)),
        "matched_records": int(matched_records),
        "unmatched_records": int(len(records) - matched_records),
    }
    return merged_records, merge_summary


def score_component(record: dict[str, object], weights: dict[str, float], component_name: str) -> tuple[float | None, dict[str, dict], list[str]]:
    contributions: dict[str, dict] = {}
    missing_models = []
    available = []

    for key, weight in weights.items():
        raw_value = record.get(key)
        normalized = normalize_unit_score(raw_value)
        if normalized is None:
            missing_models.append(key)
            continue
        available.append((key, float(weight), float(raw_value), normalized))

    if not available:
        return None, contributions, missing_models

    total_weight = sum(weight for _, weight, _, _ in available)
    component_score = 0.0
    for key, weight, raw_value, normalized in available:
        effective_weight = weight / total_weight
        contribution = effective_weight * normalized
        component_score += contribution
        contributions[key] = {
            "raw_score": raw_value,
            "normalized_score": normalized,
            "configured_weight": weight,
            "effective_weight": effective_weight,
            "weighted_contribution": contribution,
        }

    return float(component_score), contributions, missing_models


def selector_notes_for_record(
    aesthetic_component,
    technical_component,
    pairwise_used: bool,
    vila_available: bool,
) -> list[str]:
    notes = []
    if aesthetic_component is None and technical_component is None:
        notes.append("No usable model scores were available for ranking.")
    elif aesthetic_component is None:
        notes.append("Aesthetic models missing; ranking falls back to technical-only scoring.")
    elif technical_component is None:
        notes.append("Technical models missing; ranking falls back to aesthetic-only scoring.")
    else:
        notes.append("Final score mixes aesthetic and technical components with default project weights.")
    if pairwise_used:
        notes.append("Pairwise recovered score was used as a light reranking signal inside the rerank pool.")
    if vila_available:
        notes.append("VILA prompt scores are available for optional reranking and explanations.")
    return notes


def append_selector_note(row: dict[str, object], note: str) -> None:
    if note not in row["selector_notes"]:
        row["selector_notes"].append(note)


def compute_selector_row(record: dict[str, object], weights: dict[str, object]) -> dict[str, object]:
    aesthetic_component, aesthetic_contrib, aesthetic_missing = score_component(
        record,
        weights=weights["aesthetic"],
        component_name="aesthetic",
    )
    technical_component, technical_contrib, technical_missing = score_component(
        record,
        weights=weights["technical"],
        component_name="technical",
    )

    mix = weights["component_mix"]
    final_score = None
    if aesthetic_component is not None and technical_component is not None:
        final_score = float(mix["aesthetic"] * aesthetic_component + mix["technical"] * technical_component)
    elif aesthetic_component is not None:
        final_score = float(aesthetic_component)
    elif technical_component is not None:
        final_score = float(technical_component)

    raw_vila_score = pd.to_numeric(record.get("vila_score"), errors="coerce")
    vila_score_raw = None if pd.isna(raw_vila_score) else float(raw_vila_score)
    vila_prompt_scores = extract_prompt_scores(record)
    row = {
        "image_path": record.get("image_path"),
        "pre_pairwise_score": final_score,
        "pairwise_rerank_applied": False,
        "pairwise_rerank_delta": 0.0,
        "final_score_before_vila": final_score,
        "vila_score_raw": vila_score_raw,
        "vila_score": vila_score_raw,
        "vila_prompt_scores": vila_prompt_scores or None,
        "vila_score_normalized_in_pool": None,
        "vila_in_rerank_pool": False,
        "vila_rerank_applied": False,
        "vila_rerank_delta": 0.0,
        "vila_explanation": None if is_missing(record.get("vila_explanation")) else str(record.get("vila_explanation")),
        "vila_explanation_signals": maybe_parse_json_like(record.get("explanation_signals")),
        "base_score": final_score,
        "final_score_after_rerank": final_score,
        "final_score": final_score,
        "aesthetic_component": aesthetic_component,
        "technical_component": technical_component,
        "base_rank": None,
        "rerank_rank": None,
        "diversity_penalty": 0.0,
        "similarity_to_higher_ranked": None,
        "similarity_components_to_higher_ranked": None,
        "most_similar_higher_ranked_image": None,
        "reason_selected": None,
        "acut_short_reason": None,
        "acut_detailed_reason": None,
        "acut_comparison_reason": None,
        "acut_rejection_reason": None,
        "acut_explanation_structured": None,
        "per_model_contributions": {
            "aesthetic": aesthetic_contrib,
            "technical": technical_contrib,
        },
        "per_model_scores": {
            key: record.get(key)
            for key in [
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
                "vila_score",
                "baseline_final_score",
            ]
            if not is_missing(record.get(key))
        },
        "missing_models": sorted(set(aesthetic_missing + technical_missing)),
        "selector_notes": selector_notes_for_record(
            aesthetic_component,
            technical_component,
            pairwise_used=False,
            vila_available=vila_score_raw is not None,
        ),
    }
    if vila_score_raw is not None:
        row["per_model_scores"]["vila_score_raw"] = vila_score_raw
    return row


def apply_optional_pairwise_rerank(rows: list[dict[str, object]], weights: dict[str, object], enable_pairwise_rerank: bool, rerank_pool_size: int | None) -> bool:
    rerank_cfg = weights["pairwise_rerank"]
    enabled = enable_pairwise_rerank or bool(rerank_cfg.get("enabled", False))
    if not enabled:
        return False

    pool_size = int(rerank_pool_size or rerank_cfg.get("pool_size", 10))
    rerank_weight = float(rerank_cfg.get("weight", 0.15))
    ranked = sorted(
        rows,
        key=lambda row: score_sort_key_for(row, primary_score_key="final_score_before_vila", secondary_score_key="base_score"),
        reverse=True,
    )
    pool = ranked[:pool_size]
    if not pool:
        return False

    any_pairwise = False
    for row in pool:
        pairwise_score = normalize_unit_score(row["per_model_scores"].get("pairwise_recovered_score"))
        if pairwise_score is None or row["final_score_before_vila"] is None:
            continue
        previous_score = float(row["final_score_before_vila"])
        any_pairwise = True
        row["final_score_before_vila"] = float((1.0 - rerank_weight) * previous_score + rerank_weight * pairwise_score)
        row["final_score_after_rerank"] = row["final_score_before_vila"]
        row["pairwise_rerank_applied"] = True
        row["pairwise_rerank_delta"] = float(row["final_score_before_vila"] - previous_score)
        append_selector_note(row, "Top-pool rerank applied with pairwise recovered score.")

    return any_pairwise


def apply_optional_vila_rerank(
    rows: list[dict[str, object]],
    weights: dict[str, object],
    enable_vila_rerank: bool,
    vila_rerank_weight: float | None,
    rerank_pool_size: int | None,
) -> bool:
    rerank_cfg = weights["vila_rerank"]
    enabled = enable_vila_rerank or bool(rerank_cfg.get("enabled", False))
    if not enabled:
        return False

    pool_size = int(rerank_pool_size or rerank_cfg.get("pool_size", 10))
    rerank_weight = float(vila_rerank_weight if vila_rerank_weight is not None else rerank_cfg.get("weight", 0.10))
    for row in rows:
        row["final_score_after_rerank"] = row["final_score_before_vila"]
        row["vila_in_rerank_pool"] = False
        row["vila_score_normalized_in_pool"] = None
        row["vila_rerank_applied"] = False
        row["vila_rerank_delta"] = 0.0

    ranked = sorted(
        rows,
        key=lambda row: score_sort_key_for(row, primary_score_key="final_score_before_vila", secondary_score_key="base_score"),
        reverse=True,
    )
    pool = ranked[:pool_size]
    if not pool:
        return False

    valid_rows: list[dict[str, object]] = []
    valid_scores: list[float] = []
    for row in pool:
        row["vila_in_rerank_pool"] = True
        vila_score = normalize_unit_score(row.get("vila_score_raw"))
        if vila_score is None or row["final_score_before_vila"] is None:
            if row.get("vila_score_raw") is None:
                append_selector_note(row, "VILA rerank pool included this image, but no usable VILA score was available.")
            continue
        valid_rows.append(row)
        valid_scores.append(vila_score)

    if not valid_rows:
        return False

    normalized_entries = centered_pool_deltas(valid_scores, weight=rerank_weight)
    for row, normalized_entry in zip(valid_rows, normalized_entries):
        previous_score = float(row["final_score_before_vila"])
        vila_delta = float(normalized_entry["delta"])
        row["vila_score_normalized_in_pool"] = float(normalized_entry["normalized"])
        row["vila_rerank_delta"] = vila_delta
        row["final_score_after_rerank"] = float(previous_score + vila_delta)
        row["vila_rerank_applied"] = rerank_weight > 0.0 and abs(vila_delta) > 0.0
        if rerank_weight > 0.0:
            append_selector_note(
                row,
                "Top-pool normalized VILA rerank applied as a zero-centered pool-relative delta.",
            )
        else:
            append_selector_note(
                row,
                "Pool-normalized VILA signal was computed, but zero rerank weight left the score unchanged.",
            )

    for row in rows:
        if row.get("vila_score_raw") is None:
            continue
        if row["vila_in_rerank_pool"]:
            continue
        append_selector_note(
            row,
            "VILA prompt score was available for explanation support, but rerank usage stayed limited to the top pool.",
        )

    return True


def score_sort_key_for(
    row: dict[str, object],
    primary_score_key: str,
    secondary_score_key: str = "base_score",
    rank_key: str = "base_rank",
) -> tuple[int, float, float, int]:
    final_score = row.get(primary_score_key)
    base_score = row.get(secondary_score_key)
    base_rank = row.get(rank_key)
    return (
        0 if final_score is None else 1,
        float("-inf") if final_score is None else float(final_score),
        float("-inf") if base_score is None else float(base_score),
        -(int(base_rank) if base_rank is not None else 10**9),
    )


def score_sort_key(row: dict[str, object]) -> tuple[int, float, float, int]:
    final_score = row.get("final_score")
    base_score = row.get("final_score_after_rerank")
    base_rank = row.get("base_rank")
    return (
        0 if final_score is None else 1,
        float("-inf") if final_score is None else float(final_score),
        float("-inf") if base_score is None else float(base_score),
        -(int(base_rank) if base_rank is not None else 10**9),
    )


def build_diversity_config(args: argparse.Namespace) -> SimilarityConfig:
    return sanitize_similarity_config(
        SimilarityConfig(
            threshold=args.diversity_threshold,
            duplicate_threshold=args.hard_duplicate_threshold,
            penalty_strength=args.diversity_penalty_strength,
        )
    )


def build_similarity_feature_map(
    rows: list[dict[str, object]],
    config: SimilarityConfig,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    features_by_path: dict[str, object] = {}
    failures = []
    for row in rows:
        image_path = row.get("image_path")
        if image_path is None:
            failures.append({"image_path": "<missing>", "error": "Missing image_path in score record."})
            continue
        image_path = str(image_path)
        if image_path in features_by_path:
            continue
        try:
            features_by_path[image_path] = compute_similarity_features(
                image_path,
                thumbnail_size=config.thumbnail_size,
                hash_size=config.hash_size,
                histogram_bins=config.histogram_bins,
            )
        except Exception as exc:  # pragma: no cover - smoke-tested through CLI
            failures.append(
                {
                    "image_path": image_path,
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
    return features_by_path, failures


def get_cached_similarity(
    image_a: str | None,
    image_b: str | None,
    features_by_path: dict[str, object],
    similarity_cache: dict[tuple[str, str], dict[str, float] | None],
) -> dict[str, float] | None:
    if not image_a or not image_b:
        return None
    if image_a == image_b:
        return {
            "combined": 1.0,
            "thumbnail_similarity": 1.0,
            "histogram_similarity": 1.0,
            "dhash_similarity": 1.0,
        }
    key = tuple(sorted((image_a, image_b)))
    if key not in similarity_cache:
        features_a = features_by_path.get(image_a)
        features_b = features_by_path.get(image_b)
        if features_a is None or features_b is None:
            similarity_cache[key] = None
        else:
            similarity_cache[key] = compare_similarity_features(features_a, features_b)
    return similarity_cache[key]


def compute_diversity_penalty(max_similarity: float | None, config: SimilarityConfig) -> float:
    if max_similarity is None or max_similarity < config.threshold or config.penalty_strength <= 0.0:
        return 0.0
    span = max(1e-6, config.duplicate_threshold - config.threshold)
    severity = float(np.clip((max_similarity - config.threshold) / span, 0.0, 1.0))
    if max_similarity >= config.duplicate_threshold:
        return float(config.penalty_strength)
    return float(config.penalty_strength * severity)


def evaluate_candidate_for_diversity(
    row: dict[str, object],
    selected_rows: list[dict[str, object]],
    features_by_path: dict[str, object],
    similarity_cache: dict[tuple[str, str], dict[str, float] | None],
    config: SimilarityConfig,
) -> dict[str, object]:
    rerank_score = row.get("final_score_after_rerank")
    image_path = str(row["image_path"]) if row.get("image_path") is not None else None
    if not selected_rows:
        return {
            "adjusted_score": rerank_score,
            "diversity_penalty": 0.0,
            "similarity_to_higher_ranked": None,
            "similarity_components_to_higher_ranked": None,
            "most_similar_higher_ranked_image": None,
            "reason_selected": "Highest pre-diversity rerank score seeded the diversity-aware ranking.",
            "selector_note": "Diversity-aware ranking started from the strongest pre-diversity rerank score.",
        }

    best_similarity = None
    best_components = None
    best_match_path = None
    for higher_row in selected_rows:
        higher_path = str(higher_row["image_path"]) if higher_row.get("image_path") is not None else None
        similarity = get_cached_similarity(
            image_path,
            higher_path,
            features_by_path=features_by_path,
            similarity_cache=similarity_cache,
        )
        if similarity is None:
            continue
        if best_similarity is None or similarity["combined"] > best_similarity:
            best_similarity = float(similarity["combined"])
            best_components = similarity
            best_match_path = higher_path

    if rerank_score is None:
        return {
            "adjusted_score": None,
            "diversity_penalty": 0.0,
            "similarity_to_higher_ranked": best_similarity,
            "similarity_components_to_higher_ranked": best_components,
            "most_similar_higher_ranked_image": best_match_path,
            "reason_selected": "No usable pre-diversity rerank score was available, so this image stayed in scoreless fallback order.",
            "selector_note": "Diversity did not change ordering because no pre-diversity rerank score was available.",
        }

    if best_similarity is None:
        return {
            "adjusted_score": rerank_score,
            "diversity_penalty": 0.0,
            "similarity_to_higher_ranked": None,
            "similarity_components_to_higher_ranked": None,
            "most_similar_higher_ranked_image": None,
            "reason_selected": "Pre-diversity rerank score was retained because similarity features were unavailable against higher-ranked selections.",
            "selector_note": "Similarity features were unavailable for diversity comparison, so the pre-diversity rerank score was kept.",
        }

    diversity_penalty = compute_diversity_penalty(best_similarity, config)
    adjusted_score = float(max(0.0, float(rerank_score) - diversity_penalty))
    if diversity_penalty <= 0.0:
        return {
            "adjusted_score": adjusted_score,
            "diversity_penalty": 0.0,
            "similarity_to_higher_ranked": best_similarity,
            "similarity_components_to_higher_ranked": best_components,
            "most_similar_higher_ranked_image": best_match_path,
            "reason_selected": (
                f"Selected on pre-diversity rerank score because the strongest similarity to a higher-ranked image "
                f"({best_similarity:.3f}) stayed below the diversity threshold."
            ),
            "selector_note": f"Most similar higher-ranked image stayed below the diversity threshold at {best_similarity:.3f}.",
        }

    penalty_label = "near-duplicate penalty" if best_similarity >= config.duplicate_threshold else "diversity penalty"
    return {
        "adjusted_score": adjusted_score,
        "diversity_penalty": diversity_penalty,
        "similarity_to_higher_ranked": best_similarity,
        "similarity_components_to_higher_ranked": best_components,
        "most_similar_higher_ranked_image": best_match_path,
        "reason_selected": (
            f"Selected after a {penalty_label} of {diversity_penalty:.3f}; "
            f"the strongest similarity to a higher-ranked image was {best_similarity:.3f}."
        ),
        "selector_note": (
            f"{penalty_label.capitalize()} applied because similarity to a higher-ranked image reached "
            f"{best_similarity:.3f}."
        ),
    }


def apply_optional_diversity_rerank(
    rows: list[dict[str, object]],
    config: SimilarityConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    features_by_path, failures = build_similarity_feature_map(rows, config)
    failed_paths = {failure["image_path"] for failure in failures}
    for row in rows:
        if str(row.get("image_path")) in failed_paths:
            row["selector_notes"].append(
                "Diversity features were unavailable for this image; base score was used without similarity penalties."
            )

    fallback_reason = None
    if len(features_by_path) < 2:
        fallback_reason = "Fewer than two images had usable similarity features, so diversity reranking was skipped."
        for row in rows:
            row["final_score"] = row["final_score_after_rerank"]
            row["diversity_penalty"] = 0.0
            row["similarity_to_higher_ranked"] = None
            row["similarity_components_to_higher_ranked"] = None
            row["most_similar_higher_ranked_image"] = None
            row["reason_selected"] = "Pre-diversity rerank score was retained because diversity reranking could not be computed reliably."
            append_selector_note(row, fallback_reason)
        return rows, build_duplicate_groups_report(
            rows=rows,
            config=config,
            features_by_path=features_by_path,
            failures=failures,
            similarity_cache={},
            fallback_reason=fallback_reason,
        )

    selected_rows: list[dict[str, object]] = []
    remaining_rows = list(rows)
    similarity_cache: dict[tuple[str, str], dict[str, float] | None] = {}
    reranked_rows = []

    while remaining_rows:
        best_index = 0
        best_metrics = None
        best_sort_key = None
        for index, row in enumerate(remaining_rows):
            metrics = evaluate_candidate_for_diversity(
                row,
                selected_rows=selected_rows,
                features_by_path=features_by_path,
                similarity_cache=similarity_cache,
                config=config,
            )
            adjusted_score = metrics["adjusted_score"]
            sort_key = (
                0 if adjusted_score is None else 1,
                float("-inf") if adjusted_score is None else float(adjusted_score),
                float("-inf") if row["final_score_after_rerank"] is None else float(row["final_score_after_rerank"]),
                -(int(row["rerank_rank"]) if row["rerank_rank"] is not None else 10**9),
            )
            if best_sort_key is None or sort_key > best_sort_key:
                best_index = index
                best_metrics = metrics
                best_sort_key = sort_key

        chosen = remaining_rows.pop(best_index)
        chosen["final_score"] = best_metrics["adjusted_score"]
        chosen["diversity_penalty"] = best_metrics["diversity_penalty"]
        chosen["similarity_to_higher_ranked"] = best_metrics["similarity_to_higher_ranked"]
        chosen["similarity_components_to_higher_ranked"] = best_metrics["similarity_components_to_higher_ranked"]
        chosen["most_similar_higher_ranked_image"] = best_metrics["most_similar_higher_ranked_image"]
        chosen["reason_selected"] = best_metrics["reason_selected"]
        selector_note = best_metrics["selector_note"]
        append_selector_note(chosen, selector_note)
        selected_rows.append(chosen)
        reranked_rows.append(chosen)

    return reranked_rows, build_duplicate_groups_report(
        rows=reranked_rows,
        config=config,
        features_by_path=features_by_path,
        failures=failures,
        similarity_cache=similarity_cache,
        fallback_reason=fallback_reason,
    )


def build_duplicate_groups_report(
    rows: list[dict[str, object]],
    config: SimilarityConfig,
    features_by_path: dict[str, object],
    failures: list[dict[str, str]],
    similarity_cache: dict[tuple[str, str], dict[str, float] | None],
    fallback_reason: str | None,
) -> dict[str, object]:
    rows_by_path = {
        str(row["image_path"]): row
        for row in rows
        if row.get("image_path") is not None and str(row["image_path"]) in features_by_path
    }
    candidate_paths = list(rows_by_path.keys())
    adjacency = {image_path: set() for image_path in candidate_paths}

    for index, image_a in enumerate(candidate_paths):
        for image_b in candidate_paths[index + 1 :]:
            similarity = get_cached_similarity(
                image_a,
                image_b,
                features_by_path=features_by_path,
                similarity_cache=similarity_cache,
            )
            if similarity is not None and similarity["combined"] >= config.duplicate_threshold:
                adjacency[image_a].add(image_b)
                adjacency[image_b].add(image_a)

    duplicate_groups = []
    visited = set()
    for image_path in candidate_paths:
        if image_path in visited or not adjacency[image_path]:
            continue
        component = []
        stack = [image_path]
        visited.add(image_path)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        component.sort(key=lambda path: rows_by_path[path]["base_rank"])
        leader_path = component[0]
        max_pair_similarity = 0.0
        members = []
        for member_path in component:
            if member_path == leader_path:
                similarity_to_leader = 1.0
            else:
                similarity_to_leader = float(
                    get_cached_similarity(
                        leader_path,
                        member_path,
                        features_by_path=features_by_path,
                        similarity_cache=similarity_cache,
                    )["combined"]
                )
            members.append(
                {
                    "image_path": member_path,
                    "base_rank": rows_by_path[member_path]["base_rank"],
                    "rerank_rank": rows_by_path[member_path].get("rerank_rank"),
                    "rank": rows_by_path[member_path].get("rank"),
                    "base_score": rows_by_path[member_path]["base_score"],
                    "final_score_before_vila": rows_by_path[member_path].get("final_score_before_vila"),
                    "final_score_after_rerank": rows_by_path[member_path].get("final_score_after_rerank"),
                    "final_score": rows_by_path[member_path]["final_score"],
                    "diversity_penalty": rows_by_path[member_path]["diversity_penalty"],
                    "similarity_to_group_leader": similarity_to_leader,
                }
            )

        for index_a, member_a in enumerate(component):
            for member_b in component[index_a + 1 :]:
                similarity = get_cached_similarity(
                    member_a,
                    member_b,
                    features_by_path=features_by_path,
                    similarity_cache=similarity_cache,
                )
                if similarity is not None:
                    max_pair_similarity = max(max_pair_similarity, float(similarity["combined"]))

        duplicate_groups.append(
            {
                "group_id": len(duplicate_groups) + 1,
                "leader_image": leader_path,
                "group_size": len(component),
                "max_pair_similarity": max_pair_similarity,
                "members": members,
            }
        )

    selection_trace = [
        {
            "rank": row.get("rank"),
            "base_rank": row.get("base_rank"),
            "rerank_rank": row.get("rerank_rank"),
            "image_path": row.get("image_path"),
            "pre_pairwise_score": row.get("pre_pairwise_score"),
            "pairwise_rerank_applied": row.get("pairwise_rerank_applied"),
            "pairwise_rerank_delta": row.get("pairwise_rerank_delta"),
            "base_score": row.get("base_score"),
            "final_score_before_vila": row.get("final_score_before_vila"),
            "vila_score_raw": row.get("vila_score_raw"),
            "vila_score_normalized_in_pool": row.get("vila_score_normalized_in_pool"),
            "vila_rerank_delta": row.get("vila_rerank_delta"),
            "final_score_after_rerank": row.get("final_score_after_rerank"),
            "final_score": row.get("final_score"),
            "diversity_penalty": row.get("diversity_penalty"),
            "similarity_to_higher_ranked": row.get("similarity_to_higher_ranked"),
            "most_similar_higher_ranked_image": row.get("most_similar_higher_ranked_image"),
            "reason_selected": row.get("reason_selected"),
        }
        for row in rows
    ]
    return {
        "enabled": True,
        "fallback_reason": fallback_reason,
        "config": {
            "diversity_threshold": config.threshold,
            "hard_duplicate_threshold": config.duplicate_threshold,
            "diversity_penalty_strength": config.penalty_strength,
            "thumbnail_size": config.thumbnail_size,
            "hash_size": config.hash_size,
            "histogram_bins": config.histogram_bins,
        },
        "feature_stats": {
            "num_images": len(rows),
            "num_featured_images": len(features_by_path),
            "num_failures": len(failures),
        },
        "feature_failures": failures,
        "duplicate_groups": duplicate_groups,
        "selection_trace": selection_trace,
    }


def rank_records(
    records: list[dict[str, object]],
    weights: dict[str, object],
    enable_pairwise_rerank: bool,
    enable_vila_rerank: bool,
    vila_rerank_weight: float | None,
    rerank_pool_size: int | None,
    diversity_config: SimilarityConfig | None = None,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    rows = [compute_selector_row(record, weights=weights) for record in records]
    pairwise_used = apply_optional_pairwise_rerank(
        rows,
        weights=weights,
        enable_pairwise_rerank=enable_pairwise_rerank,
        rerank_pool_size=rerank_pool_size,
    )
    vila_used = apply_optional_vila_rerank(
        rows,
        weights=weights,
        enable_vila_rerank=enable_vila_rerank,
        vila_rerank_weight=vila_rerank_weight,
        rerank_pool_size=rerank_pool_size,
    )
    for row in rows:
        if pairwise_used and "pairwise_recovered_score" in row["per_model_scores"]:
            if "Top-pool rerank applied with pairwise recovered score." not in row["selector_notes"]:
                append_selector_note(row, "Pairwise recovered score was available but not applied outside the rerank pool.")
        if (not vila_used) and row.get("vila_score_raw") is not None:
            append_selector_note(row, "VILA prompt score is available for explanation support even when rerank is disabled.")

    rows.sort(
        key=lambda row: score_sort_key_for(row, primary_score_key="base_score", secondary_score_key="base_score", rank_key="base_rank"),
        reverse=True,
    )
    for index, row in enumerate(rows, start=1):
        row["base_rank"] = index

    rows.sort(
        key=lambda row: score_sort_key_for(
            row,
            primary_score_key="final_score_after_rerank",
            secondary_score_key="base_score",
            rank_key="base_rank",
        ),
        reverse=True,
    )
    for index, row in enumerate(rows, start=1):
        row["rerank_rank"] = index

    diversity_report = None
    if diversity_config is not None:
        rows, diversity_report = apply_optional_diversity_rerank(rows, config=diversity_config)
    else:
        for row in rows:
            row["final_score"] = row["final_score_after_rerank"]
            row["diversity_penalty"] = 0.0
            row["similarity_to_higher_ranked"] = None
            row["similarity_components_to_higher_ranked"] = None
            row["most_similar_higher_ranked_image"] = None
            row["reason_selected"] = "Selected directly from the pre-diversity rerank order because diversity control was disabled."
            append_selector_note(row, "Diversity control disabled; final ranking follows the pre-diversity rerank score.")

    rows.sort(key=score_sort_key, reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index

    if diversity_report is not None:
        rows_by_path = {
            str(row["image_path"]): row
            for row in rows
            if row.get("image_path") is not None
        }
        for group in diversity_report.get("duplicate_groups", []):
            for member in group.get("members", []):
                current_row = rows_by_path.get(str(member.get("image_path")))
                if current_row is None:
                    continue
                member["rank"] = current_row.get("rank")
                member["base_rank"] = current_row.get("base_rank")
                member["rerank_rank"] = current_row.get("rerank_rank")
                member["base_score"] = current_row.get("base_score")
                member["final_score_before_vila"] = current_row.get("final_score_before_vila")
                member["final_score_after_rerank"] = current_row.get("final_score_after_rerank")
                member["final_score"] = current_row.get("final_score")
                member["diversity_penalty"] = current_row.get("diversity_penalty")
        diversity_report["selection_trace"] = [
            {
                "rank": row.get("rank"),
                "base_rank": row.get("base_rank"),
                "rerank_rank": row.get("rerank_rank"),
                "image_path": row.get("image_path"),
                "pre_pairwise_score": row.get("pre_pairwise_score"),
                "pairwise_rerank_applied": row.get("pairwise_rerank_applied"),
                "pairwise_rerank_delta": row.get("pairwise_rerank_delta"),
                "base_score": row.get("base_score"),
                "final_score_before_vila": row.get("final_score_before_vila"),
                "vila_score_raw": row.get("vila_score_raw"),
                "vila_score_normalized_in_pool": row.get("vila_score_normalized_in_pool"),
                "vila_rerank_delta": row.get("vila_rerank_delta"),
                "final_score_after_rerank": row.get("final_score_after_rerank"),
                "final_score": row.get("final_score"),
                "diversity_penalty": row.get("diversity_penalty"),
                "similarity_to_higher_ranked": row.get("similarity_to_higher_ranked"),
                "most_similar_higher_ranked_image": row.get("most_similar_higher_ranked_image"),
                "reason_selected": row.get("reason_selected"),
            }
            for row in rows
        ]
    return rows, diversity_report


def populate_vila_explanations(
    rows: list[dict[str, object]],
    top_k: int,
    enable_vila_explanations: bool,
) -> None:
    if not enable_vila_explanations:
        return

    for row in rows:
        prompt_scores = row.get("vila_prompt_scores") or {}
        if prompt_scores:
            explanation = explain_prompt_scores(
                prompt_scores=prompt_scores,
                vila_score=normalize_unit_score(row.get("vila_score_raw")),
                selected=(row.get("rank") is not None and int(row["rank"]) <= top_k),
            )
            row["vila_explanation"] = explanation["text"]
            row["vila_explanation_signals"] = explanation["signals"]
            continue

        if row.get("vila_score_raw") is None or row.get("vila_explanation") is not None:
            continue

        if row.get("rank") is not None and int(row["rank"]) <= top_k:
            row["vila_explanation"] = "Selected because the prompt-based aesthetic signal stayed strong overall."
        else:
            row["vila_explanation"] = "Rejected because the prompt-based aesthetic signal trailed the top selections."


def write_ranked_outputs(rows: list[dict[str, object]], output_dir: Path, top_k: int) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked_jsonl = output_dir / "ranked_results.jsonl"
    ranked_csv = output_dir / "ranked_results.csv"
    top_k_list = output_dir / "top_k.txt"

    with ranked_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = [
        "rank",
        "base_rank",
        "rerank_rank",
        "image_path",
        "pre_pairwise_score",
        "pairwise_rerank_applied",
        "pairwise_rerank_delta",
        "final_score_before_vila",
        "vila_score_raw",
        "vila_score",
        "vila_score_normalized_in_pool",
        "vila_rerank_applied",
        "vila_rerank_delta",
        "base_score",
        "final_score_after_rerank",
        "final_score",
        "aesthetic_component",
        "technical_component",
        "diversity_penalty",
        "similarity_to_higher_ranked",
        "most_similar_higher_ranked_image",
        "reason_selected",
        "acut_short_reason",
        "acut_detailed_reason",
        "acut_comparison_reason",
        "acut_rejection_reason",
        "acut_explanation_structured",
        "vila_explanation",
        "vila_explanation_signals",
        "missing_models",
        "selector_notes",
        "similarity_components_to_higher_ranked",
        "per_model_scores",
        "vila_prompt_scores",
    ]
    with ranked_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "rank": row["rank"],
                    "base_rank": row["base_rank"],
                    "rerank_rank": row["rerank_rank"],
                    "image_path": row["image_path"],
                    "pre_pairwise_score": row["pre_pairwise_score"],
                    "pairwise_rerank_applied": row["pairwise_rerank_applied"],
                    "pairwise_rerank_delta": row["pairwise_rerank_delta"],
                    "final_score_before_vila": row["final_score_before_vila"],
                    "vila_score_raw": row["vila_score_raw"],
                    "vila_score": row["vila_score"],
                    "vila_score_normalized_in_pool": row["vila_score_normalized_in_pool"],
                    "vila_rerank_applied": row["vila_rerank_applied"],
                    "vila_rerank_delta": row["vila_rerank_delta"],
                    "base_score": row["base_score"],
                    "final_score_after_rerank": row["final_score_after_rerank"],
                    "final_score": row["final_score"],
                    "aesthetic_component": row["aesthetic_component"],
                    "technical_component": row["technical_component"],
                    "diversity_penalty": row["diversity_penalty"],
                    "similarity_to_higher_ranked": row["similarity_to_higher_ranked"],
                    "most_similar_higher_ranked_image": row["most_similar_higher_ranked_image"],
                    "reason_selected": row["reason_selected"],
                    "acut_short_reason": row["acut_short_reason"],
                    "acut_detailed_reason": row["acut_detailed_reason"],
                    "acut_comparison_reason": row["acut_comparison_reason"],
                    "acut_rejection_reason": row["acut_rejection_reason"],
                    "acut_explanation_structured": json.dumps(row["acut_explanation_structured"], ensure_ascii=False),
                    "vila_explanation": row["vila_explanation"],
                    "vila_explanation_signals": json.dumps(row["vila_explanation_signals"], ensure_ascii=False),
                    "missing_models": json.dumps(row["missing_models"], ensure_ascii=False),
                    "selector_notes": json.dumps(row["selector_notes"], ensure_ascii=False),
                    "similarity_components_to_higher_ranked": json.dumps(
                        row["similarity_components_to_higher_ranked"], ensure_ascii=False
                    ),
                    "per_model_scores": json.dumps(row["per_model_scores"], ensure_ascii=False),
                    "vila_prompt_scores": json.dumps(row["vila_prompt_scores"], ensure_ascii=False),
                }
            )

    with top_k_list.open("w", encoding="utf-8") as f:
        for row in rows[:top_k]:
            f.write(f"{row['rank']}\t{row['image_path']}\t{row['final_score']}\n")

    return ranked_jsonl, ranked_csv, top_k_list


def write_duplicate_groups_report(report: dict[str, object] | None, output_dir: Path) -> Path | None:
    if report is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "duplicate_groups.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report_path


def materialize_top_k(rows: list[dict[str, object]], output_dir: Path, top_k: int, copy_top_k: bool, symlink_top_k: bool) -> Path | None:
    if not copy_top_k and not symlink_top_k:
        return None

    target_dir = output_dir / "top_k"
    target_dir.mkdir(parents=True, exist_ok=True)
    for row in rows[:top_k]:
        source = Path(str(row["image_path"]))
        target = target_dir / f"{int(row['rank']):03d}_{source.name}"
        if target.exists() or target.is_symlink():
            target.unlink()
        if symlink_top_k:
            target.symlink_to(source.resolve())
        else:
            shutil.copy2(source, target)
    return target_dir


def load_or_compute_scores(args: argparse.Namespace, output_dir: Path) -> tuple[list[dict[str, object]], Path | None, Path | None]:
    if args.scores_jsonl:
        return load_records_from_jsonl(Path(args.scores_jsonl)), Path(args.scores_jsonl), None
    if args.scores_csv:
        return load_records_from_csv(Path(args.scores_csv)), None, Path(args.scores_csv)
    if not args.input_dir:
        raise ValueError("Provide one of --input_dir, --scores_jsonl, or --scores_csv.")

    args.input_dir = str(Path(args.input_dir))
    args.recursive = args.score_recursive
    args.extensions = args.score_extensions
    results = score_folder_results(args)
    scores_jsonl = output_dir / "scores.jsonl"
    scores_csv = output_dir / "scores.csv"
    write_score_outputs(results, output_jsonl=scores_jsonl, output_csv=scores_csv)
    return results, scores_jsonl, scores_csv


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.copy_top_k and args.symlink_top_k:
        raise ValueError("Choose only one of --copy_top_k or --symlink_top_k.")

    weights = load_weights(args.weights_config)
    records, source_jsonl, source_csv = load_or_compute_scores(args, output_dir)
    records, vila_merge_summary = merge_vila_scores(records, args.vila_scores_csv)
    diversity_config = build_diversity_config(args) if args.enable_diversity else None
    ranked_rows, duplicate_groups_report = rank_records(
        records,
        weights=weights,
        enable_pairwise_rerank=args.enable_pairwise_rerank,
        enable_vila_rerank=args.enable_vila_rerank,
        vila_rerank_weight=args.vila_rerank_weight,
        rerank_pool_size=args.rerank_pool_size,
        diversity_config=diversity_config,
    )
    populate_vila_explanations(
        ranked_rows,
        top_k=args.top_k,
        enable_vila_explanations=args.enable_vila_explanations,
    )
    if args.enable_acut_reasoning:
        synthesize_acut_selection_explanations(
            ranked_rows,
            top_k=args.top_k,
            reference_mode=args.reason_reference_mode,
            detail_level=args.reason_detail_level,
            include_model_contributions=args.reason_include_model_contributions,
            include_vila_signals=args.reason_include_vila_signals,
            include_comparison=args.reason_include_comparison,
        )

    ranked_jsonl, ranked_csv, top_k_list = write_ranked_outputs(ranked_rows, output_dir=output_dir, top_k=args.top_k)
    duplicate_groups_json = write_duplicate_groups_report(duplicate_groups_report, output_dir=output_dir)
    top_k_dir = materialize_top_k(
        ranked_rows,
        output_dir=output_dir,
        top_k=args.top_k,
        copy_top_k=args.copy_top_k,
        symlink_top_k=args.symlink_top_k,
    )

    summary = {
        "num_images": len(ranked_rows),
        "top_k": args.top_k,
        "scores_jsonl": str(source_jsonl) if source_jsonl else None,
        "scores_csv": str(source_csv) if source_csv else None,
        "vila_scores_csv": str(args.vila_scores_csv) if args.vila_scores_csv else None,
        "vila_merge_summary": vila_merge_summary,
        "ranked_jsonl": str(ranked_jsonl),
        "ranked_csv": str(ranked_csv),
        "duplicate_groups_json": str(duplicate_groups_json) if duplicate_groups_json else None,
        "top_k_list": str(top_k_list),
        "top_k_dir": str(top_k_dir) if top_k_dir else None,
    }
    print(json.dumps(summary, indent=args.json_indent, ensure_ascii=False))


if __name__ == "__main__":
    main()
