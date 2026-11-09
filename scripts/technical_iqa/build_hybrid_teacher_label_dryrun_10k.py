# Phase 5 하이브리드 teacher label 10k dry-run manifest와 audit를 생성한다.
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PHASE4_2K_MANIFEST = ROOT / "outputs" / "hybrid_teacher_label_design_review_2k_20260603" / "hybrid_label_dryrun_2k_manifest.csv"
MDIQA_PREDS = ROOT / "outputs" / "mdiqa_correct_protocol_full_benchmark_20260601" / "predictions.csv"
PAIRED_PREDS = ROOT / "outputs" / "paired_production_mdiqa_significance_expanded_20260603" / "paired_predictions_expanded.csv"
FLIVE_PROD = ROOT / "outputs" / "technical_iqa_combo_benchmark_20260524" / "fusion_predictions_flive.csv"
BASE_MANIFEST = ROOT / "outputs" / "research_protocol_iqa_manifests_20260530" / "benchmark_all_research_protocol_iqa.csv"
RECOMMENDED_POLICY = ROOT / "outputs" / "hybrid_teacher_adjusted_calibration_2k_20260603" / "recommended_adjusted_policy.json"
ADJUSTED_2K_RANGE = ROOT / "outputs" / "hybrid_teacher_adjusted_calibration_2k_20260603" / "adjusted_calibration_range_audit.csv"

OUTPUT_DIR_FULL = ROOT / "outputs" / "hybrid_teacher_label_dryrun_10k_20260603"
OUTPUT_DIR_SMOKE = ROOT / "outputs" / "hybrid_teacher_label_dryrun_10k_20260603_smoke"

SEED = 20260603
SELECTED_POLICY = "policy_f_shift_cap3"
TARGET_TOTAL = 10_000
ALLOCATIONS_FULL = {
    "koniq10k": 2500,
    "spaq": 2500,
    "flive": 4000,
}
ALLOCATIONS_SMOKE = {
    "koniq10k": 20,
    "spaq": 10,
    "flive": 15,
}
MDIQA_SOURCES = {"mdiqa_seed1", "mdiqa_xd"}
PRODUCTION_SOURCES = {"production", "production_safety_fallback"}
MIN_VALID_FOR_READY = 8000
BASELINE_2K = {
    "mean_shift": 19.62347284692526,
    "gap_mismatch": 34.99398746788501,
    "severe_fp": 0,
    "fp": 0,
    "fn": 0,
    "clipping_count": 0,
    "max_srcc_drop_vs_policy_a": 0.0,
    "koniq10k_srcc": 0.948713772081456,
    "spaq_srcc": 0.9086102286792862,
    "flive_srcc": 0.6212216409771099,
}


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "manifest": output_dir / "hybrid_label_dryrun_10k_manifest.csv",
        "validation": output_dir / "hybrid_label_dryrun_10k_validation.csv",
        "distribution": output_dir / "hybrid_label_distribution_summary_10k.csv",
        "safety": output_dir / "hybrid_label_safety_cases_10k.csv",
        "failures": output_dir / "hybrid_label_failure_cases_10k.csv",
        "range": output_dir / "range_calibration_audit_10k.csv",
        "policy_summary": output_dir / "policy_f_shift_cap3_summary_10k.csv",
        "report": output_dir / "report.md",
        "next": output_dir / "next_actions.csv",
    }


def canonical_path(value: Any) -> str:
    try:
        return str(Path(str(value)).resolve(strict=False))
    except OSError:
        return str(value)


def source_group_from_path(path: str) -> str:
    name = Path(path).name
    if "_patch_" in name:
        return name.split("_patch_", 1)[0]
    return Path(name).stem


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def clip_0_100(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 100.0)


def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if denom == 0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return pearson(rank_average(x), rank_average(y))


def linear_aligned_mae(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 2 or float(np.std(pred)) == 0.0:
        return float("nan")
    slope, intercept = np.polyfit(pred, target, 1)
    aligned = slope * pred + intercept
    return float(np.mean(np.abs(aligned - target)))


def distribution(values: pd.Series) -> dict[str, Any]:
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float).to_numpy()
    if len(vals) == 0:
        return {
            "n": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "p99": np.nan,
        }
    return {
        "n": len(vals),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
        "p01": float(np.percentile(vals, 1)),
        "p05": float(np.percentile(vals, 5)),
        "p25": float(np.percentile(vals, 25)),
        "p50": float(np.percentile(vals, 50)),
        "p75": float(np.percentile(vals, 75)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
    }


def metric_row(df: pd.DataFrame, target_col: str, extra: dict[str, Any]) -> dict[str, Any]:
    sub = df.dropna(subset=[target_col, "quality_score_0_100"]).copy()
    pred = sub[target_col].astype(float).to_numpy()
    mos = sub["quality_score_0_100"].astype(float).to_numpy()
    severe_fp = int(((sub["quality_score_0_100"] < 40) & (sub[target_col] > 65)).sum())
    fp = int(((sub["quality_score_0_100"] < 40) & (sub[target_col] > 60)).sum())
    fn = int(((sub["quality_score_0_100"] > 75) & (sub[target_col] < 45)).sum())
    unclipped_col = target_col.replace("_0_100", "_unclipped_0_100")
    if unclipped_col in sub.columns:
        clipped = int(((sub[unclipped_col] < 0) | (sub[unclipped_col] > 100)).fillna(False).sum())
    else:
        clipped = 0
    return {
        **extra,
        "n": len(sub),
        "srcc": spearman(pred, mos) if len(sub) else np.nan,
        "plcc": pearson(pred, mos) if len(sub) else np.nan,
        "mae": float(np.mean(np.abs(pred - mos))) if len(sub) else np.nan,
        "linear_aligned_mae": linear_aligned_mae(pred, mos) if len(sub) else np.nan,
        "fp": fp,
        "severe_fp": severe_fp,
        "fn": fn,
        "clipping_count": clipped,
        "boundary_0_or_100_count": int(((sub[target_col] <= 1) | (sub[target_col] >= 99)).sum()) if len(sub) else 0,
        "boundary_0_or_100_pct": float(((sub[target_col] <= 1) | (sub[target_col] >= 99)).mean()) if len(sub) else np.nan,
    }


def write_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(out)


def load_recommended_policy() -> dict[str, Any]:
    data = json.loads(RECOMMENDED_POLICY.read_text(encoding="utf-8"))
    policy = data.get("recommended_policy", {}).get("policy")
    if data.get("final_decision") != "ADJUSTED_CALIBRATION_READY_FOR_10K_DRYRUN" or policy != SELECTED_POLICY:
        raise RuntimeError(f"Expected selected policy {SELECTED_POLICY}, found decision={data.get('final_decision')} policy={policy}")
    return data


def load_base_metadata() -> pd.DataFrame:
    base = pd.read_csv(BASE_MANIFEST)
    base["image_path"] = base["image_path"].map(canonical_path)
    base["quality_score_0_100"] = pd.to_numeric(base["quality_score_0_100"], errors="coerce")
    base["split_preference"] = base["split"].astype(str).eq("test").astype(int)
    base = base.sort_values(["image_path", "split_preference"], ascending=[True, False])
    base = base.drop_duplicates("image_path", keep="first").copy()
    base["source_group"] = base["image_path"].map(source_group_from_path)
    return base[
        [
            "dataset",
            "split",
            "content_type",
            "image_path",
            "source_group",
            "quality_score_0_100",
        ]
    ].copy()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    previous_2k = pd.read_csv(PHASE4_2K_MANIFEST)
    base = load_base_metadata()
    mdiqa = pd.read_csv(MDIQA_PREDS)
    paired = pd.read_csv(PAIRED_PREDS)
    flive_prod = pd.read_csv(FLIVE_PROD)

    for df in [previous_2k, mdiqa, paired]:
        df["image_path"] = df["image_path"].map(canonical_path)
    mdiqa["quality_score_0_100"] = pd.to_numeric(mdiqa["quality_score_0_100"], errors="coerce")
    paired["quality_score_0_100"] = pd.to_numeric(paired["quality_score_0_100"], errors="coerce")
    return previous_2k, base, mdiqa, paired, flive_prod


def dedupe_mdiqa(mdiqa: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, image_path), sub in mdiqa.groupby(["dataset", "image_path"], dropna=False):
        rows.append(
            {
                "dataset": dataset,
                "image_path": image_path,
                "quality_score_0_100": float(sub["quality_score_0_100"].mean()),
                "mdiqa_pred_0_100": float(sub["pred_0_100"].mean()),
                "mdiqa_duplicate_rows": len(sub),
            }
        )
    return pd.DataFrame(rows)


def sample_rows(pool: pd.DataFrame, n: int, label: str, failures: list[dict[str, Any]]) -> pd.DataFrame:
    pool = pool.drop_duplicates("image_path").copy()
    if len(pool) < n:
        failures.append({"stage": "sampling", "dataset": label, "image_path": "", "error": f"requested {n}, available {len(pool)}"})
    take = min(n, len(pool))
    if take == 0:
        return pool.head(0).copy()
    return pool.sample(n=take, random_state=SEED).copy()


def build_candidate_pools(
    base: pd.DataFrame,
    mdiqa: pd.DataFrame,
    paired: pd.DataFrame,
    flive_prod: pd.DataFrame,
    allocations: dict[str, int],
    total_target: int,
    failures: list[dict[str, Any]],
) -> pd.DataFrame:
    mdiqa_unique = dedupe_mdiqa(mdiqa)
    paired_prod = paired.drop_duplicates("image_path").set_index("image_path")["production_pred_0_100"].to_dict()
    paired_mdiqa = paired.drop_duplicates("image_path").set_index("image_path")["mdiqa_pred_0_100"].to_dict()
    flive_prod_map = flive_prod.set_index("image_path")["flive_score"].to_dict()

    existing_paths: set[str] = set()

    koniq_pool = mdiqa_unique[mdiqa_unique["dataset"].eq("koniq10k")].merge(
        base.drop(columns=["dataset", "quality_score_0_100"], errors="ignore"),
        on="image_path",
        how="left",
    )
    koniq_pool["dataset"] = "koniq10k"
    koniq_pool["split"] = koniq_pool["split"].fillna("unknown")
    koniq_pool["content_type"] = koniq_pool["content_type"].fillna("image")
    koniq_pool["source_group"] = koniq_pool["source_group"].fillna(koniq_pool["image_path"].map(source_group_from_path))
    koniq_pool["production_pred_0_100"] = koniq_pool["image_path"].map(paired_prod)
    koniq_pool["paired_mdiqa_pred_0_100"] = koniq_pool["image_path"].map(paired_mdiqa)
    koniq_pool["path_exists"] = koniq_pool["image_path"].map(lambda p: Path(p).exists())
    koniq_pool = koniq_pool[koniq_pool["path_exists"]].dropna(subset=["mdiqa_pred_0_100", "quality_score_0_100"]).copy()
    koniq = sample_rows(koniq_pool, allocations["koniq10k"], "koniq10k_mdiqa_seed1", failures)
    koniq["mdiqa_variant"] = "seed1"
    existing_paths.update(koniq["image_path"].tolist())

    spaq_pool = mdiqa_unique[mdiqa_unique["dataset"].eq("spaq")].merge(
        base.drop(columns=["dataset", "quality_score_0_100"], errors="ignore"),
        on="image_path",
        how="left",
    )
    spaq_pool["dataset"] = "spaq"
    spaq_pool["split"] = spaq_pool["split"].fillna("unknown")
    spaq_pool["content_type"] = spaq_pool["content_type"].fillna("image")
    spaq_pool["source_group"] = spaq_pool["source_group"].fillna(spaq_pool["image_path"].map(source_group_from_path))
    spaq_pool["production_pred_0_100"] = spaq_pool["image_path"].map(paired_prod)
    spaq_pool["paired_mdiqa_pred_0_100"] = spaq_pool["image_path"].map(paired_mdiqa)
    spaq_pool["path_exists"] = spaq_pool["image_path"].map(lambda p: Path(p).exists())
    spaq_pool = spaq_pool[spaq_pool["path_exists"]].dropna(subset=["mdiqa_pred_0_100", "quality_score_0_100"]).copy()
    spaq = sample_rows(spaq_pool, allocations["spaq"], "spaq_mdiqa_xd", failures)
    spaq["mdiqa_variant"] = "xd"
    existing_paths.update(spaq["image_path"].tolist())

    flive_base = base[(base["dataset"].eq("flive")) & (base["content_type"].eq("image"))].copy()
    flive_base["prod_key"] = flive_base["image_path"].map(lambda p: Path(p).name)
    flive_base["production_pred_0_100"] = flive_base["prod_key"].map(flive_prod_map)
    flive_base = flive_base.dropna(subset=["production_pred_0_100", "quality_score_0_100"]).copy()
    flive_base["mdiqa_pred_0_100"] = flive_base["image_path"].map(
        mdiqa_unique.drop_duplicates("image_path").set_index("image_path")["mdiqa_pred_0_100"].to_dict()
    )
    flive_base["paired_mdiqa_pred_0_100"] = np.nan
    flive_base["path_exists"] = flive_base["image_path"].map(lambda p: Path(p).exists())
    flive_base = flive_base[flive_base["path_exists"]].copy()
    flive_pool = flive_base[~flive_base["image_path"].isin(existing_paths)].copy()
    flive = sample_rows(flive_pool, allocations["flive"], "flive_image_production", failures)
    flive["mdiqa_variant"] = ""
    existing_paths.update(flive["image_path"].tolist())

    fallback_target = max(0, total_target - len(koniq) - len(spaq) - len(flive))
    fallback_pool = flive_base[~flive_base["image_path"].isin(existing_paths)].copy()
    fallback = sample_rows(fallback_pool, fallback_target, "fallback_proxy_flive_production", failures)
    fallback["original_dataset"] = fallback["dataset"]
    fallback["dataset"] = "fallback_proxy"
    fallback["source_group"] = "flive_production_proxy__" + fallback["source_group"].astype(str)
    fallback["mdiqa_variant"] = ""

    dry = pd.concat([koniq, spaq, flive, fallback], ignore_index=True)
    dry = dry.drop_duplicates("image_path", keep="first").copy()
    dry["original_dataset"] = dry.get("original_dataset", pd.Series(index=dry.index, dtype=object))
    return dry


def apply_policy_f_shift_cap3(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    out["teacher_target_policy_a_unclipped_0_100"] = np.nan
    out["teacher_source_policy_a"] = ""
    out["teacher_target_policy_a_0_100"] = np.nan

    koniq = out["dataset"].eq("koniq10k")
    spaq = out["dataset"].eq("spaq")
    flive = out["dataset"].eq("flive")
    fallback = out["dataset"].eq("fallback_proxy")

    out.loc[koniq, "teacher_target_policy_a_unclipped_0_100"] = out.loc[koniq, "mdiqa_pred_0_100"]
    out.loc[koniq, "teacher_source_policy_a"] = "mdiqa_seed1"
    out.loc[spaq, "teacher_target_policy_a_unclipped_0_100"] = out.loc[spaq, "mdiqa_pred_0_100"]
    out.loc[spaq, "teacher_source_policy_a"] = "mdiqa_xd"
    out.loc[flive, "teacher_target_policy_a_unclipped_0_100"] = out.loc[flive, "production_pred_0_100"]
    out.loc[flive, "teacher_source_policy_a"] = "production"
    out.loc[fallback, "teacher_target_policy_a_unclipped_0_100"] = out.loc[fallback, "production_pred_0_100"]
    out.loc[fallback, "teacher_source_policy_a"] = "production_safety_fallback"
    out["teacher_target_policy_a_0_100"] = clip_0_100(out["teacher_target_policy_a_unclipped_0_100"])

    mdiqa_mask = out["teacher_source_policy_a"].isin(MDIQA_SOURCES)
    prod_mask = out["teacher_source_policy_a"].isin(PRODUCTION_SOURCES)
    mdiqa_mean = out.loc[mdiqa_mask, "teacher_target_policy_a_0_100"].astype(float).mean()
    prod_mean = out.loc[prod_mask, "teacher_target_policy_a_0_100"].astype(float).mean()
    raw_shift = prod_mean - mdiqa_mean
    applied_shift = float(np.clip(raw_shift, -3.0, 3.0)) if math.isfinite(raw_shift) else 0.0

    out["policy_f_shift_cap3_unclipped_0_100"] = out["teacher_target_policy_a_0_100"].astype(float)
    out.loc[mdiqa_mask, "policy_f_shift_cap3_unclipped_0_100"] = (
        out.loc[mdiqa_mask, "policy_f_shift_cap3_unclipped_0_100"].astype(float) + applied_shift
    )
    out["policy_f_shift_cap3_0_100"] = clip_0_100(out["policy_f_shift_cap3_unclipped_0_100"])
    out["policy_f_shift_cap3_applied_shift"] = np.where(mdiqa_mask, applied_shift, 0.0)

    out["selected_teacher_target_unclipped_0_100"] = out["policy_f_shift_cap3_unclipped_0_100"]
    out["selected_teacher_target_0_100"] = out["policy_f_shift_cap3_0_100"]
    out["selected_teacher_source"] = out["teacher_source_policy_a"]
    out.loc[koniq, "selected_teacher_source"] = "mdiqa_seed1_policy_f_shift_cap3"
    out.loc[spaq, "selected_teacher_source"] = "mdiqa_xd_policy_f_shift_cap3"
    out["selected_policy"] = SELECTED_POLICY
    out["target_available"] = out["selected_teacher_target_0_100"].notna()
    out["target_error"] = ""
    out.loc[~out["target_available"], "target_error"] = "missing selected teacher target"

    params = {
        "selected_policy": SELECTED_POLICY,
        "adjustment_type": "constant_shift_toward_production_mean",
        "raw_shift": raw_shift,
        "applied_shift": applied_shift,
        "cap": 3.0,
        "mdiqa_policy_a_mean": mdiqa_mean,
        "production_policy_a_mean": prod_mean,
    }
    return out, params


def build_validation(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    count_cols = ["dataset", "split", "content_type", "selected_teacher_source"]
    for key, sub in df.groupby(count_cols, dropna=False):
        rows.append({"section": "row_count", **{col: val for col, val in zip(count_cols, key)}, "n": len(sub)})
    for dataset, sub in df.groupby("dataset", dropna=False):
        rows.append(metric_row(sub, "selected_teacher_target_0_100", {"section": "mos_alignment", "dataset": dataset}))
    for source, sub in df.groupby("selected_teacher_source", dropna=False):
        rows.append(metric_row(sub, "selected_teacher_target_0_100", {"section": "mos_alignment_by_source", "selected_teacher_source": source}))
    return pd.DataFrame(rows)


def build_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_cols in [["dataset"], ["selected_teacher_source"], ["dataset", "selected_teacher_source"]]:
        for key, sub in df.groupby(group_cols, dropna=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            row = {"grouping": "+".join(group_cols), **{col: val for col, val in zip(group_cols, key_tuple)}}
            row.update(distribution(sub["selected_teacher_target_0_100"]))
            rows.append(row)
    return pd.DataFrame(rows)


def build_safety_cases(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=["selected_teacher_target_0_100", "quality_score_0_100"]).copy()
    target = sub["selected_teacher_target_0_100"]
    mos = sub["quality_score_0_100"]
    severe = (mos < 40) & (target > 65)
    fp = (mos < 40) & (target > 60)
    fn = (mos > 75) & (target < 45)
    sub["safety_case_type"] = ""
    sub.loc[fp, "safety_case_type"] = "FP"
    sub.loc[severe, "safety_case_type"] = "severe_fp"
    sub.loc[fn, "safety_case_type"] = "FN"
    keep = [
        "safety_case_type",
        "dataset",
        "split",
        "content_type",
        "image_path",
        "quality_score_0_100",
        "selected_teacher_target_0_100",
        "selected_teacher_source",
        "mdiqa_variant",
        "mdiqa_pred_0_100",
        "production_pred_0_100",
        "policy_f_shift_cap3_applied_shift",
    ]
    return sub[sub["safety_case_type"].ne("")][keep].copy()


def build_range_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    mdiqa = df[df["teacher_source_policy_a"].isin(MDIQA_SOURCES)]["selected_teacher_target_0_100"].dropna().astype(float)
    prod = df[df["teacher_source_policy_a"].isin(PRODUCTION_SOURCES)]["selected_teacher_target_0_100"].dropna().astype(float)
    if len(mdiqa) and len(prod):
        mdiqa_gap = float(np.percentile(mdiqa, 95) - np.percentile(mdiqa, 5))
        prod_gap = float(np.percentile(prod, 95) - np.percentile(prod, 5))
        std_ratio = float(mdiqa.std() / prod.std()) if float(prod.std()) != 0.0 else np.inf
        rows.append(
            {
                "audit": "mdiqa_vs_production_source_range",
                "group_a": "mdiqa_sources",
                "group_b": "production_sources",
                "mean_a": float(mdiqa.mean()),
                "mean_b": float(prod.mean()),
                "mean_shift": float(abs(mdiqa.mean() - prod.mean())),
                "std_a": float(mdiqa.std()),
                "std_b": float(prod.std()),
                "std_ratio": std_ratio,
                "p05_a": float(np.percentile(mdiqa, 5)),
                "p05_b": float(np.percentile(prod, 5)),
                "p95_a": float(np.percentile(mdiqa, 95)),
                "p95_b": float(np.percentile(prod, 95)),
                "p95_p05_gap_mismatch": float(abs(mdiqa_gap - prod_gap)),
                "baseline_2k_mean_shift": BASELINE_2K["mean_shift"],
                "baseline_2k_gap_mismatch": BASELINE_2K["gap_mismatch"],
                "delta_vs_2k_mean_shift": float(abs(mdiqa.mean() - prod.mean()) - BASELINE_2K["mean_shift"]),
                "delta_vs_2k_gap_mismatch": float(abs(mdiqa_gap - prod_gap) - BASELINE_2K["gap_mismatch"]),
                "flag_mean_shift_gt10": abs(mdiqa.mean() - prod.mean()) > 10,
                "flag_std_ratio": std_ratio > 2.0 or std_ratio < 0.5,
                "flag_gap_mismatch_gt20": abs(mdiqa_gap - prod_gap) > 20,
            }
        )

    dist = {dataset: distribution(sub["selected_teacher_target_0_100"]) for dataset, sub in df.groupby("dataset")}
    requested_pairs = [
        ("fallback_proxy", "flive"),
        ("fallback_proxy", "koniq10k"),
        ("fallback_proxy", "spaq"),
        ("flive", "koniq10k"),
        ("flive", "spaq"),
        ("koniq10k", "spaq"),
    ]
    for a, b in requested_pairs:
        if a not in dist or b not in dist or dist[a]["n"] == 0 or dist[b]["n"] == 0:
            rows.append({"audit": "dataset_boundary_mean_gap", "group_a": a, "group_b": b, "error": "missing group"})
            continue
        std_ratio = dist[a]["std"] / dist[b]["std"] if dist[b]["std"] else np.inf
        gap_a = dist[a]["p95"] - dist[a]["p05"]
        gap_b = dist[b]["p95"] - dist[b]["p05"]
        rows.append(
            {
                "audit": "dataset_boundary_mean_gap",
                "group_a": a,
                "group_b": b,
                "mean_a": dist[a]["mean"],
                "mean_b": dist[b]["mean"],
                "mean_shift": abs(dist[a]["mean"] - dist[b]["mean"]),
                "std_a": dist[a]["std"],
                "std_b": dist[b]["std"],
                "std_ratio": std_ratio,
                "p05_a": dist[a]["p05"],
                "p05_b": dist[b]["p05"],
                "p95_a": dist[a]["p95"],
                "p95_b": dist[b]["p95"],
                "p95_p05_gap_mismatch": abs(gap_a - gap_b),
                "flag_mean_shift_gt10": abs(dist[a]["mean"] - dist[b]["mean"]) > 10,
                "flag_std_ratio": std_ratio > 2.0 or std_ratio < 0.5,
                "flag_gap_mismatch_gt20": abs(gap_a - gap_b) > 20,
            }
        )

    total = len(df)
    unclipped = df["selected_teacher_target_unclipped_0_100"].astype(float)
    target = df["selected_teacher_target_0_100"].astype(float)
    clipped_count = int(((unclipped < 0) | (unclipped > 100)).sum())
    rows.append(
        {
            "audit": "boundary_and_clipping",
            "group_a": SELECTED_POLICY,
            "rows_total": total,
            "clipped_count": clipped_count,
            "clipped_pct": clipped_count / total if total else np.nan,
            "near_0_or_100_count": int(((target <= 1) | (target >= 99)).sum()),
            "near_0_or_100_pct": float(((target <= 1) | (target >= 99)).mean()) if total else np.nan,
            "near_5_or_95_count": int(((target <= 5) | (target >= 95)).sum()),
            "near_5_or_95_pct": float(((target <= 5) | (target >= 95)).mean()) if total else np.nan,
            "flag_clipped_pct_gt5": (clipped_count / total) > 0.05 if total else False,
        }
    )
    return pd.DataFrame(rows)


def build_policy_summary(
    df: pd.DataFrame,
    validation: pd.DataFrame,
    safety_cases: pd.DataFrame,
    range_audit: pd.DataFrame,
    params: dict[str, Any],
    failures: pd.DataFrame,
    smoke: bool,
) -> tuple[pd.DataFrame, str]:
    rows = []
    source_range = range_audit[range_audit["audit"].eq("mdiqa_vs_production_source_range")]
    source_row = source_range.iloc[0].to_dict() if len(source_range) else {}
    valid_rows = int(df["target_available"].sum())
    severe_fp = int((safety_cases["safety_case_type"].eq("severe_fp")).sum()) if len(safety_cases) else 0
    fp = int((safety_cases["safety_case_type"].isin(["FP", "severe_fp"])).sum()) if len(safety_cases) else 0
    fn = int((safety_cases["safety_case_type"].eq("FN")).sum()) if len(safety_cases) else 0
    clipped_count = int(range_audit.loc[range_audit["audit"].eq("boundary_and_clipping"), "clipped_count"].fillna(0).sum())
    clipped_pct = clipped_count / len(df) if len(df) else np.nan

    dataset_metrics = validation[validation["section"].eq("mos_alignment")].copy()
    srcc_drop_flags = []
    for dataset, baseline_key in [("koniq10k", "koniq10k_srcc"), ("spaq", "spaq_srcc"), ("flive", "flive_srcc")]:
        row = dataset_metrics[dataset_metrics["dataset"].eq(dataset)]
        srcc = safe_float(row.iloc[0]["srcc"]) if len(row) else np.nan
        drop = BASELINE_2K[baseline_key] - srcc if math.isfinite(srcc) else np.nan
        srcc_drop_flags.append(math.isfinite(drop) and drop > 0.02)
        rows.append(
            {
                "metric": f"{dataset}_srcc",
                "value": srcc,
                "baseline_2k_policy_f_shift_cap3": BASELINE_2K[baseline_key],
                "delta_vs_2k": srcc - BASELINE_2K[baseline_key] if math.isfinite(srcc) else np.nan,
            }
        )

    range_no_worse = (
        safe_float(source_row.get("mean_shift")) <= BASELINE_2K["mean_shift"]
        and safe_float(source_row.get("p95_p05_gap_mismatch")) <= BASELINE_2K["gap_mismatch"]
    )
    rows.extend(
        [
            {"metric": "selected_policy", "value": SELECTED_POLICY},
            {"metric": "valid_rows", "value": valid_rows},
            {"metric": "row_total", "value": len(df)},
            {"metric": "applied_shift", "value": params.get("applied_shift")},
            {"metric": "source_mean_shift", "value": source_row.get("mean_shift"), "baseline_2k_policy_f_shift_cap3": BASELINE_2K["mean_shift"], "delta_vs_2k": source_row.get("delta_vs_2k_mean_shift")},
            {"metric": "source_std_ratio", "value": source_row.get("std_ratio")},
            {"metric": "source_gap_mismatch", "value": source_row.get("p95_p05_gap_mismatch"), "baseline_2k_policy_f_shift_cap3": BASELINE_2K["gap_mismatch"], "delta_vs_2k": source_row.get("delta_vs_2k_gap_mismatch")},
            {"metric": "severe_fp", "value": severe_fp},
            {"metric": "fp", "value": fp},
            {"metric": "fn", "value": fn},
            {"metric": "clipping_count", "value": clipped_count},
            {"metric": "clipping_pct", "value": clipped_pct},
            {"metric": "failure_rows", "value": len(failures)},
            {"metric": "range_no_worse_than_2k", "value": range_no_worse},
        ]
    )
    summary = pd.DataFrame(rows)

    if smoke:
        decision = "SMOKE_ONLY_NOT_A_FULL_10K_DECISION"
    elif len(failures) and valid_rows == 0:
        decision = "HYBRID_LABEL_10K_DRYRUN_FAILED"
    elif valid_rows < MIN_VALID_FOR_READY:
        decision = "HYBRID_LABEL_10K_NEEDS_REAL_MOBILE_FALLBACK_DATA"
    elif severe_fp > 0 or fp > BASELINE_2K["fp"] + max(5, int(0.005 * valid_rows)):
        decision = "HYBRID_LABEL_10K_TOO_RISKY"
    elif clipped_pct > 0.05 or any(srcc_drop_flags) or not range_no_worse:
        decision = "HYBRID_LABEL_10K_NEEDS_MORE_CALIBRATION"
    else:
        decision = "HYBRID_LABEL_10K_READY_FOR_FULL_LABEL_DRYRUN"
    return summary, decision


def write_report(
    paths: dict[str, Path],
    smoke: bool,
    df: pd.DataFrame,
    validation: pd.DataFrame,
    distribution_df: pd.DataFrame,
    safety_cases: pd.DataFrame,
    range_audit: pd.DataFrame,
    policy_summary: pd.DataFrame,
    failures: pd.DataFrame,
    decision: str,
    params: dict[str, Any],
) -> None:
    counts = df.groupby(["dataset", "selected_teacher_source"], dropna=False).size().reset_index(name="n")
    metric_rows = validation[validation["section"].eq("mos_alignment")].copy()
    safety_counts = safety_cases.groupby(["dataset", "safety_case_type"], dropna=False).size().reset_index(name="n") if len(safety_cases) else pd.DataFrame(columns=["dataset", "safety_case_type", "n"])
    source_range = range_audit[range_audit["audit"].eq("mdiqa_vs_production_source_range")]
    source_row = source_range.iloc[0] if len(source_range) else pd.Series(dtype=object)

    lines = [
        "# Phase 5 policy_f_shift_cap3 10k Hybrid Teacher-label Dry-run Report",
        "",
        "## 1. Executive Summary",
        f"- Run mode: `{'smoke' if smoke else 'full_10k'}`.",
        f"- Final decision: `{decision}`.",
        f"- Selected policy: `{SELECTED_POLICY}`.",
        f"- Rows: `{len(df)}` total, `{int(df['target_available'].sum())}` valid.",
        f"- Applied MDIQA shift: `{params.get('applied_shift')}`.",
        "",
        "## 2. Inputs",
        f"- `{PHASE4_2K_MANIFEST}`",
        f"- `{MDIQA_PREDS}`",
        f"- `{PAIRED_PREDS}`",
        f"- `{FLIVE_PROD}`",
        f"- `{BASE_MANIFEST}`",
        f"- `{RECOMMENDED_POLICY}`",
        "",
        "## 3. Confirmed Policy",
        "- KonIQ-10k: MDIQA seed1 plus cap3 source-step smoothing.",
        "- SPAQ: MDIQA xd plus cap3 source-step smoothing.",
        "- FLIVE: Production technical score.",
        "- Fallback/Unknown: Production safety fallback.",
        "",
        "## 4. Dataset and Source Counts",
        write_table(counts.to_dict(orient="records"), ["dataset", "selected_teacher_source", "n"]),
        "",
        "## 5. Target Distribution Summary",
        f"- Distribution CSV: `{paths['distribution']}` rows: `{len(distribution_df)}`.",
        "",
        "## 6. MOS Alignment",
    ]
    for _, row in metric_rows.iterrows():
        lines.append(
            f"- {row.get('dataset')}: n={int(row.get('n', 0))}, SRCC={safe_float(row.get('srcc')):.4f}, "
            f"PLCC={safe_float(row.get('plcc')):.4f}, MAE={safe_float(row.get('mae')):.4f}, "
            f"aligned_MAE={safe_float(row.get('linear_aligned_mae')):.4f}"
        )
    lines.extend(["", "## 7. Safety Cases"])
    if len(safety_counts):
        lines.append(write_table(safety_counts.to_dict(orient="records"), ["dataset", "safety_case_type", "n"]))
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## 8. Range Calibration Audit",
            f"- Source mean shift: `{safe_float(source_row.get('mean_shift')):.4f}` vs 2k `{BASELINE_2K['mean_shift']:.4f}`.",
            f"- Source std ratio: `{safe_float(source_row.get('std_ratio')):.4f}`.",
            f"- Source p95/p05 gap mismatch: `{safe_float(source_row.get('p95_p05_gap_mismatch')):.4f}` vs 2k `{BASELINE_2K['gap_mismatch']:.4f}`.",
            f"- Range audit CSV: `{paths['range']}`.",
            "",
            "## 9. Fallback Proxy Limitation",
            "- Real mobile/unknown rows were not available in the requested cached sources.",
            "- Fallback proxy uses non-overlapping FLIVE image rows with existing production scores when available.",
            "- This is not representative real mobile fallback data.",
            "",
            "## 10. Failure Cases",
        ]
    )
    if len(failures):
        lines.append(f"- Failure rows: `{len(failures)}`. See `{paths['failures']}`.")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## 11. Policy Summary",
            f"- Summary CSV: `{paths['policy_summary']}`.",
            "",
            "## 12. Final Decision",
            decision,
            "",
            "## 13. Recommended Next Step",
        ]
    )
    if smoke:
        lines.append("- Run the full 10k dry-run command only after reviewing smoke output.")
    elif decision == "HYBRID_LABEL_10K_READY_FOR_FULL_LABEL_DRYRUN":
        lines.append("- Proceed to full-label dry-run planning; do not create final labels yet.")
    elif decision == "HYBRID_LABEL_10K_NEEDS_REAL_MOBILE_FALLBACK_DATA":
        lines.append("- Define or score real mobile/unknown fallback rows before judging the full-label dry-run gate.")
    elif decision == "HYBRID_LABEL_10K_NEEDS_MORE_CALIBRATION":
        lines.append("- Revisit calibration because 10k range/ranking/clipping gate is worse than the 2k cap3 baseline.")
    elif decision == "HYBRID_LABEL_10K_TOO_RISKY":
        lines.append("- Stop larger dry-runs until safety cases are explained and fixed.")
    else:
        lines.append("- Fix dry-run construction and rerun this script only.")
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    output_dir = OUTPUT_DIR_SMOKE if args.smoke else OUTPUT_DIR_FULL
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    failures: list[dict[str, Any]] = []

    load_recommended_policy()
    previous_2k, base, mdiqa, paired, flive_prod = load_inputs()
    allocations = ALLOCATIONS_SMOKE if args.smoke else ALLOCATIONS_FULL
    total_target = sum(ALLOCATIONS_SMOKE.values()) + 5 if args.smoke else TARGET_TOTAL

    dry = build_candidate_pools(base, mdiqa, paired, flive_prod, allocations, total_target, failures)
    dry, params = apply_policy_f_shift_cap3(dry)
    missing = dry[~dry["target_available"]]
    for row in missing.to_dict(orient="records"):
        failures.append(
            {
                "stage": "target_build",
                "dataset": row.get("dataset", ""),
                "image_path": row.get("image_path", ""),
                "error": row.get("target_error", "missing selected teacher target"),
            }
        )

    manifest_cols = [
        "dataset",
        "split",
        "content_type",
        "image_path",
        "source_group",
        "quality_score_0_100",
        "mdiqa_variant",
        "mdiqa_pred_0_100",
        "paired_mdiqa_pred_0_100",
        "production_pred_0_100",
        "teacher_target_policy_a_0_100",
        "teacher_source_policy_a",
        "policy_f_shift_cap3_unclipped_0_100",
        "policy_f_shift_cap3_0_100",
        "policy_f_shift_cap3_applied_shift",
        "selected_teacher_target_unclipped_0_100",
        "selected_teacher_target_0_100",
        "selected_teacher_source",
        "selected_policy",
        "target_available",
        "target_error",
        "original_dataset",
    ]
    for col in manifest_cols:
        if col not in dry.columns:
            dry[col] = np.nan
    manifest = dry[manifest_cols].copy()

    validation = build_validation(manifest)
    distribution_df = build_distribution(manifest)
    safety_cases = build_safety_cases(manifest)
    range_audit = build_range_audit(manifest)
    failure_df = pd.DataFrame(failures, columns=["stage", "dataset", "image_path", "error"])
    policy_summary, decision = build_policy_summary(manifest, validation, safety_cases, range_audit, params, failure_df, args.smoke)

    manifest.to_csv(paths["manifest"], index=False)
    validation.to_csv(paths["validation"], index=False)
    distribution_df.to_csv(paths["distribution"], index=False)
    safety_cases.to_csv(paths["safety"], index=False)
    failure_df.to_csv(paths["failures"], index=False)
    range_audit.to_csv(paths["range"], index=False)
    policy_summary.to_csv(paths["policy_summary"], index=False)
    pd.DataFrame(
        [
            {
                "priority": 1,
                "action": "Run full 10k dry-run after smoke review.",
                "status": "pending" if args.smoke else ("recommended" if decision == "HYBRID_LABEL_10K_READY_FOR_FULL_LABEL_DRYRUN" else "blocked"),
            },
            {
                "priority": 2,
                "action": "Do not run 50k/full label generation until 10k gate passes.",
                "status": "open",
            },
            {
                "priority": 3,
                "action": "Replace fallback proxy with real mobile/unknown data before final label policy.",
                "status": "open",
            },
        ]
    ).to_csv(paths["next"], index=False)

    write_report(paths, args.smoke, manifest, validation, distribution_df, safety_cases, range_audit, policy_summary, failure_df, decision, params)
    for key in ["manifest", "validation", "distribution", "safety", "failures", "range", "policy_summary", "report", "next"]:
        print(f"Wrote {paths[key]}")
    print(f"Rows: {len(manifest)} valid: {int(manifest['target_available'].sum())}")
    print(f"Final decision: {decision}")


if __name__ == "__main__":
    main()
