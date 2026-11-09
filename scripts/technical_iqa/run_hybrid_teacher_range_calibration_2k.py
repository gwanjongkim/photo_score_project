# 하이브리드 teacher target의 2k range calibration 정책을 비교 감사한다.
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_MANIFEST = ROOT / "outputs" / "hybrid_teacher_label_design_review_2k_20260603" / "hybrid_label_dryrun_2k_manifest.csv"
OUTPUT_DIR = ROOT / "outputs" / "hybrid_teacher_range_calibration_2k_20260603"

OUT_MANIFEST = OUTPUT_DIR / "calibrated_2k_manifest.csv"
OUT_PARAMS = OUTPUT_DIR / "calibration_parameters.csv"
OUT_METRICS = OUTPUT_DIR / "calibration_policy_metrics.csv"
OUT_RANGE = OUTPUT_DIR / "calibration_range_audit.csv"
OUT_SAFETY = OUTPUT_DIR / "calibration_safety_cases.csv"
OUT_COMPARE = OUTPUT_DIR / "calibration_policy_comparison.csv"
OUT_FAILURES = OUTPUT_DIR / "calibration_failure_cases.csv"
OUT_RECOMMENDED = OUTPUT_DIR / "recommended_policy.json"
OUT_REPORT = OUTPUT_DIR / "report.md"
OUT_NEXT = OUTPUT_DIR / "next_actions.csv"

SEED = 20260603
FOLDS = 5
SOURCE_TO_DATASET = {
    "mdiqa_seed1": "koniq10k",
    "mdiqa_xd": "spaq",
    "production": "flive",
    "production_safety_fallback": "fallback_proxy",
}
POLICY_TARGETS = {
    "policy_a_identity": "policy_a_target_0_100",
    "policy_b_blend_80_20": "policy_b_target_0_100",
    "policy_c_crossfit_affine": "policy_c_target_0_100",
    "policy_d_group_mean_std": "policy_d_target_0_100",
    "policy_e_global_conservative": "policy_e_target_0_100",
}


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
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    denom = np.sqrt(np.sum(x0**2) * np.sum(y0**2))
    if denom == 0:
        return float("nan")
    return float(np.sum(x0 * y0) / denom)


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


def fit_affine(raw: np.ndarray, target: np.ndarray) -> tuple[float, float, str]:
    raw = np.asarray(raw, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(raw) < 2 or float(np.std(raw)) == 0.0:
        return 1.0, float(np.mean(target) - np.mean(raw)), "mean_shift_fallback"
    slope, intercept = np.polyfit(raw, target, 1)
    note = "polyfit"
    if slope <= 0:
        raw_std = float(np.std(raw, ddof=1))
        target_std = float(np.std(target, ddof=1))
        slope = target_std / raw_std if raw_std else 1.0
        intercept = float(np.mean(target) - slope * np.mean(raw))
        note = "positive_slope_std_fallback"
    return float(slope), float(intercept), note


def distribution(values: pd.Series) -> dict[str, Any]:
    v = values.dropna().astype(float).to_numpy()
    if len(v) == 0:
        return {
            "n": 0,
            "target_min": np.nan,
            "target_max": np.nan,
            "target_mean": np.nan,
            "target_std": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "p99": np.nan,
        }
    return {
        "n": len(v),
        "target_min": float(np.min(v)),
        "target_max": float(np.max(v)),
        "target_mean": float(np.mean(v)),
        "target_std": float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
        "p01": float(np.percentile(v, 1)),
        "p05": float(np.percentile(v, 5)),
        "p25": float(np.percentile(v, 25)),
        "p50": float(np.percentile(v, 50)),
        "p75": float(np.percentile(v, 75)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
    }


def add_policy_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    params: list[dict[str, Any]] = []

    out["policy_a_target_unclipped_0_100"] = out["selected_teacher_target_0_100"].astype(float)
    out["policy_a_target_0_100"] = clip_0_100(out["policy_a_target_unclipped_0_100"])

    out["policy_b_target_unclipped_0_100"] = out["teacher_target_policy_b_0_100"].astype(float)
    out["policy_b_target_0_100"] = np.where(
        out["policy_b_target_unclipped_0_100"].notna(),
        clip_0_100(out["policy_b_target_unclipped_0_100"].fillna(0.0)),
        np.nan,
    )

    out["policy_c_target_unclipped_0_100"] = np.nan
    rng = np.random.default_rng(SEED)
    for source, idx in out.groupby("selected_teacher_source").groups.items():
        idx_arr = np.array(list(idx))
        rng.shuffle(idx_arr)
        folds = np.array_split(idx_arr, FOLDS)
        for fold_id, eval_idx in enumerate(folds):
            train_idx = np.setdiff1d(idx_arr, eval_idx)
            train = out.loc[train_idx].dropna(subset=["selected_teacher_target_0_100", "quality_score_0_100"])
            eval_rows = out.loc[eval_idx]
            if len(train) < 30:
                slope, intercept, note = 1.0, 0.0, "insufficient_rows_identity"
            else:
                slope, intercept, note = fit_affine(
                    train["selected_teacher_target_0_100"].to_numpy(),
                    train["quality_score_0_100"].to_numpy(),
                )
            out.loc[eval_idx, "policy_c_target_unclipped_0_100"] = (
                slope * eval_rows["selected_teacher_target_0_100"].astype(float) + intercept
            )
            params.append(
                {
                    "policy": "policy_c_crossfit_affine",
                    "source": source,
                    "fold": fold_id,
                    "n_train": len(train),
                    "slope": slope,
                    "intercept": intercept,
                    "note": note,
                }
            )
    out["policy_c_target_0_100"] = clip_0_100(out["policy_c_target_unclipped_0_100"])

    out["policy_d_target_unclipped_0_100"] = np.nan
    out["policy_e_target_unclipped_0_100"] = out["selected_teacher_target_0_100"].astype(float)
    for source, sub in out.groupby("selected_teacher_source"):
        raw = sub["selected_teacher_target_0_100"].astype(float).to_numpy()
        mos = sub["quality_score_0_100"].astype(float).to_numpy()
        raw_mean, raw_std = float(np.mean(raw)), float(np.std(raw, ddof=1))
        mos_mean, mos_std = float(np.mean(mos)), float(np.std(mos, ddof=1))
        if raw_std == 0:
            slope_d = 1.0
        else:
            slope_d = mos_std / raw_std
        intercept_d = mos_mean - slope_d * raw_mean
        out.loc[sub.index, "policy_d_target_unclipped_0_100"] = slope_d * sub["selected_teacher_target_0_100"].astype(float) + intercept_d
        params.append(
            {
                "policy": "policy_d_group_mean_std",
                "source": source,
                "fold": "all",
                "n_train": len(sub),
                "slope": slope_d,
                "intercept": intercept_d,
                "raw_mean": raw_mean,
                "raw_std": raw_std,
                "mos_mean": mos_mean,
                "mos_std": mos_std,
                "note": "source_mean_std_to_own_mos",
            }
        )

        if source in {"mdiqa_seed1", "mdiqa_xd"}:
            mean_shift = float(np.clip(mos_mean - raw_mean, -5.0, 5.0))
            if raw_std == 0:
                scale = 1.0
            else:
                scale = float(np.clip(mos_std / raw_std, 0.67, 1.5))
            out.loc[sub.index, "policy_e_target_unclipped_0_100"] = (
                (sub["selected_teacher_target_0_100"].astype(float) - raw_mean) * scale + raw_mean + mean_shift
            )
            params.append(
                {
                    "policy": "policy_e_global_conservative",
                    "source": source,
                    "fold": "all",
                    "n_train": len(sub),
                    "slope": scale,
                    "intercept": raw_mean + mean_shift - scale * raw_mean,
                    "raw_mean": raw_mean,
                    "raw_std": raw_std,
                    "mos_mean": mos_mean,
                    "mos_std": mos_std,
                    "mean_shift_applied": mean_shift,
                    "note": "mdiqa_only_capped_mean_std_shift",
                }
            )
        else:
            params.append(
                {
                    "policy": "policy_e_global_conservative",
                    "source": source,
                    "fold": "all",
                    "n_train": len(sub),
                    "slope": 1.0,
                    "intercept": 0.0,
                    "raw_mean": raw_mean,
                    "raw_std": raw_std,
                    "mos_mean": mos_mean,
                    "mos_std": mos_std,
                    "mean_shift_applied": 0.0,
                    "note": "production_source_unchanged",
                }
            )
    out["policy_d_target_0_100"] = clip_0_100(out["policy_d_target_unclipped_0_100"])
    out["policy_e_target_0_100"] = clip_0_100(out["policy_e_target_unclipped_0_100"])
    return out, pd.DataFrame(params)


def safety_counts(df: pd.DataFrame, target_col: str) -> dict[str, int]:
    sub = df.dropna(subset=[target_col, "quality_score_0_100"])
    severe = (sub["quality_score_0_100"] < 40) & (sub[target_col] > 65)
    fp = (sub["quality_score_0_100"] < 40) & (sub[target_col] > 60)
    fn = (sub["quality_score_0_100"] > 75) & (sub[target_col] < 45)
    return {"severe_fp": int(severe.sum()), "fp": int(fp.sum()), "fn": int(fn.sum())}


def clipping_counts(df: pd.DataFrame, unclipped_col: str, clipped_col: str) -> dict[str, Any]:
    sub = df.dropna(subset=[unclipped_col, clipped_col])
    clipped = ((sub[unclipped_col] < 0) | (sub[unclipped_col] > 100))
    boundary = ((sub[clipped_col] <= 1) | (sub[clipped_col] >= 99))
    return {
        "clipping_count": int(clipped.sum()),
        "clipping_pct": float(clipped.mean()) if len(sub) else np.nan,
        "boundary_0_or_100_count": int(boundary.sum()),
        "boundary_0_or_100_pct": float(boundary.mean()) if len(sub) else np.nan,
    }


def build_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy, target_col in POLICY_TARGETS.items():
        unclipped_col = target_col.replace("_target_0_100", "_target_unclipped_0_100")
        for group_cols in [["dataset"], ["dataset", "selected_teacher_source"]]:
            for key, sub in df.groupby(group_cols, dropna=False):
                key_tuple = key if isinstance(key, tuple) else (key,)
                group = {col: val for col, val in zip(group_cols, key_tuple)}
                valid = sub.dropna(subset=[target_col, "quality_score_0_100"])
                pred = valid[target_col].astype(float).to_numpy()
                mos = valid["quality_score_0_100"].astype(float).to_numpy()
                row = {"policy": policy, "grouping": "+".join(group_cols), **group}
                row.update(distribution(valid[target_col]))
                row.update(
                    {
                        "srcc": spearman(pred, mos) if len(valid) else np.nan,
                        "plcc": pearson(pred, mos) if len(valid) else np.nan,
                        "mae": float(np.mean(np.abs(pred - mos))) if len(valid) else np.nan,
                        "linear_aligned_mae": linear_aligned_mae(pred, mos) if len(valid) else np.nan,
                    }
                )
                row.update(safety_counts(valid, target_col))
                row.update(clipping_counts(sub, unclipped_col, target_col))
                rows.append(row)
    return pd.DataFrame(rows)


def build_safety_cases(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy, target_col in POLICY_TARGETS.items():
        sub = df.dropna(subset=[target_col, "quality_score_0_100"])
        for _, row in sub.iterrows():
            case_type = ""
            if row["quality_score_0_100"] < 40 and row[target_col] > 65:
                case_type = "severe_fp"
            elif row["quality_score_0_100"] < 40 and row[target_col] > 60:
                case_type = "FP"
            elif row["quality_score_0_100"] > 75 and row[target_col] < 45:
                case_type = "FN"
            if case_type:
                rows.append(
                    {
                        "policy": policy,
                        "dataset": row["dataset"],
                        "selected_teacher_source": row["selected_teacher_source"],
                        "image_path": row["image_path"],
                        "quality_score_0_100": row["quality_score_0_100"],
                        "target_0_100": row[target_col],
                        "safety_case_type": case_type,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "policy",
            "dataset",
            "selected_teacher_source",
            "image_path",
            "quality_score_0_100",
            "target_0_100",
            "safety_case_type",
        ],
    )


def range_row(policy: str, audit: str, group_a: str, group_b: str, a: pd.Series, b: pd.Series) -> dict[str, Any]:
    a = a.dropna().astype(float)
    b = b.dropna().astype(float)
    if len(a) == 0 or len(b) == 0:
        return {
            "policy": policy,
            "audit": audit,
            "group_a": group_a,
            "group_b": group_b,
            "mean_shift": np.nan,
            "std_ratio": np.nan,
            "p95_p05_gap_mismatch": np.nan,
        }
    a_gap = float(np.percentile(a, 95) - np.percentile(a, 5))
    b_gap = float(np.percentile(b, 95) - np.percentile(b, 5))
    std_ratio = float(a.std() / b.std()) if float(b.std()) else np.inf
    return {
        "policy": policy,
        "audit": audit,
        "group_a": group_a,
        "group_b": group_b,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_shift": float(abs(a.mean() - b.mean())),
        "std_a": float(a.std()),
        "std_b": float(b.std()),
        "std_ratio": std_ratio,
        "p05_a": float(np.percentile(a, 5)),
        "p05_b": float(np.percentile(b, 5)),
        "p95_a": float(np.percentile(a, 95)),
        "p95_b": float(np.percentile(b, 95)),
        "p95_p05_gap_mismatch": float(abs(a_gap - b_gap)),
        "flag_mean_shift_gt10": abs(a.mean() - b.mean()) > 10,
        "flag_std_ratio": std_ratio > 2.0 or std_ratio < 0.5,
        "flag_gap_mismatch_gt20": abs(a_gap - b_gap) > 20,
    }


def build_range_audit(df: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs = [
        ("fallback_proxy", "flive"),
        ("fallback_proxy", "koniq10k"),
        ("fallback_proxy", "spaq"),
        ("flive", "koniq10k"),
        ("flive", "spaq"),
        ("koniq10k", "spaq"),
    ]
    for policy, target_col in POLICY_TARGETS.items():
        mdiqa = df[df["selected_teacher_source"].isin(["mdiqa_seed1", "mdiqa_xd"])][target_col]
        prod = df[df["selected_teacher_source"].isin(["production", "production_safety_fallback"])][target_col]
        rows.append(range_row(policy, "mdiqa_vs_production_source_range", "mdiqa_sources", "production_sources", mdiqa, prod))
        for a_name, b_name in pairs:
            rows.append(
                range_row(
                    policy,
                    "dataset_boundary_mean_gap",
                    a_name,
                    b_name,
                    df[df["dataset"] == a_name][target_col],
                    df[df["dataset"] == b_name][target_col],
                )
            )
        policy_metrics = metrics[(metrics["policy"] == policy) & (metrics["grouping"] == "dataset")]
        severe_fp = int(policy_metrics["severe_fp"].sum())
        clipped = int(policy_metrics["clipping_count"].sum())
        n = int(policy_metrics["n"].sum())
        boundary = int(policy_metrics["boundary_0_or_100_count"].sum())
        baseline = metrics[(metrics["policy"] == "policy_a_identity") & (metrics["grouping"] == "dataset")]
        max_srcc_drop = 0.0
        for _, row in policy_metrics.iterrows():
            base = baseline[baseline["dataset"] == row["dataset"]]
            if len(base) and math.isfinite(safe_float(row["srcc"])) and math.isfinite(safe_float(base.iloc[0]["srcc"])):
                max_srcc_drop = max(max_srcc_drop, safe_float(base.iloc[0]["srcc"]) - safe_float(row["srcc"]))
        rows.append(
            {
                "policy": policy,
                "audit": "policy_global_safety",
                "group_a": policy,
                "group_b": "",
                "severe_fp": severe_fp,
                "clipping_count": clipped,
                "clipping_pct": clipped / n if n else np.nan,
                "boundary_0_or_100_count": boundary,
                "boundary_0_or_100_pct": boundary / n if n else np.nan,
                "max_srcc_drop_vs_policy_a": max_srcc_drop,
                "flag_clipped_pct_gt5": (clipped / n) > 0.05 if n else False,
                "flag_severe_fp_gt0": severe_fp > 0,
                "flag_srcc_drop_gt_0_02": max_srcc_drop > 0.02,
            }
        )
    return pd.DataFrame(rows)


def build_policy_comparison(metrics: pd.DataFrame, range_audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_metrics = metrics[(metrics["policy"] == "policy_a_identity") & (metrics["grouping"] == "dataset")]
    base_range = range_audit[
        (range_audit["policy"] == "policy_a_identity")
        & (range_audit["audit"] == "mdiqa_vs_production_source_range")
    ].iloc[0]
    for policy in POLICY_TARGETS:
        policy_metrics = metrics[(metrics["policy"] == policy) & (metrics["grouping"] == "dataset")]
        policy_range = range_audit[
            (range_audit["policy"] == policy)
            & (range_audit["audit"] == "mdiqa_vs_production_source_range")
        ].iloc[0]
        global_row = range_audit[(range_audit["policy"] == policy) & (range_audit["audit"] == "policy_global_safety")].iloc[0]
        row: dict[str, Any] = {
            "policy": policy,
            "source_mean_shift": policy_range.get("mean_shift"),
            "source_mean_shift_improvement": base_range.get("mean_shift") - policy_range.get("mean_shift"),
            "source_std_ratio": policy_range.get("std_ratio"),
            "source_gap_mismatch": policy_range.get("p95_p05_gap_mismatch"),
            "source_gap_mismatch_improvement": base_range.get("p95_p05_gap_mismatch") - policy_range.get("p95_p05_gap_mismatch"),
            "severe_fp": global_row.get("severe_fp"),
            "clipping_count": global_row.get("clipping_count"),
            "clipping_pct": global_row.get("clipping_pct"),
            "max_srcc_drop_vs_policy_a": global_row.get("max_srcc_drop_vs_policy_a"),
        }
        for _, m in policy_metrics.iterrows():
            dataset = m["dataset"]
            base = base_metrics[base_metrics["dataset"] == dataset]
            base_srcc = safe_float(base.iloc[0]["srcc"]) if len(base) else np.nan
            base_mae = safe_float(base.iloc[0]["mae"]) if len(base) else np.nan
            row[f"{dataset}_srcc"] = m["srcc"]
            row[f"{dataset}_srcc_drop_vs_policy_a"] = base_srcc - safe_float(m["srcc"])
            row[f"{dataset}_mae"] = m["mae"]
            row[f"{dataset}_mae_delta_vs_policy_a"] = safe_float(m["mae"]) - base_mae
        rows.append(row)
    return pd.DataFrame(rows)


def select_policy(comparison: pd.DataFrame, range_audit: pd.DataFrame) -> tuple[dict[str, Any], str]:
    base = comparison[comparison["policy"] == "policy_a_identity"].iloc[0]
    candidates = comparison[comparison["policy"] != "policy_a_identity"].copy()
    eligible = candidates[
        (candidates["severe_fp"] == 0)
        & (candidates["clipping_pct"] <= 0.05)
        & (candidates["max_srcc_drop_vs_policy_a"] <= 0.02)
        & (candidates["source_mean_shift_improvement"] > 2.0)
        & (candidates["source_gap_mismatch_improvement"] > 2.0)
    ].copy()
    fallback_limitation = True
    if len(eligible):
        eligible["selection_score"] = eligible["source_mean_shift_improvement"] + 0.5 * eligible["source_gap_mismatch_improvement"]
        selected = eligible.sort_values("selection_score", ascending=False).iloc[0].to_dict()
        decision = "CALIBRATION_POLICY_READY_FOR_10K_DRYRUN"
        if fallback_limitation:
            selected["fallback_limitation"] = "fallback_proxy_not_real_mobile"
        return selected, decision

    # If calibration helps but misses one hard gate, mark adjustment rather than reverting blindly.
    helps = candidates[
        (candidates["source_mean_shift_improvement"] > 2.0)
        | (candidates["source_gap_mismatch_improvement"] > 2.0)
    ]
    if len(helps):
        selected = helps.sort_values(["source_mean_shift_improvement", "source_gap_mismatch_improvement"], ascending=False).iloc[0].to_dict()
        selected["rejection_reason"] = "helpful_but_failed_selection_gate"
        return selected, "CALIBRATION_POLICY_NEEDS_ADJUSTMENT"

    selected = base.to_dict()
    selected["rejection_reason"] = "calibration_did_not_improve_range_without_risk"
    return selected, "CALIBRATION_REJECTED_KEEP_POLICY_A_WITH_RANGE_RISK"


def fmt(value: Any, digits: int = 4) -> str:
    value = safe_float(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def write_report(
    metrics: pd.DataFrame,
    range_audit: pd.DataFrame,
    comparison: pd.DataFrame,
    params: pd.DataFrame,
    safety_cases: pd.DataFrame,
    failures: pd.DataFrame,
    recommended: dict[str, Any],
    decision: str,
) -> None:
    base_range = range_audit[
        (range_audit["policy"] == "policy_a_identity")
        & (range_audit["audit"] == "mdiqa_vs_production_source_range")
    ].iloc[0]
    selected_policy = recommended.get("policy", "")
    selected_range = range_audit[
        (range_audit["policy"] == selected_policy)
        & (range_audit["audit"] == "mdiqa_vs_production_source_range")
    ]
    if len(selected_range):
        selected_range_row = selected_range.iloc[0]
    else:
        selected_range_row = base_range

    lines = [
        "# Hybrid Teacher Range Calibration 2k Audit Report",
        "",
        "## 1. Executive Summary",
        f"- Final decision: `{decision}`.",
        f"- Selected/recommended policy: `{selected_policy}`.",
        f"- Baseline source mean shift/std ratio/gap mismatch: `{fmt(base_range['mean_shift'])}` / `{fmt(base_range['std_ratio'])}` / `{fmt(base_range['p95_p05_gap_mismatch'])}`.",
        f"- Recommended source mean shift/std ratio/gap mismatch: `{fmt(selected_range_row['mean_shift'])}` / `{fmt(selected_range_row['std_ratio'])}` / `{fmt(selected_range_row['p95_p05_gap_mismatch'])}`.",
        "- This is a calibration design dry-run only; no final labels were created.",
        "",
        "## 2. Sources Inspected",
        f"- `{INPUT_MANIFEST}`",
        "- `outputs/hybrid_teacher_label_design_review_2k_20260603/report.md`",
        "- `outputs/hybrid_teacher_label_design_review_2k_20260603/range_calibration_audit.csv`",
        "- `outputs/paired_production_mdiqa_significance_expanded_20260603/paired_predictions_expanded.csv`",
        "- `outputs/mdiqa_correct_protocol_full_benchmark_20260601/predictions.csv`",
        "- `outputs/technical_iqa_combo_benchmark_20260524/fusion_predictions_flive.csv`",
        "",
        "## 3. Baseline Range Problem",
        f"- Policy A MDIQA vs production mean shift: `{fmt(base_range['mean_shift'])}`.",
        f"- Policy A std ratio: `{fmt(base_range['std_ratio'])}`.",
        f"- Policy A p95/p05 gap mismatch: `{fmt(base_range['p95_p05_gap_mismatch'])}`.",
        "",
        "## 4. Calibration Policies Evaluated",
        "- Policy A: identity strict hybrid baseline.",
        "- Policy B: 80/20 conservative blend where production exists.",
        "- Policy C: source-wise 5-fold cross-fit affine MOS calibration.",
        "- Policy D: source-wise group mean/std alignment to own MOS distribution.",
        "- Policy E: MDIQA-only capped conservative mean/std calibration.",
        "",
        "## 5. Calibration Parameters",
        f"- Parameters CSV: `{OUT_PARAMS}`.",
        f"- Parameter rows: `{len(params)}`.",
        "",
        "## 6. Policy Metrics Against MOS",
    ]
    dataset_metrics = metrics[metrics["grouping"] == "dataset"]
    for _, row in dataset_metrics.iterrows():
        lines.append(
            f"- {row['policy']} / {row['dataset']}: n={int(row['n'])}, SRCC={fmt(row['srcc'])}, "
            f"PLCC={fmt(row['plcc'])}, MAE={fmt(row['mae'])}, severeFP={int(row['severe_fp'])}, "
            f"FP={int(row['fp'])}, FN={int(row['fn'])}, clipped={int(row['clipping_count'])}"
        )

    lines.extend(["", "## 7. Safety Cases"])
    if len(safety_cases):
        counts = safety_cases.groupby(["policy", "safety_case_type"]).size().reset_index(name="n")
        for _, row in counts.iterrows():
            lines.append(f"- {row['policy']} / {row['safety_case_type']}: {row['n']}")
    else:
        lines.append("- No severe FP, FP, or FN cases for any evaluated policy.")

    lines.extend(["", "## 8. Range Calibration Comparison"])
    for _, row in comparison.iterrows():
        lines.append(
            f"- {row['policy']}: mean_shift={fmt(row['source_mean_shift'])}, "
            f"std_ratio={fmt(row['source_std_ratio'])}, gap_mismatch={fmt(row['source_gap_mismatch'])}, "
            f"max_SRCC_drop={fmt(row['max_srcc_drop_vs_policy_a'])}, clipped={int(row['clipping_count'])}"
        )

    lines.extend(["", "## 9. Selected / Rejected Policy"])
    lines.append(json.dumps(recommended, ensure_ascii=False, indent=2, default=str))

    lines.extend(["", "## 10. Remaining Risks"])
    lines.append("- Fallback group remains a FLIVE production proxy, not real mobile/unknown data.")
    lines.append("- Policy C/D use MOS-derived calibration; future use must freeze parameters and validate on larger dry-run rows.")
    lines.append("- Calibration can improve numeric range continuity while preserving rank, but it changes target semantics and should stay diagnostic until 10k dry-run confirms safety.")

    lines.extend(["", "## 11. Final Decision", decision, "", "## 12. Recommended Next Step"])
    if decision == "CALIBRATION_POLICY_READY_FOR_10K_DRYRUN":
        lines.append("- Run a 10k dry-run using the recommended deterministic calibration parameters; do not create final labels yet.")
    elif decision == "CALIBRATION_POLICY_NEEDS_ADJUSTMENT":
        lines.append("- Adjust calibration policy and rerun this 2k audit before any 10k dry-run.")
    elif decision == "CALIBRATION_NEEDS_REAL_MOBILE_FALLBACK_DATA":
        lines.append("- Acquire or define real production-scored mobile/unknown fallback rows before selecting calibration.")
    elif decision == "CALIBRATION_REJECTED_KEEP_POLICY_A_WITH_RANGE_RISK":
        lines.append("- Keep Policy A only as a baseline and document unresolved range risk; do not proceed to larger labels.")
    else:
        lines.append("- Fix script or artifact failures and rerun this calibration audit only.")
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    try:
        manifest = pd.read_csv(INPUT_MANIFEST)
    except Exception as exc:
        failures.append({"stage": "input_load", "error": repr(exc)})
        pd.DataFrame(failures).to_csv(OUT_FAILURES, index=False)
        raise

    required = {"dataset", "quality_score_0_100", "selected_teacher_source", "selected_teacher_target_0_100"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        failures.append({"stage": "input_schema", "error": f"missing columns: {missing}"})

    calibrated, params = add_policy_targets(manifest)
    metrics = build_metrics(calibrated)
    safety_cases = build_safety_cases(calibrated)
    range_audit = build_range_audit(calibrated, metrics)
    comparison = build_policy_comparison(metrics, range_audit)
    recommended, decision = select_policy(comparison, range_audit)

    failure_df = pd.DataFrame(failures, columns=["stage", "error"])
    calibrated.to_csv(OUT_MANIFEST, index=False)
    params.to_csv(OUT_PARAMS, index=False)
    metrics.to_csv(OUT_METRICS, index=False)
    range_audit.to_csv(OUT_RANGE, index=False)
    safety_cases.to_csv(OUT_SAFETY, index=False)
    comparison.to_csv(OUT_COMPARE, index=False)
    failure_df.to_csv(OUT_FAILURES, index=False)
    recommended_payload = {"final_decision": decision, "recommended_policy": recommended}
    OUT_RECOMMENDED.write_text(json.dumps(recommended_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "priority": 1,
                "action": "Run 10k calibrated dry-run only if decision is ready.",
                "status": "recommended" if decision == "CALIBRATION_POLICY_READY_FOR_10K_DRYRUN" else "blocked",
            },
            {
                "priority": 2,
                "action": "Replace fallback proxy with real production-scored mobile/unknown data before final label design.",
                "status": "open",
            },
            {
                "priority": 3,
                "action": "Freeze calibration parameters before any larger dry-run.",
                "status": "open",
            },
        ]
    ).to_csv(OUT_NEXT, index=False)
    write_report(metrics, range_audit, comparison, params, safety_cases, failure_df, recommended, decision)
    print(f"Wrote {OUT_MANIFEST}")
    print(f"Wrote {OUT_PARAMS}")
    print(f"Wrote {OUT_METRICS}")
    print(f"Wrote {OUT_RANGE}")
    print(f"Wrote {OUT_SAFETY}")
    print(f"Wrote {OUT_COMPARE}")
    print(f"Wrote {OUT_FAILURES}")
    print(f"Wrote {OUT_RECOMMENDED}")
    print(f"Wrote {OUT_REPORT}")
    print(f"Final decision: {decision}")


if __name__ == "__main__":
    main()
