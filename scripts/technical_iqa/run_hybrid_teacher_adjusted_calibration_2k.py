# 조정된 하이브리드 teacher calibration 후보를 2k manifest에서 재검증한다.
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_MANIFEST = ROOT / "outputs" / "hybrid_teacher_label_design_review_2k_20260603" / "hybrid_label_dryrun_2k_manifest.csv"
DESIGN_DIR = ROOT / "outputs" / "hybrid_teacher_adjusted_calibration_design_20260603"
OUTPUT_DIR_FULL = ROOT / "outputs" / "hybrid_teacher_adjusted_calibration_2k_20260603"
OUTPUT_DIR_SMOKE = ROOT / "outputs" / "hybrid_teacher_adjusted_calibration_2k_20260603_smoke"

SEED = 20260603
PRODUCTION_SOURCES = {"production", "production_safety_fallback"}
MDIQA_SOURCES = {"mdiqa_seed1", "mdiqa_xd"}
DATASET_PAIRS = [
    ("fallback_proxy", "flive"),
    ("fallback_proxy", "koniq10k"),
    ("fallback_proxy", "spaq"),
    ("flive", "koniq10k"),
    ("flive", "spaq"),
    ("koniq10k", "spaq"),
]


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "manifest": output_dir / "adjusted_calibrated_2k_manifest.csv",
        "params": output_dir / "adjusted_calibration_parameters.csv",
        "metrics": output_dir / "adjusted_calibration_policy_metrics.csv",
        "range": output_dir / "adjusted_calibration_range_audit.csv",
        "safety": output_dir / "adjusted_calibration_safety_cases.csv",
        "comparison": output_dir / "adjusted_calibration_policy_comparison.csv",
        "failures": output_dir / "adjusted_calibration_failure_cases.csv",
        "recommended": output_dir / "recommended_adjusted_policy.json",
        "report": output_dir / "report.md",
        "next": output_dir / "next_actions.csv",
    }


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def clip(values: pd.Series | np.ndarray) -> np.ndarray:
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


def select_smoke_rows(df: pd.DataFrame, rows: int) -> pd.DataFrame:
    per_dataset = max(1, rows // max(1, df["dataset"].nunique()))
    parts = []
    for _, sub in df.groupby("dataset", sort=True):
        parts.append(sub.head(per_dataset))
    smoke = pd.concat(parts, ignore_index=True)
    if len(smoke) < rows:
        rest = df[~df["image_path"].isin(smoke["image_path"])]
        smoke = pd.concat([smoke, rest.head(rows - len(smoke))], ignore_index=True)
    return smoke.head(rows).copy()


def safety_counts(df: pd.DataFrame, target_col: str) -> dict[str, int]:
    sub = df.dropna(subset=[target_col, "quality_score_0_100"])
    severe = (sub["quality_score_0_100"] < 40) & (sub[target_col] > 65)
    fp = (sub["quality_score_0_100"] < 40) & (sub[target_col] > 60)
    fn = (sub["quality_score_0_100"] > 75) & (sub[target_col] < 45)
    return {"severe_fp": int(severe.sum()), "fp": int(fp.sum()), "fn": int(fn.sum())}


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


def clipping_counts(df: pd.DataFrame, unclipped_col: str, clipped_col: str) -> dict[str, Any]:
    sub = df.dropna(subset=[unclipped_col, clipped_col])
    clipped = (sub[unclipped_col] < 0) | (sub[unclipped_col] > 100)
    boundary = (sub[clipped_col] <= 1) | (sub[clipped_col] >= 99)
    return {
        "clipping_count": int(clipped.sum()),
        "clipping_pct": float(clipped.mean()) if len(sub) else np.nan,
        "boundary_0_or_100_count": int(boundary.sum()),
        "boundary_0_or_100_pct": float(boundary.mean()) if len(sub) else np.nan,
    }


def add_target(df: pd.DataFrame, policy: str, values: pd.Series | np.ndarray) -> None:
    df[f"{policy}_unclipped_0_100"] = np.asarray(values, dtype=float)
    df[f"{policy}_target_0_100"] = clip(df[f"{policy}_unclipped_0_100"])


def build_policy_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    out = df.copy()
    params: list[dict[str, Any]] = []
    policies: list[str] = []

    add_target(out, "policy_a_identity", out["selected_teacher_target_0_100"].astype(float))
    policies.append("policy_a_identity")

    b_values = out["teacher_target_policy_b_0_100"].astype(float)
    out["policy_b_blend_80_20_unclipped_0_100"] = b_values
    out["policy_b_blend_80_20_target_0_100"] = np.where(b_values.notna(), clip(b_values.fillna(0)), np.nan)
    policies.append("policy_b_blend_80_20")

    base = out["selected_teacher_target_0_100"].astype(float)
    prod_mean = out[out["selected_teacher_source"].isin(PRODUCTION_SOURCES)]["selected_teacher_target_0_100"].astype(float).mean()
    mdiqa_mean = out[out["selected_teacher_source"].isin(MDIQA_SOURCES)]["selected_teacher_target_0_100"].astype(float).mean()
    raw_shift = prod_mean - mdiqa_mean
    mdiqa_mask = out["selected_teacher_source"].isin(MDIQA_SOURCES)
    for cap in [3, 5, 8]:
        policy = f"policy_f_shift_cap{cap}"
        shift = float(np.clip(raw_shift, -cap, cap))
        values = base.copy()
        values.loc[mdiqa_mask] = values.loc[mdiqa_mask] + shift
        add_target(out, policy, values)
        policies.append(policy)
        params.append(
            {
                "policy": policy,
                "source": "mdiqa_sources",
                "adjustment_type": "constant_shift_toward_production_mean",
                "raw_shift": raw_shift,
                "applied_shift": shift,
                "cap": cap,
            }
        )

    baseline_fp_by_source = {
        source: safety_counts(sub, "policy_a_identity_target_0_100")["fp"]
        for source, sub in out.groupby("selected_teacher_source")
    }
    for fraction in [0.25, 0.50]:
        policy = f"policy_g_residual_{int(fraction * 100)}_cap5"
        values = base.copy()
        for source, sub in out.groupby("selected_teacher_source"):
            bias = float(np.median(sub["selected_teacher_target_0_100"].astype(float) - sub["quality_score_0_100"].astype(float)))
            adjustment = float(np.clip(-fraction * bias, -5.0, 5.0))
            candidate = values.loc[sub.index] + adjustment
            tmp = sub.copy()
            tmp[f"{policy}_candidate"] = clip(candidate)
            candidate_fp = safety_counts(tmp, f"{policy}_candidate")["fp"]
            applied = adjustment if candidate_fp <= baseline_fp_by_source.get(source, 0) else 0.0
            values.loc[sub.index] = values.loc[sub.index] + applied
            params.append(
                {
                    "policy": policy,
                    "source": source,
                    "adjustment_type": "median_residual_fraction",
                    "bias_median_target_minus_mos": bias,
                    "fraction": fraction,
                    "candidate_adjustment": adjustment,
                    "applied_adjustment": applied,
                    "candidate_fp": candidate_fp,
                    "baseline_fp": baseline_fp_by_source.get(source, 0),
                }
            )
        add_target(out, policy, values)
        policies.append(policy)

    f5 = out["policy_f_shift_cap5_target_0_100"].astype(float)
    h_specs = {
        "policy_h_a_fp_clamp": out["policy_a_identity_target_0_100"].astype(float),
        "policy_h_f5_fp_clamp": f5,
    }
    for policy, values in h_specs.items():
        clamped = values.copy()
        low_mos = out["quality_score_0_100"].astype(float) < 40
        clamped.loc[low_mos & (clamped > 60)] = 60.0
        add_target(out, policy, clamped)
        policies.append(policy)
        params.append(
            {
                "policy": policy,
                "source": "all",
                "adjustment_type": "diagnostic_mos_fp_clamp",
                "clamp_rule": "if MOS < 40 and target > 60 then target = 60",
                "diagnostic_only": True,
            }
        )

    for alpha in [0.25, 0.50]:
        policy = f"policy_i_zsoft_{int(alpha * 100)}_cap5"
        values = base.copy()
        for dataset, sub in out.groupby("dataset"):
            raw = sub["selected_teacher_target_0_100"].astype(float)
            mos = sub["quality_score_0_100"].astype(float)
            raw_mean = float(raw.mean())
            raw_std = float(raw.std())
            mos_mean = float(mos.mean())
            mos_std = float(mos.std())
            ratio = mos_std / raw_std if raw_std else 1.0
            z_target = raw_mean + (raw - raw_mean) * ratio + (mos_mean - raw_mean)
            adjustment = alpha * (z_target - raw)
            adjustment = adjustment.clip(lower=-5.0, upper=5.0)
            values.loc[sub.index] = raw + adjustment
            params.append(
                {
                    "policy": policy,
                    "source": dataset,
                    "adjustment_type": "dataset_local_soft_zscore",
                    "alpha": alpha,
                    "cap": 5,
                    "raw_mean": raw_mean,
                    "raw_std": raw_std,
                    "mos_mean": mos_mean,
                    "mos_std": mos_std,
                    "std_ratio": ratio,
                }
            )
        add_target(out, policy, values)
        policies.append(policy)

    return out, pd.DataFrame(params), policies


def build_metrics(df: pd.DataFrame, policies: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy in policies:
        target_col = f"{policy}_target_0_100"
        raw_col = f"{policy}_unclipped_0_100"
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
                row.update(clipping_counts(sub, raw_col, target_col))
                rows.append(row)
    return pd.DataFrame(rows)


def build_safety_cases(df: pd.DataFrame, policies: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy in policies:
        target_col = f"{policy}_target_0_100"
        for _, row in df.dropna(subset=[target_col, "quality_score_0_100"]).iterrows():
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
        columns=["policy", "dataset", "selected_teacher_source", "image_path", "quality_score_0_100", "target_0_100", "safety_case_type"],
    )


def range_row(policy: str, audit: str, group_a: str, group_b: str, a: pd.Series, b: pd.Series) -> dict[str, Any]:
    a = a.dropna().astype(float)
    b = b.dropna().astype(float)
    if len(a) == 0 or len(b) == 0:
        return {"policy": policy, "audit": audit, "group_a": group_a, "group_b": group_b}
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


def build_range_audit(df: pd.DataFrame, metrics: pd.DataFrame, policies: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline = metrics[(metrics["policy"] == "policy_a_identity") & (metrics["grouping"] == "dataset")]
    for policy in policies:
        target_col = f"{policy}_target_0_100"
        rows.append(
            range_row(
                policy,
                "mdiqa_vs_production_source_range",
                "mdiqa_sources",
                "production_sources",
                df[df["selected_teacher_source"].isin(MDIQA_SOURCES)][target_col],
                df[df["selected_teacher_source"].isin(PRODUCTION_SOURCES)][target_col],
            )
        )
        for a_name, b_name in DATASET_PAIRS:
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
        fp = int(policy_metrics["fp"].sum())
        clipped = int(policy_metrics["clipping_count"].sum())
        n = int(policy_metrics["n"].sum())
        max_drop = 0.0
        for _, row in policy_metrics.iterrows():
            base = baseline[baseline["dataset"] == row["dataset"]]
            if len(base) and math.isfinite(safe_float(base.iloc[0]["srcc"])) and math.isfinite(safe_float(row["srcc"])):
                max_drop = max(max_drop, safe_float(base.iloc[0]["srcc"]) - safe_float(row["srcc"]))
        rows.append(
            {
                "policy": policy,
                "audit": "policy_global_safety",
                "group_a": policy,
                "severe_fp": severe_fp,
                "fp": fp,
                "clipping_count": clipped,
                "clipping_pct": clipped / n if n else np.nan,
                "max_srcc_drop_vs_policy_a": max_drop,
                "flag_clipped_pct_gt5": clipped / n > 0.05 if n else False,
                "flag_severe_fp_gt0": severe_fp > 0,
                "flag_fp_gt0": fp > 0,
                "flag_srcc_drop_gt_0_02": max_drop > 0.02,
            }
        )
    return pd.DataFrame(rows)


def build_comparison(metrics: pd.DataFrame, range_audit: pd.DataFrame, policies: list[str]) -> pd.DataFrame:
    base_metrics = metrics[(metrics["policy"] == "policy_a_identity") & (metrics["grouping"] == "dataset")]
    base_range = range_audit[(range_audit["policy"] == "policy_a_identity") & (range_audit["audit"] == "mdiqa_vs_production_source_range")].iloc[0]
    rows = []
    for policy in policies:
        policy_metrics = metrics[(metrics["policy"] == policy) & (metrics["grouping"] == "dataset")]
        policy_range = range_audit[(range_audit["policy"] == policy) & (range_audit["audit"] == "mdiqa_vs_production_source_range")].iloc[0]
        global_row = range_audit[(range_audit["policy"] == policy) & (range_audit["audit"] == "policy_global_safety")].iloc[0]
        row = {
            "policy": policy,
            "source_mean_shift": policy_range.get("mean_shift"),
            "source_mean_shift_improvement": base_range.get("mean_shift") - policy_range.get("mean_shift"),
            "source_std_ratio": policy_range.get("std_ratio"),
            "source_gap_mismatch": policy_range.get("p95_p05_gap_mismatch"),
            "source_gap_mismatch_improvement": base_range.get("p95_p05_gap_mismatch") - policy_range.get("p95_p05_gap_mismatch"),
            "severe_fp": global_row.get("severe_fp"),
            "fp": global_row.get("fp"),
            "clipping_count": global_row.get("clipping_count"),
            "clipping_pct": global_row.get("clipping_pct"),
            "max_srcc_drop_vs_policy_a": global_row.get("max_srcc_drop_vs_policy_a"),
        }
        for _, m in policy_metrics.iterrows():
            dataset = m["dataset"]
            base = base_metrics[base_metrics["dataset"] == dataset]
            base_srcc = safe_float(base.iloc[0]["srcc"]) if len(base) else np.nan
            row[f"{dataset}_srcc"] = m["srcc"]
            row[f"{dataset}_srcc_drop_vs_policy_a"] = base_srcc - safe_float(m["srcc"])
            row[f"{dataset}_mae"] = m["mae"]
        rows.append(row)
    return pd.DataFrame(rows)


def select_policy(comparison: pd.DataFrame) -> tuple[dict[str, Any], str]:
    policy_b = comparison[comparison["policy"] == "policy_b_blend_80_20"].iloc[0]
    candidates = comparison[~comparison["policy"].isin(["policy_a_identity", "policy_h_a_fp_clamp", "policy_h_f5_fp_clamp"])].copy()
    eligible = candidates[
        (candidates["severe_fp"] == 0)
        & (candidates["fp"] == 0)
        & (candidates["clipping_pct"] <= 0.05)
        & (candidates["max_srcc_drop_vs_policy_a"] <= 0.02)
        & (candidates["source_mean_shift_improvement"] > 0)
        & (candidates["source_gap_mismatch_improvement"] >= 0)
        & (candidates["source_mean_shift_improvement"] > policy_b["source_mean_shift_improvement"])
    ].copy()
    if len(eligible):
        eligible["selection_score"] = eligible["source_mean_shift_improvement"] + 0.5 * eligible["source_gap_mismatch_improvement"]
        selected = eligible.sort_values("selection_score", ascending=False).iloc[0].to_dict()
        return selected, "ADJUSTED_CALIBRATION_READY_FOR_10K_DRYRUN"

    helps = candidates[
        (candidates["source_mean_shift_improvement"] > 0)
        & (candidates["max_srcc_drop_vs_policy_a"] <= 0.02)
        & (candidates["severe_fp"] == 0)
    ].copy()
    if len(helps):
        helps["risk_score"] = helps["fp"].fillna(0) + (helps["source_gap_mismatch_improvement"] < 0).astype(int)
        selected = helps.sort_values(["risk_score", "source_mean_shift_improvement"], ascending=[True, False]).iloc[0].to_dict()
        selected["rejection_reason"] = "improves_some_range_metric_but_fails_recommendation_gate"
        return selected, "ADJUSTED_CALIBRATION_NEEDS_ANOTHER_ITERATION"

    selected = comparison[comparison["policy"] == "policy_a_identity"].iloc[0].to_dict()
    selected["rejection_reason"] = "no_adjusted_policy_improved_range_safely"
    return selected, "ADJUSTED_CALIBRATION_REJECTED_KEEP_POLICY_A_WITH_RANGE_RISK"


def fmt(value: Any, digits: int = 4) -> str:
    value = safe_float(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def write_report(
    paths: dict[str, Path],
    output_dir: Path,
    smoke: bool,
    metrics: pd.DataFrame,
    range_audit: pd.DataFrame,
    comparison: pd.DataFrame,
    params: pd.DataFrame,
    safety_cases: pd.DataFrame,
    failures: pd.DataFrame,
    recommended: dict[str, Any],
    decision: str,
) -> None:
    base_range = range_audit[(range_audit["policy"] == "policy_a_identity") & (range_audit["audit"] == "mdiqa_vs_production_source_range")].iloc[0]
    selected_policy = recommended.get("policy", "")
    selected_range = range_audit[(range_audit["policy"] == selected_policy) & (range_audit["audit"] == "mdiqa_vs_production_source_range")]
    selected_range_row = selected_range.iloc[0] if len(selected_range) else base_range
    lines = [
        "# Adjusted Hybrid Teacher Calibration 2k Revalidation Report",
        "",
        "## 1. Executive Summary",
        f"- Run mode: `{'smoke' if smoke else 'full_2k'}`.",
        f"- Final decision: `{decision}`.",
        f"- Selected/recommended policy: `{selected_policy}`.",
        f"- Policy A source mean shift/gap mismatch: `{fmt(base_range.get('mean_shift'))}` / `{fmt(base_range.get('p95_p05_gap_mismatch'))}`.",
        f"- Recommended source mean shift/gap mismatch: `{fmt(selected_range_row.get('mean_shift'))}` / `{fmt(selected_range_row.get('p95_p05_gap_mismatch'))}`.",
        "",
        "## 2. Sources Inspected",
        f"- `{INPUT_MANIFEST}`",
        f"- `{DESIGN_DIR / 'previous_calibration_failure_analysis.md'}`",
        f"- `{DESIGN_DIR / 'adjusted_calibration_policy_spec.md'}`",
        "",
        "## 3. Previous Calibration Failure Analysis",
        "- Prior strong calibration reduced mean shift but expanded MDIQA-source tails and created two SPAQ FP cases.",
        "- Adjusted policies avoid aggressive variance expansion and add FP guards or diagnostic clamps.",
        "",
        "## 4. Adjusted Policies Evaluated",
    ]
    for policy in comparison["policy"]:
        lines.append(f"- `{policy}`")
    lines.extend(["", "## 5. Calibration Parameters", f"- `{paths['params']}` rows: `{len(params)}`.", "", "## 6. Policy Metrics Against MOS"])
    for _, row in metrics[metrics["grouping"] == "dataset"].iterrows():
        lines.append(
            f"- {row['policy']} / {row['dataset']}: n={int(row['n'])}, SRCC={fmt(row['srcc'])}, "
            f"PLCC={fmt(row['plcc'])}, MAE={fmt(row['mae'])}, severeFP={int(row['severe_fp'])}, "
            f"FP={int(row['fp'])}, FN={int(row['fn'])}, clipped={int(row['clipping_count'])}"
        )
    lines.extend(["", "## 7. Safety Cases"])
    if len(safety_cases):
        for _, row in safety_cases.groupby(["policy", "safety_case_type"]).size().reset_index(name="n").iterrows():
            lines.append(f"- {row['policy']} / {row['safety_case_type']}: {row['n']}")
    else:
        lines.append("- None.")
    lines.extend(["", "## 8. Range Calibration Comparison"])
    for _, row in comparison.iterrows():
        lines.append(
            f"- {row['policy']}: mean_shift={fmt(row['source_mean_shift'])}, "
            f"gap_mismatch={fmt(row['source_gap_mismatch'])}, max_SRCC_drop={fmt(row['max_srcc_drop_vs_policy_a'])}, "
            f"FP={int(row['fp'])}, clipped={int(row['clipping_count'])}"
        )
    lines.extend(["", "## 9. Selected / Rejected Policy", json.dumps(recommended, ensure_ascii=False, indent=2, default=str)])
    lines.extend(
        [
            "",
            "## 10. Remaining Risks",
            "- Fallback proxy remains FLIVE-like, not real mobile/unknown data.",
            "- MOS-guarded policies are diagnostic only and cannot directly label unlabeled future rows.",
            "- Full 2k revalidation is required before any 10k dry-run.",
            "",
            "## 11. Final Decision",
            decision,
            "",
            "## 12. Recommended Next Step",
        ]
    )
    if smoke:
        lines.append("- Run the full 2k revalidation command after reviewing smoke output.")
    elif decision == "ADJUSTED_CALIBRATION_READY_FOR_10K_DRYRUN":
        lines.append("- Proceed to a 10k dry-run using the recommended adjusted policy; do not create final labels yet.")
    elif decision == "ADJUSTED_CALIBRATION_NEEDS_ANOTHER_ITERATION":
        lines.append("- Iterate adjusted calibration before 10k dry-run.")
    elif decision == "ADJUSTED_CALIBRATION_NEEDS_REAL_MOBILE_FALLBACK_DATA":
        lines.append("- Define real mobile/unknown fallback rows before selecting calibration.")
    elif decision == "ADJUSTED_CALIBRATION_REJECTED_KEEP_POLICY_A_WITH_RANGE_RISK":
        lines.append("- Keep Policy A as baseline but do not proceed to larger labels while range risk remains unresolved.")
    else:
        lines.append("- Fix revalidation failure and rerun this script only.")
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke_rows", type=int, default=50)
    args = parser.parse_args()

    output_dir = OUTPUT_DIR_SMOKE if args.smoke else OUTPUT_DIR_FULL
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    failures: list[dict[str, Any]] = []
    try:
        df = pd.read_csv(INPUT_MANIFEST)
    except Exception as exc:
        failures.append({"stage": "input_load", "error": repr(exc)})
        pd.DataFrame(failures).to_csv(paths["failures"], index=False)
        raise
    if args.smoke:
        df = select_smoke_rows(df, args.smoke_rows)

    calibrated, params, policies = build_policy_targets(df)
    metrics = build_metrics(calibrated, policies)
    safety_cases = build_safety_cases(calibrated, policies)
    range_audit = build_range_audit(calibrated, metrics, policies)
    comparison = build_comparison(metrics, range_audit, policies)
    recommended, decision = select_policy(comparison)
    if args.smoke:
        recommended["full_2k_decision_if_pattern_holds"] = decision
        decision = "SMOKE_ONLY_NOT_A_FULL_2K_DECISION"
        recommended["smoke_only"] = True

    failure_df = pd.DataFrame(failures, columns=["stage", "error"])
    calibrated.to_csv(paths["manifest"], index=False)
    params.to_csv(paths["params"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    range_audit.to_csv(paths["range"], index=False)
    safety_cases.to_csv(paths["safety"], index=False)
    comparison.to_csv(paths["comparison"], index=False)
    failure_df.to_csv(paths["failures"], index=False)
    paths["recommended"].write_text(
        json.dumps({"final_decision": decision, "recommended_policy": recommended}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "priority": 1,
                "action": "Run full 2k adjusted calibration revalidation.",
                "status": "pending" if args.smoke else ("recommended" if decision == "ADJUSTED_CALIBRATION_READY_FOR_10K_DRYRUN" else "blocked"),
            },
            {
                "priority": 2,
                "action": "Do not run 10k until full 2k adjusted revalidation passes.",
                "status": "open",
            },
            {
                "priority": 3,
                "action": "Replace fallback proxy with real mobile/unknown rows before final label design.",
                "status": "open",
            },
        ]
    ).to_csv(paths["next"], index=False)
    write_report(paths, output_dir, args.smoke, metrics, range_audit, comparison, params, safety_cases, failure_df, recommended, decision)
    for key in ["manifest", "params", "metrics", "range", "safety", "comparison", "failures", "recommended", "report"]:
        print(f"Wrote {paths[key]}")
    print(f"Final decision: {decision}")


if __name__ == "__main__":
    main()
