# Phase 4 하이브리드 teacher label 2k dry-run과 range calibration audit를 생성한다.
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "hybrid_teacher_label_design_review_2k_20260603"

PHASE3_REPORT = ROOT / "outputs" / "hybrid_teacher_label_design_review_20260603" / "report.md"
PHASE3_SPEC = ROOT / "outputs" / "hybrid_teacher_label_design_review_20260603" / "hybrid_teacher_label_spec.md"
PAIRED_PREDS = ROOT / "outputs" / "paired_production_mdiqa_significance_expanded_20260603" / "paired_predictions_expanded.csv"
PAIRED_RESULTS = ROOT / "outputs" / "paired_production_mdiqa_significance_expanded_20260603" / "paired_significance_results_expanded.csv"
MDIQA_PREDS = ROOT / "outputs" / "mdiqa_correct_protocol_full_benchmark_20260601" / "predictions.csv"
MDIQA_METRICS = ROOT / "outputs" / "mdiqa_correct_protocol_full_benchmark_20260601" / "dataset_metrics.csv"
FLIVE_PROD = ROOT / "outputs" / "technical_iqa_combo_benchmark_20260524" / "fusion_predictions_flive.csv"
BASE_MANIFEST = ROOT / "outputs" / "research_protocol_iqa_manifests_20260530" / "benchmark_all_research_protocol_iqa.csv"
SELECTOR_SOURCE = ROOT / "src" / "infer" / "select_best_shots.py"

OUT_MANIFEST = OUTPUT_DIR / "hybrid_label_dryrun_2k_manifest.csv"
OUT_VALIDATION = OUTPUT_DIR / "hybrid_label_dryrun_2k_validation.csv"
OUT_DISTRIBUTION = OUTPUT_DIR / "hybrid_label_distribution_summary_2k.csv"
OUT_SAFETY = OUTPUT_DIR / "hybrid_label_safety_cases_2k.csv"
OUT_POLICY = OUTPUT_DIR / "hybrid_label_policy_comparison_2k.csv"
OUT_RANGE_AUDIT = OUTPUT_DIR / "range_calibration_audit.csv"
OUT_RANGE_RECS = OUTPUT_DIR / "range_calibration_recommendations.csv"
OUT_FAILURES = OUTPUT_DIR / "hybrid_label_failure_cases_2k.csv"
OUT_NEXT = OUTPUT_DIR / "next_actions.csv"
OUT_REPORT = OUTPUT_DIR / "report.md"

SEED = 20260603
TARGET_ALLOCATIONS = {
    "koniq10k": 500,
    "spaq": 500,
    "flive": 700,
    "fallback_proxy": 300,
}
PRODUCTION_BASELINES = {
    "flive": 0.574,
    "koniq10k": 0.934,
    "spaq": 0.873,
}
REQUIRED_MANIFEST_COLUMNS = [
    "dataset",
    "split",
    "content_type",
    "image_path",
    "source_group",
    "quality_score_0_100",
    "mdiqa_variant",
    "mdiqa_pred_0_100",
    "production_pred_0_100",
    "teacher_target_policy_a_0_100",
    "teacher_source_policy_a",
    "teacher_target_policy_b_0_100",
    "teacher_source_policy_b",
    "policy_a_minus_policy_b",
    "selected_teacher_target_0_100",
    "selected_teacher_source",
    "target_available",
    "target_error",
]


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


def clip_0_100(value: Any) -> float:
    value_float = safe_float(value)
    if not math.isfinite(value_float):
        return float("nan")
    return float(np.clip(value_float, 0.0, 100.0))


def is_readable_image(path: str) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


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


def metric_row(df: pd.DataFrame, pred_col: str, group: dict[str, Any]) -> dict[str, Any]:
    sub = df.dropna(subset=[pred_col, "quality_score_0_100"]).copy()
    pred = sub[pred_col].astype(float).to_numpy()
    target = sub["quality_score_0_100"].astype(float).to_numpy()
    row = dict(group)
    row.update(
        {
            "n": len(sub),
            "srcc": spearman(pred, target) if len(sub) else float("nan"),
            "plcc": pearson(pred, target) if len(sub) else float("nan"),
            "mae": float(np.mean(np.abs(pred - target))) if len(sub) else float("nan"),
            "linear_aligned_mae": linear_aligned_mae(pred, target) if len(sub) else float("nan"),
        }
    )
    return row


def distribution_rows(df: pd.DataFrame, group_cols: list[str], value_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, sub in df.dropna(subset=[value_col]).groupby(group_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        values = sub[value_col].astype(float).to_numpy()
        row = {col: val for col, val in zip(group_cols, key_tuple)}
        row.update(
            {
                "n": len(values),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                "p01": float(np.percentile(values, 1)),
                "p05": float(np.percentile(values, 5)),
                "p25": float(np.percentile(values, 25)),
                "p50": float(np.percentile(values, 50)),
                "p75": float(np.percentile(values, 75)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
            }
        )
        rows.append(row)
    return rows


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(BASE_MANIFEST)
    mdiqa = pd.read_csv(MDIQA_PREDS)
    paired = pd.read_csv(PAIRED_PREDS)
    flive_prod = pd.read_csv(FLIVE_PROD)
    paired_results = pd.read_csv(PAIRED_RESULTS)

    for df in [base, mdiqa, paired]:
        df["image_path"] = df["image_path"].map(canonical_path)
    base["source_group"] = base["image_path"].map(source_group_from_path)
    return base, mdiqa, paired, flive_prod, paired_results


def sample_rows(pool: pd.DataFrame, n: int, reason: str, failures: list[dict[str, Any]]) -> pd.DataFrame:
    pool = pool.drop_duplicates("image_path").copy()
    if len(pool) < n:
        failures.append(
            {
                "stage": "sampling",
                "dataset": reason,
                "image_path": "",
                "error": f"requested {n}, available {len(pool)}",
            }
        )
    take = min(n, len(pool))
    if take == 0:
        return pool.head(0)
    return pool.sample(n=take, random_state=SEED).copy()


def is_true_flag(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def attach_policy_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["teacher_target_policy_a_unclipped_0_100"] = np.nan
    out["teacher_source_policy_a"] = ""
    out["teacher_target_policy_b_unclipped_0_100"] = np.nan
    out["teacher_source_policy_b"] = ""

    is_koniq = out["dataset"].eq("koniq10k")
    is_spaq = out["dataset"].eq("spaq")
    is_flive = out["dataset"].eq("flive")
    is_fallback = out["dataset"].eq("fallback_proxy")

    out.loc[is_koniq, "teacher_target_policy_a_unclipped_0_100"] = out.loc[is_koniq, "mdiqa_pred_0_100"]
    out.loc[is_koniq, "teacher_source_policy_a"] = "mdiqa_seed1"
    out.loc[is_spaq, "teacher_target_policy_a_unclipped_0_100"] = out.loc[is_spaq, "mdiqa_pred_0_100"]
    out.loc[is_spaq, "teacher_source_policy_a"] = "mdiqa_xd"
    out.loc[is_flive, "teacher_target_policy_a_unclipped_0_100"] = out.loc[is_flive, "production_pred_0_100"]
    out.loc[is_flive, "teacher_source_policy_a"] = "production"
    out.loc[is_fallback, "teacher_target_policy_a_unclipped_0_100"] = out.loc[is_fallback, "production_pred_0_100"]
    out.loc[is_fallback, "teacher_source_policy_a"] = "production_safety_fallback"

    both = out["mdiqa_pred_0_100"].notna() & out["production_pred_0_100"].notna()
    blendable = (is_koniq | is_spaq) & both
    out.loc[blendable, "teacher_target_policy_b_unclipped_0_100"] = (
        0.8 * out.loc[blendable, "mdiqa_pred_0_100"] + 0.2 * out.loc[blendable, "production_pred_0_100"]
    )
    out.loc[blendable, "teacher_source_policy_b"] = "mdiqa_production_blend_80_20"
    prod_only = is_flive | is_fallback
    out.loc[prod_only, "teacher_target_policy_b_unclipped_0_100"] = out.loc[prod_only, "production_pred_0_100"]
    out.loc[prod_only & is_flive, "teacher_source_policy_b"] = "production"
    out.loc[prod_only & is_fallback, "teacher_source_policy_b"] = "production_safety_fallback"
    out.loc[(is_koniq | is_spaq) & ~both, "teacher_source_policy_b"] = "unavailable_missing_pair"

    out["teacher_target_policy_a_0_100"] = out["teacher_target_policy_a_unclipped_0_100"].map(clip_0_100)
    out["teacher_target_policy_b_0_100"] = out["teacher_target_policy_b_unclipped_0_100"].map(clip_0_100)
    out["policy_a_minus_policy_b"] = out["teacher_target_policy_a_0_100"] - out["teacher_target_policy_b_0_100"]
    out["selected_teacher_target_0_100"] = out["teacher_target_policy_a_0_100"]
    out["selected_teacher_source"] = out["teacher_source_policy_a"]
    out["target_available"] = out["selected_teacher_target_0_100"].notna()
    out["target_error"] = ""
    out.loc[~out["target_available"], "target_error"] = "missing selected teacher target"
    return out


def build_manifest() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    failures: list[dict[str, Any]] = []
    base, mdiqa, paired, flive_prod, paired_results = load_inputs()

    mdiqa_map = mdiqa.set_index("image_path")["pred_0_100"].to_dict()
    paired_mdiqa_map = paired.set_index("image_path")["mdiqa_pred_0_100"].to_dict()
    prod_pair_map = paired.set_index("image_path")["production_pred_0_100"].to_dict()
    flive_prod_map = flive_prod.set_index("image_path")["flive_score"].to_dict()

    base = base.copy()
    base["path_exists"] = base["image_path"].map(lambda p: Path(p).exists())
    base = base[base["path_exists"]].copy()
    base["quality_score_0_100"] = pd.to_numeric(base["quality_score_0_100"], errors="coerce")
    base = base.dropna(subset=["quality_score_0_100"]).copy()

    koniq_pool = base[(base["dataset"] == "koniq10k") & (base["split"] == "test")].copy()
    koniq_pool["mdiqa_pred_0_100"] = koniq_pool["image_path"].map(mdiqa_map)
    koniq_pool["production_pred_0_100"] = koniq_pool["image_path"].map(prod_pair_map)
    koniq_pool = koniq_pool.dropna(subset=["mdiqa_pred_0_100"])
    paired_koniq = koniq_pool[koniq_pool["production_pred_0_100"].notna()].drop_duplicates("image_path").copy()
    extra_koniq = koniq_pool[~koniq_pool["image_path"].isin(paired_koniq["image_path"])].drop_duplicates("image_path").copy()
    paired_take = min(TARGET_ALLOCATIONS["koniq10k"], len(paired_koniq))
    extra_take = max(0, TARGET_ALLOCATIONS["koniq10k"] - paired_take)
    koniq_selected = pd.concat(
        [
            sample_rows(paired_koniq, paired_take, "koniq10k_paired", failures),
            sample_rows(
                extra_koniq,
                extra_take,
                "koniq10k_mdiqa_only",
                failures,
            ),
        ],
        ignore_index=True,
    )
    koniq_selected["mdiqa_variant"] = "seed1"

    spaq_pool = base[(base["dataset"] == "spaq") & (base["split"] == "test")].copy()
    spaq_pool["mdiqa_pred_0_100"] = spaq_pool["image_path"].map(paired_mdiqa_map)
    spaq_pool["production_pred_0_100"] = spaq_pool["image_path"].map(prod_pair_map)
    spaq_pool = spaq_pool.dropna(subset=["mdiqa_pred_0_100"])
    spaq_selected = sample_rows(spaq_pool, TARGET_ALLOCATIONS["spaq"], "spaq", failures)
    spaq_selected["mdiqa_variant"] = "xd"

    flive_pool = base[(base["dataset"] == "flive") & (base["content_type"] == "image")].copy()
    flive_pool["prod_key"] = flive_pool["image_path"].map(lambda p: Path(p).name)
    flive_pool["production_pred_0_100"] = flive_pool["prod_key"].map(flive_prod_map)
    flive_pool["mdiqa_pred_0_100"] = flive_pool["image_path"].map(mdiqa_map)
    flive_pool = flive_pool.dropna(subset=["production_pred_0_100"]).copy()
    flive_primary_pool = flive_pool.copy()
    flive_selected = sample_rows(flive_primary_pool, TARGET_ALLOCATIONS["flive"], "flive_image_production", failures)
    flive_selected["mdiqa_variant"] = ""

    fallback_pool = flive_pool[~flive_pool["image_path"].isin(flive_selected["image_path"])].copy()
    fallback_selected = sample_rows(fallback_pool, TARGET_ALLOCATIONS["fallback_proxy"], "fallback_proxy_flive_production", failures)
    fallback_selected["original_dataset"] = fallback_selected["dataset"]
    fallback_selected["dataset"] = "fallback_proxy"
    fallback_selected["source_group"] = "flive_production_proxy__" + fallback_selected["source_group"].astype(str)
    fallback_selected["mdiqa_variant"] = ""

    dry = pd.concat([koniq_selected, spaq_selected, flive_selected, fallback_selected], ignore_index=True)
    dry = dry.drop_duplicates("image_path", keep="first").copy()
    dry = attach_policy_columns(dry)
    dry["content_type"] = dry["content_type"].fillna("image")
    dry["split"] = dry["split"].fillna("")
    dry["target_available"] = dry["target_available"].astype(bool)

    missing_selected = dry[~dry["target_available"]]
    for row in missing_selected.to_dict(orient="records"):
        failures.append(
            {
                "stage": "target_build",
                "dataset": row.get("dataset", ""),
                "image_path": row.get("image_path", ""),
                "error": row.get("target_error", "missing selected teacher target"),
            }
        )

    return dry, pd.DataFrame(failures, columns=["stage", "dataset", "image_path", "error"]), paired_results


def build_validation(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    count_groups = ["dataset", "split", "content_type", "selected_teacher_source"]
    for key, sub in df.groupby(count_groups, dropna=False):
        rows.append(
            {
                "section": "row_count",
                **{col: val for col, val in zip(count_groups, key)},
                "n": len(sub),
                "srcc": np.nan,
                "plcc": np.nan,
                "mae": np.nan,
                "linear_aligned_mae": np.nan,
            }
        )
    for dataset, sub in df.groupby("dataset"):
        rows.append(metric_row(sub, "selected_teacher_target_0_100", {"section": "mos_alignment_policy_a", "dataset": dataset}))
        if sub["teacher_target_policy_b_0_100"].notna().any():
            rows.append(metric_row(sub, "teacher_target_policy_b_0_100", {"section": "mos_alignment_policy_b", "dataset": dataset}))
    return pd.DataFrame(rows)


def build_safety(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detailed: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    policies = [
        ("policy_a", "teacher_target_policy_a_0_100", "teacher_source_policy_a"),
        ("policy_b", "teacher_target_policy_b_0_100", "teacher_source_policy_b"),
        ("selected", "selected_teacher_target_0_100", "selected_teacher_source"),
    ]
    for policy, target_col, source_col in policies:
        sub = df.dropna(subset=[target_col, "quality_score_0_100"]).copy()
        sub["policy"] = policy
        sub["policy_target_0_100"] = sub[target_col]
        sub["policy_teacher_source"] = sub[source_col]
        sub["safety_case_type"] = ""
        severe = (sub["quality_score_0_100"] < 40) & (sub[target_col] > 65)
        fp = (sub["quality_score_0_100"] < 40) & (sub[target_col] > 60)
        fn = (sub["quality_score_0_100"] > 75) & (sub[target_col] < 45)
        sub.loc[fp, "safety_case_type"] = "FP"
        sub.loc[severe, "safety_case_type"] = "severe_fp"
        sub.loc[fn, "safety_case_type"] = "FN"
        cases = sub[sub["safety_case_type"] != ""].copy()
        if len(cases):
            detailed.append(cases)
        for (dataset, teacher_source), group in sub.groupby(["dataset", source_col], dropna=False):
            summary_rows.append(
                {
                    "policy": policy,
                    "dataset": dataset,
                    "teacher_source": teacher_source,
                    "n": len(group),
                    "severe_fp": int(((group["quality_score_0_100"] < 40) & (group[target_col] > 65)).sum()),
                    "fp": int(((group["quality_score_0_100"] < 40) & (group[target_col] > 60)).sum()),
                    "fn": int(((group["quality_score_0_100"] > 75) & (group[target_col] < 45)).sum()),
                }
            )
    if detailed:
        detailed_df = pd.concat(detailed, ignore_index=True)
        keep_cols = [
            "policy",
            "dataset",
            "split",
            "content_type",
            "image_path",
            "quality_score_0_100",
            "policy_target_0_100",
            "policy_teacher_source",
            "safety_case_type",
            "mdiqa_pred_0_100",
            "production_pred_0_100",
        ]
        detailed_df = detailed_df[keep_cols].copy()
    else:
        detailed_df = pd.DataFrame(
            columns=[
                "policy",
                "dataset",
                "split",
                "content_type",
                "image_path",
                "quality_score_0_100",
                "policy_target_0_100",
                "policy_teacher_source",
                "safety_case_type",
                "mdiqa_pred_0_100",
                "production_pred_0_100",
            ]
        )
    return detailed_df, pd.DataFrame(summary_rows)


def build_policy_comparison(df: pd.DataFrame, safety_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, sub in df.groupby("dataset"):
        a = sub.dropna(subset=["teacher_target_policy_a_0_100"])
        b = sub.dropna(subset=["teacher_target_policy_b_0_100"])
        a_metrics = metric_row(a, "teacher_target_policy_a_0_100", {})
        b_metrics = metric_row(b, "teacher_target_policy_b_0_100", {})
        a_safety = safety_summary[(safety_summary["policy"] == "policy_a") & (safety_summary["dataset"] == dataset)]
        b_safety = safety_summary[(safety_summary["policy"] == "policy_b") & (safety_summary["dataset"] == dataset)]
        rows.append(
            {
                "dataset": dataset,
                "policy_a_n": len(a),
                "policy_b_n": len(b),
                "policy_a_mean": float(a["teacher_target_policy_a_0_100"].mean()) if len(a) else np.nan,
                "policy_b_mean": float(b["teacher_target_policy_b_0_100"].mean()) if len(b) else np.nan,
                "policy_a_std": float(a["teacher_target_policy_a_0_100"].std()) if len(a) > 1 else np.nan,
                "policy_b_std": float(b["teacher_target_policy_b_0_100"].std()) if len(b) > 1 else np.nan,
                "mean_delta_a_minus_b": (
                    float(a["teacher_target_policy_a_0_100"].mean() - b["teacher_target_policy_b_0_100"].mean())
                    if len(a) and len(b)
                    else np.nan
                ),
                "srcc_a": a_metrics["srcc"],
                "srcc_b": b_metrics["srcc"],
                "srcc_delta_a_minus_b": a_metrics["srcc"] - b_metrics["srcc"],
                "plcc_a": a_metrics["plcc"],
                "plcc_b": b_metrics["plcc"],
                "plcc_delta_a_minus_b": a_metrics["plcc"] - b_metrics["plcc"],
                "mae_a": a_metrics["mae"],
                "mae_b": b_metrics["mae"],
                "mae_delta_a_minus_b": a_metrics["mae"] - b_metrics["mae"],
                "severe_fp_a": int(a_safety["severe_fp"].sum()) if len(a_safety) else 0,
                "severe_fp_b": int(b_safety["severe_fp"].sum()) if len(b_safety) else 0,
                "fp_a": int(a_safety["fp"].sum()) if len(a_safety) else 0,
                "fp_b": int(b_safety["fp"].sum()) if len(b_safety) else 0,
                "fn_a": int(a_safety["fn"].sum()) if len(a_safety) else 0,
                "fn_b": int(b_safety["fn"].sum()) if len(b_safety) else 0,
            }
        )
    return pd.DataFrame(rows)


def build_range_audit(df: pd.DataFrame, safety_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dist_source = pd.DataFrame(distribution_rows(df, ["selected_teacher_source"], "selected_teacher_target_0_100"))
    dist_dataset = pd.DataFrame(distribution_rows(df, ["dataset"], "selected_teacher_target_0_100"))
    rows: list[dict[str, Any]] = []

    mdiqa = df[df["selected_teacher_source"].isin(["mdiqa_seed1", "mdiqa_xd"])]["selected_teacher_target_0_100"].dropna()
    prod = df[df["selected_teacher_source"].isin(["production", "production_safety_fallback"])]["selected_teacher_target_0_100"].dropna()
    if len(mdiqa) and len(prod):
        mdiqa_std = float(mdiqa.std())
        prod_std = float(prod.std())
        std_ratio = mdiqa_std / prod_std if prod_std else np.inf
        rows.append(
            {
                "audit": "mdiqa_vs_production_source_range",
                "group_a": "mdiqa_sources",
                "group_b": "production_sources",
                "mean_a": float(mdiqa.mean()),
                "mean_b": float(prod.mean()),
                "mean_shift": float(abs(mdiqa.mean() - prod.mean())),
                "std_a": mdiqa_std,
                "std_b": prod_std,
                "std_ratio": std_ratio,
                "p05_a": float(np.percentile(mdiqa, 5)),
                "p05_b": float(np.percentile(prod, 5)),
                "p95_a": float(np.percentile(mdiqa, 95)),
                "p95_b": float(np.percentile(prod, 95)),
                "p95_p05_gap_mismatch": float(abs((np.percentile(mdiqa, 95) - np.percentile(mdiqa, 5)) - (np.percentile(prod, 95) - np.percentile(prod, 5)))),
                "flag_mean_shift_gt10": abs(mdiqa.mean() - prod.mean()) > 10,
                "flag_std_ratio": std_ratio > 2.0 or std_ratio < 0.5,
                "flag_gap_mismatch_gt20": abs((np.percentile(mdiqa, 95) - np.percentile(mdiqa, 5)) - (np.percentile(prod, 95) - np.percentile(prod, 5))) > 20,
            }
        )

    for _, a in dist_dataset.iterrows():
        for _, b in dist_dataset.iterrows():
            if str(a["dataset"]) >= str(b["dataset"]):
                continue
            rows.append(
                {
                    "audit": "dataset_boundary_mean_gap",
                    "group_a": a["dataset"],
                    "group_b": b["dataset"],
                    "mean_a": a["mean"],
                    "mean_b": b["mean"],
                    "mean_shift": abs(a["mean"] - b["mean"]),
                    "std_a": a["std"],
                    "std_b": b["std"],
                    "std_ratio": a["std"] / b["std"] if b["std"] else np.inf,
                    "p05_a": a["p05"],
                    "p05_b": b["p05"],
                    "p95_a": a["p95"],
                    "p95_b": b["p95"],
                    "p95_p05_gap_mismatch": abs((a["p95"] - a["p05"]) - (b["p95"] - b["p05"])),
                    "flag_mean_shift_gt10": abs(a["mean"] - b["mean"]) > 10,
                    "flag_std_ratio": (a["std"] / b["std"] > 2.0 or a["std"] / b["std"] < 0.5) if b["std"] else True,
                    "flag_gap_mismatch_gt20": abs((a["p95"] - a["p05"]) - (b["p95"] - b["p05"])) > 20,
                }
            )

    total = len(df)
    unclipped = df["teacher_target_policy_a_unclipped_0_100"]
    clipped = df["teacher_target_policy_a_0_100"]
    clipped_count = int(((unclipped < 0) | (unclipped > 100)).fillna(False).sum())
    near_bounds = int(((clipped <= 1) | (clipped >= 99)).fillna(False).sum())
    near_soft_bounds = int(((clipped <= 5) | (clipped >= 95)).fillna(False).sum())
    rows.append(
        {
            "audit": "boundary_and_clipping",
            "group_a": "policy_a",
            "group_b": "",
            "mean_a": float(clipped.mean()),
            "mean_b": np.nan,
            "mean_shift": np.nan,
            "std_a": float(clipped.std()),
            "std_b": np.nan,
            "std_ratio": np.nan,
            "p05_a": float(np.percentile(clipped.dropna(), 5)),
            "p05_b": np.nan,
            "p95_a": float(np.percentile(clipped.dropna(), 95)),
            "p95_b": np.nan,
            "p95_p05_gap_mismatch": np.nan,
            "flag_mean_shift_gt10": False,
            "flag_std_ratio": False,
            "flag_gap_mismatch_gt20": False,
            "rows_total": total,
            "clipped_count": clipped_count,
            "clipped_pct": clipped_count / total if total else np.nan,
            "near_0_or_100_count": near_bounds,
            "near_0_or_100_pct": near_bounds / total if total else np.nan,
            "near_5_or_95_count": near_soft_bounds,
            "near_5_or_95_pct": near_soft_bounds / total if total else np.nan,
            "flag_clipped_pct_gt5": (clipped_count / total) > 0.05 if total else False,
        }
    )

    severe_fp = int(
        safety_summary[(safety_summary["policy"] == "selected")]["severe_fp"].sum()
        if len(safety_summary)
        else 0
    )
    rows.append(
        {
            "audit": "selected_policy_safety",
            "group_a": "selected",
            "group_b": "",
            "severe_fp": severe_fp,
            "flag_severe_fp_gt0": severe_fp > 0,
        }
    )
    audit = pd.DataFrame(rows)

    recs: list[dict[str, Any]] = []
    for _, row in audit.iterrows():
        flags = [
            name
            for name in [
                "flag_mean_shift_gt10",
                "flag_std_ratio",
                "flag_gap_mismatch_gt20",
                "flag_clipped_pct_gt5",
                "flag_severe_fp_gt0",
            ]
            if is_true_flag(row.get(name, False))
        ]
        if flags:
            recs.append(
                {
                    "audit": row["audit"],
                    "group_a": row.get("group_a", ""),
                    "group_b": row.get("group_b", ""),
                    "flags": ",".join(flags),
                    "recommendation": recommendation_for(row["audit"], flags),
                }
            )
    if not recs:
        recs.append(
            {
                "audit": "overall",
                "group_a": "",
                "group_b": "",
                "flags": "",
                "recommendation": "No range calibration blocker found for this dry-run.",
            }
        )
    return audit, pd.DataFrame(recs)


def recommendation_for(audit_name: str, flags: list[str]) -> str:
    if "flag_severe_fp_gt0" in flags:
        return "Inspect severe false positives before any larger dry-run."
    if audit_name == "boundary_and_clipping":
        return "Investigate clipping or boundary concentration before larger label generation."
    if audit_name == "mdiqa_vs_production_source_range":
        return "Review source range compatibility; use MOS-alignment and dataset-distribution evidence before fitting calibration."
    if audit_name == "dataset_boundary_mean_gap":
        return "Check whether dataset mean gap follows MOS distribution or indicates artificial source switching."
    return "Review flagged range mismatch."


def decide(
    manifest: pd.DataFrame,
    failures: pd.DataFrame,
    safety_summary: pd.DataFrame,
    range_audit: pd.DataFrame,
    policy_comparison: pd.DataFrame,
) -> str:
    valid_rows = int(manifest["target_available"].sum())
    if valid_rows < 1500 or manifest["selected_teacher_target_0_100"].isna().any():
        return "HYBRID_LABEL_2K_DRYRUN_FAILED"
    if len(failures[failures["stage"].isin(["target_build", "input_load"])]) > 0:
        return "HYBRID_LABEL_2K_DRYRUN_FAILED"

    severe_fp = int(safety_summary[safety_summary["policy"] == "selected"]["severe_fp"].sum())
    if severe_fp > 0:
        return "HYBRID_LABEL_2K_TOO_RISKY"

    flive_rows = manifest[(manifest["dataset"] == "flive") & manifest["production_pred_0_100"].notna()]
    if len(flive_rows) < 500:
        return "HYBRID_LABEL_2K_NEEDS_FLIVE_PRODUCTION_COVERAGE"

    fallback_rows = manifest[manifest["dataset"] == "fallback_proxy"]
    if len(fallback_rows) == 0 or not fallback_rows["selected_teacher_source"].eq("production_safety_fallback").all():
        return "HYBRID_LABEL_2K_NEEDS_FALLBACK_DEFINITION_FIX"

    dangerous_range = False
    for _, row in range_audit.iterrows():
        if is_true_flag(row.get("flag_clipped_pct_gt5", False)):
            dangerous_range = True
        if row.get("audit") == "mdiqa_vs_production_source_range" and is_true_flag(row.get("flag_std_ratio", False)):
            dangerous_range = True
    if dangerous_range:
        return "HYBRID_LABEL_2K_NEEDS_RANGE_CALIBRATION"

    policy_b_better = False
    comparable = policy_comparison.dropna(subset=["srcc_delta_a_minus_b", "mae_delta_a_minus_b"])
    for _, row in comparable.iterrows():
        if row["srcc_delta_a_minus_b"] < -0.02 and row["mae_delta_a_minus_b"] > 0:
            policy_b_better = True
    if policy_b_better:
        return "HYBRID_LABEL_2K_NEEDS_RANGE_CALIBRATION"
    return "HYBRID_LABEL_2K_READY_FOR_10K_DRYRUN"


def fmt(value: Any, digits: int = 4) -> str:
    value_float = safe_float(value)
    if not math.isfinite(value_float):
        return "nan"
    return f"{value_float:.{digits}f}"


def write_report(
    manifest: pd.DataFrame,
    validation: pd.DataFrame,
    distribution: pd.DataFrame,
    safety_summary: pd.DataFrame,
    safety_cases: pd.DataFrame,
    policy_comparison: pd.DataFrame,
    range_audit: pd.DataFrame,
    range_recs: pd.DataFrame,
    failures: pd.DataFrame,
    decision: str,
) -> None:
    counts = manifest.groupby(["dataset", "selected_teacher_source"]).size().reset_index(name="n")
    align = validation[validation["section"].eq("mos_alignment_policy_a")].copy()
    selected_safety = safety_summary[safety_summary["policy"].eq("selected")]
    severe_fp = int(selected_safety["severe_fp"].sum()) if len(selected_safety) else 0
    fp = int(selected_safety["fp"].sum()) if len(selected_safety) else 0
    fn = int(selected_safety["fn"].sum()) if len(selected_safety) else 0

    lines = [
        "# Phase 4 Hybrid Teacher Label 2k Dry-run and Range Calibration Audit Report",
        "",
        "## 1. Executive Summary",
        f"- Final decision: `{decision}`.",
        f"- Valid selected Policy A rows: `{int(manifest['target_available'].sum())}` / `{len(manifest)}`.",
        f"- Selected policy: `Policy A strict hybrid`.",
        f"- Safety counts: severe FP `{severe_fp}`, FP `{fp}`, FN `{fn}`.",
        "- No new MDIQA inference or production inference was run.",
        "",
        "## 2. Sources Inspected",
        f"- `{PHASE3_REPORT}`",
        f"- `{PHASE3_SPEC}`",
        f"- `{PAIRED_PREDS}`",
        f"- `{PAIRED_RESULTS}`",
        f"- `{MDIQA_PREDS}`",
        f"- `{MDIQA_METRICS}`",
        f"- `{FLIVE_PROD}`",
        f"- `{BASE_MANIFEST}`",
        f"- `{SELECTOR_SOURCE}`",
        "",
        "## 3. Confirmed Label Policy",
        "- KonIQ-10k -> MDIQA `seed1`.",
        "- SPAQ -> MDIQA `xd`.",
        "- FLIVE -> Production technical score.",
        "- Fallback/Unknown -> Production technical score safety fallback.",
        "",
        "## 4. 2k Dry-run Manifest Construction",
        "- Target allocation: KonIQ 500, SPAQ 500, FLIVE image 700, fallback proxy 300.",
        "- Fallback/unknown real mobile rows were unavailable; fallback uses a documented production-scored FLIVE image proxy group not duplicated with primary FLIVE rows.",
        f"- Output manifest: `{OUT_MANIFEST}`.",
        "",
        "## 5. Dataset and Source Counts",
    ]
    for row in counts.to_dict(orient="records"):
        lines.append(f"- {row['dataset']} / {row['selected_teacher_source']}: {row['n']}")

    lines.extend(["", "## 6. Target Distribution Summary"])
    for row in distribution.to_dict(orient="records"):
        lines.append(
            f"- {row.get('dataset', '')} / {row.get('selected_teacher_source', '')}: "
            f"n={row['n']}, mean={fmt(row['mean'])}, std={fmt(row['std'])}, "
            f"p05={fmt(row['p05'])}, p50={fmt(row['p50'])}, p95={fmt(row['p95'])}"
        )

    lines.extend(["", "## 7. MOS Alignment"])
    for row in align.to_dict(orient="records"):
        lines.append(
            f"- {row['dataset']}: n={row['n']}, SRCC={fmt(row['srcc'])}, "
            f"PLCC={fmt(row['plcc'])}, MAE={fmt(row['mae'])}, aligned MAE={fmt(row['linear_aligned_mae'])}"
        )

    lines.extend(["", "## 8. Safety Cases"])
    lines.append(f"- Severe FP: `{severe_fp}`")
    lines.append(f"- FP: `{fp}`")
    lines.append(f"- FN: `{fn}`")
    lines.append(f"- Detailed cases: `{OUT_SAFETY}`.")

    lines.extend(["", "## 9. Policy A vs Policy B"])
    for row in policy_comparison.to_dict(orient="records"):
        lines.append(
            f"- {row['dataset']}: Policy A SRCC={fmt(row['srcc_a'])}, Policy B SRCC={fmt(row['srcc_b'])}, "
            f"delta={fmt(row['srcc_delta_a_minus_b'])}; MAE delta A-B={fmt(row['mae_delta_a_minus_b'])}; "
            f"severe FP A/B={row['severe_fp_a']}/{row['severe_fp_b']}"
        )

    lines.extend(["", "## 10. Range Calibration Audit"])
    for row in range_audit.to_dict(orient="records"):
        flags = [
            key
            for key in row
            if key.startswith("flag_") and str(row.get(key)).lower() == "true"
        ]
        lines.append(
            f"- {row.get('audit')} {row.get('group_a', '')} vs {row.get('group_b', '')}: "
            f"mean_shift={fmt(row.get('mean_shift'))}, std_ratio={fmt(row.get('std_ratio'))}, "
            f"gap_mismatch={fmt(row.get('p95_p05_gap_mismatch'))}, flags={','.join(flags) if flags else 'none'}"
        )
    lines.append(f"- Recommendations: `{OUT_RANGE_RECS}`.")

    lines.extend(["", "## 11. Dataset-aware Label Risks"])
    lines.append("- SPAQ remains lower-mean than KonIQ/FLIVE, but this is checked against MOS alignment and safety cases rather than automatically calibrated away.")
    lines.append("- Fallback proxy is production-scored FLIVE image data, not real mobile data; it is suitable for policy mechanics but not a representativeness claim.")
    lines.append("- FLIVE primary uses production scores and image rows only; patches are excluded.")

    lines.extend(["", "## 12. Failure Cases"])
    if len(failures):
        for row in failures.head(50).to_dict(orient="records"):
            lines.append(f"- {row.get('stage')} | {row.get('dataset')} | {row.get('image_path')} | {row.get('error')}")
    else:
        lines.append("- None.")

    lines.extend(["", "## 13. Final Decision", decision, "", "## 14. Recommended Next Step"])
    if decision == "HYBRID_LABEL_2K_READY_FOR_10K_DRYRUN":
        lines.append("- Proceed to a 10k dry-run with the same Policy A rule and a real fallback/mobile source definition when available.")
    elif decision == "HYBRID_LABEL_2K_NEEDS_RANGE_CALIBRATION":
        lines.append("- Run a calibration design pass before larger dry-runs; do not create full labels yet.")
    elif decision == "HYBRID_LABEL_2K_NEEDS_FALLBACK_DEFINITION_FIX":
        lines.append("- Define a better fallback/unknown source before larger dry-runs.")
    elif decision == "HYBRID_LABEL_2K_NEEDS_FLIVE_PRODUCTION_COVERAGE":
        lines.append("- Expand or verify FLIVE production score coverage before larger dry-runs.")
    elif decision == "HYBRID_LABEL_2K_TOO_RISKY":
        lines.append("- Inspect safety failures and revise the label policy before larger dry-runs.")
    else:
        lines.append("- Fix script/input failures and rerun this 2k dry-run only.")
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest, failures, _paired_results = build_manifest()
    distribution = pd.DataFrame(
        distribution_rows(manifest, ["dataset", "selected_teacher_source"], "selected_teacher_target_0_100")
    )
    validation = build_validation(manifest)
    safety_cases, safety_summary = build_safety(manifest)
    policy_comparison = build_policy_comparison(manifest, safety_summary)
    range_audit, range_recs = build_range_audit(manifest, safety_summary)
    decision = decide(manifest, failures, safety_summary, range_audit, policy_comparison)

    out_manifest = manifest.copy()
    for col in REQUIRED_MANIFEST_COLUMNS:
        if col not in out_manifest.columns:
            out_manifest[col] = np.nan
    extra_cols = [
        "teacher_target_policy_a_unclipped_0_100",
        "teacher_target_policy_b_unclipped_0_100",
        "original_dataset",
    ]
    manifest_cols = REQUIRED_MANIFEST_COLUMNS + [col for col in extra_cols if col in out_manifest.columns]
    out_manifest[manifest_cols].to_csv(OUT_MANIFEST, index=False)
    validation.to_csv(OUT_VALIDATION, index=False)
    distribution.to_csv(OUT_DISTRIBUTION, index=False)
    safety_cases.to_csv(OUT_SAFETY, index=False)
    policy_comparison.to_csv(OUT_POLICY, index=False)
    range_audit.to_csv(OUT_RANGE_AUDIT, index=False)
    range_recs.to_csv(OUT_RANGE_RECS, index=False)
    failures.to_csv(OUT_FAILURES, index=False)
    pd.DataFrame(
        [
            {
                "priority": 1,
                "action": "Run 10k dry-run only if final decision is ready.",
                "status": "recommended" if decision == "HYBRID_LABEL_2K_READY_FOR_10K_DRYRUN" else "blocked",
            },
            {
                "priority": 2,
                "action": "Replace fallback proxy with real mobile/unknown production-scored rows when available.",
                "status": "open",
            },
            {
                "priority": 3,
                "action": "Keep Policy B diagnostic-only unless it clearly improves MOS alignment and safety.",
                "status": "open",
            },
        ]
    ).to_csv(OUT_NEXT, index=False)
    write_report(
        out_manifest,
        validation,
        distribution,
        safety_summary,
        safety_cases,
        policy_comparison,
        range_audit,
        range_recs,
        failures,
        decision,
    )
    print(f"Wrote {OUT_MANIFEST}")
    print(f"Wrote {OUT_VALIDATION}")
    print(f"Wrote {OUT_DISTRIBUTION}")
    print(f"Wrote {OUT_SAFETY}")
    print(f"Wrote {OUT_POLICY}")
    print(f"Wrote {OUT_RANGE_AUDIT}")
    print(f"Wrote {OUT_RANGE_RECS}")
    print(f"Wrote {OUT_FAILURES}")
    print(f"Wrote {OUT_REPORT}")
    print(f"Final decision: {decision}")


if __name__ == "__main__":
    main()
