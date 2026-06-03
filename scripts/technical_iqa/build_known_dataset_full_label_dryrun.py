# Phase 6 known-dataset-only full-label dry-run manifest와 audit를 생성한다.
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

MDIQA_PREDS = ROOT / "outputs" / "mdiqa_correct_protocol_full_benchmark_20260601" / "predictions.csv"
PAIRED_PREDS = ROOT / "outputs" / "paired_production_mdiqa_significance_expanded_20260603" / "paired_predictions_expanded.csv"
FLIVE_PROD = ROOT / "outputs" / "technical_iqa_combo_benchmark_20260524" / "fusion_predictions_flive.csv"
BASE_MANIFEST = ROOT / "outputs" / "research_protocol_iqa_manifests_20260530" / "benchmark_all_research_protocol_iqa.csv"
RECOMMENDED_POLICY = ROOT / "outputs" / "hybrid_teacher_adjusted_calibration_2k_20260603" / "recommended_adjusted_policy.json"
SCOPE_DECISION = ROOT / "outputs" / "hybrid_teacher_final_label_scope_decision_20260603" / "report.md"
SCOPE_CSV = ROOT / "outputs" / "hybrid_teacher_final_label_scope_decision_20260603" / "known_dataset_label_scope.csv"
PHASE5_10K_REPORT = ROOT / "outputs" / "hybrid_teacher_label_dryrun_10k_20260603" / "report.md"

OUTPUT_DIR = ROOT / "outputs" / "known_dataset_full_label_dryrun_20260603"

SELECTED_POLICY = "policy_f_shift_cap3"
KNOWN_DATASETS = ("koniq10k", "spaq", "flive")
MDIQA_SOURCES = {"mdiqa_seed1", "mdiqa_xd"}
PRODUCTION_SOURCES = {"production"}
PHASE5_10K = {
    "valid_rows": 7606,
    "koniq10k_rows": 2500,
    "spaq_rows": 1125,
    "flive_rows": 3981,
    "source_mean_shift": 15.4364,
    "source_std_ratio": 2.9749,
    "source_gap_mismatch": 32.3943,
    "severe_fp": 0,
    "fp": 0,
    "fn": 0,
}

MANIFEST_COLUMNS = [
    "dataset",
    "split",
    "content_type",
    "image_path",
    "source_group",
    "quality_score_0_100",
    "mdiqa_variant",
    "mdiqa_pred_0_100",
    "production_pred_0_100",
    "selected_teacher_target_0_100",
    "selected_teacher_target_unclipped_0_100",
    "selected_teacher_source",
    "selected_policy",
    "target_available",
    "target_error",
    "excluded_reason",
]


def output_paths() -> dict[str, Path]:
    return {
        "manifest": OUTPUT_DIR / "known_dataset_full_label_dryrun_manifest.csv",
        "validation": OUTPUT_DIR / "known_dataset_full_label_validation.csv",
        "distribution": OUTPUT_DIR / "known_dataset_distribution_summary.csv",
        "safety": OUTPUT_DIR / "known_dataset_safety_cases.csv",
        "failures": OUTPUT_DIR / "known_dataset_failure_cases.csv",
        "range": OUTPUT_DIR / "known_dataset_range_audit.csv",
        "scope": OUTPUT_DIR / "known_dataset_scope_summary.csv",
        "report": OUTPUT_DIR / "report.md",
        "next": OUTPUT_DIR / "next_actions.csv",
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def canonical_path(value: Any) -> str:
    try:
        return str(Path(str(value)).resolve(strict=False))
    except OSError:
        return str(value)


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    val = safe_float(value)
    if math.isfinite(val):
        return f"{val:.{digits}f}"
    return ""


def source_group_from_path(path: str) -> str:
    name = Path(path).name
    if "_patch_" in name:
        return name.split("_patch_", 1)[0]
    return Path(name).stem


def clip_0_100(value: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    return min(100.0, max(0.0, value))


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def stdev(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if len(vals) < 2:
        return float("nan")
    mu = mean(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))


def percentile(values: list[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[int(pos)]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def distribution(values: list[float]) -> dict[str, Any]:
    vals = [v for v in values if math.isfinite(v)]
    return {
        "n": len(vals),
        "min": min(vals) if vals else float("nan"),
        "max": max(vals) if vals else float("nan"),
        "mean": mean(vals),
        "std": stdev(vals),
        "p01": percentile(vals, 1),
        "p05": percentile(vals, 5),
        "p25": percentile(vals, 25),
        "p50": percentile(vals, 50),
        "p75": percentile(vals, 75),
        "p95": percentile(vals, 95),
        "p99": percentile(vals, 99),
    }


def rank_average(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        for idx in order[start:end]:
            ranks[idx] = avg_rank
        start = end
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    mx = mean(x)
    my = mean(y)
    denom_x = sum((v - mx) ** 2 for v in x)
    denom_y = sum((v - my) ** 2 for v in y)
    denom = math.sqrt(denom_x * denom_y)
    if denom == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / denom


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    return pearson(rank_average(x), rank_average(y))


def linear_aligned_mae(pred: list[float], mos: list[float]) -> float:
    if len(pred) < 2:
        return float("nan")
    pred_mean = mean(pred)
    mos_mean = mean(mos)
    denom = sum((p - pred_mean) ** 2 for p in pred)
    if denom == 0:
        return float("nan")
    slope = sum((p - pred_mean) * (m - mos_mean) for p, m in zip(pred, mos)) / denom
    intercept = mos_mean - slope * pred_mean
    aligned = [slope * p + intercept for p in pred]
    return mean([abs(a - m) for a, m in zip(aligned, mos)])


def load_recommended_policy() -> dict[str, Any]:
    data = json.loads(RECOMMENDED_POLICY.read_text(encoding="utf-8"))
    policy = data.get("recommended_policy", {}).get("policy")
    if policy != SELECTED_POLICY:
        raise RuntimeError(f"Expected {SELECTED_POLICY}, found {policy}")
    return data


def load_scope_expectations() -> dict[str, dict[str, Any]]:
    expectations: dict[str, dict[str, Any]] = {}
    for row in read_csv(SCOPE_CSV):
        dataset = row.get("dataset", "")
        expectations[dataset] = {
            "teacher_source": row.get("teacher_source", ""),
            "calibration_policy": row.get("calibration_policy", ""),
            "status": row.get("status", ""),
            "estimated_rows": int(safe_float(row.get("estimated_rows", "0"))),
        }
    return expectations


def load_base_metadata() -> dict[str, dict[str, Any]]:
    rows = []
    for row in read_csv(BASE_MANIFEST):
        dataset = row.get("dataset", "")
        if dataset not in KNOWN_DATASETS:
            continue
        image_path = canonical_path(row.get("image_path", ""))
        rows.append(
            {
                "dataset": dataset,
                "split": row.get("split", "unknown") or "unknown",
                "content_type": row.get("content_type", ""),
                "image_path": image_path,
                "source_group": source_group_from_path(image_path),
                "quality_score_0_100": safe_float(row.get("quality_score_0_100")),
                "split_preference": 1 if row.get("split", "") == "test" else 0,
            }
        )
    rows.sort(key=lambda row: (row["image_path"], -row["split_preference"]))
    by_path: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_path.setdefault(row["image_path"], row)
    return by_path


def load_mdiqa_unique() -> dict[tuple[str, str], dict[str, Any]]:
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_csv(MDIQA_PREDS):
        dataset = row.get("dataset", "")
        if dataset not in {"koniq10k", "spaq"}:
            continue
        image_path = canonical_path(row.get("image_path", ""))
        key = (dataset, image_path)
        bucket = buckets.setdefault(key, {"mos": [], "pred": []})
        bucket["mos"].append(safe_float(row.get("quality_score_0_100")))
        bucket["pred"].append(safe_float(row.get("pred_0_100")))
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, bucket in buckets.items():
        out[key] = {
            "quality_score_0_100": mean(bucket["mos"]),
            "mdiqa_pred_0_100": mean(bucket["pred"]),
            "mdiqa_duplicate_rows": len(bucket["pred"]),
        }
    return out


def load_paired_production() -> dict[str, float]:
    prod: dict[str, float] = {}
    for row in read_csv(PAIRED_PREDS):
        image_path = canonical_path(row.get("image_path", ""))
        prod.setdefault(image_path, safe_float(row.get("production_pred_0_100")))
    return prod


def load_flive_production() -> dict[str, float]:
    return {row.get("image_path", ""): safe_float(row.get("flive_score")) for row in read_csv(FLIVE_PROD)}


def build_known_candidate_rows(
    base: dict[str, dict[str, Any]],
    mdiqa: dict[tuple[str, str], dict[str, Any]],
    paired_prod: dict[str, float],
    flive_prod: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for dataset, variant, source in [
        ("koniq10k", "seed1", "mdiqa_seed1"),
        ("spaq", "xd", "mdiqa_xd"),
    ]:
        for (pred_dataset, image_path), pred in sorted(mdiqa.items()):
            if pred_dataset != dataset:
                continue
            meta = base.get(image_path)
            if meta is None:
                failures.append({"stage": "metadata_join", "dataset": dataset, "image_path": image_path, "error": "missing base metadata"})
                continue
            if meta.get("content_type") != "image":
                failures.append({"stage": "scope_filter", "dataset": dataset, "image_path": image_path, "error": "non-image content_type"})
                continue
            if not Path(image_path).exists():
                failures.append({"stage": "path_check", "dataset": dataset, "image_path": image_path, "error": "image path missing"})
                continue
            target = pred["mdiqa_pred_0_100"]
            if not math.isfinite(target):
                failures.append({"stage": "target_build", "dataset": dataset, "image_path": image_path, "error": "missing mdiqa target"})
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "split": meta.get("split", "unknown"),
                    "content_type": "image",
                    "image_path": image_path,
                    "source_group": meta.get("source_group", source_group_from_path(image_path)),
                    "quality_score_0_100": pred.get("quality_score_0_100", meta.get("quality_score_0_100")),
                    "mdiqa_variant": variant,
                    "mdiqa_pred_0_100": target,
                    "production_pred_0_100": paired_prod.get(image_path, float("nan")),
                    "teacher_source_policy_a": source,
                    "teacher_target_policy_a_unclipped_0_100": target,
                    "teacher_target_policy_a_0_100": clip_0_100(target),
                    "excluded_reason": "",
                }
            )

    for image_path, meta in sorted(base.items()):
        if meta.get("dataset") != "flive" or meta.get("content_type") != "image":
            continue
        prod_key = Path(image_path).name
        prod = flive_prod.get(prod_key, float("nan"))
        if not math.isfinite(prod):
            continue
        if not Path(image_path).exists():
            failures.append({"stage": "path_check", "dataset": "flive", "image_path": image_path, "error": "image path missing"})
            continue
        rows.append(
            {
                "dataset": "flive",
                "split": meta.get("split", "unknown"),
                "content_type": "image",
                "image_path": image_path,
                "source_group": meta.get("source_group", source_group_from_path(image_path)),
                "quality_score_0_100": meta.get("quality_score_0_100"),
                "mdiqa_variant": "",
                "mdiqa_pred_0_100": float("nan"),
                "production_pred_0_100": prod,
                "teacher_source_policy_a": "production",
                "teacher_target_policy_a_unclipped_0_100": prod,
                "teacher_target_policy_a_0_100": clip_0_100(prod),
                "excluded_reason": "",
            }
        )

    return rows, failures


def apply_policy_f_shift_cap3(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mdiqa_values = [row["teacher_target_policy_a_0_100"] for row in rows if row["teacher_source_policy_a"] in MDIQA_SOURCES]
    production_values = [row["teacher_target_policy_a_0_100"] for row in rows if row["teacher_source_policy_a"] in PRODUCTION_SOURCES]
    raw_shift = mean(production_values) - mean(mdiqa_values)
    applied_shift = min(3.0, max(-3.0, raw_shift)) if math.isfinite(raw_shift) else 0.0

    for row in rows:
        source = row["teacher_source_policy_a"]
        base_target = row["teacher_target_policy_a_0_100"]
        if source in MDIQA_SOURCES:
            unclipped = base_target + applied_shift
        else:
            unclipped = base_target
        clipped = clip_0_100(unclipped)
        row["selected_teacher_target_unclipped_0_100"] = unclipped
        row["selected_teacher_target_0_100"] = clipped
        if row["dataset"] == "koniq10k":
            row["selected_teacher_source"] = "mdiqa_seed1_policy_f_shift_cap3"
        elif row["dataset"] == "spaq":
            row["selected_teacher_source"] = "mdiqa_xd_policy_f_shift_cap3"
        else:
            row["selected_teacher_source"] = "production"
        row["selected_policy"] = SELECTED_POLICY
        row["target_available"] = math.isfinite(clipped)
        row["target_error"] = "" if row["target_available"] else "missing selected teacher target"
    return rows, {
        "raw_shift": raw_shift,
        "applied_shift": applied_shift,
        "mdiqa_policy_a_mean": mean(mdiqa_values),
        "production_policy_a_mean": mean(production_values),
    }


def metric_row(rows: list[dict[str, Any]], extra: dict[str, Any]) -> dict[str, Any]:
    pairs = [
        (safe_float(row.get("selected_teacher_target_0_100")), safe_float(row.get("quality_score_0_100")))
        for row in rows
    ]
    pairs = [(pred, mos) for pred, mos in pairs if math.isfinite(pred) and math.isfinite(mos)]
    pred = [p for p, _ in pairs]
    mos = [m for _, m in pairs]
    targets = [safe_float(row.get("selected_teacher_target_0_100")) for row in rows]
    unclipped = [safe_float(row.get("selected_teacher_target_unclipped_0_100")) for row in rows]
    return {
        **extra,
        "n": len(pairs),
        "srcc": spearman(pred, mos),
        "plcc": pearson(pred, mos),
        "mae": mean([abs(p - m) for p, m in pairs]),
        "linear_aligned_mae": linear_aligned_mae(pred, mos),
        "fp": sum(1 for p, m in pairs if m < 40 and p > 60),
        "severe_fp": sum(1 for p, m in pairs if m < 40 and p > 65),
        "fn": sum(1 for p, m in pairs if m > 75 and p < 45),
        "clipping_count": sum(1 for v in unclipped if math.isfinite(v) and (v < 0 or v > 100)),
        "boundary_0_or_100_count": sum(1 for v in targets if math.isfinite(v) and (v <= 1 or v >= 99)),
        "boundary_0_or_100_pct": (
            sum(1 for v in targets if math.isfinite(v) and (v <= 1 or v >= 99)) / len(targets) if targets else float("nan")
        ),
    }


def group_rows(rows: list[dict[str, Any]], cols: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(col, "") for col in cols)].append(row)
    return dict(groups)


def build_validation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, sub in group_rows(rows, ["dataset", "split", "content_type", "selected_teacher_source"]).items():
        dataset, split, content_type, source = key
        out.append(
            {
                "section": "row_count",
                "dataset": dataset,
                "split": split,
                "content_type": content_type,
                "selected_teacher_source": source,
                "n": len(sub),
            }
        )
    for (dataset,), sub in group_rows(rows, ["dataset"]).items():
        out.append(metric_row(sub, {"section": "mos_alignment", "dataset": dataset}))
    for (source,), sub in group_rows(rows, ["selected_teacher_source"]).items():
        out.append(metric_row(sub, {"section": "mos_alignment_by_source", "selected_teacher_source": source}))
    return out


def build_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cols in [["dataset"], ["selected_teacher_source"], ["dataset", "selected_teacher_source"]]:
        for key, sub in group_rows(rows, cols).items():
            dist = distribution([safe_float(row.get("selected_teacher_target_0_100")) for row in sub])
            row = {"grouping": "+".join(cols)}
            row.update({col: val for col, val in zip(cols, key)})
            row.update(dist)
            out.append(row)
    return out


def build_safety_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        target = safe_float(row.get("selected_teacher_target_0_100"))
        mos = safe_float(row.get("quality_score_0_100"))
        if not math.isfinite(target) or not math.isfinite(mos):
            continue
        case = ""
        if mos < 40 and target > 65:
            case = "severe_fp"
        elif mos < 40 and target > 60:
            case = "FP"
        elif mos > 75 and target < 45:
            case = "FN"
        if case:
            keep = {
                "safety_case_type": case,
                "dataset": row.get("dataset"),
                "split": row.get("split"),
                "content_type": row.get("content_type"),
                "image_path": row.get("image_path"),
                "quality_score_0_100": row.get("quality_score_0_100"),
                "selected_teacher_target_0_100": target,
                "selected_teacher_source": row.get("selected_teacher_source"),
                "mdiqa_variant": row.get("mdiqa_variant"),
                "mdiqa_pred_0_100": row.get("mdiqa_pred_0_100"),
                "production_pred_0_100": row.get("production_pred_0_100"),
            }
            out.append(keep)
    return out


def build_range_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    mdiqa = [
        safe_float(row.get("selected_teacher_target_0_100"))
        for row in rows
        if row.get("teacher_source_policy_a") in MDIQA_SOURCES
    ]
    prod = [
        safe_float(row.get("selected_teacher_target_0_100"))
        for row in rows
        if row.get("teacher_source_policy_a") in PRODUCTION_SOURCES
    ]
    mdiqa = [v for v in mdiqa if math.isfinite(v)]
    prod = [v for v in prod if math.isfinite(v)]
    if mdiqa and prod:
        mdiqa_gap = percentile(mdiqa, 95) - percentile(mdiqa, 5)
        prod_gap = percentile(prod, 95) - percentile(prod, 5)
        mean_shift = abs(mean(mdiqa) - mean(prod))
        std_ratio = stdev(mdiqa) / stdev(prod) if stdev(prod) else float("inf")
        gap_mismatch = abs(mdiqa_gap - prod_gap)
        out.append(
            {
                "audit": "mdiqa_vs_production_source_range",
                "group_a": "mdiqa_sources",
                "group_b": "production_sources",
                "mean_a": mean(mdiqa),
                "mean_b": mean(prod),
                "mean_shift": mean_shift,
                "std_a": stdev(mdiqa),
                "std_b": stdev(prod),
                "std_ratio": std_ratio,
                "p05_a": percentile(mdiqa, 5),
                "p05_b": percentile(prod, 5),
                "p95_a": percentile(mdiqa, 95),
                "p95_b": percentile(prod, 95),
                "p95_p05_gap_mismatch": gap_mismatch,
                "phase5_10k_mean_shift": PHASE5_10K["source_mean_shift"],
                "phase5_10k_std_ratio": PHASE5_10K["source_std_ratio"],
                "phase5_10k_gap_mismatch": PHASE5_10K["source_gap_mismatch"],
                "delta_vs_phase5_10k_mean_shift": mean_shift - PHASE5_10K["source_mean_shift"],
                "delta_vs_phase5_10k_std_ratio": std_ratio - PHASE5_10K["source_std_ratio"],
                "delta_vs_phase5_10k_gap_mismatch": gap_mismatch - PHASE5_10K["source_gap_mismatch"],
                "flag_mean_shift_gt10": mean_shift > 10,
                "flag_std_ratio": std_ratio > 2.0 or std_ratio < 0.5,
                "flag_gap_mismatch_gt20": gap_mismatch > 20,
            }
        )

    dist_by_dataset = {
        dataset: distribution([safe_float(row.get("selected_teacher_target_0_100")) for row in sub])
        for (dataset,), sub in group_rows(rows, ["dataset"]).items()
    }
    for a, b in [("flive", "koniq10k"), ("flive", "spaq"), ("koniq10k", "spaq")]:
        da = dist_by_dataset.get(a, {})
        db = dist_by_dataset.get(b, {})
        if not da or not db or da.get("n", 0) == 0 or db.get("n", 0) == 0:
            out.append({"audit": "dataset_boundary_mean_gap", "group_a": a, "group_b": b, "error": "missing group"})
            continue
        gap_a = da["p95"] - da["p05"]
        gap_b = db["p95"] - db["p05"]
        std_ratio = da["std"] / db["std"] if db["std"] else float("inf")
        out.append(
            {
                "audit": "dataset_boundary_mean_gap",
                "group_a": a,
                "group_b": b,
                "mean_a": da["mean"],
                "mean_b": db["mean"],
                "mean_shift": abs(da["mean"] - db["mean"]),
                "std_a": da["std"],
                "std_b": db["std"],
                "std_ratio": std_ratio,
                "p05_a": da["p05"],
                "p05_b": db["p05"],
                "p95_a": da["p95"],
                "p95_b": db["p95"],
                "p95_p05_gap_mismatch": abs(gap_a - gap_b),
                "flag_mean_shift_gt10": abs(da["mean"] - db["mean"]) > 10,
                "flag_std_ratio": std_ratio > 2.0 or std_ratio < 0.5,
                "flag_gap_mismatch_gt20": abs(gap_a - gap_b) > 20,
            }
        )

    unclipped = [safe_float(row.get("selected_teacher_target_unclipped_0_100")) for row in rows]
    target = [safe_float(row.get("selected_teacher_target_0_100")) for row in rows]
    clipped = sum(1 for v in unclipped if math.isfinite(v) and (v < 0 or v > 100))
    out.append(
        {
            "audit": "boundary_and_clipping",
            "group_a": SELECTED_POLICY,
            "rows_total": len(rows),
            "clipped_count": clipped,
            "clipped_pct": clipped / len(rows) if rows else float("nan"),
            "near_0_or_100_count": sum(1 for v in target if math.isfinite(v) and (v <= 1 or v >= 99)),
            "near_0_or_100_pct": sum(1 for v in target if math.isfinite(v) and (v <= 1 or v >= 99)) / len(rows) if rows else float("nan"),
            "near_5_or_95_count": sum(1 for v in target if math.isfinite(v) and (v <= 5 or v >= 95)),
            "near_5_or_95_pct": sum(1 for v in target if math.isfinite(v) and (v <= 5 or v >= 95)) / len(rows) if rows else float("nan"),
            "flag_clipped_pct_gt5": (clipped / len(rows)) > 0.05 if rows else False,
        }
    )
    return out


def build_scope_summary(
    rows: list[dict[str, Any]],
    expectations: dict[str, dict[str, Any]],
    failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts = {dataset: len(sub) for (dataset,), sub in group_rows(rows, ["dataset"]).items()}
    duplicate_paths = len(rows) - len({row.get("image_path") for row in rows})
    out: list[dict[str, Any]] = []
    for dataset in KNOWN_DATASETS:
        expected = expectations.get(dataset, {}).get("estimated_rows", 0)
        available = counts.get(dataset, 0)
        out.append(
            {
                "dataset": dataset,
                "expected_or_estimated_rows": expected,
                "valid_rows": available,
                "coverage_gap": max(0, expected - available),
                "status": "ok" if expected and available >= expected else "coverage_gap",
                "included_in_main_labels": True,
                "excluded_reason": "",
            }
        )
        if expected and available < expected:
            failures.append(
                {
                    "stage": "coverage",
                    "dataset": dataset,
                    "image_path": "",
                    "error": f"expected_or_estimated_rows={expected}; valid_cached_rows={available}",
                }
            )
    out.extend(
        [
            {
                "dataset": "unknown_fallback",
                "expected_or_estimated_rows": expectations.get("unknown_fallback", {}).get("estimated_rows", 0),
                "valid_rows": 0,
                "coverage_gap": 0,
                "status": "excluded_future_work",
                "included_in_main_labels": False,
                "excluded_reason": "excluded by final scope decision; product-gold labels are not strict technical MOS",
            },
            {
                "dataset": "real_mobile_c6",
                "expected_or_estimated_rows": expectations.get("real_mobile_c6", {}).get("estimated_rows", 0),
                "valid_rows": 0,
                "coverage_gap": 0,
                "status": "excluded_holdout",
                "included_in_main_labels": False,
                "excluded_reason": "c6_parity real-mobile holdout only",
            },
            {
                "dataset": "test_weight_gold",
                "expected_or_estimated_rows": expectations.get("test_weight_gold", {}).get("estimated_rows", 0),
                "valid_rows": 0,
                "coverage_gap": 0,
                "status": "excluded_validation_only",
                "included_in_main_labels": False,
                "excluded_reason": "Test_weight/product-gold is aesthetic/product validation only",
            },
            {
                "dataset": "duplicate_image_path",
                "expected_or_estimated_rows": 0,
                "valid_rows": duplicate_paths,
                "coverage_gap": 0,
                "status": "ok" if duplicate_paths == 0 else "duplicate_error",
                "included_in_main_labels": False,
                "excluded_reason": "audit row",
            },
        ]
    )
    return out


def decide(
    rows: list[dict[str, Any]],
    scope_summary: list[dict[str, Any]],
    safety_cases: list[dict[str, Any]],
    range_audit: list[dict[str, Any]],
) -> str:
    counts = {dataset: len(sub) for (dataset,), sub in group_rows(rows, ["dataset"]).items()}
    duplicate_count = len(rows) - len({row.get("image_path") for row in rows})
    severe_fp = sum(1 for row in safety_cases if row.get("safety_case_type") == "severe_fp")
    fp = sum(1 for row in safety_cases if row.get("safety_case_type") in {"FP", "severe_fp"})
    clipped_row = next((row for row in range_audit if row.get("audit") == "boundary_and_clipping"), {})
    clipped_pct = safe_float(clipped_row.get("clipped_pct"))
    coverage_gap = any(row.get("dataset") in KNOWN_DATASETS and row.get("status") == "coverage_gap" for row in scope_summary)
    flive_non_image = any(row.get("dataset") == "flive" and row.get("content_type") != "image" for row in rows)
    bad_scope = any(row.get("dataset") not in KNOWN_DATASETS for row in rows)

    if not rows or bad_scope:
        return "KNOWN_DATASET_FULL_LABEL_DRYRUN_FAILED"
    if duplicate_count or flive_non_image:
        return "KNOWN_DATASET_FULL_LABEL_DRYRUN_FAILED"
    if any(counts.get(dataset, 0) == 0 for dataset in KNOWN_DATASETS):
        return "KNOWN_DATASET_FULL_LABEL_NEEDS_COVERAGE_FIX"
    if severe_fp > 0 or fp > max(5, int(0.005 * len(rows))):
        return "KNOWN_DATASET_FULL_LABEL_TOO_RISKY"
    if math.isfinite(clipped_pct) and clipped_pct > 0.05:
        return "KNOWN_DATASET_FULL_LABEL_NEEDS_RANGE_REVIEW"
    if coverage_gap:
        return "KNOWN_DATASET_FULL_LABEL_NEEDS_COVERAGE_FIX"
    if any(str(row.get("flag_mean_shift_gt10")) == "True" for row in range_audit):
        return "KNOWN_DATASET_FULL_LABEL_NEEDS_RANGE_REVIEW"
    return "KNOWN_DATASET_FULL_LABEL_DRYRUN_READY_FOR_REVIEW"


def write_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def write_report(
    paths: dict[str, Path],
    rows: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
    safety_cases: list[dict[str, Any]],
    range_audit: list[dict[str, Any]],
    scope_summary: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    decision: str,
    params: dict[str, Any],
) -> None:
    dataset_counts = [{"dataset": key[0], "n": len(sub)} for key, sub in sorted(group_rows(rows, ["dataset"]).items())]
    source_counts = [
        {"selected_teacher_source": key[0], "n": len(sub)}
        for key, sub in sorted(group_rows(rows, ["selected_teacher_source"]).items())
    ]
    metrics = [row for row in validation if row.get("section") == "mos_alignment"]
    safety_count_rows = []
    for key, sub in sorted(group_rows(safety_cases, ["dataset", "safety_case_type"]).items()):
        safety_count_rows.append({"dataset": key[0], "safety_case_type": key[1], "n": len(sub)})
    source_range = next((row for row in range_audit if row.get("audit") == "mdiqa_vs_production_source_range"), {})
    clipping = next((row for row in range_audit if row.get("audit") == "boundary_and_clipping"), {})

    lines = [
        "# Known-dataset-only Full-label Dry-run Report",
        "",
        "## 1. Executive Summary",
        f"- Final decision: `{decision}`.",
        f"- Rows: `{len(rows)}` total, `{sum(1 for row in rows if row.get('target_available'))}` valid.",
        f"- Selected policy: `{SELECTED_POLICY}`.",
        f"- Applied MDIQA shift: `{fmt(params.get('applied_shift'))}`.",
        f"- Coverage note: cached MDIQA availability is below scope estimate for KonIQ/SPAQ.",
        "",
        "## 2. Sources Inspected",
        f"- `{MDIQA_PREDS}`",
        f"- `{PAIRED_PREDS}`",
        f"- `{FLIVE_PROD}`",
        f"- `{BASE_MANIFEST}`",
        f"- `{RECOMMENDED_POLICY}`",
        f"- `{SCOPE_CSV}`",
        f"- `{SCOPE_DECISION}`",
        f"- `{PHASE5_10K_REPORT}`",
        "",
        "## 3. Confirmed Label Scope",
        "- KonIQ-10k: include, MDIQA seed1 + policy_f_shift_cap3.",
        "- SPAQ: include, MDIQA xd + policy_f_shift_cap3.",
        "- FLIVE: include, production technical score, image rows only.",
        "- Unknown/fallback: excluded from main labels.",
        "- c6_parity real-mobile: holdout only.",
        "- Test_weight/product-gold: aesthetic/product validation only.",
        "",
        "## 4. Manifest Construction",
        "- Base metadata deduped by `image_path`, preferring `test` split when repeated across research protocol runs.",
        "- KonIQ/SPAQ rows require cached MDIQA prediction, image content_type, existing image path, finite target.",
        "- FLIVE rows require image content_type and cached production `flive_score` keyed by basename.",
        "- Duplicate image_path rows are not allowed.",
        "",
        "## 5. Dataset Counts",
        write_table(dataset_counts, ["dataset", "n"]),
        "",
        "Source counts:",
        write_table(source_counts, ["selected_teacher_source", "n"]),
        "",
        "## 6. Target Distribution",
        f"- Distribution CSV: `{paths['distribution']}` rows: `{len(distribution_rows)}`.",
        "",
        "## 7. MOS Alignment",
    ]
    for row in metrics:
        lines.append(
            f"- {row.get('dataset')}: n={row.get('n')}, SRCC={fmt(row.get('srcc'))}, PLCC={fmt(row.get('plcc'))}, "
            f"MAE={fmt(row.get('mae'))}, aligned_MAE={fmt(row.get('linear_aligned_mae'))}, "
            f"FP={row.get('fp')}, severe_FP={row.get('severe_fp')}, FN={row.get('fn')}."
        )
    lines.extend(["", "## 8. Safety Cases"])
    if safety_count_rows:
        lines.append(write_table(safety_count_rows, ["dataset", "safety_case_type", "n"]))
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## 9. Range Audit",
            f"- MDIQA vs Production mean shift: `{fmt(source_range.get('mean_shift'))}`; Phase 5 10k: `{PHASE5_10K['source_mean_shift']}`.",
            f"- Std ratio: `{fmt(source_range.get('std_ratio'))}`; Phase 5 10k: `{PHASE5_10K['source_std_ratio']}`.",
            f"- p95/p05 gap mismatch: `{fmt(source_range.get('p95_p05_gap_mismatch'))}`; Phase 5 10k: `{PHASE5_10K['source_gap_mismatch']}`.",
            f"- Clipping count: `{clipping.get('clipped_count', 0)}`; clipping pct: `{fmt(clipping.get('clipped_pct'))}`.",
            f"- Range audit CSV: `{paths['range']}`.",
            "",
            "## 10. Excluded Holdout Data",
        ]
    )
    excluded = [row for row in scope_summary if not row.get("included_in_main_labels")]
    lines.append(write_table(excluded, ["dataset", "status", "expected_or_estimated_rows", "excluded_reason"]))
    lines.extend(["", "## 11. Failure Cases"])
    if failures:
        lines.append(f"- Failure rows: `{len(failures)}`. See `{paths['failures']}`.")
        coverage = [row for row in failures if row.get("stage") == "coverage"]
        if coverage:
            lines.append(write_table(coverage, ["stage", "dataset", "error"]))
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## 12. Final Decision",
            decision,
            "",
            "## 13. Recommended Next Step",
        ]
    )
    if decision == "KNOWN_DATASET_FULL_LABEL_DRYRUN_READY_FOR_REVIEW":
        lines.append("- Review dry-run outputs; do not create final training labels yet.")
    elif decision == "KNOWN_DATASET_FULL_LABEL_NEEDS_COVERAGE_FIX":
        lines.append("- Fill missing cached MDIQA coverage for KonIQ/SPAQ, then rerun this dry-run only.")
    elif decision == "KNOWN_DATASET_FULL_LABEL_NEEDS_RANGE_REVIEW":
        lines.append("- Review source range audit before any final label generation.")
    elif decision == "KNOWN_DATASET_FULL_LABEL_TOO_RISKY":
        lines.append("- Stop. Review safety cases before expanding labels.")
    else:
        lines.append("- Fix construction failure and rerun this dry-run only.")
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = output_paths()

    load_recommended_policy()
    expectations = load_scope_expectations()
    base = load_base_metadata()
    mdiqa = load_mdiqa_unique()
    paired_prod = load_paired_production()
    flive_prod = load_flive_production()

    rows, failures = build_known_candidate_rows(base, mdiqa, paired_prod, flive_prod)
    rows, params = apply_policy_f_shift_cap3(rows)
    rows.sort(key=lambda row: (row["dataset"], row["image_path"]))

    duplicate_paths = len(rows) - len({row["image_path"] for row in rows})
    if duplicate_paths:
        failures.append({"stage": "duplicate_check", "dataset": "", "image_path": "", "error": f"duplicate_image_path_count={duplicate_paths}"})

    validation = build_validation(rows)
    distribution_rows = build_distribution(rows)
    safety_cases = build_safety_cases(rows)
    range_audit = build_range_audit(rows)
    scope_summary = build_scope_summary(rows, expectations, failures)
    decision = decide(rows, scope_summary, safety_cases, range_audit)

    write_csv(paths["manifest"], rows, MANIFEST_COLUMNS)
    write_csv(
        paths["validation"],
        validation,
        [
            "section",
            "dataset",
            "split",
            "content_type",
            "selected_teacher_source",
            "n",
            "srcc",
            "plcc",
            "mae",
            "linear_aligned_mae",
            "fp",
            "severe_fp",
            "fn",
            "clipping_count",
            "boundary_0_or_100_count",
            "boundary_0_or_100_pct",
        ],
    )
    write_csv(
        paths["distribution"],
        distribution_rows,
        ["grouping", "dataset", "selected_teacher_source", "n", "min", "max", "mean", "std", "p01", "p05", "p25", "p50", "p75", "p95", "p99"],
    )
    write_csv(
        paths["safety"],
        safety_cases,
        [
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
        ],
    )
    write_csv(paths["failures"], failures, ["stage", "dataset", "image_path", "error"])
    write_csv(
        paths["range"],
        range_audit,
        [
            "audit",
            "group_a",
            "group_b",
            "mean_a",
            "mean_b",
            "mean_shift",
            "std_a",
            "std_b",
            "std_ratio",
            "p05_a",
            "p05_b",
            "p95_a",
            "p95_b",
            "p95_p05_gap_mismatch",
            "phase5_10k_mean_shift",
            "phase5_10k_std_ratio",
            "phase5_10k_gap_mismatch",
            "delta_vs_phase5_10k_mean_shift",
            "delta_vs_phase5_10k_std_ratio",
            "delta_vs_phase5_10k_gap_mismatch",
            "flag_mean_shift_gt10",
            "flag_std_ratio",
            "flag_gap_mismatch_gt20",
            "rows_total",
            "clipped_count",
            "clipped_pct",
            "near_0_or_100_count",
            "near_0_or_100_pct",
            "near_5_or_95_count",
            "near_5_or_95_pct",
            "flag_clipped_pct_gt5",
            "error",
        ],
    )
    write_csv(
        paths["scope"],
        scope_summary,
        [
            "dataset",
            "expected_or_estimated_rows",
            "valid_rows",
            "coverage_gap",
            "status",
            "included_in_main_labels",
            "excluded_reason",
        ],
    )
    next_actions = [
        {
            "priority": 1,
            "action": "Review known-dataset full-label dry-run outputs.",
            "status": "ready" if decision == "KNOWN_DATASET_FULL_LABEL_DRYRUN_READY_FOR_REVIEW" else "blocked",
            "notes": decision,
        },
        {
            "priority": 2,
            "action": "Fill cached MDIQA coverage for missing KonIQ/SPAQ rows before final labels.",
            "status": "required" if decision == "KNOWN_DATASET_FULL_LABEL_NEEDS_COVERAGE_FIX" else "not_required",
            "notes": "Do not invent fallback labels.",
        },
        {
            "priority": 3,
            "action": "Do not run 50k/full label generation or student distillation from this dry-run.",
            "status": "open",
            "notes": "Dry-run/audit only.",
        },
    ]
    write_csv(paths["next"], next_actions, ["priority", "action", "status", "notes"])
    write_report(paths, rows, validation, distribution_rows, safety_cases, range_audit, scope_summary, failures, decision, params)

    for key in ["manifest", "validation", "distribution", "safety", "failures", "range", "scope", "report", "next"]:
        print(f"Wrote {paths[key]}")
    counts = {dataset: len(sub) for (dataset,), sub in group_rows(rows, ["dataset"]).items()}
    print(f"Rows: {len(rows)} valid: {sum(1 for row in rows if row.get('target_available'))}")
    print(f"Counts: {counts}")
    print(f"Final decision: {decision}")


if __name__ == "__main__":
    main()
