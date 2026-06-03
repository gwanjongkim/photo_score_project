# Phase 7 학생 distillation용 14105행 teacher-cache 매니페스트를 준비한다.
from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

INPUT_MANIFEST = ROOT / "outputs" / "known_dataset_full_label_dryrun_20260603" / "known_dataset_full_label_dryrun_manifest.csv"
COVERAGE_REPORT = ROOT / "outputs" / "known_dataset_full_label_coverage_audit_20260603" / "report.md"
PHASE6_REPORT = ROOT / "outputs" / "known_dataset_full_label_dryrun_20260603" / "report.md"
OUTPUT_DIR = ROOT / "outputs" / "student_distillation_manifest_14105_20260603"

SEED = 20260603
EXPECTED_ROWS = 14105
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
SPLIT_ORDER = ["train", "val", "test"]
KNOWN_DATASETS = {"koniq10k", "spaq", "flive"}

INPUT_REQUIRED = [
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
    "selected_teacher_source",
    "selected_policy",
    "target_available",
]

MANIFEST_COLUMNS = [
    "dataset",
    "split",
    "split_original",
    "content_type",
    "image_path",
    "source_group",
    "quality_score_0_100",
    "selected_teacher_target_0_100",
    "target_norm",
    "selected_teacher_source",
    "selected_policy",
    "mdiqa_variant",
    "mdiqa_pred_0_100",
    "production_pred_0_100",
]


def output_paths() -> dict[str, Path]:
    return {
        "manifest": OUTPUT_DIR / "student_distillation_manifest.csv",
        "train": OUTPUT_DIR / "train_manifest.csv",
        "val": OUTPUT_DIR / "val_manifest.csv",
        "test": OUTPUT_DIR / "test_manifest.csv",
        "split_summary": OUTPUT_DIR / "split_summary.csv",
        "target_distribution": OUTPUT_DIR / "target_distribution_summary.csv",
        "leakage": OUTPUT_DIR / "leakage_audit.csv",
        "balance": OUTPUT_DIR / "dataset_balance_summary.csv",
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


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    val = safe_float(value)
    return f"{val:.{digits}f}" if math.isfinite(val) else ""


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


def require_columns(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError(f"input manifest has no rows: {INPUT_MANIFEST}")
    missing = [col for col in INPUT_REQUIRED if col not in rows[0]]
    if missing:
        raise RuntimeError(f"input manifest missing columns: {missing}")


def normalize_input_rows(input_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for idx, row in enumerate(input_rows, start=2):
        dataset = row.get("dataset", "")
        image_path = row.get("image_path", "")
        source_group = row.get("source_group") or image_path
        target = safe_float(row.get("selected_teacher_target_0_100"))
        target_norm = target / 100.0 if math.isfinite(target) else float("nan")
        target_available = str(row.get("target_available", "")).lower() == "true"
        reason = ""
        if dataset not in KNOWN_DATASETS:
            reason = "non-known dataset"
        elif not image_path:
            reason = "missing image_path"
        elif image_path in seen_paths:
            reason = "duplicate image_path"
        elif not target_available:
            reason = "target_available false"
        elif not (0.0 <= target <= 100.0):
            reason = "target outside [0,100]"
        elif not (0.0 <= target_norm <= 1.0):
            reason = "target_norm outside [0,1]"
        if reason:
            invalid.append({"line": idx, "dataset": dataset, "image_path": image_path, "reason": reason})
            continue
        seen_paths.add(image_path)
        rows.append(
            {
                "dataset": dataset,
                "split": "",
                "split_original": row.get("split", ""),
                "content_type": row.get("content_type", ""),
                "image_path": image_path,
                "source_group": source_group,
                "quality_score_0_100": row.get("quality_score_0_100", ""),
                "selected_teacher_target_0_100": target,
                "target_norm": target_norm,
                "selected_teacher_source": row.get("selected_teacher_source", ""),
                "selected_policy": row.get("selected_policy", ""),
                "mdiqa_variant": row.get("mdiqa_variant", ""),
                "mdiqa_pred_0_100": row.get("mdiqa_pred_0_100", ""),
                "production_pred_0_100": row.get("production_pred_0_100", ""),
            }
        )
    return rows, invalid


def split_dataset_groups(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(SEED)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["dataset"]][row["source_group"]].append(row)

    split_targets: dict[str, dict[str, int]] = {}
    split_group_counts: Counter[tuple[str, str]] = Counter()
    for dataset in sorted(grouped):
        groups = [{"source_group": sg, "rows": group_rows} for sg, group_rows in sorted(grouped[dataset].items())]
        rng.shuffle(groups)
        total = sum(len(group["rows"]) for group in groups)
        train_target = int(total * SPLIT_RATIOS["train"])
        val_target = int(total * SPLIT_RATIOS["val"])
        targets = {"train": train_target, "val": val_target, "test": total - train_target - val_target}
        split_targets[dataset] = targets
        assigned = {split: 0 for split in SPLIT_ORDER}
        for group in groups:
            deficits = {split: targets[split] - assigned[split] for split in SPLIT_ORDER}
            chosen = max(SPLIT_ORDER, key=lambda split: (deficits[split], -assigned[split], split))
            for row in group["rows"]:
                row["split"] = chosen
            assigned[chosen] += len(group["rows"])
            split_group_counts[(dataset, chosen)] += 1

    metadata = {
        "seed": SEED,
        "group_count": sum(len(groups) for groups in grouped.values()),
        "multi_row_group_count": sum(1 for groups in grouped.values() for group_rows in groups.values() if len(group_rows) > 1),
        "split_targets": split_targets,
        "split_group_counts": dict(split_group_counts),
    }
    return rows, metadata


def group_rows(rows: list[dict[str, Any]], cols: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(col, "") for col in cols)].append(row)
    return dict(grouped)


def metric_row(rows: list[dict[str, Any]], extra: dict[str, Any]) -> dict[str, Any]:
    pairs = []
    for row in rows:
        pred = safe_float(row.get("selected_teacher_target_0_100"))
        mos = safe_float(row.get("quality_score_0_100"))
        if math.isfinite(pred) and math.isfinite(mos):
            pairs.append((pred, mos))
    pred_values = [pred for pred, _ in pairs]
    mos_values = [mos for _, mos in pairs]
    return {
        **extra,
        "n": len(rows),
        "mos_n": len(pairs),
        "srcc": spearman(pred_values, mos_values),
        "plcc": pearson(pred_values, mos_values),
        "mae": mean([abs(pred - mos) for pred, mos in pairs]),
        "linear_aligned_mae": linear_aligned_mae(pred_values, mos_values),
        "fp": sum(1 for pred, mos in pairs if mos < 40 and pred > 60),
        "severe_fp": sum(1 for pred, mos in pairs if mos < 40 and pred > 65),
        "fn": sum(1 for pred, mos in pairs if mos > 75 and pred < 45),
    }


def build_split_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(rows)
    out: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
        sub = [row for row in rows if row["split"] == split]
        metric = metric_row(sub, {"split": split})
        metric.update(
            {
                "row_count": len(sub),
                "row_pct": len(sub) / total if total else float("nan"),
                "target_pct": SPLIT_RATIOS[split],
                "pct_deviation": (len(sub) / total - SPLIT_RATIOS[split]) if total else float("nan"),
                "source_group_count": len({row["source_group"] for row in sub}),
                "image_path_count": len({row["image_path"] for row in sub}),
            }
        )
        out.append(metric)
    return out


def build_distribution_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cols in [["split"], ["split", "dataset"], ["split", "selected_teacher_source"]]:
        for key, sub in sorted(group_rows(rows, cols).items()):
            row = {"grouping": "+".join(cols)}
            row.update({col: val for col, val in zip(cols, key)})
            row.update(distribution([safe_float(item["selected_teacher_target_0_100"]) for item in sub]))
            out.append(row)
    return out


def build_balance_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    total_by_dataset = Counter(row["dataset"] for row in rows)
    total_by_source = Counter(row["selected_teacher_source"] for row in rows)
    split_totals = Counter(row["split"] for row in rows)
    for split in SPLIT_ORDER:
        split_rows = [row for row in rows if row["split"] == split]
        for dataset in sorted(total_by_dataset):
            count = sum(1 for row in split_rows if row["dataset"] == dataset)
            out.append(
                {
                    "grouping": "dataset",
                    "split": split,
                    "key": dataset,
                    "row_count": count,
                    "split_pct": count / split_totals[split] if split_totals[split] else float("nan"),
                    "overall_pct": total_by_dataset[dataset] / len(rows) if rows else float("nan"),
                    "expected_count_at_split_ratio": total_by_dataset[dataset] * SPLIT_RATIOS[split],
                    "count_delta_vs_expected": count - total_by_dataset[dataset] * SPLIT_RATIOS[split],
                }
            )
        for source in sorted(total_by_source):
            count = sum(1 for row in split_rows if row["selected_teacher_source"] == source)
            out.append(
                {
                    "grouping": "selected_teacher_source",
                    "split": split,
                    "key": source,
                    "row_count": count,
                    "split_pct": count / split_totals[split] if split_totals[split] else float("nan"),
                    "overall_pct": total_by_source[source] / len(rows) if rows else float("nan"),
                    "expected_count_at_split_ratio": total_by_source[source] * SPLIT_RATIOS[split],
                    "count_delta_vs_expected": count - total_by_source[source] * SPLIT_RATIOS[split],
                }
            )
    return out


def build_leakage_audit(rows: list[dict[str, Any]], invalid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    image_paths = [row["image_path"] for row in rows]
    duplicate_count = len(image_paths) - len(set(image_paths))
    source_to_splits: dict[str, set[str]] = defaultdict(set)
    path_to_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source_to_splits[row["source_group"]].add(row["split"])
        path_to_splits[row["image_path"]].add(row["split"])
    source_group_leakage = sum(1 for splits in source_to_splits.values() if len(splits) > 1)
    image_path_split_overlap = sum(1 for splits in path_to_splits.values() if len(splits) > 1)
    target_bad = sum(1 for row in rows if not (0.0 <= safe_float(row["selected_teacher_target_0_100"]) <= 100.0))
    norm_bad = sum(1 for row in rows if not (0.0 <= safe_float(row["target_norm"]) <= 1.0))
    missing_split = sum(1 for row in rows if row["split"] not in SPLIT_ORDER)
    split_counts = Counter(row["split"] for row in rows)
    dataset_split_missing = []
    for dataset in sorted({row["dataset"] for row in rows}):
        present = {row["split"] for row in rows if row["dataset"] == dataset}
        for split in SPLIT_ORDER:
            if split not in present:
                dataset_split_missing.append(f"{dataset}:{split}")
    checks = [
        ("input_rows", len(rows) + len(invalid_rows), "pass" if len(rows) + len(invalid_rows) == EXPECTED_ROWS else "fail"),
        ("valid_rows", len(rows), "pass" if len(rows) == EXPECTED_ROWS else "warn"),
        ("removed_invalid_rows", len(invalid_rows), "pass" if not invalid_rows else "warn"),
        ("duplicate_image_path", duplicate_count, "pass" if duplicate_count == 0 else "fail"),
        ("image_path_overlap_across_splits", image_path_split_overlap, "pass" if image_path_split_overlap == 0 else "fail"),
        ("source_group_leakage_across_splits", source_group_leakage, "pass" if source_group_leakage == 0 else "fail"),
        ("target_out_of_range_0_100", target_bad, "pass" if target_bad == 0 else "fail"),
        ("target_norm_out_of_range_0_1", norm_bad, "pass" if norm_bad == 0 else "fail"),
        ("missing_student_split", missing_split, "pass" if missing_split == 0 else "fail"),
        ("dataset_missing_from_split", ";".join(dataset_split_missing), "pass" if not dataset_split_missing else "fail"),
        ("train_rows", split_counts.get("train", 0), "pass" if split_counts.get("train", 0) > 0 else "fail"),
        ("val_rows", split_counts.get("val", 0), "pass" if split_counts.get("val", 0) > 0 else "fail"),
        ("test_rows", split_counts.get("test", 0), "pass" if split_counts.get("test", 0) > 0 else "fail"),
    ]
    return [{"check": check, "value": value, "status": status} for check, value, status in checks]


def balance_is_acceptable(split_summary: list[dict[str, Any]]) -> bool:
    return all(abs(safe_float(row["pct_deviation"])) <= 0.005 for row in split_summary)


def decide(
    rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
    leakage_audit: list[dict[str, Any]],
    split_summary: list[dict[str, Any]],
    safety_rows: list[dict[str, Any]],
) -> str:
    failed = {row["check"] for row in leakage_audit if row["status"] == "fail"}
    if not rows or len(rows) + len(invalid_rows) != EXPECTED_ROWS:
        return "STUDENT_MANIFEST_14105_FAILED"
    if {"target_out_of_range_0_100", "target_norm_out_of_range_0_1"} & failed:
        return "STUDENT_MANIFEST_14105_NEEDS_TARGET_FIX"
    if {"duplicate_image_path", "image_path_overlap_across_splits", "source_group_leakage_across_splits"} & failed:
        return "STUDENT_MANIFEST_14105_NEEDS_LEAKAGE_FIX"
    if {"missing_student_split", "dataset_missing_from_split"} & failed or not balance_is_acceptable(split_summary):
        return "STUDENT_MANIFEST_14105_NEEDS_SPLIT_FIX"
    severe_fp = sum(int(row.get("severe_fp", 0)) for row in safety_rows)
    fp = sum(int(row.get("fp", 0)) for row in safety_rows)
    if severe_fp > 0 or fp > max(5, int(0.005 * len(rows))):
        return "STUDENT_MANIFEST_14105_NEEDS_TARGET_FIX"
    if invalid_rows:
        return "STUDENT_MANIFEST_14105_NEEDS_TARGET_FIX"
    return "STUDENT_MANIFEST_14105_READY_FOR_SMOKE_TRAINING"


def write_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def write_report(
    paths: dict[str, Path],
    rows: list[dict[str, Any]],
    split_summary: list[dict[str, Any]],
    balance_rows: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
    leakage_rows: list[dict[str, Any]],
    safety_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
    split_metadata: dict[str, Any],
    decision: str,
) -> None:
    dataset_counts = [
        {"dataset": key[0], "n": len(sub)}
        for key, sub in sorted(group_rows(rows, ["dataset"]).items())
    ]
    split_counts = [
        {"split": split, "n": len([row for row in rows if row["split"] == split])}
        for split in SPLIT_ORDER
    ]
    balance_dataset = [row for row in balance_rows if row["grouping"] == "dataset"]
    leakage_failures = [row for row in leakage_rows if row["status"] == "fail"]
    target_split_rows = [row for row in distribution_rows if row["grouping"] == "split"]
    lines = [
        "# Student Distillation Manifest 14105 Preparation Report",
        "",
        "## 1. Executive Summary",
        f"- Final decision: `{decision}`.",
        f"- Rows: `{len(rows)}` valid, `{len(invalid_rows)}` removed.",
        f"- Split seed: `{SEED}`.",
        "- Scope: validated known-dataset teacher-cache subset, not MDIQA official full-dataset relabeling.",
        "",
        "## 2. Sources Inspected",
        f"- `{INPUT_MANIFEST}`",
        f"- `{COVERAGE_REPORT}`",
        f"- `{PHASE6_REPORT}`",
        "- Existing scripts inspected: `make_student_training_manifest.py`, `make_student_training_manifest_50k.py`.",
        "",
        "## 3. Scope Statement",
        "- Option A: keep current validated cache-subset direction.",
        "- KonIQ-10k: MDIQA seed1 + policy_f_shift_cap3.",
        "- SPAQ: MDIQA xd + policy_f_shift_cap3.",
        "- FLIVE: Production technical score.",
        "- Unknown/fallback excluded; c6_parity holdout only; Test_weight/product-gold validation only.",
        "- No extra MDIQA inference, training, TFLite export, Flutter, production scoring, or model asset changes.",
        "",
        "## 4. Manifest Construction",
        "- Kept `selected_teacher_target_0_100` as main target.",
        "- Added `target_norm = selected_teacher_target_0_100 / 100.0`.",
        "- Preserved original split as `split_original`; new student split stored in `split`.",
        "- Split unit: `source_group` when present, falling back to `image_path`.",
        "",
        "## 5. Train/Val/Test Split",
        write_table(split_counts, ["split", "n"]),
        "",
        write_table(split_summary, ["split", "row_count", "row_pct", "target_pct", "pct_deviation", "source_group_count", "image_path_count"]),
        f"- Source groups: `{split_metadata.get('group_count')}` total; multi-row groups: `{split_metadata.get('multi_row_group_count')}`.",
        "",
        "## 6. Dataset Balance",
        write_table(dataset_counts, ["dataset", "n"]),
        "",
        write_table(balance_dataset, ["split", "key", "row_count", "split_pct", "overall_pct", "count_delta_vs_expected"]),
        "",
        "## 7. Target Distribution",
        write_table(target_split_rows, ["split", "n", "min", "max", "mean", "std", "p05", "p50", "p95"]),
        f"- Full distribution CSV: `{paths['target_distribution']}`.",
        "",
        "## 8. Leakage Audit",
    ]
    lines.append(write_table(leakage_rows, ["check", "value", "status"]))
    lines.extend(["", "## 9. Safety Checks"])
    lines.append(write_table(safety_rows, ["split", "mos_n", "srcc", "plcc", "mae", "linear_aligned_mae", "fp", "severe_fp", "fn"]))
    lines.extend(
        [
            "",
            "## 10. Limitations",
            "- This is a validated teacher-cache subset, not full official KonIQ/SPAQ/FLIVE relabeling.",
            "- SPAQ coverage follows correct-protocol cached test split only.",
            "- Real-mobile/unknown fallback labels remain excluded from main supervision.",
            "",
            "## 11. Final Decision",
            decision,
            "",
            "## 12. Recommended Next Step",
        ]
    )
    if decision == "STUDENT_MANIFEST_14105_READY_FOR_SMOKE_TRAINING":
        lines.append("- Run a bounded smoke training job only; do not start full distillation yet.")
    elif decision == "STUDENT_MANIFEST_14105_NEEDS_SPLIT_FIX":
        lines.append("- Fix split balance or missing dataset split coverage, then rerun this script.")
    elif decision == "STUDENT_MANIFEST_14105_NEEDS_LEAKAGE_FIX":
        lines.append("- Fix duplicate/path/source_group leakage before any training.")
    elif decision == "STUDENT_MANIFEST_14105_NEEDS_TARGET_FIX":
        lines.append("- Fix target validity or safety issue before any training.")
    else:
        lines.append("- Fix manifest construction failure and rerun this script only.")
    if leakage_failures:
        lines.extend(["", "Leakage failures:", write_table(leakage_failures, ["check", "value", "status"])])
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = output_paths()

    raw_rows = read_csv(INPUT_MANIFEST)
    require_columns(raw_rows)
    rows, invalid_rows = normalize_input_rows(raw_rows)
    rows, split_metadata = split_dataset_groups(rows)
    rows.sort(key=lambda row: (SPLIT_ORDER.index(row["split"]), row["dataset"], row["image_path"]))

    split_summary = build_split_summary(rows)
    distribution_rows = build_distribution_summary(rows)
    balance_rows = build_balance_summary(rows)
    leakage_rows = build_leakage_audit(rows, invalid_rows)
    safety_rows = [metric_row([row for row in rows if row["split"] == split], {"split": split}) for split in SPLIT_ORDER]
    decision = decide(rows, invalid_rows, leakage_rows, split_summary, safety_rows)

    write_csv(paths["manifest"], rows, MANIFEST_COLUMNS)
    for split in SPLIT_ORDER:
        write_csv(paths[split], [row for row in rows if row["split"] == split], MANIFEST_COLUMNS)
    write_csv(
        paths["split_summary"],
        split_summary,
        [
            "split",
            "row_count",
            "row_pct",
            "target_pct",
            "pct_deviation",
            "source_group_count",
            "image_path_count",
            "n",
            "mos_n",
            "srcc",
            "plcc",
            "mae",
            "linear_aligned_mae",
            "fp",
            "severe_fp",
            "fn",
        ],
    )
    write_csv(
        paths["target_distribution"],
        distribution_rows,
        ["grouping", "split", "dataset", "selected_teacher_source", "n", "min", "max", "mean", "std", "p01", "p05", "p25", "p50", "p75", "p95", "p99"],
    )
    write_csv(paths["leakage"], leakage_rows, ["check", "value", "status"])
    write_csv(
        paths["balance"],
        balance_rows,
        ["grouping", "split", "key", "row_count", "split_pct", "overall_pct", "expected_count_at_split_ratio", "count_delta_vs_expected"],
    )
    next_actions = [
        {
            "priority": 1,
            "action": "Review Phase 7 student distillation manifest outputs.",
            "status": "ready" if decision == "STUDENT_MANIFEST_14105_READY_FOR_SMOKE_TRAINING" else "blocked",
            "notes": decision,
        },
        {
            "priority": 2,
            "action": "Run bounded smoke training only after review.",
            "status": "candidate" if decision == "STUDENT_MANIFEST_14105_READY_FOR_SMOKE_TRAINING" else "blocked",
            "notes": "No full distillation from this task.",
        },
        {
            "priority": 3,
            "action": "Do not export TFLite, edit Flutter, or modify production scoring.",
            "status": "open",
            "notes": "Manifest preparation only.",
        },
    ]
    write_csv(paths["next"], next_actions, ["priority", "action", "status", "notes"])
    write_report(paths, rows, split_summary, balance_rows, distribution_rows, leakage_rows, safety_rows, invalid_rows, split_metadata, decision)

    counts = Counter(row["split"] for row in rows)
    dataset_by_split = {
        split: dict(Counter(row["dataset"] for row in rows if row["split"] == split))
        for split in SPLIT_ORDER
    }
    print(f"Wrote {paths['manifest']}")
    print(f"Rows: {len(rows)} valid; removed: {len(invalid_rows)}")
    print(f"Split counts: {dict(counts)}")
    print(f"Dataset by split: {dataset_by_split}")
    print(f"Final decision: {decision}")


if __name__ == "__main__":
    main()
