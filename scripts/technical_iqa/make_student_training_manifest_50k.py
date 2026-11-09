# 모바일 technical IQA v3 학습용 50k 매니페스트를 생성하는 스크립트
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_GUARDED = "outputs/topiq_teacher_label_expansion_50k_20260601/guarded_labels.csv"
DEFAULT_VALIDATION = "outputs/topiq_teacher_validation_20260531/validation_manifest.csv"
DEFAULT_HOLDOUT = "outputs/topiq_teacher_guard_holdout_20260601/holdout_manifest.csv"
DEFAULT_OUT = "outputs/student_training_manifest_50k_20260601/student_training_manifest.csv"

REQUIRED_COLUMNS = [
    "image_path",
    "dataset",
    "split",
    "quality_score_0_100",
    "topiq_raw",
    "topiq_raw_x100",
    "calibrated_topiq_0_100",
    "guarded_target_0_100",
]
MIN_OUTPUT_COLUMNS = [
    "image_path",
    "dataset",
    "split",
    "source_filename",
    "source_group",
    "student_split",
    "target_decile",
    "quality_score_0_100",
    "topiq_raw",
    "topiq_raw_x100",
    "calibrated_topiq_0_100",
    "guarded_target_0_100",
]
SCORE_COLUMNS_0_100 = [
    "quality_score_0_100",
    "topiq_raw_x100",
    "calibrated_topiq_0_100",
    "guarded_target_0_100",
]
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_ORDER = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a group-safe 50k student training manifest from guarded TOPIQ labels."
    )
    parser.add_argument("--guarded_labels_csv", default=DEFAULT_GUARDED)
    parser.add_argument("--validation_manifest_csv", default=DEFAULT_VALIDATION)
    parser.add_argument("--holdout_manifest_csv", default=DEFAULT_HOLDOUT)
    parser.add_argument("--out_csv", default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260601)
    return parser.parse_args()


def read_image_paths(path: Path) -> set[str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return {
            row.get("image_path", "")
            for row in csv.DictReader(f)
            if row.get("image_path")
        }


def source_filename_from_path(image_path: str) -> str:
    return Path(image_path).name


def source_group_from_path(image_path: str) -> str:
    filename = source_filename_from_path(image_path)
    normalized = image_path.replace("\\", "/")
    if "/patches/" in normalized and "_patch_" in filename:
        return filename.split("_patch_", 1)[0]
    return filename


def require_columns(df: pd.DataFrame) -> list[str]:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"guarded_labels_csv is missing required columns: {missing}")
    return missing


def input_validation_summary(df: pd.DataFrame) -> dict[str, Any]:
    numeric = df[SCORE_COLUMNS_0_100].apply(pd.to_numeric, errors="coerce")
    missing_values = {col: int(df[col].isna().sum()) for col in REQUIRED_COLUMNS}
    missing_values.update({col: int(numeric[col].isna().sum()) for col in SCORE_COLUMNS_0_100})
    guarded = numeric["guarded_target_0_100"]
    range_issues = {
        col: int(((numeric[col] < 0.0) | (numeric[col] > 100.0) | numeric[col].isna()).sum())
        for col in SCORE_COLUMNS_0_100
    }
    return {
        "input_row_count": int(len(df)),
        "dataset_distribution": dict(sorted(Counter(df["dataset"].astype(str)).items())),
        "missing_values": missing_values,
        "duplicate_image_path_count": int(df["image_path"].duplicated().sum()),
        "guarded_target_min": float(guarded.min()),
        "guarded_target_max": float(guarded.max()),
        "guarded_target_mean": float(guarded.mean()),
        "guarded_target_std": float(guarded.std(ddof=0)),
        "score_range_issues": range_issues,
    }


def add_target_deciles(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = df.copy()
    qcut_note = "qcut"
    try:
        deciles = pd.qcut(
            work["guarded_target_0_100"].astype(float),
            q=10,
            labels=False,
            duplicates="drop",
        )
        if deciles.isna().any():
            raise ValueError("qcut returned NaN bins")
    except Exception as exc:
        qcut_note = f"qcut failed, used rank-based cut fallback: {exc!r}"
        ranks = work["guarded_target_0_100"].astype(float).rank(method="first")
        deciles = pd.cut(ranks, bins=10, labels=False, include_lowest=True)
    work["target_decile"] = deciles.astype(int).map(lambda value: f"d{value}")
    bin_count = int(work["target_decile"].nunique())
    if bin_count < 10 and qcut_note == "qcut":
        qcut_note = f"qcut used duplicates='drop' and produced {bin_count} bins"
    return work, {"method": qcut_note, "bin_count": bin_count}


def group_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source_group, group_df in df.groupby("source_group", sort=False):
        strat_counts = Counter(group_df["strat_key"].tolist())
        strat_key = sorted(strat_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        records.append(
            {
                "source_group": source_group,
                "row_indices": list(group_df.index),
                "row_count": int(len(group_df)),
                "strat_key": strat_key,
            }
        )
    return records


def split_groups(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = random.Random(seed)
    work = df.copy()
    groups = group_records(work)
    by_strat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_strat[group["strat_key"]].append(group)

    split_group_counts: Counter[str] = Counter()
    split_by_index: dict[int, str] = {}
    bucket_sizes = {key: sum(group["row_count"] for group in value) for key, value in by_strat.items()}
    for strat_key in sorted(by_strat):
        bucket = by_strat[strat_key]
        rng.shuffle(bucket)
        bucket_total = sum(group["row_count"] for group in bucket)
        targets = {split: bucket_total * ratio for split, ratio in SPLIT_RATIOS.items()}
        assigned_counts = {split: 0 for split in SPLIT_ORDER}
        for group in sorted(bucket, key=lambda item: (-item["row_count"], item["source_group"])):
            deficits = {
                split: targets[split] - assigned_counts[split]
                for split in SPLIT_ORDER
            }
            chosen = max(SPLIT_ORDER, key=lambda split: (deficits[split], -assigned_counts[split]))
            for row_index in group["row_indices"]:
                split_by_index[row_index] = chosen
            assigned_counts[chosen] += group["row_count"]
            split_group_counts[chosen] += 1

    work["student_split"] = pd.Series(split_by_index).reindex(work.index).values

    metadata = {
        "group_count": len(groups),
        "multi_row_group_count": sum(1 for group in groups if group["row_count"] > 1),
        "max_rows_per_source_group": max((group["row_count"] for group in groups), default=0),
        "strat_key_count": len(by_strat),
        "bucket_size_min": min(bucket_sizes.values()) if bucket_sizes else 0,
        "bucket_size_max": max(bucket_sizes.values()) if bucket_sizes else 0,
        "small_buckets": {key: value for key, value in sorted(bucket_sizes.items()) if value < 30},
        "split_group_counts": dict(split_group_counts),
    }
    return work, metadata


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    for split in SPLIT_ORDER:
        split_df = df[df["student_split"] == split]
        row_count = int(len(split_df))
        group_count = int(split_df["source_group"].nunique())
        target_pct = SPLIT_RATIOS[split]
        actual_pct = row_count / total if total else 0.0
        rows.append(
            {
                "student_split": split,
                "row_count": row_count,
                "group_count": group_count,
                "actual_pct": actual_pct,
                "target_pct": target_pct,
                "pct_deviation": actual_pct - target_pct,
            }
        )
    return pd.DataFrame(rows)


def grouped_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (
        df.groupby(columns, dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values(columns)
    )


def source_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source_group, group_df in df.groupby("source_group", sort=False):
        rows.append(
            {
                "source_group": source_group,
                "row_count": int(len(group_df)),
                "dataset_count": int(group_df["dataset"].nunique()),
                "datasets": ",".join(sorted(group_df["dataset"].astype(str).unique())),
                "student_split": ",".join(sorted(group_df["student_split"].astype(str).unique())),
                "target_decile_count": int(group_df["target_decile"].nunique()),
                "target_deciles": ",".join(sorted(group_df["target_decile"].astype(str).unique())),
            }
        )
    return pd.DataFrame(rows).sort_values(["row_count", "source_group"], ascending=[False, True])


def leakage_checks(df: pd.DataFrame, validation_paths: set[str], holdout_paths: set[str]) -> dict[str, Any]:
    image_paths = set(df["image_path"].tolist())
    duplicate_count = int(df["image_path"].duplicated().sum())
    validation_overlap = len(image_paths & validation_paths)
    holdout_overlap = len(image_paths & holdout_paths)
    group_split_counts = df.groupby("source_group")["student_split"].nunique()
    source_group_overlap_count = int((group_split_counts > 1).sum())
    datasets_by_split = {
        split: sorted(df[df["student_split"] == split]["dataset"].astype(str).unique().tolist())
        for split in SPLIT_ORDER
    }
    all_splits_non_empty = all(len(df[df["student_split"] == split]) > 0 for split in SPLIT_ORDER)
    expected_datasets = set(df["dataset"].astype(str).unique())
    each_dataset_each_split = all(
        set(datasets_by_split[split]) == expected_datasets for split in SPLIT_ORDER
    )
    reasonable_split_rows = all(len(df[df["student_split"] == split]) >= 100 for split in SPLIT_ORDER)
    return {
        "image_path_duplicate_count": duplicate_count,
        "validation_image_path_overlap": validation_overlap,
        "holdout_image_path_overlap": holdout_overlap,
        "source_group_overlap_across_student_splits": source_group_overlap_count,
        "all_splits_non_empty": all_splits_non_empty,
        "each_dataset_appears_in_each_split": each_dataset_each_split,
        "reasonable_split_rows": reasonable_split_rows,
        "datasets_by_split": datasets_by_split,
    }


def ratio_is_close(summary_df: pd.DataFrame) -> bool:
    return bool(summary_df["pct_deviation"].abs().max() <= 0.03)


def decile_distribution_ok(decile_df: pd.DataFrame) -> bool:
    pivot = decile_df.pivot_table(
        index="target_decile",
        columns="student_split",
        values="row_count",
        aggfunc="sum",
        fill_value=0,
    )
    if len(pivot) < 8:
        return False
    for split in SPLIT_ORDER:
        if split not in pivot.columns or int((pivot[split] > 0).sum()) < len(pivot):
            return False
    return True


def final_decision(
    checks: dict[str, Any],
    summary_df: pd.DataFrame,
    decile_df: pd.DataFrame,
    required_missing: list[str],
) -> tuple[str, str]:
    if required_missing:
        return "DO_NOT_TRAIN_STUDENT", "Required input columns are missing."
    hard_failures = [
        checks["image_path_duplicate_count"] != 0,
        checks["validation_image_path_overlap"] != 0,
        checks["holdout_image_path_overlap"] != 0,
        checks["source_group_overlap_across_student_splits"] != 0,
    ]
    if any(hard_failures):
        return "DO_NOT_TRAIN_STUDENT", "Leakage, duplicate, or source-group overlap hard-failure check failed."
    if not checks["all_splits_non_empty"] or not checks["reasonable_split_rows"]:
        return "NEEDS_SPLIT_REVISION", "A split is empty or too small."
    if not checks["each_dataset_appears_in_each_split"]:
        return "NEEDS_SPLIT_REVISION", "At least one dataset is missing from a split."
    if not ratio_is_close(summary_df):
        return "NEEDS_SPLIT_REVISION", "Split ratio deviates by more than 3 percentage points."
    if not decile_distribution_ok(decile_df):
        return "NEEDS_SPLIT_REVISION", "Target decile distribution is severely collapsed in at least one split."
    return "READY_FOR_V3_STUDENT_TRAINING", "Manifest passed duplicate, leakage, group, dataset, ratio, and target-decile checks."


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "(empty)"
    render_df = df if max_rows is None else df.head(max_rows)
    columns = list(render_df.columns)
    lines = [
        "| " + " | ".join(str(col) for col in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in render_df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    if max_rows is not None and len(df) > max_rows:
        lines.append(f"| ... | {len(df) - max_rows} more rows |")
    return "\n".join(lines)


def write_report(
    path: Path,
    args: argparse.Namespace,
    input_summary: dict[str, Any],
    qcut_info: dict[str, Any],
    split_meta: dict[str, Any],
    split_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    decile_df: pd.DataFrame,
    dataset_decile_df: pd.DataFrame,
    source_group_df: pd.DataFrame,
    checks: dict[str, Any],
    decision: str,
    decision_reason: str,
) -> None:
    dataset_group_counts = (
        source_group_df.assign(dataset=source_group_df["datasets"].str.split(","))
        .explode("dataset")
        .groupby("dataset")
        .size()
        .reset_index(name="source_group_count")
        .sort_values("dataset")
    )
    lines = [
        "# 50k Student Training Manifest Report",
        "",
        "## 1. Summary",
        f"- guarded_labels_csv: {args.guarded_labels_csv}",
        f"- validation_manifest_csv: {args.validation_manifest_csv}",
        f"- holdout_manifest_csv: {args.holdout_manifest_csv}",
        f"- output_csv: {args.out_csv}",
        f"- seed: {args.seed}",
        f"- input rows: {input_summary['input_row_count']}",
        f"- final decision: {decision}",
        f"- decision reason: {decision_reason}",
        "",
        "## 2. Input Validation",
        f"- dataset distribution: {input_summary['dataset_distribution']}",
        f"- missing values: {input_summary['missing_values']}",
        f"- duplicate image_path count: {input_summary['duplicate_image_path_count']}",
        f"- guarded target min/max/mean/std: {input_summary['guarded_target_min']:.6f} / {input_summary['guarded_target_max']:.6f} / {input_summary['guarded_target_mean']:.6f} / {input_summary['guarded_target_std']:.6f}",
        f"- score range issues outside 0..100: {input_summary['score_range_issues']}",
        "",
        "## 3. Source Group Derivation",
        f"- total source_group count: {split_meta['group_count']}",
        f"- multi-row source_group count: {split_meta['multi_row_group_count']}",
        f"- max rows per source_group: {split_meta['max_rows_per_source_group']}",
        "- patch rule: paths containing `/patches/` use filename before `_patch_`; other paths use filename.",
        "",
        "Dataset-wise source_group counts:",
        dataframe_to_markdown(dataset_group_counts),
        "",
        "## 4. Target Decile Binning",
        f"- method: {qcut_info['method']}",
        f"- bins created: {qcut_info['bin_count']}",
        "",
        "## 5. Split Policy",
        "- target ratio: train 70%, val 15%, test 15%",
        "- split unit: source_group",
        "- stratification key: dataset + `__` + target_decile",
        f"- stratification buckets: {split_meta['strat_key_count']}",
        f"- bucket size min/max: {split_meta['bucket_size_min']} / {split_meta['bucket_size_max']}",
        f"- small buckets below 30 rows: {split_meta['small_buckets']}",
        "- assignment: deterministic greedy per stratification bucket toward row-count targets",
        "",
        "## 6. Split Counts",
        dataframe_to_markdown(split_df),
        "",
        "## 7. Dataset Distribution by Split",
        dataframe_to_markdown(dataset_df),
        "",
        "## 8. Target Decile Distribution by Split",
        dataframe_to_markdown(decile_df),
        "",
        "## 9. Dataset-Decile Distribution by Split",
        dataframe_to_markdown(dataset_decile_df, max_rows=80),
        "",
        "## 10. Leakage Checks",
        f"- image_path duplicate count: {checks['image_path_duplicate_count']}",
        f"- validation image_path overlap: {checks['validation_image_path_overlap']}",
        f"- holdout image_path overlap: {checks['holdout_image_path_overlap']}",
        f"- source_group overlap across student train/val/test: {checks['source_group_overlap_across_student_splits']}",
        f"- all splits non-empty: {checks['all_splits_non_empty']}",
        f"- every dataset appears in train/val/test: {checks['each_dataset_appears_in_each_split']}",
        f"- every split has reasonable rows: {checks['reasonable_split_rows']}",
        f"- datasets by split: {checks['datasets_by_split']}",
        "",
        "## 11. Deviations From Target Ratio",
        "- pass threshold used here: max absolute split-ratio deviation <= 0.03",
        f"- max absolute deviation: {split_df['pct_deviation'].abs().max():.6f}",
        "",
        "## 12. Risks",
        "- Exact stratification can be impossible when a source_group contains multiple patch rows.",
        "- FLIVE has many multi-row source groups, so safety against leakage is prioritized over exact distribution matching.",
        "- This manifest uses guarded teacher-label design targets; it is not a Flutter runtime formula.",
        "",
        "## 13. Final Decision",
        decision,
        "",
        "## 14. Next Training Command Recommendation",
        "Do not run until the v3 trainer exists and this manifest has been reviewed.",
        "",
        "```bash",
        "./.venv_gpu/bin/python scripts/technical_iqa/train_mobile_student_iqa_v3.py \\",
        "  --manifest_csv outputs/student_training_manifest_50k_20260601/student_training_manifest.csv \\",
        "  --out_dir outputs/mobile_student_iqa_v3_20260601 \\",
        "  --head_epochs 5 \\",
        "  --finetune_epochs 20 \\",
        "  --batch_size 32",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_csv = Path(args.out_csv)
    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.guarded_labels_csv)
    required_missing = require_columns(df)
    for col in SCORE_COLUMNS_0_100 + ["topiq_raw"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    input_summary = input_validation_summary(df)
    if required_missing:
        raise ValueError(f"Required columns missing: {required_missing}")

    df["source_filename"] = df["image_path"].astype(str).map(source_filename_from_path)
    df["source_group"] = df["image_path"].astype(str).map(source_group_from_path)
    df, qcut_info = add_target_deciles(df)
    df["strat_key"] = df["dataset"].astype(str) + "__" + df["target_decile"].astype(str)
    df, split_meta = split_groups(df, args.seed)

    validation_paths = read_image_paths(Path(args.validation_manifest_csv))
    holdout_paths = read_image_paths(Path(args.holdout_manifest_csv))
    checks = leakage_checks(df, validation_paths, holdout_paths)

    split_df = split_summary(df)
    dataset_df = grouped_summary(df, ["student_split", "dataset"])
    decile_df = grouped_summary(df, ["student_split", "target_decile"])
    dataset_decile_df = grouped_summary(df, ["student_split", "dataset", "target_decile"])
    source_group_df = source_group_summary(df)
    decision, decision_reason = final_decision(checks, split_df, decile_df, required_missing)

    output_columns = list(dict.fromkeys(MIN_OUTPUT_COLUMNS + [col for col in df.columns if col not in MIN_OUTPUT_COLUMNS]))
    df.to_csv(out_csv, columns=output_columns, index=False)
    split_df.to_csv(out_dir / "student_split_summary.csv", index=False)
    dataset_df.to_csv(out_dir / "student_split_dataset_summary.csv", index=False)
    decile_df.to_csv(out_dir / "student_split_decile_summary.csv", index=False)
    dataset_decile_df.to_csv(out_dir / "student_split_dataset_decile_summary.csv", index=False)
    source_group_df.to_csv(out_dir / "source_group_summary.csv", index=False)
    write_report(
        out_dir / "student_training_manifest_report.md",
        args,
        input_summary,
        qcut_info,
        split_meta,
        split_df,
        dataset_df,
        decile_df,
        dataset_decile_df,
        source_group_df,
        checks,
        decision,
        decision_reason,
    )

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_dir / 'student_training_manifest_report.md'}")
    print(f"Wrote {out_dir / 'student_split_summary.csv'}")
    print(f"Wrote {out_dir / 'student_split_dataset_summary.csv'}")
    print(f"Wrote {out_dir / 'student_split_decile_summary.csv'}")
    print(f"Wrote {out_dir / 'student_split_dataset_decile_summary.csv'}")
    print(f"Wrote {out_dir / 'source_group_summary.csv'}")
    print(f"final_decision={decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
