# 모바일 technical IQA 학생 모델 학습용 매니페스트를 생성하는 스크립트
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_GUARDED = "outputs/topiq_teacher_label_dryrun_20260601/guarded_labels.csv"
DEFAULT_VALIDATION = "outputs/topiq_teacher_validation_20260531/validation_manifest.csv"
DEFAULT_HOLDOUT = "outputs/topiq_teacher_guard_holdout_20260601/holdout_manifest.csv"
DEFAULT_OUT = "outputs/student_training_manifest_design_20260601/student_training_manifest.csv"
REQUIRED_COLUMNS = [
    "image_path",
    "dataset",
    "quality_score_0_100",
    "topiq_raw",
    "calibrated_topiq_0_100",
    "guarded_target_0_100",
]
MIN_OUTPUT_COLUMNS = [
    "image_path",
    "dataset",
    "source_filename",
    "source_group",
    "student_split",
    "target_decile",
    "quality_score_0_100",
    "topiq_raw",
    "calibrated_topiq_0_100",
    "guarded_target_0_100",
]
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_ORDER = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a group-safe student training manifest from guarded TOPIQ labels."
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
    work["student_split"] = ""
    groups = group_records(work)
    by_strat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_strat[group["strat_key"]].append(group)

    split_group_counts: Counter[str] = Counter()
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
            work.loc[group["row_indices"], "student_split"] = chosen
            assigned_counts[chosen] += group["row_count"]
            split_group_counts[chosen] += 1

    metadata = {
        "group_count": len(groups),
        "multi_row_group_count": sum(1 for group in groups if group["row_count"] > 1),
        "strat_key_count": len(by_strat),
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


def dataset_decile_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["student_split", "dataset", "target_decile"], dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values(["student_split", "dataset", "target_decile"])
    )
    return summary


def leakage_checks(
    df: pd.DataFrame, validation_paths: set[str], holdout_paths: set[str]
) -> dict[str, Any]:
    image_paths = set(df["image_path"].tolist())
    duplicate_count = int(df["image_path"].duplicated().sum())
    validation_overlap = len(image_paths & validation_paths)
    holdout_overlap = len(image_paths & holdout_paths)
    group_split_counts = df.groupby("source_group")["student_split"].nunique()
    source_group_overlap_count = int((group_split_counts > 1).sum())
    datasets_by_split = {
        split: sorted(df[df["student_split"] == split]["dataset"].unique().tolist())
        for split in SPLIT_ORDER
    }
    all_splits_non_empty = all(len(df[df["student_split"] == split]) > 0 for split in SPLIT_ORDER)
    expected_datasets = set(df["dataset"].unique())
    each_dataset_each_split = all(
        set(datasets_by_split[split]) == expected_datasets for split in SPLIT_ORDER
    )
    return {
        "image_path_duplicate_count": duplicate_count,
        "validation_image_path_overlap": validation_overlap,
        "holdout_image_path_overlap": holdout_overlap,
        "source_group_overlap_across_student_splits": source_group_overlap_count,
        "all_splits_non_empty": all_splits_non_empty,
        "each_dataset_appears_in_each_split": each_dataset_each_split,
        "datasets_by_split": datasets_by_split,
    }


def ratio_is_close(summary_df: pd.DataFrame) -> bool:
    max_abs_dev = summary_df["pct_deviation"].abs().max()
    return bool(max_abs_dev <= 0.03)


def final_decision(checks: dict[str, Any], summary_df: pd.DataFrame) -> tuple[str, str]:
    hard_failures = [
        checks["image_path_duplicate_count"] != 0,
        checks["validation_image_path_overlap"] != 0,
        checks["holdout_image_path_overlap"] != 0,
        checks["source_group_overlap_across_student_splits"] != 0,
    ]
    if any(hard_failures):
        return "DO_NOT_TRAIN_STUDENT", "Leakage or duplicate hard-failure check failed."
    if not checks["all_splits_non_empty"] or not checks["each_dataset_appears_in_each_split"]:
        return "NEEDS_SPLIT_REVISION", "A split is empty or a dataset is missing from at least one split."
    if not ratio_is_close(summary_df):
        return "NEEDS_SPLIT_REVISION", "Split ratio deviates by more than 3 percentage points."
    return "READY_FOR_FIRST_STUDENT_TRAINING", "Manifest passed duplicate, leakage, group, dataset, and ratio checks."


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(str(col) for col in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    guarded_labels_csv: Path,
    validation_manifest_csv: Path,
    holdout_manifest_csv: Path,
    out_csv: Path,
    input_rows: int,
    required_missing: list[str],
    qcut_info: dict[str, Any],
    split_meta: dict[str, Any],
    summary_df: pd.DataFrame,
    dataset_decile_df: pd.DataFrame,
    checks: dict[str, Any],
    decision: str,
    decision_reason: str,
) -> None:
    dataset_distribution = dataset_decile_df.groupby(["student_split", "dataset"])["row_count"].sum().reset_index()
    decile_distribution = dataset_decile_df.groupby(["student_split", "target_decile"])["row_count"].sum().reset_index()
    lines = [
        "# Student Training Manifest Implementation Report",
        "",
        "## 1. Summary",
        f"- guarded_labels_csv: {guarded_labels_csv}",
        f"- validation_manifest_csv: {validation_manifest_csv}",
        f"- holdout_manifest_csv: {holdout_manifest_csv}",
        f"- output_csv: {out_csv}",
        f"- input rows: {input_rows}",
        f"- final decision: {decision}",
        f"- decision reason: {decision_reason}",
        "",
        "## 2. Input Validation",
        f"- required columns missing: {required_missing}",
        f"- qcut method: {qcut_info['method']}",
        f"- target decile bins: {qcut_info['bin_count']}",
        "",
        "## 3. Source Group Derivation",
        f"- source groups: {split_meta['group_count']}",
        f"- multi-row source groups: {split_meta['multi_row_group_count']}",
        "- patch rule: paths containing `/patches/` use filename before `_patch_`; other paths use filename.",
        "",
        "## 4. Split Policy",
        "- target ratio: train 70%, val 15%, test 15%",
        "- split unit: source_group",
        "- stratification key: dataset + `__` + target_decile",
        f"- stratification buckets: {split_meta['strat_key_count']}",
        "- assignment: deterministic greedy per stratification bucket toward row-count targets",
        "",
        "## 5. Split Counts",
        dataframe_to_markdown(summary_df),
        "",
        "## 6. Dataset Distribution by Split",
        dataframe_to_markdown(dataset_distribution),
        "",
        "## 7. Target Decile Distribution by Split",
        dataframe_to_markdown(decile_distribution),
        "",
        "## 8. Leakage Checks",
        f"- image_path duplicate count: {checks['image_path_duplicate_count']}",
        f"- validation image_path overlap: {checks['validation_image_path_overlap']}",
        f"- holdout image_path overlap: {checks['holdout_image_path_overlap']}",
        f"- source_group overlap across student train/val/test: {checks['source_group_overlap_across_student_splits']}",
        f"- all splits non-empty: {checks['all_splits_non_empty']}",
        f"- each dataset appears in each split: {checks['each_dataset_appears_in_each_split']}",
        f"- datasets by split: {checks['datasets_by_split']}",
        "",
        "## 9. Deviations From Target Ratio",
        "- pass threshold used here: max absolute split-ratio deviation <= 0.03",
        f"- max absolute deviation: {summary_df['pct_deviation'].abs().max():.6f}",
        "",
        "## 10. Risks",
        "- Greedy group stratification optimizes row-count balance inside each dataset/decile bucket but does not guarantee exact global ratios.",
        "- FLIVE patch groups are protected by source_group, which can create small unavoidable stratification deviations.",
        "- This manifest is based on the 9k guarded-label dry run only; it is not the full technical IQA training corpus.",
        "",
        "## 11. Final Decision",
        decision,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    guarded_labels_csv = Path(args.guarded_labels_csv)
    validation_manifest_csv = Path(args.validation_manifest_csv)
    holdout_manifest_csv = Path(args.holdout_manifest_csv)
    out_csv = Path(args.out_csv)
    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(guarded_labels_csv)
    required_missing = require_columns(df)
    df["source_filename"] = df["image_path"].astype(str).map(source_filename_from_path)
    df["source_group"] = df["image_path"].astype(str).map(source_group_from_path)
    df, qcut_info = add_target_deciles(df)
    df["strat_key"] = df["dataset"].astype(str) + "__" + df["target_decile"].astype(str)
    df, split_meta = split_groups(df, args.seed)

    validation_paths = read_image_paths(validation_manifest_csv)
    holdout_paths = read_image_paths(holdout_manifest_csv)
    checks = leakage_checks(df, validation_paths, holdout_paths)
    summary_df = split_summary(df)
    dataset_decile_df = dataset_decile_summary(df)
    decision, decision_reason = final_decision(checks, summary_df)

    report_md = out_dir / "student_training_manifest_report.md"
    split_summary_csv = out_dir / "student_split_summary.csv"
    dataset_decile_csv = out_dir / "student_split_dataset_decile_summary.csv"

    output_columns = list(dict.fromkeys(MIN_OUTPUT_COLUMNS + [col for col in df.columns if col not in MIN_OUTPUT_COLUMNS]))
    df.to_csv(out_csv, columns=output_columns, index=False)
    summary_df.to_csv(split_summary_csv, index=False)
    dataset_decile_df.to_csv(dataset_decile_csv, index=False)
    write_report(
        report_md,
        guarded_labels_csv,
        validation_manifest_csv,
        holdout_manifest_csv,
        out_csv,
        len(df),
        required_missing,
        qcut_info,
        split_meta,
        summary_df,
        dataset_decile_df,
        checks,
        decision,
        decision_reason,
    )

    print(f"Wrote {out_csv}")
    print(f"Wrote {report_md}")
    print(f"Wrote {split_summary_csv}")
    print(f"Wrote {dataset_decile_csv}")
    print(f"final_decision={decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
