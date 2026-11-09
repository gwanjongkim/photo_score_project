# TOPIQ 50k 확장 교사 라벨 매니페스트를 생성하는 스크립트
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT = (
    "outputs/research_protocol_iqa_manifests_20260530/"
    "benchmark_all_research_protocol_iqa.csv"
)
DEFAULT_VALIDATION = "outputs/topiq_teacher_validation_20260531/validation_manifest.csv"
DEFAULT_HOLDOUT = "outputs/topiq_teacher_guard_holdout_20260601/holdout_manifest.csv"
DEFAULT_DRYRUN = "outputs/topiq_teacher_label_dryrun_20260601/dryrun_manifest.csv"
DEFAULT_OUTPUT = "outputs/topiq_teacher_label_expansion_50k_20260601/expansion_manifest.csv"
SEED = 20260601
DATASET_CAPS = {"koniq10k": 9000, "spaq": 9000, "flive": 32000}


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a bounded 50k TOPIQ label expansion manifest.")
    parser.add_argument("--input_csv", default=DEFAULT_INPUT)
    parser.add_argument("--validation_manifest_csv", default=DEFAULT_VALIDATION)
    parser.add_argument("--holdout_manifest_csv", default=DEFAULT_HOLDOUT)
    parser.add_argument("--dryrun_manifest_csv", default=DEFAULT_DRYRUN)
    parser.add_argument("--out_csv", default=DEFAULT_OUTPUT)
    parser.add_argument("--exclude_dryrun", type=parse_bool, default=False)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def read_paths(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open("r", newline="", encoding="utf-8") as f:
        return {
            row.get("image_path", "")
            for row in csv.DictReader(f)
            if row.get("image_path")
        }


def source_group_from_path(image_path: str) -> str:
    filename = Path(image_path).name
    normalized = image_path.replace("\\", "/")
    if "/patches/" in normalized and "_patch_" in filename:
        return filename.split("_patch_", 1)[0]
    return filename


def read_deduped_rows(
    input_csv: Path, excluded_paths: set[str]
) -> tuple[list[dict[str, str]], list[str], int, int, int, int]:
    by_path: dict[str, dict[str, str]] = {}
    all_paths: dict[str, dict[str, str]] = {}
    total_rows = 0
    excluded_raw_rows = 0
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if "image_path" not in fieldnames or "dataset" not in fieldnames:
            raise ValueError("Input CSV must include dataset and image_path columns")
        for row in reader:
            total_rows += 1
            image_path = row.get("image_path", "")
            if not image_path:
                continue
            all_paths.setdefault(image_path, row)
            if image_path in excluded_paths:
                excluded_raw_rows += 1
                continue
            by_path.setdefault(image_path, row)
    deduped_before = len(all_paths)
    duplicate_rows_dropped = total_rows - deduped_before
    return list(by_path.values()), fieldnames, total_rows, deduped_before, duplicate_rows_dropped, excluded_raw_rows


def sample_rows(rows: list[dict[str, str]], rng: random.Random) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    selected_rows: list[dict[str, str]] = []
    summaries: list[dict[str, Any]] = []
    for dataset, requested in DATASET_CAPS.items():
        dataset_rows = [row for row in rows if row.get("dataset") == dataset]
        if len(dataset_rows) <= requested:
            selected = list(dataset_rows)
            took_all_available = len(dataset_rows) < requested
        else:
            selected = rng.sample(dataset_rows, requested)
            took_all_available = False
        rng.shuffle(selected)
        selected_rows.extend(selected)
        groups = [source_group_from_path(row.get("image_path", "")) for row in selected]
        group_counts = Counter(groups)
        summaries.append(
            {
                "dataset": dataset,
                "available_after_exclusion": len(dataset_rows),
                "requested": requested,
                "selected": len(selected),
                "took_all_available": took_all_available,
                "unique_source_groups": len(group_counts),
                "multi_row_source_groups": sum(1 for count in group_counts.values() if count > 1),
                "split_counts": dict(sorted(Counter(row.get("split", "") for row in selected).items())),
                "content_type_counts": dict(sorted(Counter(row.get("content_type", "") for row in selected).items())),
            }
        )
    rng.shuffle(selected_rows)
    return selected_rows, summaries


def write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    output_fields = list(fieldnames)
    if "source_group" not in output_fields:
        output_fields.append("source_group")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        for row in rows:
            item = {col: row.get(col, "") for col in fieldnames}
            item["source_group"] = source_group_from_path(row.get("image_path", ""))
            writer.writerow(item)


def write_report(
    path: Path,
    args: argparse.Namespace,
    total_rows: int,
    deduped_before: int,
    duplicate_rows_dropped: int,
    validation_paths: int,
    holdout_paths: int,
    dryrun_paths: int,
    excluded_union_paths: int,
    excluded_raw_rows: int,
    deduped_after_exclusion: int,
    selected_count: int,
    summaries: list[dict[str, Any]],
) -> None:
    lines = [
        "# TOPIQ 50k Label Expansion Manifest Report",
        "",
        f"- input_csv: {args.input_csv}",
        f"- validation_manifest_csv: {args.validation_manifest_csv}",
        f"- holdout_manifest_csv: {args.holdout_manifest_csv}",
        f"- dryrun_manifest_csv: {args.dryrun_manifest_csv}",
        f"- exclude_dryrun: {args.exclude_dryrun}",
        f"- output_csv: {args.out_csv}",
        f"- seed: {args.seed}",
        f"- input rows: {total_rows}",
        f"- deduped rows before exclusion: {deduped_before}",
        f"- duplicate rows dropped: {duplicate_rows_dropped}",
        f"- validation image paths excluded: {validation_paths}",
        f"- holdout image paths excluded: {holdout_paths}",
        f"- dry-run image paths considered for exclusion: {dryrun_paths if args.exclude_dryrun else 0}",
        f"- excluded union image paths: {excluded_union_paths}",
        f"- raw input rows excluded by selected exclusion paths: {excluded_raw_rows}",
        f"- deduped rows after exclusion: {deduped_after_exclusion}",
        f"- selected rows: {selected_count}",
        "",
        "## Dataset Sampling",
        "",
        "| dataset | available after exclusion | requested | selected | took all available | unique source groups | multi-row source groups | split distribution | content type distribution |",
        "|---|---:|---:|---:|---|---:|---:|---|---|",
    ]
    for summary in summaries:
        lines.append(
            "| {dataset} | {available_after_exclusion} | {requested} | {selected} | {took_all_available} | "
            "{unique_source_groups} | {multi_row_source_groups} | {split_counts} | {content_type_counts} |".format(**summary)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    validation_paths = read_paths(Path(args.validation_manifest_csv))
    holdout_paths = read_paths(Path(args.holdout_manifest_csv))
    dryrun_paths = read_paths(Path(args.dryrun_manifest_csv)) if args.exclude_dryrun else set()
    excluded_paths = validation_paths | holdout_paths | dryrun_paths
    rows, fieldnames, total_rows, deduped_before, duplicate_rows_dropped, excluded_raw_rows = read_deduped_rows(
        Path(args.input_csv), excluded_paths
    )
    selected_rows, summaries = sample_rows(rows, rng)
    out_csv = Path(args.out_csv)
    report_md = out_csv.with_name("expansion_manifest_report.md")
    write_manifest(out_csv, selected_rows, fieldnames)
    write_report(
        report_md,
        args,
        total_rows,
        deduped_before,
        duplicate_rows_dropped,
        len(validation_paths),
        len(holdout_paths),
        len(dryrun_paths),
        len(excluded_paths),
        excluded_raw_rows,
        len(rows),
        len(selected_rows),
        summaries,
    )
    print(f"Wrote {out_csv} ({len(selected_rows)} rows)")
    print(f"Wrote {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
