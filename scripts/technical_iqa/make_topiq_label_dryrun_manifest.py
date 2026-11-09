# TOPIQ 가드 교사 라벨 드라이런용 매니페스트를 생성하는 스크립트
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
DEFAULT_OUTPUT = "outputs/topiq_teacher_label_dryrun_20260601/dryrun_manifest.csv"
SEED = 20260601
DATASET_CAPS = {"koniq10k": 3000, "spaq": 3000, "flive": 3000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a bounded guarded TOPIQ teacher-label dry-run manifest."
    )
    parser.add_argument("--input_csv", default=DEFAULT_INPUT)
    parser.add_argument("--validation_manifest_csv", default=DEFAULT_VALIDATION)
    parser.add_argument("--holdout_manifest_csv", default=DEFAULT_HOLDOUT)
    parser.add_argument("--out_csv", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def read_paths(path: Path) -> set[str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return {
            row.get("image_path", "")
            for row in csv.DictReader(f)
            if row.get("image_path")
        }


def choose_first(current: dict[str, str] | None, candidate: dict[str, str]) -> dict[str, str]:
    return candidate if current is None else current


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
            all_paths[image_path] = choose_first(all_paths.get(image_path), row)
            if image_path in excluded_paths:
                excluded_raw_rows += 1
                continue
            by_path[image_path] = choose_first(by_path.get(image_path), row)

    deduped_before = len(all_paths)
    duplicate_rows_dropped = total_rows - deduped_before
    return (
        list(by_path.values()),
        fieldnames,
        total_rows,
        deduped_before,
        duplicate_rows_dropped,
        excluded_raw_rows,
    )


def sample_rows(
    rows: list[dict[str, str]], rng: random.Random
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
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
        summaries.append(
            {
                "dataset": dataset,
                "available_after_exclusion": len(dataset_rows),
                "requested": requested,
                "selected": len(selected),
                "took_all_available": took_all_available,
                "split_counts": dict(sorted(Counter(row.get("split", "") for row in selected).items())),
                "content_type_counts": dict(sorted(Counter(row.get("content_type", "") for row in selected).items())),
            }
        )
    rng.shuffle(selected_rows)
    return selected_rows, summaries


def write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{col: row.get(col, "") for col in fieldnames} for row in rows])


def write_report(
    path: Path,
    input_csv: Path,
    validation_manifest_csv: Path,
    holdout_manifest_csv: Path,
    out_csv: Path,
    seed: int,
    total_rows: int,
    deduped_before: int,
    duplicate_rows_dropped: int,
    validation_paths: int,
    holdout_paths: int,
    excluded_union_paths: int,
    excluded_raw_rows: int,
    deduped_after_exclusion: int,
    summaries: list[dict[str, Any]],
) -> None:
    lines = [
        "# TOPIQ Label Dry-Run Manifest Report",
        "",
        f"- input_csv: {input_csv}",
        f"- validation_manifest_csv: {validation_manifest_csv}",
        f"- holdout_manifest_csv: {holdout_manifest_csv}",
        f"- output_csv: {out_csv}",
        f"- seed: {seed}",
        f"- input rows: {total_rows}",
        f"- deduped rows before exclusion: {deduped_before}",
        f"- duplicate rows dropped: {duplicate_rows_dropped}",
        f"- validation image paths excluded: {validation_paths}",
        f"- holdout image paths excluded: {holdout_paths}",
        f"- excluded union image paths: {excluded_union_paths}",
        f"- raw input rows excluded by validation/holdout paths: {excluded_raw_rows}",
        f"- deduped rows after exclusion: {deduped_after_exclusion}",
        "",
        "## Dataset Sampling",
        "",
        "| dataset | available after exclusion | requested | selected | took all available | split distribution | content type distribution |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for summary in summaries:
        lines.append(
            "| {dataset} | {available_after_exclusion} | {requested} | {selected} | "
            "{took_all_available} | {split_counts} | {content_type_counts} |".format(**summary)
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv)
    validation_manifest_csv = Path(args.validation_manifest_csv)
    holdout_manifest_csv = Path(args.holdout_manifest_csv)
    out_csv = Path(args.out_csv)
    report_md = out_csv.with_name("dryrun_manifest_report.md")
    rng = random.Random(args.seed)

    validation_paths = read_paths(validation_manifest_csv)
    holdout_paths = read_paths(holdout_manifest_csv)
    excluded_paths = validation_paths | holdout_paths
    (
        rows,
        fieldnames,
        total_rows,
        deduped_before,
        duplicate_rows_dropped,
        excluded_raw_rows,
    ) = read_deduped_rows(input_csv, excluded_paths)
    selected_rows, summaries = sample_rows(rows, rng)
    write_manifest(out_csv, selected_rows, fieldnames)
    write_report(
        report_md,
        input_csv,
        validation_manifest_csv,
        holdout_manifest_csv,
        out_csv,
        args.seed,
        total_rows,
        deduped_before,
        duplicate_rows_dropped,
        len(validation_paths),
        len(holdout_paths),
        len(excluded_paths),
        excluded_raw_rows,
        len(rows),
        summaries,
    )
    print(f"Wrote {out_csv} ({len(selected_rows)} rows)")
    print(f"Wrote {report_md}")
    for summary in summaries:
        print(
            "{dataset}: selected={selected}, available={available_after_exclusion}, took_all={took_all_available}".format(
                **summary
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
