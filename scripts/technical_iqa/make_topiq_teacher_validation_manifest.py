# TOPIQ 교사 모델 검증용 균형 매니페스트를 생성하는 스크립트
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
DEFAULT_OUTPUT = "outputs/topiq_teacher_validation_20260531/validation_manifest.csv"
SEED = 20260531
DATASET_CAPS = {
    "koniq10k": 300,
    "spaq": 300,
    "flive": 300,
}
PREFERRED_SPLITS = {"val", "test"}
PRESERVED_COLUMNS = [
    "dataset",
    "protocol",
    "run_id",
    "split",
    "content_type",
    "image_id",
    "image_path",
    "quality_score_raw",
    "quality_score_0_100",
    "source_annotation_file",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a balanced TOPIQ teacher validation manifest."
    )
    parser.add_argument("--input_csv", default=DEFAULT_INPUT)
    parser.add_argument("--out_csv", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def split_priority(row: dict[str, str]) -> tuple[int, int]:
    split = row.get("split", "")
    if split == "val":
        return (0, 0)
    if split == "test":
        return (0, 1)
    return (1, 0)


def choose_preferred_duplicate(
    current: dict[str, str] | None, candidate: dict[str, str]
) -> dict[str, str]:
    if current is None:
        return candidate
    if split_priority(candidate) < split_priority(current):
        return candidate
    return current


def read_deduped_rows(input_csv: Path) -> tuple[list[dict[str, str]], int]:
    by_path: dict[str, dict[str, str]] = {}
    total_rows = 0
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [col for col in PRESERVED_COLUMNS if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {missing}")

        for row in reader:
            total_rows += 1
            image_path = row.get("image_path", "")
            if not image_path:
                continue
            by_path[image_path] = choose_preferred_duplicate(by_path.get(image_path), row)

    return list(by_path.values()), total_rows


def sample_dataset_rows(
    rows: list[dict[str, str]], dataset: str, cap: int, rng: random.Random
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    dataset_rows = [row for row in rows if row.get("dataset") == dataset]
    preferred = [row for row in dataset_rows if row.get("split") in PREFERRED_SPLITS]
    train = [row for row in dataset_rows if row.get("split") == "train"]

    if len(preferred) >= cap:
        selected = rng.sample(preferred, cap)
        train_fill_count = 0
    else:
        selected = list(preferred)
        train_needed = cap - len(selected)
        train_fill = rng.sample(train, min(train_needed, len(train)))
        selected.extend(train_fill)
        train_fill_count = len(train_fill)

    rng.shuffle(selected)
    split_counts = Counter(row.get("split", "") for row in selected)
    summary = {
        "dataset": dataset,
        "available_deduped": len(dataset_rows),
        "available_preferred_val_test": len(preferred),
        "available_train": len(train),
        "selected": len(selected),
        "train_fill_count": train_fill_count,
        "split_counts": dict(sorted(split_counts.items())),
    }
    return selected, summary


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PRESERVED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in PRESERVED_COLUMNS})


def write_report(
    path: Path,
    input_csv: Path,
    output_csv: Path,
    total_rows: int,
    deduped_rows: int,
    summaries: list[dict[str, Any]],
    seed: int,
) -> None:
    lines = [
        "# TOPIQ Teacher Validation Manifest Report",
        "",
        f"- input_csv: {input_csv}",
        f"- output_csv: {output_csv}",
        f"- seed: {seed}",
        f"- input rows: {total_rows}",
        f"- deduped image_path rows: {deduped_rows}",
        f"- duplicate rows dropped: {total_rows - deduped_rows}",
        "",
        "## Dataset Sampling",
        "",
        "| dataset | available deduped | preferred val/test | train available | selected | train fill | selected splits |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        lines.append(
            "| {dataset} | {available_deduped} | {available_preferred_val_test} | "
            "{available_train} | {selected} | {train_fill_count} | {split_counts} |".format(
                **item
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.out_csv)
    report_md = output_csv.with_name("validation_manifest_report.md")
    rng = random.Random(args.seed)

    rows, total_rows = read_deduped_rows(input_csv)
    selected_rows: list[dict[str, str]] = []
    summaries: list[dict[str, Any]] = []

    for dataset, cap in DATASET_CAPS.items():
        dataset_selected, summary = sample_dataset_rows(rows, dataset, cap, rng)
        selected_rows.extend(dataset_selected)
        summaries.append(summary)

    rng.shuffle(selected_rows)
    write_csv(output_csv, selected_rows)
    write_report(report_md, input_csv, output_csv, total_rows, len(rows), summaries, args.seed)

    print(f"Wrote {output_csv} ({len(selected_rows)} rows)")
    print(f"Wrote {report_md}")
    for summary in summaries:
        print(
            "{dataset}: selected={selected}, train_fill={train_fill_count}, splits={split_counts}".format(
                **summary
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
