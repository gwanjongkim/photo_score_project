from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from src.distill.aggregation import compute_teacher_target, load_teacher_config
from src.distill.data import canonicalize_image_key, summarize_cross_split_duplicates


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEACHER_CONFIG = "configs/student_distill_teacher_conservative_aadb.json"
CONSERVATIVE_FORMULA_TEXT = (
    "0.35 * rgnet_aadb + 0.20 * alamp_aadb + 0.30 * aadb_composition + 0.15 * nima_ava_unit"
)
REQUIRED_SCORE_COLUMNS = {"image_path", "split", "aadb_score", "alamp_score", "rgnet_score", "nima_mean_score"}
EXTRA_COLUMNS = [
    "previous_teacher_aesthetic_score",
    "conservative_teacher_score",
    "conservative_teacher_formula",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a weighted teacher manifest from existing component-score teacher CSVs."
    )
    parser.add_argument("--input_csv", action="append", required=True, help="Teacher CSV. Repeat this flag.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--teacher_config", default=DEFAULT_TEACHER_CONFIG)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_every", type=int, default=500)
    return parser


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path.resolve(strict=False)
    return (PROJECT_ROOT / path).resolve(strict=False)


def summarize_series(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "min": None, "max": None, "range": None, "mean": None, "std": None}
    return {
        "count": int(values.count()),
        "min": float(values.min()),
        "max": float(values.max()),
        "range": float(values.max() - values.min()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
    }


def load_input_frame(path_value: str | Path) -> pd.DataFrame:
    path = resolve_repo_path(path_value)
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_SCORE_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Input teacher CSV is missing required columns: {path} :: {', '.join(missing)}")
    frame = frame.copy()
    frame["image_path"] = frame["image_path"].map(lambda value: str(resolve_repo_path(value)))
    frame["split"] = frame["split"].astype(str)
    frame["source_weighted_teacher_input_csv"] = str(path)
    frame["__image_key"] = frame["image_path"].map(canonicalize_image_key)
    frame = frame.dropna(subset=["__image_key"]).reset_index(drop=True)
    return frame


def load_existing_frame(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "image_path" in frame.columns:
        frame["__image_key"] = frame["image_path"].map(canonicalize_image_key)
    return frame


def write_split_outputs(frame: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_paths: dict[str, str] = {}
    for split_name, split_frame in frame.groupby("split", sort=True):
        split_key = str(split_name).strip() or "unspecified"
        split_path = output_dir / f"teacher_labels_{split_key}.csv"
        split_frame.drop(columns=["__image_key"], errors="ignore").to_csv(split_path, index=False)
        output_paths[split_key] = str(split_path)
    return output_paths


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "teacher_labels_trainval.csv"
    summary_path = output_dir / "summary.json"
    config_copy_path = output_dir / "resolved_teacher_config.json"

    if output_csv.exists() and not args.resume and not args.overwrite:
        raise ValueError(f"Output already exists. Use --resume or --overwrite: {output_csv}")
    if args.resume and args.overwrite:
        raise ValueError("Choose only one of --resume or --overwrite.")
    if args.overwrite and output_csv.exists():
        output_csv.unlink()

    teacher_config, teacher_config_path = load_teacher_config(args.teacher_config)
    config_copy_path.write_text(json.dumps(teacher_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    frames = [load_input_frame(path_value) for path_value in args.input_csv]
    source_frame = pd.concat(frames, ignore_index=True)
    source_frame = source_frame.drop_duplicates(subset=["__image_key", "split"], keep="first").reset_index(drop=True)
    duplicate_summary = summarize_cross_split_duplicates(source_frame.drop(columns=["__image_key"]))

    existing_output = load_existing_frame(output_csv) if args.resume else pd.DataFrame()
    existing_keys = set(existing_output.get("__image_key", pd.Series(dtype=str)).dropna().astype(str).tolist())
    rows_written = 0
    rows_skipped_existing = 0
    rows_skipped_missing_teacher = 0
    rows_total = int(len(source_frame))

    fieldnames = list(
        dict.fromkeys(
            source_frame.drop(columns=["__image_key"], errors="ignore").columns.tolist()
            + EXTRA_COLUMNS
            + [
                "teacher_aesthetic_score",
                "teacher_model_count",
                "teacher_models_used",
                "missing_teacher_models",
                "teacher_contributions_json",
                "teacher_config_name",
                "teacher_config_path",
            ]
        )
    )
    if not existing_output.empty:
        fieldnames = existing_output.drop(columns=["__image_key"], errors="ignore").columns.tolist()

    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
    with output_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for index, row in enumerate(source_frame.to_dict(orient="records"), start=1):
            image_key = row.get("__image_key")
            if image_key in existing_keys:
                rows_skipped_existing += 1
                continue

            teacher_score, contributions, missing_teacher_models = compute_teacher_target(
                record=row,
                teacher_config=teacher_config,
            )
            if teacher_score is None:
                rows_skipped_missing_teacher += 1
                continue

            output_row = {key: row.get(key) for key in fieldnames}
            output_row["previous_teacher_aesthetic_score"] = row.get("teacher_aesthetic_score")
            output_row["teacher_aesthetic_score"] = round(float(teacher_score), 6)
            output_row["conservative_teacher_score"] = round(float(teacher_score), 6)
            output_row["conservative_teacher_formula"] = CONSERVATIVE_FORMULA_TEXT
            output_row["teacher_model_count"] = len(contributions)
            output_row["teacher_models_used"] = json.dumps(sorted(contributions), ensure_ascii=False)
            output_row["missing_teacher_models"] = json.dumps(sorted(missing_teacher_models), ensure_ascii=False)
            output_row["teacher_contributions_json"] = json.dumps(contributions, ensure_ascii=False)
            output_row["teacher_config_name"] = teacher_config.get("name")
            output_row["teacher_config_path"] = str(resolve_repo_path(teacher_config_path))
            writer.writerow(output_row)
            handle.flush()
            rows_written += 1

            if index % max(1, int(args.log_every)) == 0 or index == rows_total:
                print(
                    json.dumps(
                        {
                            "processed": index,
                            "total": rows_total,
                            "rows_written": rows_written,
                            "rows_skipped_existing": rows_skipped_existing,
                            "rows_skipped_missing_teacher": rows_skipped_missing_teacher,
                            "output_csv": str(output_csv),
                        },
                        ensure_ascii=False,
                    )
                )

    final_frame = pd.read_csv(output_csv)
    final_frame["__image_key"] = final_frame["image_path"].map(canonicalize_image_key)
    split_output_paths = write_split_outputs(final_frame, output_dir=output_dir)
    split_stats = {
        str(split_name): summarize_series(split_frame["teacher_aesthetic_score"])
        for split_name, split_frame in final_frame.groupby("split", sort=True)
    }

    summary = {
        "output_dir": str(output_dir),
        "output_csv": str(output_csv),
        "input_csvs": [str(resolve_repo_path(path_value)) for path_value in args.input_csv],
        "teacher_config_path": str(resolve_repo_path(teacher_config_path)),
        "teacher_config_name": teacher_config.get("name"),
        "teacher_formula": CONSERVATIVE_FORMULA_TEXT,
        "rows_total_input": rows_total,
        "rows_written": rows_written,
        "rows_skipped_existing": rows_skipped_existing,
        "rows_skipped_missing_teacher": rows_skipped_missing_teacher,
        "final_rows": int(len(final_frame)),
        "split_counts": {str(key): int(value) for key, value in final_frame["split"].value_counts().to_dict().items()},
        "split_output_paths": split_output_paths,
        "split_teacher_score_summary": split_stats,
        "cross_split_duplicate_summary": duplicate_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
