from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import pandas as pd

from src.distill.aggregation import compute_teacher_target, load_teacher_config
from src.distill.data import canonicalize_image_key, load_source_manifests, summarize_cross_split_duplicates
from src.infer.predict_quality_bundle import load_bundle_models, predict_bundle_for_image
from src.infer.stage5_reference import load_stage5_reference_config, repo_relative_path


OUTPUT_FIELDNAMES = [
    "image_path",
    "split",
    "dataset",
    "relative_path",
    "source_manifest",
    "source_row_index",
    "teacher_aesthetic_score",
    "teacher_model_count",
    "teacher_models_used",
    "missing_teacher_models",
    "teacher_contributions_json",
    "teacher_stage_failure_count",
    "teacher_stage_failures_json",
    "teacher_label_error",
    "aadb_score",
    "nima_mean_score",
    "alamp_score",
    "musiq_score",
    "rgnet_score",
    "pairwise_recovered_score",
    "teacher_config_name",
    "teacher_config_path",
    "stage5_config_path",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate stage5-teacher labels for student distillation.")
    parser.add_argument(
        "--input_csv",
        action="append",
        required=True,
        help="Input manifest CSV. Repeat this flag to combine train/val/test manifests.",
    )
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--stage5_config", default="configs/stage5_reference.json")
    parser.add_argument("--teacher_config", default="configs/student_distill_teacher.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--allow_partial_teacher_scores",
        action="store_true",
        help="Emit labels when some teacher scores are missing. By default such rows are skipped.",
    )
    return parser


def prepare_runtime_environment() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/photo_score_project_mplconfig")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def build_teacher_bundle_namespace(stage5_config: dict[str, object]) -> argparse.Namespace:
    bundle_args = dict(stage5_config.get("bundle_args") or {})
    return argparse.Namespace(
        aadb_model=bundle_args.get("aadb_model"),
        koniq_model=bundle_args.get("koniq_model"),
        flive_image_model=bundle_args.get("flive_image_model"),
        flive_patch_model=bundle_args.get("flive_patch_model"),
        nima_model=bundle_args.get("nima_model"),
        alamp_model=bundle_args.get("alamp_model"),
        musiq_model=bundle_args.get("musiq_model"),
        rgnet_model=bundle_args.get("rgnet_model"),
        pairwise_model=bundle_args.get("pairwise_model"),
        pairwise_reference_csv=bundle_args.get("pairwise_reference_csv"),
        pairwise_target_col=bundle_args.get("pairwise_target_col", "score"),
        pairwise_max_references=int(bundle_args.get("pairwise_max_references", 16)),
        image_size=int(bundle_args.get("image_size", 224)),
        rgnet_image_size=int(bundle_args.get("rgnet_image_size", 256)),
        alamp_global_size=int(bundle_args.get("alamp_global_size", 384)),
        alamp_patch_size=int(bundle_args.get("alamp_patch_size", 224)),
        alamp_num_patches=int(bundle_args.get("alamp_num_patches", 5)),
        alamp_patch_scale=float(bundle_args.get("alamp_patch_scale", 0.35)),
        alamp_include_debug=False,
        include_debug=False,
        musiq_patch_size=int(bundle_args.get("musiq_patch_size", 32)),
        musiq_scale_sizes=str(bundle_args.get("musiq_scale_sizes", "224,384,512")),
        musiq_patches_per_scale=int(bundle_args.get("musiq_patches_per_scale", 16)),
        json_indent=2,
        _auto_resolved_model_fields=[],
        _model_load_warnings=[],
    )


def load_existing_output(output_csv: Path) -> pd.DataFrame:
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return pd.DataFrame(columns=OUTPUT_FIELDNAMES)
    frame = pd.read_csv(output_csv)
    if "image_path" in frame.columns:
        frame["__image_key"] = frame["image_path"].apply(canonicalize_image_key)
    return frame


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    prepare_runtime_environment()

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = output_csv.resolve(strict=False)

    if output_csv.exists() and not args.resume and not args.overwrite:
        raise ValueError(f"Output already exists. Use --resume or --overwrite: {output_csv}")
    if args.resume and args.overwrite:
        raise ValueError("Choose only one of --resume or --overwrite.")
    if args.overwrite and output_csv.exists():
        output_csv.unlink()

    source_frame = load_source_manifests(args.input_csv)
    if args.limit:
        source_frame = source_frame.head(int(args.limit)).copy()
    duplicate_summary = summarize_cross_split_duplicates(source_frame)
    if duplicate_summary["num_cross_split_duplicates"] > 0:
        print(json.dumps({"cross_split_duplicate_warning": duplicate_summary}, ensure_ascii=False))

    teacher_config, teacher_config_path = load_teacher_config(args.teacher_config)
    stage5_config, stage5_config_path = load_stage5_reference_config(args.stage5_config)
    runtime_args = build_teacher_bundle_namespace(stage5_config)
    teacher_models = load_bundle_models(runtime_args)

    existing_output = load_existing_output(output_csv) if args.resume else pd.DataFrame(columns=OUTPUT_FIELDNAMES)
    existing_keys = set(existing_output.get("__image_key", pd.Series(dtype=str)).dropna().astype(str).tolist())
    source_split_counts = {
        str(key): int(value) for key, value in source_frame["split"].astype(str).value_counts().to_dict().items()
    }
    existing_split_counts = (
        {str(key): int(value) for key, value in existing_output["split"].astype(str).value_counts().to_dict().items()}
        if not existing_output.empty and "split" in existing_output.columns
        else {}
    )
    print(
        json.dumps(
            {
                "generator_start": {
                    "input_rows": int(len(source_frame)),
                    "input_split_counts": source_split_counts,
                    "resume_enabled": bool(args.resume),
                    "existing_output_rows": int(len(existing_output)),
                    "existing_output_split_counts": existing_split_counts,
                    "output_csv": str(output_csv),
                }
            },
            ensure_ascii=False,
        )
    )

    rows_written = 0
    rows_skipped_existing = 0
    rows_skipped_missing_teacher = 0
    rows_failed = 0
    rows_total = int(len(source_frame))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
    with output_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDNAMES)
        if write_header:
            writer.writeheader()

        for index, row in enumerate(source_frame.to_dict(orient="records"), start=1):
            image_key = canonicalize_image_key(row.get("image_path"))
            if image_key is not None and image_key in existing_keys:
                rows_skipped_existing += 1
                continue

            try:
                result = predict_bundle_for_image(row["image_path"], args=runtime_args, models=teacher_models)
            except Exception as exc:
                rows_failed += 1
                print(
                    json.dumps(
                        {
                            "processed": index,
                            "total": rows_total,
                            "image_path": row.get("image_path"),
                            "error": f"{exc.__class__.__name__}: {exc}",
                            "rows_failed": rows_failed,
                        },
                        ensure_ascii=False,
                    )
                )
                continue
            effective_teacher_config = dict(teacher_config)
            if args.allow_partial_teacher_scores:
                effective_teacher_config["allow_partial_teacher_scores"] = True
            teacher_score, contributions, missing_teacher_models = compute_teacher_target(
                record=result,
                teacher_config=effective_teacher_config,
            )
            if teacher_score is None:
                rows_skipped_missing_teacher += 1
                continue

            stage_failures = result.get("model_stage_failures") or []
            output_row = {
                "image_path": str(row.get("image_path")),
                "split": row.get("split"),
                "dataset": row.get("dataset"),
                "relative_path": row.get("relative_path"),
                "source_manifest": row.get("source_manifest"),
                "source_row_index": row.get("source_row_index"),
                "teacher_aesthetic_score": round(float(teacher_score), 6),
                "teacher_model_count": len(contributions),
                "teacher_models_used": json.dumps(sorted(contributions), ensure_ascii=False),
                "missing_teacher_models": json.dumps(sorted(missing_teacher_models), ensure_ascii=False),
                "teacher_contributions_json": json.dumps(contributions, ensure_ascii=False),
                "teacher_stage_failure_count": len(stage_failures),
                "teacher_stage_failures_json": json.dumps(stage_failures, ensure_ascii=False),
                "teacher_label_error": None,
                "aadb_score": result.get("aadb_score"),
                "nima_mean_score": result.get("nima_mean_score"),
                "alamp_score": result.get("alamp_score"),
                "musiq_score": result.get("musiq_score"),
                "rgnet_score": result.get("rgnet_score"),
                "pairwise_recovered_score": result.get("pairwise_recovered_score"),
                "teacher_config_name": teacher_config.get("name"),
                "teacher_config_path": repo_relative_path(teacher_config_path),
                "stage5_config_path": repo_relative_path(stage5_config_path),
            }
            writer.writerow(output_row)
            handle.flush()
            rows_written += 1

            if index % 50 == 0 or index == rows_total:
                print(
                    json.dumps(
                        {
                            "processed": index,
                            "total": rows_total,
                            "rows_written": rows_written,
                            "rows_skipped_existing": rows_skipped_existing,
                            "rows_skipped_missing_teacher": rows_skipped_missing_teacher,
                            "rows_failed": rows_failed,
                            "output_csv": str(output_csv),
                        },
                        ensure_ascii=False,
                    )
                )

    summary = {
        "input_rows": rows_total,
        "rows_written": rows_written,
        "rows_skipped_existing": rows_skipped_existing,
        "rows_skipped_missing_teacher": rows_skipped_missing_teacher,
        "rows_failed": rows_failed,
        "output_csv": str(output_csv),
        "teacher_config": repo_relative_path(teacher_config_path),
        "stage5_config": repo_relative_path(stage5_config_path),
        "cross_split_duplicate_summary": duplicate_summary,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
