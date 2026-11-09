from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.distill.aggregation import compute_production_aesthetic_target, compute_teacher_target, load_teacher_config
from src.distill.data import load_source_manifests
from src.distill.evaluation import compute_regression_metrics
from src.infer.predict_quality_bundle import load_bundle_models, predict_bundle_for_image
from src.infer.stage5_reference import load_stage5_reference_config, repo_relative_path
from src.distill.generate_teacher_labels import build_teacher_bundle_namespace


FIELDNAMES = [
    "image_path",
    "split",
    "dataset",
    "production_aesthetic_score",
    "distill_teacher_aesthetic_score",
    "signed_diff",
    "abs_diff",
    "production_missing_models",
    "distill_missing_models",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare production aesthetic aggregation vs distill teacher target.")
    parser.add_argument("--input_csv", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stage5_config", default="configs/stage5_reference.json")
    parser.add_argument("--teacher_config", default="configs/student_distill_teacher.json")
    parser.add_argument("--limit", type=int)
    return parser


def prepare_runtime_environment() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/photo_score_project_mplconfig")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    prepare_runtime_environment()

    source_frame = load_source_manifests(args.input_csv)
    if args.limit:
        source_frame = source_frame.head(int(args.limit)).copy()

    teacher_config, teacher_config_path = load_teacher_config(args.teacher_config)
    stage5_config, stage5_config_path = load_stage5_reference_config(args.stage5_config)
    runtime_args = build_teacher_bundle_namespace(stage5_config)
    models = load_bundle_models(runtime_args)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = output_dir.resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows: list[dict[str, object]] = []
    for row in source_frame.to_dict(orient="records"):
        record = predict_bundle_for_image(row["image_path"], args=runtime_args, models=models)
        production_score, _, production_missing = compute_production_aesthetic_target(
            record=record,
            selector_weights_config=teacher_config.get("production_selector_weights_config"),
        )
        distill_score, _, distill_missing = compute_teacher_target(record=record, teacher_config=teacher_config)
        signed_diff = None
        abs_diff = None
        if production_score is not None and distill_score is not None:
            signed_diff = float(distill_score - production_score)
            abs_diff = abs(signed_diff)
        comparison_rows.append(
            {
                "image_path": row.get("image_path"),
                "split": row.get("split"),
                "dataset": row.get("dataset"),
                "production_aesthetic_score": production_score,
                "distill_teacher_aesthetic_score": distill_score,
                "signed_diff": signed_diff,
                "abs_diff": abs_diff,
                "production_missing_models": json.dumps(sorted(production_missing), ensure_ascii=False),
                "distill_missing_models": json.dumps(sorted(distill_missing), ensure_ascii=False),
            }
        )

    comparison_csv = output_dir / "teacher_consistency_comparison.csv"
    with comparison_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(comparison_rows)

    comparison_frame = pd.DataFrame(comparison_rows)
    usable_frame = comparison_frame.dropna(subset=["production_aesthetic_score", "distill_teacher_aesthetic_score"]).copy()
    if usable_frame.empty:
        summary = {
            "num_images": int(len(comparison_rows)),
            "num_usable_images": 0,
            "comparison_csv": str(comparison_csv),
            "teacher_config": repo_relative_path(teacher_config_path),
            "stage5_config": repo_relative_path(stage5_config_path),
            "error": "No usable rows had both production and distill scores.",
        }
    else:
        y_true = usable_frame["production_aesthetic_score"].astype("float32").to_numpy()
        y_pred = usable_frame["distill_teacher_aesthetic_score"].astype("float32").to_numpy()
        agreement_metrics = compute_regression_metrics(y_true=y_true, y_pred=y_pred)
        abs_diff = np.abs(y_pred - y_true)
        summary = {
            "num_images": int(len(comparison_rows)),
            "num_usable_images": int(len(usable_frame)),
            "comparison_csv": str(comparison_csv),
            "teacher_config": repo_relative_path(teacher_config_path),
            "stage5_config": repo_relative_path(stage5_config_path),
            "agreement_metrics": agreement_metrics,
            "mean_abs_diff": float(abs_diff.mean()),
            "max_abs_diff": float(abs_diff.max()),
            "exact_match_count_at_1e-6": int(np.sum(abs_diff <= 1e-6)),
            "exact_match_rate_at_1e-6": float(np.mean(abs_diff <= 1e-6)),
        }

    summary_path = output_dir / "teacher_consistency_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
