# TFLite 기반 미적 가중치 실험 도구의 CLI 진입점입니다.
from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from html_report import build_report_html, prepare_report_assets
from model_registry import (
    ModelSpec,
    assert_model_files_exist,
    build_weight_presets,
    enabled_model_specs,
    load_weights,
    normalized_weighted_score,
)
from tflite_model_runner import SUPPORTED_EXTENSIONS, TfliteModelRunner, iter_image_paths, load_image_rgb


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TFLite Aesthetic Weight Lab on a folder of images.")
    parser.add_argument("--input_dir", required=True, help="Folder containing private test images.")
    parser.add_argument("--config", required=True, help="YAML config defining enabled TFLite models and weights.")
    parser.add_argument("--output_dir", required=True, help="New output directory for scores, assets, and report.html.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input_dir.")
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(SUPPORTED_EXTENSIONS)),
        help="Comma-separated image extensions. Defaults to jpg,jpeg,png,webp.",
    )
    parser.add_argument("--num_threads", type=int, default=1, help="TFLite interpreter thread count.")
    return parser


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def ensure_new_or_empty_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_extensions(value: str) -> set[str]:
    extensions = set()
    for part in value.split(","):
        item = part.strip().lower()
        if not item:
            continue
        extensions.add(item if item.startswith(".") else f".{item}")
    if not extensions:
        raise ValueError("At least one image extension is required.")
    return extensions


def rank_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        records,
        key=lambda row: (
            row.get("final_score") is None,
            0.0 if row.get("final_score") is None else -float(row["final_score"]),
            str(row.get("image_name") or ""),
        ),
    )
    for index, record in enumerate(ranked, start=1):
        record["rank"] = index
    return ranked


def score_images(
    *,
    image_paths: list[Path],
    specs: list[ModelSpec],
    weights: dict[str, float],
    num_threads: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runners = {spec.model_id: TfliteModelRunner(spec, num_threads=num_threads) for spec in specs}
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for image_path in image_paths:
        try:
            image = load_image_rgb(image_path)
        except Exception as exc:
            failures.append(
                {
                    "image_path": str(image_path),
                    "stage": "image_decode",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
            continue

        record: dict[str, Any] = {
            "image_path": str(image_path.resolve()),
            "image_name": image_path.name,
            "width": image.width,
            "height": image.height,
            "model_details": {},
            "model_errors": [],
        }
        for spec in specs:
            try:
                prediction = runners[spec.model_id].score_image(image)
                record[spec.score_column] = float(prediction.score)
                record["model_details"][spec.model_id] = prediction.details
            except Exception as exc:
                record[spec.score_column] = None
                record["model_errors"].append(
                    {
                        "model_id": spec.model_id,
                        "stage": "model_inference",
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                )
        record["final_score"] = normalized_weighted_score(record, specs, weights)
        records.append(record)
    return rank_records(records), failures


def write_raw_scores_csv(path: Path, records: list[dict[str, Any]], specs: list[ModelSpec]) -> None:
    fieldnames = [
        "rank",
        "image_name",
        "image_path",
        "width",
        "height",
        *[spec.score_column for spec in specs],
        "final_score",
        "thumbnail_path",
        "copied_image_path",
        "model_errors_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in fieldnames}
            row["model_errors_json"] = json.dumps(record.get("model_errors") or [], ensure_ascii=False, sort_keys=True)
            writer.writerow(row)


def write_raw_scores_json(
    path: Path,
    *,
    config_path: Path,
    input_dir: Path,
    output_dir: Path,
    specs: list[ModelSpec],
    weights: dict[str, float],
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "models": [
            {
                "model_id": spec.model_id,
                "display_name": spec.display_name,
                "type": spec.model_type,
                "model_path": str(spec.model_path),
                "score_column": spec.score_column,
                "score_index": spec.config.get("score_index"),
                "weight": float(weights.get(spec.model_id, 0.0)),
            }
            for spec in specs
        ],
        "weights": weights,
        "rows": records,
        "failures": failures,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_command_log(
    path: Path,
    *,
    args: argparse.Namespace,
    config_path: Path,
    input_dir: Path,
    output_dir: Path,
    image_paths: list[Path],
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    specs: list[ModelSpec],
) -> None:
    lines = [
        f"created_at: {datetime.now().isoformat(timespec='seconds')}",
        f"cwd: {Path.cwd()}",
        "command: " + " ".join(shlex.quote(part) for part in sys.argv),
        f"python: {sys.executable}",
        f"python_version: {platform.python_version()}",
        f"config: {config_path}",
        f"input_dir: {input_dir}",
        f"output_dir: {output_dir}",
        f"recursive: {bool(args.recursive)}",
        f"extensions: {args.extensions}",
        f"num_threads: {int(args.num_threads)}",
        f"input_images_found: {len(image_paths)}",
        f"images_scored: {len(records)}",
        f"image_failures: {len(failures)}",
        "models:",
    ]
    for spec in specs:
        detail = f"  - {spec.model_id}: {spec.model_path}; type={spec.model_type}; score_column={spec.score_column}"
        if "score_index" in spec.config:
            detail += f"; score_index={spec.config['score_index']}"
        lines.append(detail)
    if failures:
        lines.append("failures:")
        lines.extend(f"  - {failure}" for failure in failures)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    input_dir = resolve_repo_path(args.input_dir)
    config_path = resolve_repo_path(args.config)
    output_dir = resolve_repo_path(args.output_dir)
    ensure_new_or_empty_output_dir(output_dir)

    config = load_yaml_config(config_path)
    specs = enabled_model_specs(config, REPO_ROOT)
    assert_model_files_exist(specs)
    weights = load_weights(config, specs)
    presets = build_weight_presets(specs, weights)

    image_paths = iter_image_paths(
        input_dir,
        recursive=bool(args.recursive),
        extensions=parse_extensions(args.extensions),
    )
    records, failures = score_images(
        image_paths=image_paths,
        specs=specs,
        weights=weights,
        num_threads=int(args.num_threads),
    )

    report_config = dict(config.get("report") or {})
    prepare_report_assets(
        records,
        output_dir=output_dir,
        thumbnail_size=int(report_config.get("thumbnail_size", 240)),
        copy_original_images=bool(report_config.get("copy_original_images", True)),
    )

    (output_dir / "weight_presets.json").write_text(
        json.dumps(presets, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_raw_scores_csv(output_dir / "raw_scores.csv", records, specs)
    write_raw_scores_json(
        output_dir / "raw_scores.json",
        config_path=config_path,
        input_dir=input_dir,
        output_dir=output_dir,
        specs=specs,
        weights=weights,
        records=records,
        failures=failures,
    )
    build_report_html(
        records=records,
        specs=specs,
        weights=weights,
        presets=presets,
        report_config=report_config,
        config_path=config_path,
        output_path=output_dir / "report.html",
    )
    write_command_log(
        output_dir / "command_log.txt",
        args=args,
        config_path=config_path,
        input_dir=input_dir,
        output_dir=output_dir,
        image_paths=image_paths,
        records=records,
        failures=failures,
        specs=specs,
    )

    summary = {
        "input_images_found": len(image_paths),
        "images_scored": len(records),
        "image_failures": len(failures),
        "output_dir": str(output_dir),
        "report_html": str(output_dir / "report.html"),
        "raw_scores_csv": str(output_dir / "raw_scores.csv"),
        "raw_scores_json": str(output_dir / "raw_scores.json"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
