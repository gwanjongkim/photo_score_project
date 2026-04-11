from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import traceback

from src.infer.predict_quality_bundle import (
    BundleInferenceError,
    add_bundle_arguments,
    load_bundle_models,
    predict_bundle_for_image,
)


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a folder of images with the bundle inference pipeline.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_csv")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--extensions", default=".jpg,.jpeg,.png,.webp,.bmp")
    return add_bundle_arguments(parser)


def iter_image_paths(input_dir: Path, recursive: bool, extensions: set[str]) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    paths = []
    for path in input_dir.glob(pattern):
        if path.is_file() and path.suffix.lower() in extensions:
            paths.append(path)
    return sorted(paths)


def flatten_for_csv(result: dict[str, object]) -> dict[str, object]:
    per_model = result.get("per_model", {})
    summary = result.get("summary", {})
    fused = result.get("fused", {})
    return {
        "image_path": result.get("image_path"),
        "available_models": ",".join(summary.get("available_models", [])),
        "reranking_ready": summary.get("reranking_ready"),
        "baseline_final_score": fused.get("final_score"),
        "baseline_aadb_score": result.get("aadb_score"),
        "baseline_koniq_score": result.get("koniq_score"),
        "baseline_flive_image_score": result.get("flive_image_score"),
        "baseline_flive_patch_mean": result.get("flive_patch_mean"),
        "baseline_flive_patch_min": result.get("flive_patch_min"),
        "nima_mean_score": result.get("nima_mean_score"),
        "alamp_score": result.get("alamp_score"),
        "musiq_score": result.get("musiq_score"),
        "rgnet_score": result.get("rgnet_score"),
        "pairwise_recovered_score": result.get("pairwise_recovered_score"),
        "num_models": len(per_model),
    }


def score_folder_results(args: argparse.Namespace) -> list[dict[str, object]]:
    input_dir = Path(args.input_dir)
    extensions = {part.strip().lower() for part in args.extensions.split(",") if part.strip()}
    image_paths = iter_image_paths(input_dir, recursive=args.recursive, extensions=extensions)
    models = load_bundle_models(args)

    results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    recoverable_stage_failure_count = 0

    for index, image_path in enumerate(image_paths, start=1):
        logger.warning("Scoring image %d/%d: %s", index, len(image_paths), image_path)
        try:
            result = predict_bundle_for_image(image_path, args=args, models=models)
        except BundleInferenceError as exc:
            failure = {
                "image_path": str(image_path),
                "stage": exc.stage,
                "recoverable": exc.recoverable,
                "error": str(exc),
                "exception_type": exc.__class__.__name__,
                "traceback": traceback.format_exc(limit=8),
            }
            failures.append(failure)
            logger.error(
                "Skipping image after bundle inference error: image=%s stage=%s recoverable=%s error=%s",
                image_path,
                exc.stage,
                exc.recoverable,
                exc,
            )
            continue
        except Exception as exc:
            failure = {
                "image_path": str(image_path),
                "stage": "unexpected",
                "recoverable": False,
                "error": f"{exc.__class__.__name__}: {exc}",
                "exception_type": exc.__class__.__name__,
                "traceback": traceback.format_exc(limit=8),
            }
            failures.append(failure)
            logger.error(
                "Skipping image after unexpected scoring error: image=%s stage=unexpected recoverable=False error=%s",
                image_path,
                failure["error"],
            )
            continue

        stage_failures = result.get("model_stage_failures")
        stage_failure_len = len(stage_failures) if isinstance(stage_failures, list) else 0
        recoverable_stage_failure_count += stage_failure_len
        logger.warning(
            "Scored image %d/%d: %s models=%d recoverable_stage_failures=%d",
            index,
            len(image_paths),
            image_path,
            len(result.get("per_model") or {}),
            stage_failure_len,
        )
        results.append(result)

    args._score_failures = failures
    args._score_num_input_images = len(image_paths)
    args._score_num_scored_images = len(results)
    args._score_num_skipped_images = len(failures)
    args._score_num_recoverable_stage_failures = int(recoverable_stage_failure_count)

    if failures:
        logger.warning(
            "Completed scoring with skipped images: input=%d scored=%d skipped=%d",
            len(image_paths),
            len(results),
            len(failures),
        )
    else:
        logger.warning(
            "Completed scoring without skipped images: input=%d scored=%d recoverable_stage_failures=%d",
            len(image_paths),
            len(results),
            recoverable_stage_failure_count,
        )
    return results


def write_score_outputs(results: list[dict[str, object]], output_jsonl: Path, output_csv: Path | None = None) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows = [flatten_for_csv(result) for result in results]

    with output_jsonl.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "image_path",
            "available_models",
            "reranking_ready",
            "baseline_final_score",
            "baseline_aadb_score",
            "baseline_koniq_score",
            "baseline_flive_image_score",
            "baseline_flive_patch_mean",
            "baseline_flive_patch_min",
            "aadb_score",
            "koniq_score",
            "flive_image_score",
            "flive_patch_mean",
            "flive_patch_min",
            "nima_mean_score",
            "alamp_score",
            "musiq_score",
            "rgnet_score",
            "pairwise_recovered_score",
            "num_models",
        ]
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_jsonl = Path(args.output_jsonl)
    output_csv = Path(args.output_csv) if args.output_csv else None

    results = score_folder_results(args)
    write_score_outputs(results, output_jsonl=output_jsonl, output_csv=output_csv)

    summary = {
        "input_dir": str(input_dir),
        "num_images": len(results),
        "num_input_images": int(getattr(args, "_score_num_input_images", len(results))),
        "num_skipped_images": int(getattr(args, "_score_num_skipped_images", 0)),
        "num_recoverable_stage_failures": int(getattr(args, "_score_num_recoverable_stage_failures", 0)),
        "output_jsonl": str(output_jsonl),
        "output_csv": str(args.output_csv) if args.output_csv else None,
    }
    print(json.dumps(summary, indent=args.json_indent, ensure_ascii=False))


if __name__ == "__main__":
    main()
