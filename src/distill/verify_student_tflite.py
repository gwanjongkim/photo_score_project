from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.distill.config import resolve_repo_path
from src.distill.data import canonicalize_image_key
from src.distill.evaluation import compute_regression_metrics, sample_pairwise_accuracy, top_k_overlap
from src.distill.preprocessing import decode_resize_normalize_student_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify stage5 student Keras/SavedModel/TFLite parity.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--tflite_path", required=True)
    parser.add_argument("--image_path", action="append", help="Repeatable explicit image path.")
    parser.add_argument(
        "--image_glob",
        action="append",
        help="Repeatable glob. Defaults to test_samples/*.jpg when no explicit images are provided.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--output_path")
    parser.add_argument("--teacher_csv")
    parser.add_argument("--teacher_image_col", default="image_path")
    parser.add_argument("--teacher_score_col")
    parser.add_argument("--baseline_score_col")
    parser.add_argument("--pairwise_margin", type=float, default=0.03)
    parser.add_argument("--pairwise_max_pairs", type=int, default=10000)
    parser.add_argument("--top_k", type=int, default=10)
    return parser


def resolve_image_paths(args: argparse.Namespace) -> list[Path]:
    explicit = [Path(path) for path in args.image_path or []]
    globs = list(args.image_glob or [])
    csv_paths: list[Path] = []
    if not explicit and not globs and args.teacher_csv:
        teacher_csv = resolve_repo_path(args.teacher_csv)
        if teacher_csv is not None:
            with teacher_csv.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    raw_path = row.get(str(args.teacher_image_col))
                    if not raw_path:
                        continue
                    resolved = resolve_repo_path(raw_path)
                    if resolved is not None:
                        csv_paths.append(resolved)

    if not explicit and not globs and not csv_paths:
        globs = ["test_samples/*.jpg"]

    resolved: list[Path] = []
    seen: set[str] = set()
    for path in explicit:
        candidate = resolve_repo_path(path)
        if candidate is None:
            continue
        key = str(candidate.resolve(strict=False))
        if key not in seen:
            resolved.append(candidate)
            seen.add(key)

    for pattern in globs:
        candidate_pattern = resolve_repo_path(pattern)
        if candidate_pattern is None:
            continue
        for path in sorted(candidate_pattern.parent.glob(candidate_pattern.name)):
            key = str(path.resolve(strict=False))
            if key not in seen:
                resolved.append(path)
                seen.add(key)

    for path in csv_paths:
        key = str(path.resolve(strict=False))
        if key not in seen:
            resolved.append(path)
            seen.add(key)

    if args.limit is not None:
        resolved = resolved[: max(0, int(args.limit))]
    return resolved


def load_image_array(image_path: Path, image_size: int) -> np.ndarray:
    image = decode_resize_normalize_student_image(tf.convert_to_tensor(str(image_path)), image_size=image_size)
    return np.asarray(image.numpy(), dtype=np.float32)


def tensor_detail_to_json(detail: dict[str, object]) -> dict[str, object]:
    return {
        "name": str(detail["name"]),
        "shape": [int(dim) for dim in detail["shape"]],
        "shape_signature": [int(dim) for dim in detail["shape_signature"]],
        "dtype": np.dtype(detail["dtype"]).name,
        "index": int(detail["index"]),
    }


def run_saved_model(signature, image_batch: np.ndarray) -> float:
    outputs = signature(image=tf.convert_to_tensor(image_batch, dtype=tf.float32))
    return float(np.asarray(next(iter(outputs.values())), dtype=np.float32).reshape(-1)[0])


def run_tflite(
    interpreter: tf.lite.Interpreter,
    input_index: int,
    output_index: int,
    input_dtype,
    image_batch: np.ndarray,
) -> float:
    interpreter.set_tensor(input_index, image_batch.astype(input_dtype, copy=False))
    interpreter.invoke()
    return float(np.asarray(interpreter.get_tensor(output_index), dtype=np.float32).reshape(-1)[0])


def summarize_diffs(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    diff = np.abs(np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32))
    return {
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "p95_abs_diff": float(np.percentile(diff, 95)) if diff.size else 0.0,
    }


def score_order(labels: list[str], scores: np.ndarray) -> list[str]:
    indices = np.argsort(-np.asarray(scores, dtype=np.float32))
    return [labels[int(index)] for index in indices]


def compute_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(scores, dtype=np.float32), kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.float32)
    return ranks


def pairwise_order_agreement(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
    if reference.size < 2:
        return None
    total = 0
    agreed = 0
    for left in range(reference.size):
        for right in range(left + 1, reference.size):
            ref_delta = float(reference[left] - reference[right])
            cand_delta = float(candidate[left] - candidate[right])
            if ref_delta == 0.0 and cand_delta == 0.0:
                agreed += 1
            elif ref_delta == 0.0 or cand_delta == 0.0:
                pass
            elif ref_delta * cand_delta > 0.0:
                agreed += 1
            total += 1
    if total == 0:
        return None
    return float(agreed / total)


def compare_rankings(labels: list[str], reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    reference_order = score_order(labels, reference)
    candidate_order = score_order(labels, candidate)
    reference_ranks = compute_ranks(reference)
    candidate_ranks = compute_ranks(candidate)
    spearman = None
    if reference_ranks.size >= 2:
        spearman = float(np.corrcoef(reference_ranks, candidate_ranks)[0, 1])
    top_k = min(5, len(reference_order))
    return {
        "reference_order": reference_order,
        "candidate_order": candidate_order,
        "exact_match": reference_order == candidate_order,
        "top_1_match": reference_order[:1] == candidate_order[:1],
        "top_5_match": reference_order[:top_k] == candidate_order[:top_k],
        "spearman_rank_corr": spearman,
        "pairwise_order_agreement": pairwise_order_agreement(reference, candidate),
    }


def summarize_scores(scores: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {"count": 0, "min": None, "max": None, "range": None, "mean": None, "std": None}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "range": float(values.max() - values.min()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
    }


def load_teacher_lookup(
    *,
    teacher_csv: Path,
    image_col: str,
    teacher_score_col: str,
    baseline_score_col: str | None,
) -> dict[str, dict[str, float | None]]:
    with teacher_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted({image_col, teacher_score_col} - fieldnames)
        if missing:
            raise ValueError(f"Teacher CSV is missing required columns: {', '.join(missing)}")
        if baseline_score_col and baseline_score_col not in fieldnames:
            baseline_score_col = None

        lookup: dict[str, dict[str, float | None]] = {}
        for row in reader:
            resolved_image_path = resolve_repo_path(row[image_col])
            image_key = canonicalize_image_key(resolved_image_path or row[image_col])
            if image_key is None:
                continue
            try:
                teacher_score = float(row[teacher_score_col])
            except (TypeError, ValueError):
                teacher_score = None
            baseline_score = None
            if baseline_score_col:
                try:
                    baseline_score = float(row[baseline_score_col])
                except (TypeError, ValueError):
                    baseline_score = None
            lookup[image_key] = {
                "teacher_score": teacher_score,
                "baseline_score": baseline_score,
            }
    return lookup


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    checkpoint_dir = resolve_repo_path(args.checkpoint_dir)
    tflite_path = resolve_repo_path(args.tflite_path)
    output_path = resolve_repo_path(args.output_path) if args.output_path else None
    if checkpoint_dir is None or tflite_path is None:
        raise ValueError("Could not resolve checkpoint_dir or tflite_path.")

    resolved_config = json.loads((checkpoint_dir / "resolved_config.json").read_text(encoding="utf-8"))
    image_size = int(resolved_config["data"]["image_size"])
    image_paths = resolve_image_paths(args)
    if not image_paths:
        raise ValueError("No images were resolved for verification.")

    keras_model = tf.keras.models.load_model(checkpoint_dir / "final_model.keras", compile=False)
    saved_model = tf.saved_model.load(str(checkpoint_dir / "saved_model"))
    signature = saved_model.signatures["serving_default"]

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=args.num_threads)
    input_details = interpreter.get_input_details()
    if len(input_details) != 1:
        raise ValueError(f"Expected a single TFLite input, found {len(input_details)}.")
    interpreter.resize_tensor_input(input_details[0]["index"], [1, image_size, image_size, 3], strict=False)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(output_details) != 1:
        raise ValueError(f"Expected a single TFLite output, found {len(output_details)}.")

    input_index = int(input_details[0]["index"])
    output_index = int(output_details[0]["index"])
    input_dtype = input_details[0]["dtype"]

    records: list[dict[str, object]] = []
    skipped_images: list[dict[str, str]] = []
    keras_scores = []
    saved_model_scores = []
    tflite_scores = []

    for image_path in image_paths:
        try:
            image_array = load_image_array(image_path, image_size=image_size)
        except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError, FileNotFoundError, OSError, ValueError) as exc:
            skipped_images.append(
                {
                    "image_path": str(image_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        image_batch = image_array[None, ...].astype(np.float32)
        keras_score = float(np.asarray(keras_model.predict(image_batch, verbose=0), dtype=np.float32).reshape(-1)[0])
        saved_model_score = run_saved_model(signature, image_batch)
        tflite_score = run_tflite(interpreter, input_index, output_index, input_dtype, image_batch)

        keras_scores.append(keras_score)
        saved_model_scores.append(saved_model_score)
        tflite_scores.append(tflite_score)

        records.append(
            {
                "image_path": str(image_path),
                "keras_score": keras_score,
                "saved_model_score": saved_model_score,
                "tflite_score": tflite_score,
                "abs_diff_tflite_vs_keras": abs(tflite_score - keras_score),
                "abs_diff_tflite_vs_saved_model": abs(tflite_score - saved_model_score),
                "abs_diff_keras_vs_saved_model": abs(keras_score - saved_model_score),
            }
        )

    if not records:
        raise ValueError("No readable images were available for verification.")

    keras_scores_np = np.asarray(keras_scores, dtype=np.float32)
    saved_model_scores_np = np.asarray(saved_model_scores, dtype=np.float32)
    tflite_scores_np = np.asarray(tflite_scores, dtype=np.float32)
    labels = [Path(record["image_path"]).name for record in records]
    teacher_lookup = None
    if args.teacher_csv and args.teacher_score_col:
        teacher_csv = resolve_repo_path(args.teacher_csv)
        if teacher_csv is None:
            raise ValueError("Could not resolve teacher_csv.")
        teacher_lookup = load_teacher_lookup(
            teacher_csv=teacher_csv,
            image_col=str(args.teacher_image_col),
            teacher_score_col=str(args.teacher_score_col),
            baseline_score_col=str(args.baseline_score_col) if args.baseline_score_col else None,
        )
        for record in records:
            image_key = canonicalize_image_key(record["image_path"])
            teacher_payload = teacher_lookup.get(image_key) if teacher_lookup is not None else None
            if teacher_payload is not None:
                record["teacher_score"] = teacher_payload.get("teacher_score")
                if args.baseline_score_col:
                    record["baseline_score"] = teacher_payload.get("baseline_score")

    result = {
        "checkpoint_dir": str(checkpoint_dir),
        "tflite_path": str(tflite_path),
        "num_images": len(records),
        "num_skipped_images": len(skipped_images),
        "image_paths": [str(path) for path in image_paths],
        "processed_image_paths": [str(record["image_path"]) for record in records],
        "input_details": [tensor_detail_to_json(detail) for detail in input_details],
        "output_details": [tensor_detail_to_json(detail) for detail in output_details],
        "score_ranges": {
            "keras": {
                "min": float(keras_scores_np.min()),
                "max": float(keras_scores_np.max()),
            },
            "saved_model": {
                "min": float(saved_model_scores_np.min()),
                "max": float(saved_model_scores_np.max()),
            },
            "tflite": {
                "min": float(tflite_scores_np.min()),
                "max": float(tflite_scores_np.max()),
            },
        },
        "diff_summary": {
            "tflite_vs_keras": summarize_diffs(tflite_scores_np, keras_scores_np),
            "tflite_vs_saved_model": summarize_diffs(tflite_scores_np, saved_model_scores_np),
            "keras_vs_saved_model": summarize_diffs(keras_scores_np, saved_model_scores_np),
        },
        "ranking_summary": {
            "tflite_vs_keras": compare_rankings(labels, keras_scores_np, tflite_scores_np),
            "tflite_vs_saved_model": compare_rankings(labels, saved_model_scores_np, tflite_scores_np),
            "keras_vs_saved_model": compare_rankings(labels, keras_scores_np, saved_model_scores_np),
        },
        "skipped_images": skipped_images,
        "per_image": records,
    }

    if teacher_lookup is not None:
        teacher_records = [record for record in records if record.get("teacher_score") is not None]
        if teacher_records:
            teacher_scores_np = np.asarray([record["teacher_score"] for record in teacher_records], dtype=np.float32)
            keras_teacher_np = np.asarray([record["keras_score"] for record in teacher_records], dtype=np.float32)
            tflite_teacher_np = np.asarray([record["tflite_score"] for record in teacher_records], dtype=np.float32)
            teacher_tracking = {
                "teacher_csv": str(resolve_repo_path(args.teacher_csv)),
                "teacher_score_col": args.teacher_score_col,
                "baseline_score_col": args.baseline_score_col,
                "num_images_with_teacher": int(len(teacher_records)),
                "teacher_score_summary": summarize_scores(teacher_scores_np),
                "keras_score_summary": summarize_scores(keras_teacher_np),
                "tflite_score_summary": summarize_scores(tflite_teacher_np),
                "keras_vs_teacher": compute_regression_metrics(y_true=teacher_scores_np, y_pred=keras_teacher_np),
                "tflite_vs_teacher": compute_regression_metrics(y_true=teacher_scores_np, y_pred=tflite_teacher_np),
            }
            teacher_tracking["keras_vs_teacher"].update(
                sample_pairwise_accuracy(
                    y_true=teacher_scores_np,
                    y_pred=keras_teacher_np,
                    margin=float(args.pairwise_margin),
                    max_pairs=int(args.pairwise_max_pairs),
                    seed=42,
                )
            )
            teacher_tracking["keras_vs_teacher"].update(
                top_k_overlap(y_true=teacher_scores_np, y_pred=keras_teacher_np, k=int(args.top_k))
            )
            teacher_tracking["tflite_vs_teacher"].update(
                sample_pairwise_accuracy(
                    y_true=teacher_scores_np,
                    y_pred=tflite_teacher_np,
                    margin=float(args.pairwise_margin),
                    max_pairs=int(args.pairwise_max_pairs),
                    seed=42,
                )
            )
            teacher_tracking["tflite_vs_teacher"].update(
                top_k_overlap(y_true=teacher_scores_np, y_pred=tflite_teacher_np, k=int(args.top_k))
            )

            baseline_scores = [record.get("baseline_score") for record in teacher_records]
            if any(score is not None for score in baseline_scores):
                usable = [
                    (float(record["teacher_score"]), float(record["baseline_score"]))
                    for record in teacher_records
                    if record.get("baseline_score") is not None
                ]
                baseline_teacher_np = np.asarray([item[0] for item in usable], dtype=np.float32)
                baseline_scores_np = np.asarray([item[1] for item in usable], dtype=np.float32)
                teacher_tracking["baseline_score_summary"] = summarize_scores(baseline_scores_np)
                teacher_tracking["baseline_vs_teacher"] = compute_regression_metrics(
                    y_true=baseline_teacher_np,
                    y_pred=baseline_scores_np,
                )
                teacher_tracking["baseline_vs_teacher"].update(
                    sample_pairwise_accuracy(
                        y_true=baseline_teacher_np,
                        y_pred=baseline_scores_np,
                        margin=float(args.pairwise_margin),
                        max_pairs=int(args.pairwise_max_pairs),
                        seed=42,
                    )
                )
                teacher_tracking["baseline_vs_teacher"].update(
                    top_k_overlap(y_true=baseline_teacher_np, y_pred=baseline_scores_np, k=int(args.top_k))
                )

            result["teacher_tracking"] = teacher_tracking

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
