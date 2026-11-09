from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageOps
import tensorflow as tf

from src.distill.config import resolve_repo_path
from src.distill.data import load_teacher_manifest, select_splits
from src.distill.evaluation import compute_regression_metrics, sample_pairwise_accuracy, top_k_overlap
from src.export.tflite_presets import load_source_model
from src.models.nima_distribution import distribution_mean_score


DEFAULT_CANDIDATES = {
    "aadb_mobile": {
        "preset": "aadb",
        "checkpoint_path": "checkpoints/composition_aadb_gpu/final_model.keras",
        "tflite_path": "exports/tflite/aadb_mobile.tflite",
        "metadata_path": "exports/tflite/aadb_mobile.metadata.json",
        "input_size": 224,
        "output_semantics": "scalar_[0,1]",
    },
    "nima_mobile": {
        "preset": "nima",
        "checkpoint_path": "checkpoints/nima_ava_gpu/final_model.keras",
        "tflite_path": "exports/tflite/nima_mobile.tflite",
        "metadata_path": "exports/tflite/nima_mobile.metadata.json",
        "input_size": 224,
        "output_semantics": "distribution_mean_normalized_to_[0,1]",
    },
    "rgnet_aadb_gpu": {
        "preset": "rgnet",
        "checkpoint_path": "checkpoints/rgnet_aadb_gpu/final_model.keras",
        "tflite_path": "exports/tflite/rgnet_aadb_gpu.tflite",
        "metadata_path": "exports/tflite/rgnet_aadb_gpu.metadata.json",
        "input_size": 256,
        "output_semantics": "scalar_[0,1]",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare stage5 student against existing aesthetic mobile candidates.")
    parser.add_argument("--student_checkpoint_dir", required=True)
    parser.add_argument("--student_tflite_path")
    parser.add_argument("--manifest_csv")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output_path")
    return parser


def load_image_array(image_path: str | Path, image_size: int) -> np.ndarray:
    with Image.open(image_path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype="float32") / 255.0


def iter_batches(image_paths: list[str], image_size: int, batch_size: int):
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch = np.stack([load_image_array(path, image_size=image_size) for path in batch_paths], axis=0)
        yield batch_paths, batch.astype(np.float32, copy=False)


def predict_student(model: tf.keras.Model, image_paths: list[str], image_size: int, batch_size: int) -> np.ndarray:
    outputs = []
    for _, batch in iter_batches(image_paths, image_size=image_size, batch_size=batch_size):
        batch_output = np.asarray(model.predict(batch, verbose=0), dtype=np.float32).reshape(-1)
        outputs.append(batch_output)
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0,), dtype=np.float32)


def predict_candidate(
    *,
    preset: str,
    checkpoint_path: Path,
    image_paths: list[str],
    image_size: int,
    batch_size: int,
) -> np.ndarray:
    model = load_source_model(preset, checkpoint_path)
    outputs = []
    for _, batch in iter_batches(image_paths, image_size=image_size, batch_size=batch_size):
        raw_output = np.asarray(model.predict(batch, verbose=0), dtype=np.float32)
        if preset == "nima":
            mean_scores = distribution_mean_score(tf.convert_to_tensor(raw_output, dtype=tf.float32)).numpy()
            normalized = (np.asarray(mean_scores, dtype=np.float32) - 1.0) / 9.0
            outputs.append(np.clip(normalized, 0.0, 1.0))
        else:
            outputs.append(raw_output.reshape(-1))
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0,), dtype=np.float32)


def load_existing_metadata(path_value: str | Path) -> dict[str, object] | None:
    path = resolve_repo_path(path_value)
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_candidate_result(
    *,
    scores: np.ndarray,
    y_true: np.ndarray,
    eval_pairwise_margin: float,
    eval_pairwise_max_pairs: int,
    eval_top_k: int,
    metadata: dict[str, object] | None,
    tflite_path: Path | None,
    input_size: int,
    output_semantics: str,
) -> dict[str, object]:
    metrics = compute_regression_metrics(y_true=y_true, y_pred=scores)
    metrics.update(
        sample_pairwise_accuracy(
            y_true=y_true,
            y_pred=scores,
            margin=eval_pairwise_margin,
            max_pairs=eval_pairwise_max_pairs,
            seed=42,
        )
    )
    metrics.update(top_k_overlap(y_true=y_true, y_pred=scores, k=eval_top_k))

    tflite_size_bytes = None
    if tflite_path is not None and tflite_path.exists():
        tflite_size_bytes = int(tflite_path.stat().st_size)

    return {
        "metrics": metrics,
        "prediction_summary": {
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
        },
        "artifact": {
            "tflite_path": str(tflite_path) if tflite_path is not None else None,
            "tflite_size_bytes": tflite_size_bytes,
            "input_size": input_size,
            "output_semantics": output_semantics,
        },
        "metadata_excerpt": {
            "display_name": metadata.get("display_name") if metadata else None,
            "task": metadata.get("task") if metadata else None,
            "export": metadata.get("export") if metadata else None,
        },
    }


def load_existing_score_frame(student_checkpoint_dir: Path, split_name: str, limit: int | None) -> pd.DataFrame | None:
    predictions_path = student_checkpoint_dir / "evaluation" / f"{split_name}_predictions.csv"
    if not predictions_path.exists():
        return None
    frame = pd.read_csv(predictions_path)
    if limit is not None:
        frame = frame.head(max(0, int(limit))).reset_index(drop=True)
    return frame


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    student_checkpoint_dir = resolve_repo_path(args.student_checkpoint_dir)
    student_tflite_path = resolve_repo_path(args.student_tflite_path) if args.student_tflite_path else None
    output_path = resolve_repo_path(args.output_path) if args.output_path else None
    if student_checkpoint_dir is None:
        raise ValueError("Could not resolve student_checkpoint_dir.")

    resolved_config = json.loads((student_checkpoint_dir / "resolved_config.json").read_text(encoding="utf-8"))
    manifest_csv = resolve_repo_path(args.manifest_csv or resolved_config["manifest_csv"])
    if manifest_csv is None:
        raise ValueError("Could not resolve manifest_csv.")

    data_cfg = dict(resolved_config["data"])
    eval_cfg = dict(resolved_config["eval"])
    target_col = str(data_cfg["target_col"])
    frame = load_teacher_manifest(manifest_csv, target_col=target_col)
    split_frame = select_splits(frame, [args.split])
    if args.limit is not None:
        split_frame = split_frame.head(max(0, int(args.limit))).reset_index(drop=True)

    image_paths = split_frame["image_path"].astype(str).tolist()
    y_true = split_frame[target_col].astype("float32").to_numpy()
    existing_score_frame = load_existing_score_frame(student_checkpoint_dir, args.split, args.limit)

    results: dict[str, object] = {}

    if existing_score_frame is not None and "student_prediction" in existing_score_frame.columns:
        student_scores = existing_score_frame["student_prediction"].astype("float32").to_numpy()
    else:
        student_model = tf.keras.models.load_model(student_checkpoint_dir / "final_model.keras", compile=False)
        student_scores = predict_student(
            student_model,
            image_paths=image_paths,
            image_size=int(data_cfg["image_size"]),
            batch_size=int(args.batch_size),
        )
    student_metadata = None
    if student_tflite_path is not None:
        metadata_path = student_tflite_path.with_suffix(".metadata.json")
        if metadata_path.exists():
            student_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    results["stage5_student_full_aadb"] = build_candidate_result(
        scores=student_scores,
        y_true=y_true,
        eval_pairwise_margin=float(eval_cfg["pairwise_margin"]),
        eval_pairwise_max_pairs=int(eval_cfg["pairwise_max_pairs"]),
        eval_top_k=int(eval_cfg["top_k"]),
        metadata=student_metadata,
        tflite_path=student_tflite_path,
        input_size=int(data_cfg["image_size"]),
        output_semantics="scalar_[0,1]_stage5_aesthetic_component",
    )

    for candidate_name, candidate_cfg in DEFAULT_CANDIDATES.items():
        tflite_path = resolve_repo_path(candidate_cfg["tflite_path"])
        if existing_score_frame is not None and candidate_name == "aadb_mobile" and "aadb_score" in existing_score_frame.columns:
            scores = existing_score_frame["aadb_score"].astype("float32").to_numpy()
        elif existing_score_frame is not None and candidate_name == "nima_mobile" and "nima_mean_score" in existing_score_frame.columns:
            scores = np.clip((existing_score_frame["nima_mean_score"].astype("float32").to_numpy() - 1.0) / 9.0, 0.0, 1.0)
        elif existing_score_frame is not None and candidate_name == "rgnet_aadb_gpu" and "rgnet_score" in existing_score_frame.columns:
            scores = existing_score_frame["rgnet_score"].astype("float32").to_numpy()
        else:
            checkpoint_path = resolve_repo_path(candidate_cfg["checkpoint_path"])
            if checkpoint_path is None or not checkpoint_path.exists():
                continue
            scores = predict_candidate(
                preset=str(candidate_cfg["preset"]),
                checkpoint_path=checkpoint_path,
                image_paths=image_paths,
                image_size=int(candidate_cfg["input_size"]),
                batch_size=int(args.batch_size),
            )
        metadata = load_existing_metadata(candidate_cfg["metadata_path"])
        results[candidate_name] = build_candidate_result(
            scores=scores,
            y_true=y_true,
            eval_pairwise_margin=float(eval_cfg["pairwise_margin"]),
            eval_pairwise_max_pairs=int(eval_cfg["pairwise_max_pairs"]),
            eval_top_k=int(eval_cfg["top_k"]),
            metadata=metadata,
            tflite_path=tflite_path,
            input_size=int(candidate_cfg["input_size"]),
            output_semantics=str(candidate_cfg["output_semantics"]),
        )

    leaderboard = []
    for name, payload in results.items():
        metrics = payload["metrics"]
        leaderboard.append(
            {
                "candidate": name,
                "spearman": metrics.get("spearman"),
                "pearson": metrics.get("pearson"),
                "mae": metrics.get("mae"),
                "pairwise_accuracy": metrics.get("pairwise_accuracy"),
                "top_k_overlap": metrics.get("top_k_overlap"),
                "tflite_size_bytes": payload["artifact"].get("tflite_size_bytes"),
            }
        )
    leaderboard.sort(
        key=lambda item: (
            float("-inf") if item["spearman"] is None else float(item["spearman"]),
            float("-inf") if item["pairwise_accuracy"] is None else float(item["pairwise_accuracy"]),
        ),
        reverse=True,
    )

    result = {
        "comparison_target": {
            "target_column": target_col,
            "target_semantics": (
                "teacher_aesthetic_score / stage5_production_aesthetic_target_v2. "
                "This is the project aesthetic target the student was trained to match."
            ),
        },
        "manifest_csv": str(manifest_csv),
        "split": args.split,
        "num_samples": int(len(split_frame)),
        "leaderboard": leaderboard,
        "candidates": results,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
