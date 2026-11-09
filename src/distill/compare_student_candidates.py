from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.distill.config import resolve_repo_path, validate_training_config, write_json
from src.distill.data import canonicalize_image_key, load_teacher_manifest, select_splits
from src.distill.evaluation import compute_regression_metrics, sample_pairwise_accuracy, top_k_overlap
from src.distill.modeling import build_mobile_student_model
from src.distill.preprocessing import decode_resize_normalize_student_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare a new student against a teacher and an older baseline student.")
    parser.add_argument("--student_checkpoint_dir", required=True)
    parser.add_argument("--baseline_checkpoint_dir")
    parser.add_argument("--teacher_manifest_csv")
    parser.add_argument("--teacher_split", default="val")
    parser.add_argument("--real_image_score_csv")
    parser.add_argument("--real_image_teacher_col", default="ensemble_conservative_score")
    parser.add_argument("--real_image_baseline_col", default="stage5_student_full_aadb_score")
    parser.add_argument("--real_image_label", default="test_vila")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pairwise_margin", type=float)
    parser.add_argument("--pairwise_max_pairs", type=int)
    parser.add_argument("--top_k", type=int)
    return parser


def load_checkpoint_model(checkpoint_dir: Path) -> tuple[tf.keras.Model, dict[str, object], int]:
    resolved_config = json.loads((checkpoint_dir / "resolved_config.json").read_text(encoding="utf-8"))
    validate_training_config(resolved_config)
    image_size = int(resolved_config["data"]["image_size"])
    final_model_path = checkpoint_dir / "final_model.keras"
    if final_model_path.exists():
        return tf.keras.models.load_model(final_model_path, compile=False), resolved_config, image_size

    data_cfg = dict(resolved_config["data"])
    model_cfg = dict(resolved_config["model"])
    model, _ = build_mobile_student_model(
        input_shape=(int(data_cfg["image_size"]), int(data_cfg["image_size"]), 3),
        alpha=float(model_cfg["alpha"]),
        head_units=int(model_cfg["head_units"]),
        dropout_rate=float(model_cfg["dropout_rate"]),
        backbone_trainable=bool(model_cfg["backbone_trainable"]),
        backbone_weights=None,
    )
    model.load_weights(checkpoint_dir / "best.weights.h5")
    return model, resolved_config, image_size


def load_image_array(image_path: str | Path, image_size: int) -> np.ndarray:
    image = decode_resize_normalize_student_image(tf.convert_to_tensor(str(image_path)), image_size=image_size)
    return np.asarray(image.numpy(), dtype=np.float32)


def predict_model(
    model: tf.keras.Model,
    image_paths: list[str],
    image_size: int,
    batch_size: int,
) -> tuple[np.ndarray, list[int], list[dict[str, str]]]:
    outputs: list[np.ndarray] = []
    kept_indices: list[int] = []
    skipped_images: list[dict[str, str]] = []
    batch_arrays: list[np.ndarray] = []

    def flush_batch() -> None:
        if not batch_arrays:
            return
        batch = np.stack(batch_arrays, axis=0)
        batch_output = np.asarray(model.predict(batch, verbose=0), dtype=np.float32).reshape(-1)
        outputs.append(batch_output)
        batch_arrays.clear()

    for index, path in enumerate(image_paths):
        try:
            batch_arrays.append(load_image_array(path, image_size=image_size))
            kept_indices.append(index)
        except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError, FileNotFoundError, OSError, ValueError) as exc:
            skipped_images.append(
                {
                    "image_path": str(path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue

        if len(batch_arrays) >= batch_size:
            flush_batch()

    flush_batch()
    return (
        np.concatenate(outputs, axis=0) if outputs else np.empty((0,), dtype=np.float32),
        kept_indices,
        skipped_images,
    )


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


def build_candidate_payload(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pairwise_margin: float,
    pairwise_max_pairs: int,
    top_k: int,
) -> dict[str, object]:
    metrics = compute_regression_metrics(y_true=y_true, y_pred=y_pred)
    metrics.update(
        sample_pairwise_accuracy(
            y_true=y_true,
            y_pred=y_pred,
            margin=pairwise_margin,
            max_pairs=pairwise_max_pairs,
            seed=42,
        )
    )
    metrics.update(top_k_overlap(y_true=y_true, y_pred=y_pred, k=top_k))
    score_summary = summarize_scores(y_pred)
    teacher_summary = summarize_scores(y_true)
    signed_error = np.asarray(y_pred, dtype=np.float32) - np.asarray(y_true, dtype=np.float32)
    return {
        "metrics": metrics,
        "score_summary": score_summary,
        "compression_vs_teacher": {
            "range_ratio": (
                None
                if not teacher_summary["range"]
                else float(score_summary["range"] / teacher_summary["range"])
            ),
            "std_ratio": None if not teacher_summary["std"] else float(score_summary["std"] / teacher_summary["std"]),
            "mean_bias": float(np.mean(signed_error)),
        },
    }


def attach_candidate_scores(frame: pd.DataFrame, column_name: str, scores: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    out[column_name] = np.asarray(scores, dtype=np.float32)
    out[f"{column_name}_signed_error"] = out[column_name] - out["teacher_score"]
    out[f"{column_name}_abs_error"] = out[f"{column_name}_signed_error"].abs()
    out[f"{column_name}_rank"] = out[column_name].rank(method="first", ascending=False).astype(int)
    return out


def write_diagnostic_csv(frame: pd.DataFrame, output_path: Path, primary_column: str) -> None:
    ordered = frame.sort_values(f"{primary_column}_abs_error", ascending=False)
    ordered.to_csv(output_path, index=False)


def compare_against_teacher(
    *,
    frame: pd.DataFrame,
    teacher_col: str,
    student_scores: np.ndarray,
    baseline_scores: np.ndarray | None,
    label: str,
    output_dir: Path,
    pairwise_margin: float,
    pairwise_max_pairs: int,
    top_k: int,
) -> dict[str, object]:
    work_frame = frame.copy().reset_index(drop=True)
    work_frame["teacher_score"] = pd.to_numeric(work_frame[teacher_col], errors="coerce").astype("float32")
    work_frame = attach_candidate_scores(work_frame, "student_score", student_scores)
    if baseline_scores is not None:
        work_frame = attach_candidate_scores(work_frame, "baseline_score", baseline_scores)

    output_dir.mkdir(parents=True, exist_ok=True)
    work_frame.to_csv(output_dir / f"{label}_scores.csv", index=False)
    write_diagnostic_csv(work_frame, output_dir / f"{label}_student_disagreements.csv", "student_score")
    if baseline_scores is not None:
        write_diagnostic_csv(work_frame, output_dir / f"{label}_baseline_disagreements.csv", "baseline_score")

    y_true = work_frame["teacher_score"].to_numpy(dtype=np.float32)
    summary = {
        "label": label,
        "num_images": int(len(work_frame)),
        "teacher_score_column": teacher_col,
        "teacher_summary": summarize_scores(y_true),
        "student": build_candidate_payload(
            y_true=y_true,
            y_pred=work_frame["student_score"].to_numpy(dtype=np.float32),
            pairwise_margin=pairwise_margin,
            pairwise_max_pairs=pairwise_max_pairs,
            top_k=top_k,
        ),
    }
    if baseline_scores is not None:
        summary["baseline"] = build_candidate_payload(
            y_true=y_true,
            y_pred=work_frame["baseline_score"].to_numpy(dtype=np.float32),
            pairwise_margin=pairwise_margin,
            pairwise_max_pairs=pairwise_max_pairs,
            top_k=top_k,
        )
        summary["delta_student_minus_baseline"] = {
            "mae": (
                float(summary["student"]["metrics"]["mae"] - summary["baseline"]["metrics"]["mae"])
                if summary["baseline"]["metrics"]["mae"] is not None
                else None
            ),
            "rmse": (
                float(summary["student"]["metrics"]["rmse"] - summary["baseline"]["metrics"]["rmse"])
                if summary["baseline"]["metrics"]["rmse"] is not None
                else None
            ),
            "pearson": (
                float(summary["student"]["metrics"]["pearson"] - summary["baseline"]["metrics"]["pearson"])
                if summary["baseline"]["metrics"]["pearson"] is not None
                else None
            ),
            "spearman": (
                float(summary["student"]["metrics"]["spearman"] - summary["baseline"]["metrics"]["spearman"])
                if summary["baseline"]["metrics"]["spearman"] is not None
                else None
            ),
            "pairwise_accuracy": (
                float(summary["student"]["metrics"]["pairwise_accuracy"] - summary["baseline"]["metrics"]["pairwise_accuracy"])
                if summary["baseline"]["metrics"]["pairwise_accuracy"] is not None
                else None
            ),
            "top_k_overlap": (
                float(summary["student"]["metrics"]["top_k_overlap"] - summary["baseline"]["metrics"]["top_k_overlap"])
                if summary["baseline"]["metrics"]["top_k_overlap"] is not None
                else None
            ),
            "range_ratio_delta": (
                None
                if summary["baseline"]["compression_vs_teacher"]["range_ratio"] is None
                else float(
                    summary["student"]["compression_vs_teacher"]["range_ratio"]
                    - summary["baseline"]["compression_vs_teacher"]["range_ratio"]
                )
            ),
            "std_ratio_delta": (
                None
                if summary["baseline"]["compression_vs_teacher"]["std_ratio"] is None
                else float(
                    summary["student"]["compression_vs_teacher"]["std_ratio"]
                    - summary["baseline"]["compression_vs_teacher"]["std_ratio"]
                )
            ),
        }
    return summary


def prepare_real_image_frame(path: Path, teacher_col: str, baseline_col: str | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "image_path" not in frame.columns:
        raise ValueError(f"real_image_score_csv is missing image_path: {path}")
    required = {"image_path", teacher_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"real_image_score_csv is missing columns: {', '.join(missing)}")
    if baseline_col and baseline_col not in frame.columns:
        baseline_col = None
    frame = frame.copy()
    frame["image_path"] = frame["image_path"].map(lambda value: str(resolve_repo_path(value)))
    frame["__image_key"] = frame["image_path"].map(canonicalize_image_key)
    return frame.dropna(subset=["__image_key"]).reset_index(drop=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    student_checkpoint_dir = resolve_repo_path(args.student_checkpoint_dir)
    baseline_checkpoint_dir = resolve_repo_path(args.baseline_checkpoint_dir) if args.baseline_checkpoint_dir else None
    output_dir = resolve_repo_path(args.output_dir)
    if student_checkpoint_dir is None or output_dir is None:
        raise ValueError("Could not resolve student_checkpoint_dir or output_dir.")
    output_dir.mkdir(parents=True, exist_ok=True)

    student_model, student_config, student_image_size = load_checkpoint_model(student_checkpoint_dir)
    eval_cfg = dict(student_config["eval"])
    pairwise_margin = float(args.pairwise_margin if args.pairwise_margin is not None else eval_cfg["pairwise_margin"])
    pairwise_max_pairs = int(
        args.pairwise_max_pairs if args.pairwise_max_pairs is not None else eval_cfg["pairwise_max_pairs"]
    )
    top_k = int(args.top_k if args.top_k is not None else eval_cfg["top_k"])

    baseline_model = None
    baseline_image_size = None
    if baseline_checkpoint_dir is not None:
        baseline_model, _, baseline_image_size = load_checkpoint_model(baseline_checkpoint_dir)

    final_summary: dict[str, object] = {
        "student_checkpoint_dir": str(student_checkpoint_dir),
        "baseline_checkpoint_dir": str(baseline_checkpoint_dir) if baseline_checkpoint_dir is not None else None,
        "pairwise_margin": pairwise_margin,
        "pairwise_max_pairs": pairwise_max_pairs,
        "top_k": top_k,
        "comparisons": {},
    }

    if args.teacher_manifest_csv:
        manifest_csv = resolve_repo_path(args.teacher_manifest_csv)
        if manifest_csv is None:
            raise ValueError("Could not resolve teacher_manifest_csv.")
        manifest_frame = load_teacher_manifest(manifest_csv, target_col="teacher_aesthetic_score")
        split_frame = select_splits(manifest_frame, [args.teacher_split]).reset_index(drop=True)
        image_paths = split_frame["image_path"].astype(str).tolist()
        student_scores, kept_indices, student_skipped = predict_model(
            student_model,
            image_paths,
            student_image_size,
            int(args.batch_size),
        )
        if len(kept_indices) != len(split_frame):
            split_frame = split_frame.iloc[kept_indices].reset_index(drop=True)
        baseline_scores = None
        baseline_skipped: list[dict[str, str]] = []
        if baseline_model is not None and baseline_image_size is not None:
            baseline_scores, baseline_kept_indices, baseline_skipped = predict_model(
                baseline_model,
                split_frame["image_path"].astype(str).tolist(),
                baseline_image_size,
                int(args.batch_size),
            )
            if len(baseline_kept_indices) != len(split_frame):
                split_frame = split_frame.iloc[baseline_kept_indices].reset_index(drop=True)
                student_scores = student_scores[np.asarray(baseline_kept_indices, dtype=np.int64)]
        manifest_summary = compare_against_teacher(
            frame=split_frame,
            teacher_col="teacher_aesthetic_score",
            student_scores=student_scores,
            baseline_scores=baseline_scores,
            label=f"manifest_{args.teacher_split}",
            output_dir=output_dir,
            pairwise_margin=pairwise_margin,
            pairwise_max_pairs=pairwise_max_pairs,
            top_k=top_k,
        )
        manifest_summary["manifest_csv"] = str(manifest_csv)
        manifest_summary["skipped_images"] = {
            "student": student_skipped,
            "baseline": baseline_skipped,
        }
        final_summary["comparisons"]["manifest"] = manifest_summary

    if args.real_image_score_csv:
        real_image_csv = resolve_repo_path(args.real_image_score_csv)
        if real_image_csv is None:
            raise ValueError("Could not resolve real_image_score_csv.")
        real_frame = prepare_real_image_frame(
            real_image_csv,
            teacher_col=args.real_image_teacher_col,
            baseline_col=args.real_image_baseline_col,
        )
        image_paths = real_frame["image_path"].astype(str).tolist()
        student_scores, kept_indices, student_skipped = predict_model(
            student_model,
            image_paths,
            student_image_size,
            int(args.batch_size),
        )
        if len(kept_indices) != len(real_frame):
            real_frame = real_frame.iloc[kept_indices].reset_index(drop=True)

        baseline_scores = None
        baseline_skipped: list[dict[str, str]] = []
        if args.real_image_baseline_col in real_frame.columns:
            baseline_scores = pd.to_numeric(real_frame[args.real_image_baseline_col], errors="coerce").to_numpy(
                dtype=np.float32
            )
        elif baseline_model is not None and baseline_image_size is not None:
            baseline_scores, baseline_kept_indices, baseline_skipped = predict_model(
                baseline_model,
                real_frame["image_path"].astype(str).tolist(),
                baseline_image_size,
                int(args.batch_size),
            )
            if len(baseline_kept_indices) != len(real_frame):
                real_frame = real_frame.iloc[baseline_kept_indices].reset_index(drop=True)
                student_scores = student_scores[np.asarray(baseline_kept_indices, dtype=np.int64)]
        real_summary = compare_against_teacher(
            frame=real_frame,
            teacher_col=args.real_image_teacher_col,
            student_scores=student_scores,
            baseline_scores=baseline_scores,
            label=args.real_image_label,
            output_dir=output_dir,
            pairwise_margin=pairwise_margin,
            pairwise_max_pairs=pairwise_max_pairs,
            top_k=top_k,
        )
        real_summary["real_image_score_csv"] = str(real_image_csv)
        real_summary["baseline_score_column"] = (
            args.real_image_baseline_col if args.real_image_baseline_col in real_frame.columns else None
        )
        real_summary["skipped_images"] = {
            "student": student_skipped,
            "baseline": baseline_skipped,
        }
        final_summary["comparisons"]["real_images"] = real_summary

    write_json(output_dir / "summary.json", final_summary)
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
