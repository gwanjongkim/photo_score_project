from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from src.distill.config import resolve_repo_path, validate_training_config, write_json
from src.distill.data import load_teacher_manifest, select_splits, summarize_cross_split_duplicates
from src.distill.evaluation import evaluate_model_on_frame, write_evaluation_artifacts
from src.distill.modeling import build_mobile_student_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained student against teacher labels.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--manifest_csv")
    parser.add_argument("--splits", default="val")
    parser.add_argument("--output_dir")
    return parser


def load_student_model(checkpoint_dir: Path, resolved_config: dict[str, object]) -> tf.keras.Model:
    final_model_path = checkpoint_dir / "final_model.keras"
    if final_model_path.exists():
        return tf.keras.models.load_model(final_model_path, compile=False)

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
    return model


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    checkpoint_dir = resolve_repo_path(args.checkpoint_dir)
    if checkpoint_dir is None:
        raise ValueError("Could not resolve checkpoint_dir.")
    resolved_config_path = checkpoint_dir / "resolved_config.json"
    resolved_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    validate_training_config(resolved_config)

    manifest_csv = resolve_repo_path(args.manifest_csv or resolved_config["manifest_csv"])
    if manifest_csv is None:
        raise ValueError("Could not resolve manifest_csv.")
    output_dir = resolve_repo_path(args.output_dir) if args.output_dir else checkpoint_dir / "manual_evaluation"
    if output_dir is None:
        raise ValueError("Could not resolve output_dir.")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = dict(resolved_config["data"])
    eval_cfg = dict(resolved_config["eval"])
    target_col = str(data_cfg["target_col"])
    manifest_frame = load_teacher_manifest(manifest_csv, target_col=target_col)
    duplicate_summary = summarize_cross_split_duplicates(manifest_frame)
    if duplicate_summary["num_cross_split_duplicates"] > 0:
        print(json.dumps({"cross_split_duplicate_warning": duplicate_summary}, ensure_ascii=False))
    model = load_student_model(checkpoint_dir, resolved_config=resolved_config)

    split_names = [part.strip() for part in str(args.splits).split(",") if part.strip()]

    summary: dict[str, object] = {
        "checkpoint_dir": str(checkpoint_dir),
        "manifest_csv": str(manifest_csv),
        "splits": {},
        "cross_split_duplicate_summary": duplicate_summary,
    }
    for split_name in split_names:
        split_frame = select_splits(manifest_frame, [split_name])
        metrics, predictions = evaluate_model_on_frame(
            model=model,
            frame=split_frame,
            image_size=int(data_cfg["image_size"]),
            batch_size=int(data_cfg["batch_size"]),
            target_col=target_col,
            pairwise_margin=float(eval_cfg["pairwise_margin"]),
            pairwise_max_pairs=int(eval_cfg["pairwise_max_pairs"]),
            top_k=int(eval_cfg["top_k"]),
            seed=int(resolved_config.get("seed", 42)),
        )
        metrics_path, predictions_path = write_evaluation_artifacts(output_dir, split_name, metrics, predictions)
        summary["splits"][split_name] = {
            "rows": int(len(split_frame)),
            "metrics": metrics,
            "metrics_path": str(metrics_path),
            "predictions_path": str(predictions_path),
        }

    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
