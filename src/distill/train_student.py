from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.distill.config import load_training_config, resolve_repo_path, to_serializable, write_json
from src.distill.data import (
    load_teacher_manifest,
    make_student_dataset,
    select_splits,
    summarize_cross_split_duplicates,
)
from src.distill.evaluation import (
    ValidationRankingMetricsCallback,
    evaluate_model_on_frame,
    write_evaluation_artifacts,
)
from src.distill.modeling import build_distillation_loss, build_mobile_student_model, build_optimizer
from src.distill.preprocessing import student_preprocessing_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a first-pass mobile student on stage5 teacher labels.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest_csv")
    parser.add_argument("--output_dir")
    parser.add_argument("--resume", action="store_true", help="Resume from output_dir/best.weights.h5 when available.")
    parser.add_argument("--resume_weights_path")
    return parser


def configure_tensorflow() -> tuple[list[tf.config.PhysicalDevice], bool]:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"✅ GPU available: {len(gpus)} devices found.")
        for gpu in gpus:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"  - {details.get('device_name', 'Unknown GPU')}")
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as exc:
                print(f"    Memory growth setup failed: {exc}", file=sys.stderr)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("  Mixed precision enabled.")
        return gpus, True
    print("⚠️ GPU not available. Training will run on CPU.", file=sys.stderr)
    return gpus, False


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def export_saved_model(model: tf.keras.Model, export_dir: Path) -> Path:
    if hasattr(model, "export"):
        model.export(export_dir)
    else:
        tf.saved_model.save(model, export_dir)
    return export_dir


def maybe_load_bootstrap_weights(
    *,
    model: tf.keras.Model,
    output_dir: Path,
    model_cfg: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    resume_weights_path = resolve_repo_path(args.resume_weights_path) if args.resume_weights_path else None
    output_best_weights = output_dir / "best.weights.h5"
    if args.resume and output_best_weights.exists():
        resume_weights_path = output_best_weights
    elif args.resume and not output_best_weights.exists():
        print(f"Resume requested but no existing checkpoint was found: {output_best_weights}")

    initial_weights_path = resolve_repo_path(model_cfg.get("initial_weights_path"))

    loaded_weights_path = None
    loaded_weights_reason = None

    if resume_weights_path is not None:
        if not resume_weights_path.exists():
            raise FileNotFoundError(f"Resume weights not found: {resume_weights_path}")
        model.load_weights(resume_weights_path)
        loaded_weights_path = resume_weights_path
        loaded_weights_reason = "resume"
        print(f"Loaded resume weights: {resume_weights_path}")
    elif initial_weights_path is not None:
        if not initial_weights_path.exists():
            raise FileNotFoundError(f"Initial weights not found: {initial_weights_path}")
        model.load_weights(initial_weights_path)
        loaded_weights_path = initial_weights_path
        loaded_weights_reason = "initial_weights"
        print(f"Loaded initial weights: {initial_weights_path}")

    return {
        "loaded_weights": loaded_weights_path is not None,
        "loaded_weights_path": str(loaded_weights_path) if loaded_weights_path is not None else None,
        "loaded_weights_reason": loaded_weights_reason,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    gpus, gpu_available = configure_tensorflow()

    config, config_path = load_training_config(args.config)
    if args.manifest_csv:
        config["manifest_csv"] = args.manifest_csv
    if args.output_dir:
        config["output_dir"] = args.output_dir

    manifest_csv = resolve_repo_path(config["manifest_csv"])
    output_dir = resolve_repo_path(config["output_dir"])
    if manifest_csv is None or output_dir is None:
        raise ValueError("manifest_csv and output_dir must resolve to real paths.")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    data_cfg = dict(config["data"])
    model_cfg = dict(config["model"])
    train_cfg = dict(config["train"])
    eval_cfg = dict(config["eval"])
    target_col = str(data_cfg["target_col"])

    manifest_frame = load_teacher_manifest(manifest_csv, target_col=target_col)
    duplicate_summary = summarize_cross_split_duplicates(manifest_frame)
    if duplicate_summary["num_cross_split_duplicates"] > 0:
        print(json.dumps({"cross_split_duplicate_warning": duplicate_summary}, ensure_ascii=False))
    train_frame = select_splits(manifest_frame, data_cfg["train_splits"])
    val_frame = select_splits(manifest_frame, data_cfg["val_splits"])
    test_splits = list(data_cfg.get("test_splits") or [])
    test_frame = select_splits(manifest_frame, test_splits) if test_splits else None

    print("\n" + "=" * 80)
    print("TRAINING PRE-FLIGHT CHECK")
    print("-" * 80)
    print(f"Manifest path: {manifest_csv}")
    print(f"Train split size: {len(train_frame):,}")
    print(f"Val split size: {len(val_frame):,}")
    print(f"Test split size: {len(test_frame):,}" if test_frame is not None else "Test split: Not used")
    print(f"Image size: {data_cfg['image_size']}x{data_cfg['image_size']}")
    print(f"Batch size: {data_cfg['batch_size']}")
    print("-" * 80)

    ranking_weight = float(train_cfg["ranking_weight"]) if bool(train_cfg["use_ranking_loss"]) else 0.0
    model, pretrained_weights_loaded = build_mobile_student_model(
        input_shape=(int(data_cfg["image_size"]), int(data_cfg["image_size"]), 3),
        alpha=float(model_cfg["alpha"]),
        head_units=int(model_cfg["head_units"]),
        dropout_rate=float(model_cfg["dropout_rate"]),
        backbone_trainable=bool(model_cfg["backbone_trainable"]),
        backbone_weights=model_cfg.get("backbone_weights"),
        backbone_weights_path=resolve_repo_path(model_cfg.get("backbone_weights_path")),
    )

    print(f"GPU available for training: {gpu_available}")
    print(f"Pretrained backbone weights loaded: {pretrained_weights_loaded}")
    print("=" * 80 + "\n")

    if not gpu_available:
        print(
            "CRITICAL WARNING: No GPU detected. Full training runs will be extremely slow. "
            "It is recommended to stop and resolve the host environment.",
            file=sys.stderr,
        )
    if not pretrained_weights_loaded:
        print(
            "CRITICAL WARNING: Training with a randomly initialized backbone. "
            "The model is unlikely to converge to a useful state without pretrained weights.",
            file=sys.stderr,
        )

    weight_bootstrap = maybe_load_bootstrap_weights(
        model=model,
        output_dir=output_dir,
        model_cfg=model_cfg,
        args=args,
    )

    model.compile(
        optimizer=build_optimizer(
            learning_rate=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
        ),
        loss=build_distillation_loss(
            loss_name=str(train_cfg["loss"]),
            huber_delta=float(train_cfg["huber_delta"]),
            ranking_weight=ranking_weight,
            ranking_margin=float(train_cfg["ranking_margin"]),
        ),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    train_dataset = make_student_dataset(
        frame=train_frame,
        image_size=int(data_cfg["image_size"]),
        batch_size=int(data_cfg["batch_size"]),
        target_col=target_col,
        training=True,
        shuffle_buffer=int(data_cfg["shuffle_buffer"]),
        augment_horizontal_flip=bool(data_cfg["augment_horizontal_flip"]),
        cache=bool(data_cfg["cache"]),
    )
    val_dataset = make_student_dataset(
        frame=val_frame,
        image_size=int(data_cfg["image_size"]),
        batch_size=int(data_cfg["batch_size"]),
        target_col=target_col,
        training=False,
        cache=bool(data_cfg["cache"]),
    )

    best_weights_path = output_dir / "best.weights.h5"
    ranking_metrics_callback = ValidationRankingMetricsCallback(
        frame=val_frame,
        image_size=int(data_cfg["image_size"]),
        batch_size=int(data_cfg["batch_size"]),
        target_col=target_col,
        pairwise_margin=float(eval_cfg["pairwise_margin"]),
        pairwise_max_pairs=int(eval_cfg["pairwise_max_pairs"]),
        top_k=int(eval_cfg["top_k"]),
        seed=seed,
    )
    checkpoint_monitor = str(train_cfg["checkpoint_monitor"])
    checkpoint_monitor_mode = str(train_cfg["checkpoint_monitor_mode"]).lower()
    callbacks: list[tf.keras.callbacks.Callback] = [
        ranking_metrics_callback,
        tf.keras.callbacks.ModelCheckpoint(
            str(best_weights_path),
            save_best_only=True,
            save_weights_only=True,
            monitor=checkpoint_monitor,
            mode=checkpoint_monitor_mode,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=checkpoint_monitor,
            patience=int(train_cfg["early_stopping_patience"]),
            restore_best_weights=True,
            mode=checkpoint_monitor_mode,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=int(train_cfg["reduce_lr_patience"]),
            min_lr=float(train_cfg["min_learning_rate"]),
        ),
    ]

    resolved_config = dict(config)
    resolved_config["manifest_csv"] = str(manifest_csv)
    resolved_config["output_dir"] = str(output_dir)
    resolved_config["config_source"] = str(config_path)
    resolved_config["preprocessing"] = student_preprocessing_metadata(int(data_cfg["image_size"]))
    resolved_config["weight_bootstrap"] = weight_bootstrap
    write_json(output_dir / "resolved_config.json", resolved_config)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=int(train_cfg["epochs"]),
        callbacks=callbacks,
    )

    final_model_path = output_dir / "final_model.keras"
    saved_model_dir = output_dir / "saved_model"
    model.save(final_model_path)
    export_saved_model(model, saved_model_dir)
    write_json(output_dir / "history.json", history.history)
    write_json(output_dir / "validation_ranking_history.json", {"epochs": ranking_metrics_callback.metric_history})

    val_metrics, val_predictions = evaluate_model_on_frame(
        model=model,
        frame=val_frame,
        image_size=int(data_cfg["image_size"]),
        batch_size=int(data_cfg["batch_size"]),
        target_col=target_col,
        pairwise_margin=float(eval_cfg["pairwise_margin"]),
        pairwise_max_pairs=int(eval_cfg["pairwise_max_pairs"]),
        top_k=int(eval_cfg["top_k"]),
        seed=seed,
    )
    val_metrics_path, val_predictions_path = write_evaluation_artifacts(output_dir / "evaluation", "val", val_metrics, val_predictions)

    test_metrics = None
    test_metrics_path = None
    test_predictions_path = None
    if test_frame is not None:
        test_metrics, test_predictions = evaluate_model_on_frame(
            model=model,
            frame=test_frame,
            image_size=int(data_cfg["image_size"]),
            batch_size=int(data_cfg["batch_size"]),
            target_col=target_col,
            pairwise_margin=float(eval_cfg["pairwise_margin"]),
            pairwise_max_pairs=int(eval_cfg["pairwise_max_pairs"]),
            top_k=int(eval_cfg["top_k"]),
            seed=seed,
        )
        test_metrics_path, test_predictions_path = write_evaluation_artifacts(
            output_dir / "evaluation",
            "test",
            test_metrics,
            test_predictions,
        )

    summary = {
        "experiment_name": config.get("experiment_name"),
        "train_rows": int(len(train_frame)),
        "val_rows": int(len(val_frame)),
        "test_rows": int(len(test_frame)) if test_frame is not None else 0,
        "manifest_csv": str(manifest_csv),
        "output_dir": str(output_dir),
        "final_model": str(final_model_path),
        "saved_model": str(saved_model_dir),
        "best_weights": str(best_weights_path),
        "resolved_config": str(output_dir / "resolved_config.json"),
        "history": str(output_dir / "history.json"),
        "validation_ranking_history": str(output_dir / "validation_ranking_history.json"),
        "val_metrics": to_serializable(val_metrics),
        "val_metrics_path": str(val_metrics_path),
        "val_predictions_path": str(val_predictions_path),
        "test_metrics": to_serializable(test_metrics) if test_metrics is not None else None,
        "test_metrics_path": str(test_metrics_path) if test_metrics_path else None,
        "test_predictions_path": str(test_predictions_path) if test_predictions_path else None,
        "ranking_loss_enabled": ranking_weight > 0.0,
        "checkpoint_monitor": checkpoint_monitor,
        "checkpoint_monitor_mode": checkpoint_monitor_mode,
        "weight_bootstrap": weight_bootstrap,
        "cross_split_duplicate_summary": duplicate_summary,
    }
    print(json.dumps(to_serializable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
