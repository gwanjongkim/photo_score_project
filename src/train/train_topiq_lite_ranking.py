# TOPIQ-lite FLIVE 순위 손실 실험 학습 루프를 실행한다.
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.datasets.arp_dataset import inspect_csv
from src.datasets.topiq_ranking_dataset import (
    load_iqa_csv,
    make_flive_pairs,
    make_pair_dataset,
    make_regression_dataset,
)
from src.models.topiq_lite import build_topiq_lite


IMAGE_SIZE = 384


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def _configure_tensorflow() -> None:
    tf.keras.mixed_precision.set_global_policy("float32")
    gpus = tf.config.list_physical_devices("GPU")
    print("Visible GPUs:", gpus)
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            print("memory growth setup failed:", exc)
    print("mixed precision policy:", tf.keras.mixed_precision.global_policy().name)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_training_log(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _command_for_summary() -> str:
    command = " ".join([sys.executable, *sys.argv])
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        return f"PYTHONPATH={pythonpath} {command}"
    return command


def _decode_resize_with_pad(path: tf.Tensor) -> tf.Tensor:
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)
    image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
    return image


def _make_eval_dataset(
    paths: list[str],
    targets: list[float],
    batch_size: int,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((paths, targets))

    def _map_fn(path: tf.Tensor, target: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = _decode_resize_with_pad(path)
        return image, tf.reshape(tf.cast(target, tf.float32), (1,))

    return dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)


def _evaluate_regression_loss(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    loss_fn: tf.keras.losses.Loss,
) -> dict[str, Any]:
    metric = tf.keras.metrics.Mean()
    batch_count = 0
    sample_count = 0
    for images, targets in dataset:
        predictions = model(images, training=False)
        metric.update_state(loss_fn(targets, predictions))
        batch_count += 1
        sample_count += int(tf.shape(images)[0].numpy())
    return {
        "loss": float(metric.result().numpy()),
        "batch_count": int(batch_count),
        "sample_count": int(sample_count),
    }


def _correlation_stats(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    try:
        from scipy.stats import spearmanr
    except Exception as exc:
        return {
            "srcc": None,
            "correlation_error": str(exc),
        }
    if len(targets) < 2:
        return {
            "srcc": None,
            "correlation_error": "Need at least two samples for SRCC.",
        }
    srcc = spearmanr(targets, predictions).correlation
    return {
        "srcc": None if np.isnan(srcc) else float(srcc),
        "correlation_error": None,
    }


def _quick_flive_val_diagnostics(
    model: tf.keras.Model,
    val_csv: str | Path,
    target_col: str,
    batch_size: int,
    limit: int = 512,
) -> dict[str, Any]:
    df = load_iqa_csv(val_csv, target_col=target_col)
    source = "all_val"
    if "dataset" in df.columns:
        mask = df["dataset"].astype(str).str.lower().str.contains("flive", na=False)
        if mask.any():
            df = df[mask].copy()
            source = "flive_subset"

    df = df.head(limit).reset_index(drop=True)
    if len(df) == 0:
        return {"sample_count": 0, "source": source}

    dataset = _make_eval_dataset(
        paths=df["image_path"].astype(str).tolist(),
        targets=df[target_col].astype("float32").tolist(),
        batch_size=batch_size,
    )
    predictions: list[float] = []
    targets: list[float] = []
    for images, batch_targets in dataset:
        batch_predictions = model(images, training=False).numpy().reshape(-1)
        predictions.extend(float(value) for value in batch_predictions)
        targets.extend(float(value) for value in batch_targets.numpy().reshape(-1))

    pred = np.asarray(predictions, dtype=np.float32)
    target = np.asarray(targets, dtype=np.float32)
    pred_std = float(np.std(pred)) if len(pred) else 0.0
    diagnostics: dict[str, Any] = {
        "sample_count": int(len(pred)),
        "source": source,
        "pred_min": float(np.min(pred)) if len(pred) else None,
        "pred_max": float(np.max(pred)) if len(pred) else None,
        "pred_mean": float(np.mean(pred)) if len(pred) else None,
        "pred_std": pred_std,
        "target_std": float(np.std(target)) if len(target) else None,
        "possible_mode_collapse": bool(pred_std < 0.01),
    }
    diagnostics.update(_correlation_stats(target, pred))
    return diagnostics


def _count_csv_rows(csv_path: str | Path) -> int:
    with Path(csv_path).expanduser().open("r", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimentally fine-tune TOPIQ-lite with FLIVE pair ranking loss."
    )
    parser.add_argument("--regression_train_csv", required=True)
    parser.add_argument("--regression_val_csv", required=True)
    parser.add_argument("--pair_csv", required=True)
    parser.add_argument("--flive_pair_source_csv")
    parser.add_argument("--make_pairs", action="store_true")
    parser.add_argument("--init_weights")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--target_col", default="normalized_mos")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pair_batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--ranking_lambda", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--min_pair_gap", type=float, default=0.05)
    parser.add_argument("--max_pairs", type=int, default=10000)
    parser.add_argument(
        "--max_steps_per_epoch",
        type=int,
        default=100,
        help="Safety cap for smoke runs; set <=0 to use the full paired epoch.",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=256,
        help="Safety cap for regression validation; set <=0 to use full validation.",
    )
    parser.add_argument(
        "--quick_flive_limit",
        type=int,
        default=256,
        help="Maximum FLIVE validation samples for quick SRCC diagnostics.",
    )
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@tf.function
def _train_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: tf.keras.losses.Loss,
    reg_images: tf.Tensor,
    reg_targets: tf.Tensor,
    pair_images_a: tf.Tensor,
    pair_images_b: tf.Tensor,
    pair_signs: tf.Tensor,
    ranking_lambda: float,
    tau: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as tape:
        reg_predictions = model(reg_images, training=True)
        reg_loss = loss_fn(reg_targets, reg_predictions)

        pred_a = model(pair_images_a, training=True)
        pred_b = model(pair_images_b, training=True)
        signs = tf.reshape(tf.cast(pair_signs, tf.float32), (-1, 1))
        rank_loss = tf.reduce_mean(
            tf.nn.softplus(-signs * (pred_a - pred_b) / tf.cast(tau, tf.float32))
        )
        total_loss = reg_loss + tf.cast(ranking_lambda, tf.float32) * rank_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    grad_pairs = [
        (gradient, variable)
        for gradient, variable in zip(gradients, model.trainable_variables)
        if gradient is not None
    ]
    optimizer.apply_gradients(grad_pairs)
    return reg_loss, rank_loss, total_loss


def main() -> None:
    args = _parse_args()
    if args.tau <= 0.0:
        raise ValueError("--tau must be positive.")
    if args.ranking_lambda < 0.0:
        raise ValueError("--ranking_lambda must be non-negative.")
    if args.quick_flive_limit <= 0:
        raise ValueError("--quick_flive_limit must be positive.")
    if args.make_pairs and not args.flive_pair_source_csv:
        raise ValueError("--flive_pair_source_csv is required with --make_pairs.")

    _set_seed(args.seed)
    _configure_tensorflow()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_summary = None
    if args.make_pairs:
        pair_summary = make_flive_pairs(
            csv_path=args.flive_pair_source_csv,
            output_csv=args.pair_csv,
            target_col=args.target_col,
            max_pairs=args.max_pairs,
            min_gap=args.min_pair_gap,
            seed=args.seed,
        )

    pair_csv_path = Path(args.pair_csv).expanduser().resolve()
    if not pair_csv_path.is_file():
        raise FileNotFoundError(f"Pair CSV not found: {pair_csv_path}")

    train_summary = inspect_csv(args.regression_train_csv, target_col=args.target_col)
    val_summary = inspect_csv(args.regression_val_csv, target_col=args.target_col)
    pair_count = _count_csv_rows(pair_csv_path)
    if pair_count == 0:
        raise ValueError(f"Pair CSV has no pairs: {pair_csv_path}")

    train_dataset = make_regression_dataset(
        args.regression_train_csv,
        target_col=args.target_col,
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_dataset = make_regression_dataset(
        args.regression_val_csv,
        target_col=args.target_col,
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        shuffle=False,
    )
    pair_dataset = make_pair_dataset(
        pair_csv_path,
        image_size=IMAGE_SIZE,
        batch_size=args.pair_batch_size,
        shuffle=True,
        seed=args.seed,
    )

    reg_steps = int(np.ceil(train_summary["row_count"] / args.batch_size))
    pair_steps = int(np.ceil(pair_count / args.pair_batch_size))
    full_steps_per_epoch = max(1, min(reg_steps, pair_steps))
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(full_steps_per_epoch, args.max_steps_per_epoch)
    else:
        steps_per_epoch = full_steps_per_epoch
    max_val_batches = None
    if args.max_val_samples > 0:
        max_val_batches = max(1, int(np.ceil(args.max_val_samples / args.batch_size)))

    backbone_weights = None if args.init_weights else "imagenet"
    model = build_topiq_lite(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        weights=backbone_weights,
        dropout_rate=0.3,
        dense_units=256,
        freeze_backbone=args.freeze_backbone,
    )
    if args.init_weights:
        init_weights_path = Path(args.init_weights).expanduser().resolve()
        if not init_weights_path.is_file():
            raise FileNotFoundError(f"Initial weights file not found: {init_weights_path}")
        model.load_weights(str(init_weights_path))
        print(f"Initialized model weights from {init_weights_path}")

    print(
        json.dumps(
            {
                "train_rows": train_summary["row_count"],
                "val_rows": val_summary["row_count"],
                "pair_count": pair_count,
                "full_steps_per_epoch": full_steps_per_epoch,
                "max_steps_per_epoch": args.max_steps_per_epoch,
                "max_val_samples": args.max_val_samples,
                "steps_per_epoch": steps_per_epoch,
                "freeze_backbone": bool(args.freeze_backbone),
                "ranking_lambda": args.ranking_lambda,
                "tau": args.tau,
            },
            indent=2,
            sort_keys=True,
        )
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_fn = tf.keras.losses.Huber(delta=0.1)
    best_val_loss = float("inf")
    best_epoch = 0
    log_rows: list[dict[str, Any]] = []
    best_weights_path = out_dir / "best.weights.h5"

    for epoch in range(1, args.epochs + 1):
        reg_iter = iter(train_dataset.repeat())
        pair_iter = iter(pair_dataset.repeat())
        reg_metric = tf.keras.metrics.Mean()
        rank_metric = tf.keras.metrics.Mean()
        total_metric = tf.keras.metrics.Mean()

        for step in range(steps_per_epoch):
            reg_images, reg_targets = next(reg_iter)
            (pair_images_a, pair_images_b), pair_signs = next(pair_iter)
            reg_loss, rank_loss, total_loss = _train_step(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                reg_images=reg_images,
                reg_targets=reg_targets,
                pair_images_a=pair_images_a,
                pair_images_b=pair_images_b,
                pair_signs=pair_signs,
                ranking_lambda=args.ranking_lambda,
                tau=args.tau,
            )
            reg_metric.update_state(reg_loss)
            rank_metric.update_state(rank_loss)
            total_metric.update_state(total_loss)
            if (step + 1) % 100 == 0 or (step + 1) == steps_per_epoch:
                print(
                    f"epoch {epoch}/{args.epochs} step {step + 1}/{steps_per_epoch} "
                    f"reg_loss={float(reg_metric.result().numpy()):.6f} "
                    f"rank_loss={float(rank_metric.result().numpy()):.6f} "
                    f"total_loss={float(total_metric.result().numpy()):.6f}"
                )

        val_eval_dataset = val_dataset
        if max_val_batches is not None:
            val_eval_dataset = val_eval_dataset.take(max_val_batches)
        val_eval = _evaluate_regression_loss(
            model=model,
            dataset=val_eval_dataset,
            loss_fn=loss_fn,
        )
        flive_diag = _quick_flive_val_diagnostics(
            model=model,
            val_csv=args.regression_val_csv,
            target_col=args.target_col,
            batch_size=args.batch_size,
            limit=args.quick_flive_limit,
        )
        row = {
            "epoch": epoch,
            "reg_loss": float(reg_metric.result().numpy()),
            "rank_loss": float(rank_metric.result().numpy()),
            "total_loss": float(total_metric.result().numpy()),
            "val_reg_loss": val_eval["loss"],
            "val_reg_loss_samples": val_eval["sample_count"],
            "quick_flive_val_srcc": flive_diag.get("srcc"),
            "quick_flive_pred_std": flive_diag.get("pred_std"),
            "quick_flive_possible_mode_collapse": flive_diag.get(
                "possible_mode_collapse"
            ),
        }
        log_rows.append(row)
        print("epoch summary:", json.dumps(row, sort_keys=True))

        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_epoch = epoch
            model.save_weights(best_weights_path)
            print(f"Saved best weights: {best_weights_path}")

    final_model_path = out_dir / "final_model.keras"
    model.save(final_model_path)
    _write_training_log(out_dir / "training_log.csv", log_rows)

    final_flive_diag = _quick_flive_val_diagnostics(
        model=model,
        val_csv=args.regression_val_csv,
        target_col=args.target_col,
        batch_size=args.batch_size,
        limit=args.quick_flive_limit,
    )
    summary = {
        "command": _command_for_summary(),
        "artifacts": {
            "best_weights": str(best_weights_path),
            "final_model": str(final_model_path),
            "training_log": str(out_dir / "training_log.csv"),
            "training_summary": str(out_dir / "training_summary.json"),
        },
        "args": vars(args),
        "best_epoch": int(best_epoch),
        "best_val_reg_loss": float(best_val_loss),
        "final_metrics": log_rows[-1] if log_rows else {},
        "final_quick_flive_val": final_flive_diag,
        "pair_count": int(pair_count),
        "pair_summary": pair_summary,
        "train_csv": train_summary,
        "val_csv": val_summary,
    }
    _write_json(out_dir / "training_summary.json", summary)
    print("Saved:", final_model_path)
    print("Saved:", best_weights_path)
    print("Saved:", out_dir / "training_log.csv")
    print("Saved:", out_dir / "training_summary.json")


if __name__ == "__main__":
    main()
