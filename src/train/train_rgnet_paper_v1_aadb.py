# RGNet 논문 지향 v1 AADB 회귀 실험을 학습한다.
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from src.models.rgnet_paper_v1 import (
    MODEL_VARIANT,
    build_rgnet_paper_v1_model,
    get_rgnet_paper_v1_custom_objects,
)


DEFAULT_OUTPUT_DIR = "outputs/rgnet_paper_v1_aadb_regression_20260509/full_train"


def _read_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return loaded


def _cfg(config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    value = config.get(section, {})
    if isinstance(value, dict) and key in value:
        return value[key]
    return default


def _resolve(value: Any, config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return value if value is not None else _cfg(config, section, key, default)


def _normalize_weights(value: str | None) -> str | None:
    if value is None:
        return None
    if str(value).lower() in {"none", "null", "false", "random"}:
        return None
    return str(value)


def _parse_int_tuple(value: str | list[int] | tuple[int, ...] | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    return tuple(int(part) for part in value)


def _setup_tensorflow(seed: int) -> dict[str, object]:
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.utils.set_random_seed(seed)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            print("memory growth setup failed:", exc)
    return {
        "visible_gpus": [str(gpu) for gpu in gpus],
        "mixed_precision_policy": str(tf.keras.mixed_precision.global_policy()),
        "tensorflow_version": tf.__version__,
    }


def _load_frame(csv_path: str, image_col: str, target_col: str, max_samples: int | None) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required = {image_col, target_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")
    frame = frame.dropna(subset=[image_col, target_col]).reset_index(drop=True)
    if max_samples is not None:
        frame = frame.head(int(max_samples)).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{csv_path} produced an empty frame")
    return frame


def _decode_image(path: tf.Tensor, image_size: int, training: bool) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [image_size, image_size])
    if training:
        image = tf.image.random_flip_left_right(image)
    return image


def _make_dataset(
    frame: pd.DataFrame,
    image_col: str,
    target_col: str,
    image_size: int,
    batch_size: int,
    training: bool,
    shuffle: bool,
) -> tf.data.Dataset:
    paths = frame[image_col].astype(str).to_numpy()
    targets = frame[target_col].astype("float32").to_numpy().reshape(-1, 1)
    dataset = tf.data.Dataset.from_tensor_slices((paths, targets))
    if training and shuffle:
        dataset = dataset.shuffle(min(len(frame), 10000), reshuffle_each_iteration=True)

    def _map(path: tf.Tensor, target: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return _decode_image(path, image_size=image_size, training=training), target

    return dataset.map(_map, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _float_history(history: dict[str, list[float]]) -> dict[str, list[float]]:
    return {key: [float(value) for value in values] for key, values in history.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the isolated RGNet-paper-v1 AADB regression model.")
    parser.add_argument("--config")
    parser.add_argument("--train_csv")
    parser.add_argument("--val_csv")
    parser.add_argument("--image_col")
    parser.add_argument("--target_col")
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--out_dir")
    parser.add_argument("--backbone_weights")
    parser.add_argument("--region_dim", type=int)
    parser.add_argument("--graph_units", type=int)
    parser.add_argument("--graph_blocks", type=int)
    parser.add_argument("--graph_temperature", type=float)
    parser.add_argument("--graph_dropout", type=float)
    parser.add_argument("--head_dropout", type=float)
    parser.add_argument("--dilation_rates")
    parser.add_argument("--aggregation")
    parser.add_argument("--lse_r", type=float)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_val_samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no_verify_save_load", dest="verify_save_load", action="store_false", default=None)
    parser.add_argument("--verify_save_load", dest="verify_save_load", action="store_true")
    parser.add_argument("--allow_cpu_full", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _read_yaml(args.config)

    train_csv = _resolve(args.train_csv, config, "data", "train_csv", "data/processed/aadb/train.csv")
    val_csv = _resolve(args.val_csv, config, "data", "val_csv", "data/processed/aadb/val.csv")
    image_col = _resolve(args.image_col, config, "data", "image_col", "image_path")
    target_col = _resolve(args.target_col, config, "data", "target_col", "score")
    image_size = int(_resolve(args.image_size, config, "model", "image_size", 256))
    batch_size = int(_resolve(args.batch_size, config, "training", "batch_size", 8))
    epochs = int(_resolve(args.epochs, config, "training", "epochs", 20))
    learning_rate = float(_resolve(args.learning_rate, config, "training", "learning_rate", 1e-4))
    patience = int(_resolve(args.patience, config, "training", "early_stopping_patience", _cfg(config, "training", "patience", 3)))
    out_dir = Path(_resolve(args.out_dir, config, "experiment", "train_output_dir", DEFAULT_OUTPUT_DIR))
    backbone_weights = _normalize_weights(_resolve(args.backbone_weights, config, "model", "backbone_weights", "imagenet"))
    region_dim = int(_resolve(args.region_dim, config, "model", "region_dim", 256))
    graph_units = int(_resolve(args.graph_units, config, "model", "graph_units", 256))
    graph_blocks = int(_resolve(args.graph_blocks, config, "model", "graph_blocks", 3))
    graph_temperature = float(_resolve(args.graph_temperature, config, "model", "graph_temperature", 0.25))
    graph_dropout = float(_resolve(args.graph_dropout, config, "model", "graph_dropout", 0.1))
    head_dropout = float(_resolve(args.head_dropout, config, "model", "head_dropout", 0.3))
    dilation_rates = _parse_int_tuple(_resolve(args.dilation_rates, config, "model", "dilation_rates", None), (1, 3, 6, 12, 18))
    aggregation = str(_resolve(args.aggregation, config, "model", "aggregation", "lse")).lower()
    lse_r = float(_resolve(args.lse_r, config, "model", "lse_r", 4.0))
    max_train_samples = _resolve(args.max_train_samples, config, "training", "max_train_samples", None)
    max_val_samples = _resolve(args.max_val_samples, config, "training", "max_val_samples", None)
    seed = int(_resolve(args.seed, config, "training", "seed", 42))
    verify_save_load = bool(_resolve(args.verify_save_load, config, "training", "verify_save_load", True))

    out_dir.mkdir(parents=True, exist_ok=True)
    tf_info = _setup_tensorflow(seed)

    train_frame = _load_frame(str(train_csv), image_col, target_col, max_train_samples)
    val_frame = _load_frame(str(val_csv), image_col, target_col, max_val_samples)
    if not tf_info["visible_gpus"] and len(train_frame) > 1000 and epochs > 3 and not args.allow_cpu_full:
        raise RuntimeError(
            "GPU is not visible for a long/full run. Re-run with GPU access or pass --allow_cpu_full explicitly."
        )

    train_ds = _make_dataset(train_frame, image_col, target_col, image_size, batch_size, training=True, shuffle=True)
    val_ds = _make_dataset(val_frame, image_col, target_col, image_size, batch_size, training=False, shuffle=False)

    model = build_rgnet_paper_v1_model(
        input_shape=(image_size, image_size, 3),
        backbone_weights=backbone_weights,
        region_dim=region_dim,
        graph_units=graph_units,
        graph_blocks=graph_blocks,
        graph_temperature=graph_temperature,
        graph_dropout=graph_dropout,
        head_dropout=head_dropout,
        dilation_rates=dilation_rates,
        aggregation=aggregation,
        lse_r=lse_r,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    sample_images, _ = next(iter(train_ds.take(1)))
    forward_sample = model(sample_images, training=False)
    forward_check = {
        "input_shape": [int(dim) for dim in sample_images.shape],
        "output_shape": [int(dim) for dim in forward_sample.shape],
        "prediction_min": float(tf.reduce_min(forward_sample).numpy()),
        "prediction_max": float(tf.reduce_max(forward_sample).numpy()),
        "all_finite": bool(np.isfinite(forward_sample.numpy()).all()),
        "in_unit_range": bool((forward_sample.numpy() >= 0.0).all() and (forward_sample.numpy() <= 1.0).all()),
    }

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(out_dir / "best.weights.h5"),
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(str(out_dir / "training_history.csv")),
    ]

    history_obj = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    history = _float_history(history_obj.history)

    final_model_path = out_dir / "final_model.keras"
    trained_sample = model(sample_images, training=False)
    model.save(str(final_model_path))

    save_load_check: dict[str, object] | None = None
    if verify_save_load:
        loaded = tf.keras.models.load_model(
            str(final_model_path),
            compile=False,
            safe_mode=False,
            custom_objects=get_rgnet_paper_v1_custom_objects(),
        )
        loaded_sample = loaded(sample_images, training=False)
        max_abs_diff = tf.reduce_max(tf.abs(tf.cast(trained_sample, tf.float32) - tf.cast(loaded_sample, tf.float32)))
        save_load_check = {
            "loaded_model_name": loaded.name,
            "loaded_output_shape": [int(dim) for dim in loaded_sample.shape],
            "max_abs_diff_vs_trained_forward_sample": float(max_abs_diff.numpy()),
        }

    val_losses = history.get("val_loss", [])
    best_epoch = int(np.argmin(val_losses) + 1) if val_losses else None
    best_val_loss = float(min(val_losses)) if val_losses else None
    best_val_mae = None
    if best_epoch is not None and history.get("val_mae"):
        best_val_mae = float(history["val_mae"][best_epoch - 1])

    summary = {
        "created_at_local": datetime.now().astimezone().isoformat(),
        "model_variant": MODEL_VARIANT,
        "official_reproduction": False,
        "paper_comparability_note": (
            "Uses DenseNet121, ASPP approximation, spatial region nodes, cosine adjacency, "
            "residual graph convolution, region-level scores, and configurable aggregation. "
            "This is not the official paper implementation."
        ),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "image_col": image_col,
        "target_col": target_col,
        "train_samples": int(len(train_frame)),
        "val_samples": int(len(val_frame)),
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "epochs_completed": int(len(history.get("loss", []))),
        "learning_rate": learning_rate,
        "early_stopping": {
            "monitor": "val_loss",
            "patience": patience,
            "restore_best_weights": True,
        },
        "model": {
            "backbone": "DenseNet121",
            "backbone_weights_requested": backbone_weights,
            "context_module": "ASPP approximation",
            "dilation_rates": list(dilation_rates),
            "region_dim": region_dim,
            "graph_units": graph_units,
            "graph_blocks": graph_blocks,
            "graph_temperature": graph_temperature,
            "graph_dropout": graph_dropout,
            "head_dropout": head_dropout,
            "adjacency": "cosine similarity, softmax-normalized",
            "graph_convolution": "residual graph convolution",
            "region_score_head": "per-node sigmoid scalar score",
            "aggregation": aggregation,
            "lse_r": lse_r,
            "aggregation_formula": "(1 / r) * log(mean(exp(r * region_scores))) for aggregation=lse",
            "output_activation": "region sigmoid before aggregation",
        },
        "tensorflow": tf_info,
        "forward_check": forward_check,
        "save_load_check": save_load_check,
        "outputs": {
            "final_model": str(final_model_path),
            "best_weights": str(out_dir / "best.weights.h5"),
            "training_history_csv": str(out_dir / "training_history.csv"),
            "train_summary_json": str(out_dir / "train_summary.json"),
        },
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_mae": best_val_mae,
        "history": history,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
