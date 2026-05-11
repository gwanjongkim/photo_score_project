# A-LAMP 논문 지향 AVA 이진 분류 실험을 학습한다.
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
from PIL import Image

from src.datasets.native_size_dataset import prepare_alamp_inputs
from src.models.alamp_paper_ava import (
    MODEL_DESCRIPTION,
    MODEL_VARIANT,
    STYLE_DESCRIPTION,
    build_alamp_paper_ava_model,
    get_alamp_paper_ava_custom_objects,
)


DEFAULT_OUTPUT_DIR = "outputs/alamp_paper_ava_classification_20260511/mid_train/v0_a"
LABEL_RULES = {"paper_strict", "project_compatible"}


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


def _label_rule_text(score_col: str, label_threshold: float, label_rule: str) -> str:
    if label_rule == "paper_strict":
        return f"label = 1 if {score_col} > {label_threshold} else 0"
    if label_rule == "project_compatible":
        return f"label = 1 if {score_col} >= {label_threshold} else 0"
    raise ValueError(f"Unsupported label_rule: {label_rule}")


def _labels_from_scores(scores: pd.Series, label_threshold: float, label_rule: str) -> pd.Series:
    if label_rule == "paper_strict":
        return (scores.astype("float32") > float(label_threshold)).astype("float32")
    if label_rule == "project_compatible":
        return (scores.astype("float32") >= float(label_threshold)).astype("float32")
    raise ValueError(f"Unsupported label_rule: {label_rule}")


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


def _resolve_image_paths(frame: pd.DataFrame, image_col: str, image_dir: str | None) -> pd.Series:
    image_root = Path(image_dir) if image_dir else None

    def _resolve_path(value: object) -> str:
        raw = Path(str(value))
        if raw.is_absolute() or raw.exists():
            return str(raw)
        if image_root is not None:
            candidate = image_root / raw.name
            if candidate.exists():
                return str(candidate)
        return str(raw)

    return frame[image_col].map(_resolve_path)


def _verify_decodable_image(path: str) -> None:
    with Image.open(path) as image:
        image.convert("RGB").load()


def _filter_decodable_images(frame: pd.DataFrame, image_col: str) -> pd.DataFrame:
    kept_indices: list[int] = []
    skipped: list[dict[str, str]] = []
    for index, row in frame.iterrows():
        resolved_path = str(row["_resolved_image_path"])
        try:
            _verify_decodable_image(resolved_path)
        except Exception as exc:
            skipped.append(
                {
                    "row_index": str(index),
                    "image_path": str(row.get(image_col, "")),
                    "resolved_image_path": resolved_path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        kept_indices.append(index)
    filtered = frame.loc[kept_indices].reset_index(drop=True)
    filtered.attrs["skipped_images"] = skipped
    return filtered


def _load_frame(
    csv_path: str,
    image_col: str,
    score_col: str,
    image_dir: str | None,
    label_threshold: float,
    label_rule: str,
    max_samples: int | None,
) -> pd.DataFrame:
    if label_rule not in LABEL_RULES:
        raise ValueError(f"Unsupported label_rule: {label_rule}")
    frame = pd.read_csv(csv_path)
    required = {image_col, score_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")
    frame = frame.dropna(subset=[image_col, score_col]).reset_index(drop=True)
    if max_samples is not None:
        frame = frame.head(int(max_samples)).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{csv_path} produced an empty frame")
    frame = frame.copy()
    frame["_resolved_image_path"] = _resolve_image_paths(frame, image_col, image_dir)
    frame["_label"] = _labels_from_scores(frame[score_col], label_threshold, label_rule)
    frame = _filter_decodable_images(frame, image_col=image_col)
    if frame.empty:
        raise ValueError(f"{csv_path} produced an empty frame after image validation")
    return frame


def _decode_image(path: tf.Tensor, training: bool) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image.set_shape([None, None, 3])
    if training:
        image = tf.image.random_flip_left_right(image)
    return image


def _layout_features(boxes: tf.Tensor, proposal_scores: tf.Tensor) -> tf.Tensor:
    boxes = tf.cast(boxes, tf.float32)
    proposal_scores = tf.cast(proposal_scores, tf.float32)
    y1, x1, y2, x2 = tf.unstack(boxes, axis=-1)
    height = tf.maximum(y2 - y1, 0.0)
    width = tf.maximum(x2 - x1, 0.0)
    center_y = y1 + height * 0.5
    center_x = x1 + width * 0.5
    area = height * width
    return tf.stack(
        [y1, x1, y2, x2, center_y, center_x, height, width, area, proposal_scores],
        axis=-1,
    )


def _make_dataset(
    frame: pd.DataFrame,
    variant: str,
    patch_size: int,
    global_size: int,
    patch_count: int,
    batch_size: int,
    training: bool,
    shuffle: bool,
) -> tf.data.Dataset:
    paths = frame["_resolved_image_path"].astype(str).to_numpy()
    labels = frame["_label"].astype("float32").to_numpy().reshape(-1, 1)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training and shuffle:
        dataset = dataset.shuffle(min(len(frame), 10000), reshuffle_each_iteration=True)

    def _map(path: tf.Tensor, label: tf.Tensor) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
        image = _decode_image(path, training=training)
        global_view, patches, boxes, proposal_scores = prepare_alamp_inputs(
            image=image,
            global_size=global_size,
            patch_size=patch_size,
            num_patches=patch_count,
        )
        inputs: dict[str, tf.Tensor] = {"patches": patches}
        if variant == "v0_b":
            inputs["global_view"] = global_view
            inputs["layout_features"] = _layout_features(boxes, proposal_scores)
        return inputs, label

    return dataset.map(_map, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _float_history(history: dict[str, list[float]]) -> dict[str, list[float]]:
    return {key: [float(value) for value in values] for key, values in history.items()}


def _classification_metrics() -> list[tf.keras.metrics.Metric]:
    metrics: list[tf.keras.metrics.Metric] = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
        tf.keras.metrics.Precision(name="precision", thresholds=0.5),
        tf.keras.metrics.Recall(name="recall", thresholds=0.5),
    ]
    if hasattr(tf.keras.metrics, "AUC"):
        metrics.append(tf.keras.metrics.AUC(name="auc", curve="ROC"))
    return metrics


def _compile_model(
    model: tf.keras.Model,
    learning_rate: float,
) -> dict[str, object]:
    compile_kwargs: dict[str, object] = {
        "optimizer": tf.keras.optimizers.Adam(learning_rate),
        "loss": tf.keras.losses.BinaryCrossentropy(),
        "metrics": _classification_metrics(),
        "jit_compile": False,
    }
    try:
        model.compile(**compile_kwargs)
        jit_compile_supported = True
    except TypeError:
        compile_kwargs.pop("jit_compile")
        model.compile(**compile_kwargs)
        jit_compile_supported = False
    return {
        "jit_compile_false_requested": True,
        "jit_compile_false_supported": jit_compile_supported,
        "model_jit_compile": bool(getattr(model, "jit_compile", False)),
    }


def _shape_summary(inputs: dict[str, tf.Tensor]) -> dict[str, list[int]]:
    return {name: [int(dim) for dim in tensor.shape] for name, tensor in inputs.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train A-LAMP-paper-AVA-v0 binary classification models.")
    parser.add_argument("--config")
    parser.add_argument("--train_csv")
    parser.add_argument("--val_csv")
    parser.add_argument("--out_dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_val_samples", type=int)
    parser.add_argument("--backbone_weights")
    parser.add_argument("--variant")
    parser.add_argument("--label_rule")
    parser.add_argument("--patience", type=int)
    parser.add_argument("--image_dir")
    parser.add_argument("--image_col")
    parser.add_argument("--score_col")
    parser.add_argument("--label_threshold", type=float)
    parser.add_argument("--patch_size", type=int)
    parser.add_argument("--global_size", type=int)
    parser.add_argument("--patch_count", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--feature_dim", type=int)
    parser.add_argument("--train_backbone", action="store_true", default=None)
    parser.add_argument("--freeze_backbone", dest="train_backbone", action="store_false")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no_verify_save_load", dest="verify_save_load", action="store_false", default=None)
    parser.add_argument("--verify_save_load", dest="verify_save_load", action="store_true")
    parser.add_argument("--allow_cpu_mid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _read_yaml(args.config)

    train_csv = _resolve(args.train_csv, config, "data", "train_csv", "data/processed/ava/train.csv")
    val_csv = _resolve(args.val_csv, config, "data", "val_csv", "data/processed/ava/val.csv")
    image_dir = _resolve(args.image_dir, config, "data", "image_dir", "data/raw/ava/images")
    image_col = _resolve(args.image_col, config, "data", "image_col", "image_path")
    score_col = _resolve(args.score_col, config, "data", "score_col", "mean_score")
    label_threshold = float(_resolve(args.label_threshold, config, "data", "label_threshold", 5.0))
    label_rule = str(_resolve(args.label_rule, config, "data", "label_rule", "paper_strict"))
    variant = str(_resolve(args.variant, config, "model", "variant", "v0_a")).lower()
    patch_size = int(_resolve(args.patch_size, config, "model", "patch_size", 224))
    global_size = int(_resolve(args.global_size, config, "model", "global_size", 384))
    patch_count = int(_resolve(args.patch_count, config, "model", "patch_count", 5))
    batch_size = int(_resolve(args.batch_size, config, "training", "batch_size", 4))
    epochs = int(_resolve(args.epochs, config, "training", "epochs", 20))
    learning_rate = float(_resolve(args.learning_rate, config, "training", "learning_rate", 1e-4))
    patience = int(_resolve(args.patience, config, "training", "early_stopping_patience", 3))
    backbone_weights = _normalize_weights(_resolve(args.backbone_weights, config, "model", "backbone_weights", "imagenet"))
    dropout = float(_resolve(args.dropout, config, "model", "dropout", 0.3))
    feature_dim = int(_resolve(args.feature_dim, config, "model", "feature_dim", 256))
    train_backbone = bool(_resolve(args.train_backbone, config, "model", "train_backbone", False))
    seed = int(_resolve(args.seed, config, "training", "seed", 42))
    verify_save_load = bool(_resolve(args.verify_save_load, config, "training", "verify_save_load", True))
    max_train_samples = _resolve(args.max_train_samples, config, "training", "max_train_samples", None)
    max_val_samples = _resolve(args.max_val_samples, config, "training", "max_val_samples", None)
    out_dir = Path(_resolve(args.out_dir, config, "experiment", "train_output_dir", DEFAULT_OUTPUT_DIR))

    out_dir.mkdir(parents=True, exist_ok=True)
    tf_info = _setup_tensorflow(seed)

    train_frame = _load_frame(str(train_csv), image_col, score_col, image_dir, label_threshold, label_rule, max_train_samples)
    val_frame = _load_frame(str(val_csv), image_col, score_col, image_dir, label_threshold, label_rule, max_val_samples)
    long_cpu_run = (len(train_frame) > 1000 or len(val_frame) > 1000) and epochs > 1
    if not tf_info["visible_gpus"] and long_cpu_run and not args.allow_cpu_mid:
        raise RuntimeError(
            "GPU is not visible for a mid/full AVA run. Re-run with GPU access or pass --allow_cpu_mid explicitly."
        )

    train_ds = _make_dataset(train_frame, variant, patch_size, global_size, patch_count, batch_size, training=True, shuffle=True)
    val_ds = _make_dataset(val_frame, variant, patch_size, global_size, patch_count, batch_size, training=False, shuffle=False)

    model = build_alamp_paper_ava_model(
        config,
        variant=variant,
        patch_count=patch_count,
        image_size_patch=patch_size,
        image_size_global=global_size,
        backbone_weights=backbone_weights,
        train_backbone=train_backbone,
        feature_dim=feature_dim,
        dropout=dropout,
    )
    compile_info = _compile_model(model, learning_rate)

    sample_inputs, _ = next(iter(train_ds.take(1)))
    forward_sample = model(sample_inputs, training=False)
    forward_np = forward_sample.numpy()
    forward_check = {
        "input_shapes": _shape_summary(sample_inputs),
        "output_shape": [int(dim) for dim in forward_sample.shape],
        "prediction_min": float(np.min(forward_np)),
        "prediction_max": float(np.max(forward_np)),
        "all_finite": bool(np.isfinite(forward_np).all()),
        "in_unit_range": bool((forward_np >= 0.0).all() and (forward_np <= 1.0).all()),
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
    trained_sample = model(sample_inputs, training=False)
    model.save(str(final_model_path))

    save_load_check: dict[str, object] | None = None
    if verify_save_load:
        loaded = tf.keras.models.load_model(
            str(final_model_path),
            compile=False,
            safe_mode=False,
            custom_objects=get_alamp_paper_ava_custom_objects(),
        )
        loaded_sample = loaded(sample_inputs, training=False)
        max_abs_diff = tf.reduce_max(tf.abs(tf.cast(trained_sample, tf.float32) - tf.cast(loaded_sample, tf.float32)))
        save_load_check = {
            "loaded_model_name": loaded.name,
            "loaded_output_shape": [int(dim) for dim in loaded_sample.shape],
            "max_abs_diff_vs_trained_forward_sample": float(max_abs_diff.numpy()),
        }

    val_losses = history.get("val_loss", [])
    best_epoch = int(np.argmin(val_losses) + 1) if val_losses else None
    best_val_loss = float(min(val_losses)) if val_losses else None
    train_losses = history.get("loss", [])
    architecture = getattr(model, "_alamp_paper_ava_config", {})

    summary = {
        "created_at_local": datetime.now().astimezone().isoformat(),
        "model_variant": MODEL_VARIANT,
        "model_variant_full": f"{MODEL_VARIANT}-{variant.replace('_', '-')}",
        "description": MODEL_DESCRIPTION,
        "style_description": STYLE_DESCRIPTION,
        "official_reproduction": False,
        "paper_comparability_note": (
            "A-LAMP-paper-oriented approximation using shared VGG16 patch columns, "
            "adaptive patch selection approximation, and a binary AVA classifier head. "
            "Exact object/global attribute graph is not implemented in v0. Exact "
            "saliency-map pipeline and official author weights are not available locally."
        ),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "image_dir": str(image_dir),
        "image_col": image_col,
        "score_col": score_col,
        "label_rule": label_rule,
        "label_rule_text": _label_rule_text(score_col, label_threshold, label_rule),
        "label_threshold": label_threshold,
        "train_samples": int(len(train_frame)),
        "val_samples": int(len(val_frame)),
        "train_skipped_image_count": int(len(train_frame.attrs.get("skipped_images", []))),
        "val_skipped_image_count": int(len(val_frame.attrs.get("skipped_images", []))),
        "train_skipped_image_examples": train_frame.attrs.get("skipped_images", [])[:20],
        "val_skipped_image_examples": val_frame.attrs.get("skipped_images", [])[:20],
        "train_positive_count": int(train_frame["_label"].sum()),
        "train_negative_count": int(len(train_frame) - train_frame["_label"].sum()),
        "val_positive_count": int(val_frame["_label"].sum()),
        "val_negative_count": int(len(val_frame) - val_frame["_label"].sum()),
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
            **architecture,
            "backbone_weights_requested": backbone_weights,
            "dropout": dropout,
            "feature_dim": feature_dim,
            "loss": "binary_crossentropy",
            "metrics": ["binary_accuracy", "auc", "precision", "recall"],
            "optimizer": "adam",
            "patch_selection": (
                "deterministic adaptive approximation using edge strength, luminance variance, "
                "color variance, and non-max overlap reduction from native-size images"
            ),
        },
        "tensorflow": tf_info,
        "compile": compile_info,
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
        "final_train_loss": float(train_losses[-1]) if train_losses else None,
        "history": history,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
