"""A-LAMP-paper-MPNet training script."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from PIL import Image

from src.models.alamp_paper_mpnet import build_alamp_paper_mpnet_model, get_alamp_paper_mpnet_custom_objects

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_OUTPUT_DIR = "outputs/alamp_paper_mpnet_selector_compare_20260512/train"

def _read_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}

def _cfg(config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return config.get(section, {}).get(key, default)

def _resolve(value: Any, config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return value if value is not None else _cfg(config, section, key, default)

def _resolve_bool(value: Any, config: dict[str, Any], section: str, key: str, default: bool) -> bool:
    resolved = _resolve(value, config, section, key, default)
    if isinstance(resolved, str):
        return resolved.strip().lower() in {"1", "true", "yes", "on"}
    return bool(resolved)

def _labels_from_scores(scores: np.ndarray, label_threshold: float = 5.0) -> np.ndarray:
    return (scores.astype("float32") > label_threshold).astype("float32")

def _compute_label_counts(
    jsonl_path: str,
    max_samples: int | None = None,
    label_threshold: float = 5.0,
) -> dict[str, int | float]:
    records = _load_jsonl_records(jsonl_path, max_samples)
    scores = np.array([record.get("mean_score", 0.0) for record in records], dtype=np.float32)
    labels = _labels_from_scores(scores, label_threshold=label_threshold)
    positive_count = int(np.sum(labels == 1.0))
    negative_count = int(np.sum(labels == 0.0))
    return {
        "total": int(labels.size),
        "positive": positive_count,
        "negative": negative_count,
        "label_threshold": float(label_threshold),
    }

def _build_class_weight(
    use_class_weight: bool,
    negative_setting: Any,
    positive_weight: float,
    label_counts: dict[str, int | float],
) -> tuple[dict[int, float] | None, dict[str, Any]]:
    if not use_class_weight:
        return None, {
            "enabled": False,
            "weights": None,
            "negative_setting": negative_setting,
        }

    positive_count = int(label_counts["positive"])
    negative_count = int(label_counts["negative"])
    auto_negative = False
    if negative_setting is None:
        auto_negative = True
    elif isinstance(negative_setting, str) and negative_setting.strip().lower() == "auto":
        auto_negative = True
    else:
        negative_weight = float(negative_setting)
        auto_negative = negative_weight <= 0.0

    if auto_negative:
        negative_weight = positive_count / max(negative_count, 1)

    class_weight = {0: float(negative_weight), 1: float(positive_weight)}
    return class_weight, {
        "enabled": True,
        "weights": class_weight,
        "negative_setting": negative_setting,
        "auto_negative": auto_negative,
    }

def _get_vgg16_backbone(model: tf.keras.Model) -> tf.keras.Model:
    try:
        return model.get_layer("vgg16_backbone")
    except ValueError as exc:
        raise ValueError("Expected MPNet model to contain a layer named 'vgg16_backbone'.") from exc

def _set_vgg16_trainable(model: tf.keras.Model, trainable: bool) -> None:
    backbone = _get_vgg16_backbone(model)
    backbone.trainable = trainable
    for layer in backbone.layers:
        layer.trainable = trainable

def _freeze_vgg16_backbone(model: tf.keras.Model) -> None:
    _set_vgg16_trainable(model, trainable=False)

def _unfreeze_vgg16_block5(model: tf.keras.Model) -> None:
    backbone = _get_vgg16_backbone(model)
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = layer.name.startswith("block5")

def _summarize_vgg16_trainability(model: tf.keras.Model) -> dict[str, Any]:
    backbone = _get_vgg16_backbone(model)
    trainable_layers = [layer.name for layer in backbone.layers if layer.trainable]
    frozen_layers = [layer.name for layer in backbone.layers if not layer.trainable]
    return {
        "backbone_name": backbone.name,
        "trainable_layer_count": len(trainable_layers),
        "frozen_layer_count": len(frozen_layers),
        "trainable_layers": trainable_layers,
        "frozen_layers": frozen_layers,
        "model_trainable_weight_count": len(model.trainable_weights),
        "model_non_trainable_weight_count": len(model.non_trainable_weights),
    }

def _monitor_mode(monitor_metric: str) -> str:
    return "min" if monitor_metric == "val_loss" else "max"

def _compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

class _BestWeightsSaver(tf.keras.callbacks.Callback):
    def __init__(self, out_dir: Path):
        super().__init__()
        self.paths = {
            "val_loss": out_dir / "best_val_loss.weights.h5",
            "val_auc": out_dir / "best_val_auc.weights.h5",
        }
        self.best_values = {"val_loss": np.inf, "val_auc": -np.inf}
        self.best_epochs: dict[str, int | None] = {"val_loss": None, "val_auc": None}
        self.global_epoch = 0

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        del epoch
        logs = logs or {}
        self.global_epoch += 1
        for metric, path in self.paths.items():
            if metric not in logs:
                continue
            value = float(logs[metric])
            improved = value < self.best_values[metric] if metric == "val_loss" else value > self.best_values[metric]
            if improved:
                self.best_values[metric] = value
                self.best_epochs[metric] = self.global_epoch
                self.model.save_weights(str(path))

def _history_to_rows(
    history: tf.keras.callbacks.History,
    phase_name: str,
    existing_rows: int,
) -> list[dict[str, Any]]:
    metric_names = list(history.history.keys())
    if not metric_names:
        return []
    epoch_count = len(history.history[metric_names[0]])
    rows = []
    for index in range(epoch_count):
        row: dict[str, Any] = {
            "epoch": existing_rows + index + 1,
            "phase": phase_name,
        }
        for metric_name in metric_names:
            row[metric_name] = float(history.history[metric_name][index])
        rows.append(row)
    return rows

def _make_phase_callbacks(
    best_saver: _BestWeightsSaver,
    monitor_metric: str,
    patience: int,
) -> list[tf.keras.callbacks.Callback]:
    callbacks: list[tf.keras.callbacks.Callback] = [best_saver]
    if patience >= 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor_metric,
                mode=_monitor_mode(monitor_metric),
                patience=patience,
                restore_best_weights=True,
            )
        )
    return callbacks

def _fit_phase(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    phase_name: str,
    epochs: int,
    learning_rate: float,
    class_weight: dict[int, float] | None,
    best_saver: _BestWeightsSaver,
    monitor_metric: str,
    patience: int,
    existing_rows: int,
) -> list[dict[str, Any]]:
    if epochs <= 0:
        logging.info(f"Skipping {phase_name}; epochs={epochs}")
        return []

    _compile_model(model, learning_rate)
    fit_kwargs: dict[str, Any] = {
        "x": train_ds,
        "validation_data": val_ds,
        "epochs": epochs,
        "callbacks": _make_phase_callbacks(best_saver, monitor_metric, patience),
        "verbose": 2,
    }
    if class_weight is not None:
        fit_kwargs["class_weight"] = class_weight

    logging.info(f"Starting {phase_name}: epochs={epochs}, lr={learning_rate}")
    history = model.fit(**fit_kwargs)
    return _history_to_rows(history, phase_name=phase_name, existing_rows=existing_rows)

def _best_from_rows(rows: list[dict[str, Any]], metric: str) -> tuple[float | None, int | None]:
    metric_rows = [row for row in rows if metric in row]
    if not metric_rows:
        return None, None
    if metric == "val_loss":
        best_row = min(metric_rows, key=lambda row: row[metric])
    else:
        best_row = max(metric_rows, key=lambda row: row[metric])
    return float(best_row[metric]), int(best_row["epoch"])

def _setup_tensorflow(seed: int = 42, require_gpu: bool = True) -> dict[str, object]:
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.utils.set_random_seed(seed)
    gpus = tf.config.list_physical_devices("GPU")
    
    if not gpus and require_gpu:
        logging.error("No GPUs detected! Aborting as --require_gpu is True.")
        import sys
        sys.exit(1)
        
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    return {
        "visible_gpus": [str(gpu) for gpu in gpus],
        "tensorflow_version": tf.__version__,
    }

def _load_jsonl_records(jsonl_path: str, max_samples: int | None = None) -> list[dict[str, Any]]:
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            if max_samples and len(records) >= max_samples:
                break
    return records

def _process_image_crops(record: dict[str, Any], patch_size: int = 224, patch_count: int = 5) -> np.ndarray | None:
    img_path = record.get("resolved_image_path") or record.get("image_path")
    if not img_path or not Path(img_path).exists():
        return None
        
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            
            boxes = record.get("boxes_abs_xyxy")
            if not boxes:
                # Fallback to normalized
                norm_boxes = record.get("boxes_norm_xyxy", [])
                boxes = [[b[0]*w, b[1]*h, b[2]*w, b[3]*h] for b in norm_boxes]
            
            if len(boxes) < patch_count:
                return None
                
            crops = []
            for i in range(patch_count):
                b = boxes[i]
                # PIL crop expects (left, top, right, bottom)
                crop = img.crop((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
                crop = crop.resize((patch_size, patch_size), Image.BILINEAR)
                crops.append(np.array(crop, dtype=np.float32) / 255.0)
                
            return np.array(crops, dtype=np.float32)
    except Exception as e:
        logging.warning(f"Error processing {img_path}: {e}")
        return None

def _make_dataset(
    jsonl_path: str,
    patch_size: int = 224,
    patch_count: int = 5,
    batch_size: int = 4,
    max_samples: int | None = None,
    training: bool = False,
) -> tf.data.Dataset:
    records = _load_jsonl_records(jsonl_path, max_samples)
    
    def generator():
        skipped = 0
        for r in records:
            crops = _process_image_crops(r, patch_size, patch_count)
            if crops is None:
                skipped += 1
                continue
            
            label = _labels_from_scores(np.array([r.get("mean_score", 0.0)]))
            yield crops, label
        if skipped > 0:
            logging.info(f"Skipped {skipped} records in {jsonl_path}")

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(patch_count, patch_size, patch_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
        )
    )
    
    if training:
        ds = ds.shuffle(min(len(records), 100), reshuffle_each_iteration=True)
    
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--train_jsonl")
    parser.add_argument("--val_jsonl")
    parser.add_argument("--out_dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--backbone_weights")
    parser.add_argument("--label_rule", default="paper_strict")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_val_samples", type=int)
    parser.add_argument("--selector_name")
    parser.add_argument("--training_mode", choices=["single_phase", "two_phase_freeze_block5"])
    parser.add_argument("--use_class_weight", action="store_true", default=None)
    parser.add_argument("--class_weight_negative")
    parser.add_argument("--class_weight_positive", type=float)
    parser.add_argument("--phase1_epochs", type=int)
    parser.add_argument("--phase2_epochs", type=int)
    parser.add_argument("--phase1_learning_rate", type=float)
    parser.add_argument("--phase2_learning_rate", type=float)
    parser.add_argument("--monitor_metric", choices=["val_loss", "val_auc"])
    parser.add_argument("--require_gpu", action="store_true", default=True)
    parser.add_argument("--no_gpu", action="store_false", dest="require_gpu")
    args = parser.parse_args()
    
    config = _read_yaml(args.config)
    
    train_jsonl = _resolve(args.train_jsonl, config, "dataset", "train_jsonl", "")
    val_jsonl = _resolve(args.val_jsonl, config, "dataset", "val_jsonl", "")
    out_dir = Path(_resolve(args.out_dir, config, "training", "output_dir", DEFAULT_OUTPUT_DIR))
    if args.selector_name:
        out_dir = out_dir / args.selector_name
        
    epochs = int(_resolve(args.epochs, config, "training", "epochs", 10))
    batch_size = int(_resolve(args.batch_size, config, "training", "batch_size", 4))
    lr = float(_resolve(args.learning_rate, config, "training", "learning_rate", 1e-4))
    patience = int(_resolve(args.patience, config, "training", "early_stopping_patience", 3))
    training_mode = _resolve(args.training_mode, config, "training", "training_mode", "single_phase")
    monitor_metric = _resolve(args.monitor_metric, config, "training", "monitor_metric", "val_auc")
    phase1_epochs = int(_resolve(args.phase1_epochs, config, "training", "phase1_epochs", epochs))
    phase2_epochs = int(_resolve(args.phase2_epochs, config, "training", "phase2_epochs", epochs))
    phase1_lr = float(_resolve(args.phase1_learning_rate, config, "training", "phase1_learning_rate", lr))
    phase2_lr = float(_resolve(args.phase2_learning_rate, config, "training", "phase2_learning_rate", lr * 0.1))
    use_class_weight = _resolve_bool(args.use_class_weight, config, "training", "use_class_weight", False)
    class_weight_negative = _resolve(args.class_weight_negative, config, "training", "class_weight_negative", "auto")
    class_weight_positive = float(_resolve(args.class_weight_positive, config, "training", "class_weight_positive", 1.0))
    
    max_train = _resolve(args.max_train_samples, config, "training", "max_train_samples", None)
    max_val = _resolve(args.max_val_samples, config, "training", "max_val_samples", None)
    max_train = int(max_train) if max_train is not None else None
    max_val = int(max_val) if max_val is not None else None
    
    backbone_weights = _resolve(args.backbone_weights, config, "model", "backbone_weights", "imagenet")
    if str(backbone_weights).lower() == "none":
        backbone_weights = None

    out_dir.mkdir(parents=True, exist_ok=True)
    tf_info = _setup_tensorflow(require_gpu=args.require_gpu)
    
    logging.info(f"Loading datasets for selector: {args.selector_name}")
    train_label_counts = _compute_label_counts(train_jsonl, max_samples=max_train)
    class_weight, class_weight_summary = _build_class_weight(
        use_class_weight=use_class_weight,
        negative_setting=class_weight_negative,
        positive_weight=class_weight_positive,
        label_counts=train_label_counts,
    )
    if class_weight is not None:
        logging.info(f"Using class_weight={class_weight}")

    train_ds = _make_dataset(train_jsonl, batch_size=batch_size, max_samples=max_train, training=True)
    val_ds = _make_dataset(val_jsonl, batch_size=batch_size, max_samples=max_val, training=False)
    
    model = build_alamp_paper_mpnet_model(backbone_weights=backbone_weights)
    best_saver = _BestWeightsSaver(out_dir)
    layer_summary: dict[str, Any] = {}
    history_rows: list[dict[str, Any]] = []

    if training_mode == "single_phase":
        _set_vgg16_trainable(model, trainable=True)
        layer_summary["single_phase"] = _summarize_vgg16_trainability(model)
        history_rows.extend(
            _fit_phase(
                model=model,
                train_ds=train_ds,
                val_ds=val_ds,
                phase_name="single_phase",
                epochs=epochs,
                learning_rate=lr,
                class_weight=class_weight,
                best_saver=best_saver,
                monitor_metric=monitor_metric,
                patience=patience,
                existing_rows=len(history_rows),
            )
        )
    elif training_mode == "two_phase_freeze_block5":
        _freeze_vgg16_backbone(model)
        layer_summary["phase1_frozen_backbone"] = _summarize_vgg16_trainability(model)
        history_rows.extend(
            _fit_phase(
                model=model,
                train_ds=train_ds,
                val_ds=val_ds,
                phase_name="phase1_frozen_backbone",
                epochs=phase1_epochs,
                learning_rate=phase1_lr,
                class_weight=class_weight,
                best_saver=best_saver,
                monitor_metric=monitor_metric,
                patience=patience,
                existing_rows=len(history_rows),
            )
        )

        _unfreeze_vgg16_block5(model)
        layer_summary["phase2_block5_unfrozen"] = _summarize_vgg16_trainability(model)
        history_rows.extend(
            _fit_phase(
                model=model,
                train_ds=train_ds,
                val_ds=val_ds,
                phase_name="phase2_block5_unfrozen",
                epochs=phase2_epochs,
                learning_rate=phase2_lr,
                class_weight=class_weight,
                best_saver=best_saver,
                monitor_metric=monitor_metric,
                patience=patience,
                existing_rows=len(history_rows),
            )
        )
    else:
        raise ValueError(f"Unsupported training_mode: {training_mode}")

    if not history_rows:
        raise RuntimeError("Training produced no history rows.")

    pd.DataFrame(history_rows).to_csv(out_dir / "training_history.csv", index=False)

    selected_best_path = out_dir / f"best_{monitor_metric}.weights.h5"
    best_alias_path = out_dir / "best.weights.h5"
    if not selected_best_path.exists():
        raise RuntimeError(f"Expected monitor-selected weights at {selected_best_path}")
    shutil.copyfile(selected_best_path, best_alias_path)
    model.load_weights(str(best_alias_path))
    
    final_model_path = out_dir / "final_model.keras"
    model.save(str(final_model_path))
    
    # Save/load diff verification
    sample_inputs, _ = next(iter(val_ds.take(1)))
    orig_preds = model(sample_inputs, training=False).numpy()
    loaded = tf.keras.models.load_model(str(final_model_path), compile=False, custom_objects=get_alamp_paper_mpnet_custom_objects())
    load_preds = loaded(sample_inputs, training=False).numpy()
    max_diff = float(np.max(np.abs(orig_preds - load_preds)))

    best_val_auc, best_val_auc_epoch = _best_from_rows(history_rows, "val_auc")
    best_val_loss, best_val_loss_epoch = _best_from_rows(history_rows, "val_loss")
    final_row = history_rows[-1]
    
    summary = {
        "selector_name": args.selector_name,
        "model_variant": "A-LAMP-paper-MPNet",
        "official_reproduction": False,
        "label_rule": args.label_rule,
        "training_mode": training_mode,
        "epochs_completed": len(history_rows),
        "batch_size": batch_size,
        "learning_rate": lr,
        "phase1_settings": {
            "epochs": phase1_epochs,
            "learning_rate": phase1_lr,
            "freeze_vgg16_backbone": training_mode == "two_phase_freeze_block5",
        },
        "phase2_settings": {
            "epochs": phase2_epochs,
            "learning_rate": phase2_lr,
            "unfreeze_vgg16_block5_only": training_mode == "two_phase_freeze_block5",
        },
        "class_weight": class_weight_summary,
        "train_label_counts": train_label_counts,
        "frozen_unfrozen_layer_summary": layer_summary,
        "best_val_auc": best_val_auc,
        "best_val_auc_epoch": best_val_auc_epoch,
        "best_val_loss": best_val_loss,
        "best_val_loss_epoch": best_val_loss_epoch,
        "monitor_metric": monitor_metric,
        "monitor_mode": _monitor_mode(monitor_metric),
        "best_weights_alias_source": str(selected_best_path),
        "final_train_loss": float(final_row["loss"]),
        "final_val_loss": float(final_row["val_loss"]),
        "visible_gpus": tf_info["visible_gpus"],
        "save_load_max_abs_diff": max_diff,
        "save_load_max_diff": max_diff,
        "tensorflow_info": tf_info
    }
    with open(out_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Training complete. Saved to {out_dir}")

if __name__ == "__main__":
    main()
