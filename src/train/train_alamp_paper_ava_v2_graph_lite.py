"""A-LAMP-paper-AVA-v2_graph_lite training script."""
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
from src.models.alamp_paper_ava_v2_graph_lite import (
    MODEL_DESCRIPTION,
    MODEL_VARIANT,
    STYLE_DESCRIPTION,
    build_alamp_paper_ava_v2_model,
    GraphLiteBranch,
)
from src.models.alamp_paper_ava import get_alamp_paper_ava_custom_objects

def get_v2_custom_objects() -> dict[str, Any]:
    objs = get_alamp_paper_ava_custom_objects()
    objs["GraphLiteBranch"] = GraphLiteBranch
    return objs

DEFAULT_OUTPUT_DIR = "outputs/alamp_paper_ava_v2_graph_lite_20260511/mid_train"
LABEL_RULES = {"paper_strict", "project_compatible"}

def _read_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
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
        except Exception:
            pass
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

def _image_match_keys(path_value: Any) -> list[str]:
    raw = str(path_value or "").strip()
    if not raw:
        return []
    path = Path(raw)
    keys = {raw, path.as_posix(), path.name, path.stem}
    parts = [part for part in path.parts if part not in ("", "/")]
    if len(parts) >= 2:
        keys.add("/".join(parts[-2:]))
    if len(parts) >= 3:
        keys.add("/".join(parts[-3:]))
    try:
        keys.add(str(path.expanduser().resolve()))
    except Exception:
        pass
    return [key for key in keys if key]

def _graph_matches_image(graph: dict[str, list], image_paths: list[str]) -> bool:
    graph_keys = set(_image_match_keys(graph.get("image_path")))
    return any(graph_keys.intersection(_image_match_keys(path)) for path in image_paths)

def _load_graphs(jsonl_path: str) -> dict[str, dict[str, list]]:
    graphs = {}
    ambiguous_keys = set()
    if not jsonl_path or not Path(jsonl_path).exists():
        return graphs
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            for key in _image_match_keys(data.get("image_path")):
                if key in ambiguous_keys:
                    continue
                existing = graphs.get(key)
                if existing is not None and existing.get("image_path") != data.get("image_path"):
                    graphs.pop(key, None)
                    ambiguous_keys.add(key)
                    continue
                graphs[key] = data
    return graphs

def _find_graph_for_row(row: pd.Series, graphs: dict[str, dict[str, list]], image_col: str) -> dict[str, list] | None:
    image_paths = [str(row.get("_resolved_image_path", "")), str(row.get(image_col, ""))]
    for image_path in image_paths:
        for key in _image_match_keys(image_path):
            graph = graphs.get(key)
            if graph is not None and _graph_matches_image(graph, image_paths):
                return graph
    return None

def _prepare_graph_arrays(frame: pd.DataFrame, graphs: dict[str, dict[str, list]], max_objects: int, image_col: str):
    n = len(frame)
    obj_masks = np.zeros((n, max_objects), dtype=np.float32)
    boxes = np.zeros((n, max_objects, 4), dtype=np.float32)
    centers = np.zeros((n, max_objects, 2), dtype=np.float32)
    areas = np.zeros((n, max_objects), dtype=np.float32)
    class_ids = np.zeros((n, max_objects), dtype=np.int32)
    confidences = np.zeros((n, max_objects), dtype=np.float32)
    local_edges = np.zeros((n, max_objects, max_objects, 3), dtype=np.float32)
    global_edges = np.zeros((n, max_objects, 3), dtype=np.float32)
    
    missing_idx = []
    graph_match_summary = {
        "matched_count": 0,
        "missing_count": 0,
        "skipped_count": 0,
        "total_count": int(n),
    }
    for i, row in frame.iterrows():
        graph = _find_graph_for_row(row, graphs, image_col)
        if not graph:
            missing_idx.append(i)
            graph_match_summary["missing_count"] += 1
            continue
        graph_match_summary["matched_count"] += 1
        
        n_obj = min(len(graph.get("object_mask", [])), max_objects)
        if n_obj > 0:
            obj_masks[i, :n_obj] = graph["object_mask"][:n_obj]
            boxes[i, :n_obj] = graph["boxes_norm_xyxy"][:n_obj]
            centers[i, :n_obj] = graph["centers_norm_xy"][:n_obj]
            areas[i, :n_obj] = graph["area_ratio"][:n_obj]
            class_ids[i, :n_obj] = graph["class_ids"][:n_obj]
            confidences[i, :n_obj] = graph["confidences"][:n_obj]
            
            le = np.array(graph["local_edges"])
            le_n = min(le.shape[0], max_objects)
            if le_n > 0:
                local_edges[i, :le_n, :le_n] = le[:le_n, :le_n]
                
            ge = np.array(graph["global_edges"])
            ge_n = min(ge.shape[0], max_objects)
            if ge_n > 0:
                global_edges[i, :ge_n] = ge[:ge_n]
                
    return (obj_masks, boxes, centers, areas, class_ids, confidences, local_edges, global_edges), missing_idx, graph_match_summary

def _load_frame(
    csv_path: str,
    graph_jsonl: str,
    image_col: str,
    score_col: str,
    image_dir: str | None,
    label_threshold: float,
    label_rule: str,
    max_samples: int | None,
    max_objects: int,
) -> tuple[pd.DataFrame, tuple]:
    frame = pd.read_csv(csv_path)
    if score_col in frame.columns:
        frame = frame.dropna(subset=[image_col, score_col]).reset_index(drop=True)
    else:
        frame = frame.dropna(subset=[image_col]).reset_index(drop=True)
    if max_samples is not None:
        frame = frame.head(int(max_samples)).reset_index(drop=True)
    
    frame = frame.copy()
    frame["_resolved_image_path"] = _resolve_image_paths(frame, image_col, image_dir)
    
    graphs = _load_graphs(graph_jsonl)
    arrays, missing_graph_idx, graph_match_summary = _prepare_graph_arrays(frame, graphs, max_objects, image_col)
    
    if score_col not in frame.columns:
        # Fallback to jsonl mean_score if available
        scores = []
        for _, row in frame.iterrows():
            g = _find_graph_for_row(row, graphs, image_col)
            scores.append(g.get("mean_score", 0.0) if g else 0.0)
        frame[score_col] = scores

    frame["_label"] = _labels_from_scores(frame[score_col], label_threshold, label_rule)
    
    kept_indices = []
    skipped = []
    for i, row in frame.iterrows():
        if i in missing_graph_idx:
            skipped.append({"row_index": str(i), "error": "missing graph"})
            continue
        try:
            _verify_decodable_image(str(row["_resolved_image_path"]))
            kept_indices.append(i)
        except Exception as exc:
            skipped.append({"row_index": str(i), "error": str(exc)})
            
    filtered = frame.loc[kept_indices].reset_index(drop=True)
    graph_match_summary["skipped_count"] = len(skipped)
    filtered.attrs["skipped_images"] = skipped
    filtered.attrs["graph_match_summary"] = graph_match_summary
    if graph_match_summary["missing_count"]:
        print(
            "WARNING: graph matching skipped "
            f"{graph_match_summary['missing_count']} of {graph_match_summary['total_count']} rows; "
            "row-index fallback is disabled."
        )
    filtered_arrays = tuple(a[kept_indices] for a in arrays)
    return filtered, filtered_arrays

def _decode_image(path: tf.Tensor, training: bool) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image.set_shape([None, None, 3])
    if training:
        image = tf.image.random_flip_left_right(image)
    return image

def _make_dataset(
    frame: pd.DataFrame,
    arrays: tuple,
    patch_size: int,
    global_size: int,
    patch_count: int,
    batch_size: int,
    training: bool,
    shuffle: bool,
) -> tf.data.Dataset:
    paths = frame["_resolved_image_path"].astype(str).to_numpy()
    labels = frame["_label"].astype("float32").to_numpy().reshape(-1, 1)
    
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels, *arrays))
    if training and shuffle:
        dataset = dataset.shuffle(min(len(frame), 10000), reshuffle_each_iteration=True)

    def _map(path, label, obj_mask, boxes, centers, area, class_id, conf, local_edge, global_edge):
        image = _decode_image(path, training=training)
        global_view, patches, _, _ = prepare_alamp_inputs(
            image=image,
            global_size=global_size,
            patch_size=patch_size,
            num_patches=patch_count,
        )
        inputs = {
            "patches": patches,
            "global_view": global_view,
            "object_mask": obj_mask,
            "boxes_norm_xyxy": boxes,
            "centers_norm_xy": centers,
            "area_ratio": area,
            "class_ids": class_id,
            "confidences": conf,
            "local_edges": local_edge,
            "global_edges": global_edge,
        }
        return inputs, label

    return dataset.map(_map, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def _classification_metrics() -> list[tf.keras.metrics.Metric]:
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
        tf.keras.metrics.Precision(name="precision", thresholds=0.5),
        tf.keras.metrics.Recall(name="recall", thresholds=0.5),
        tf.keras.metrics.AUC(name="auc", curve="ROC"),
    ]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--train_csv")
    parser.add_argument("--val_csv")
    parser.add_argument("--train_graph_jsonl")
    parser.add_argument("--val_graph_jsonl")
    parser.add_argument("--out_dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_val_samples", type=int)
    parser.add_argument("--backbone_weights")
    parser.add_argument("--label_rule")
    parser.add_argument("--patience", type=int)
    parser.add_argument("--allow_cpu_mid", action="store_true")
    args = parser.parse_args()
    config = _read_yaml(args.config)

    train_csv = _resolve(args.train_csv, config, "dataset", "train_csv", "")
    val_csv = _resolve(args.val_csv, config, "dataset", "val_csv", "")
    train_graph = _resolve(args.train_graph_jsonl, config, "dataset", "train_graph_jsonl", "")
    val_graph = _resolve(args.val_graph_jsonl, config, "dataset", "val_graph_jsonl", "")
    image_dir = _resolve(None, config, "dataset", "image_dir", "data/raw/ava/images")
    image_col = _resolve(None, config, "dataset", "image_col", "image_path")
    score_col = _resolve(None, config, "dataset", "score_col", "mean_score")
    label_threshold = 5.0
    label_rule = str(_resolve(args.label_rule, config, "dataset", "label_rule", "paper_strict"))
    
    patch_size = int(_resolve(None, config, "model", "patch_size", 224))
    global_size = int(_resolve(None, config, "model", "global_size", 384))
    patch_count = int(_resolve(None, config, "model", "patch_count", 5))
    max_objects = int(_resolve(None, config, "model", "max_objects", 4))
    
    batch_size = int(_resolve(args.batch_size, config, "training", "batch_size", 4))
    epochs = int(_resolve(args.epochs, config, "training", "epochs", 10))
    learning_rate = float(_resolve(None, config, "training", "learning_rate", 1e-4))
    patience = int(_resolve(args.patience, config, "training", "early_stopping_patience", 3))
    backbone_weights = _normalize_weights(_resolve(args.backbone_weights, config, "model", "backbone_weights", "imagenet"))
    
    max_train_samples = _resolve(args.max_train_samples, config, "training", "max_train_samples", None)
    max_val_samples = _resolve(args.max_val_samples, config, "training", "max_val_samples", None)
    out_dir = Path(_resolve(args.out_dir, config, "experiment", "output_dir", DEFAULT_OUTPUT_DIR))

    out_dir.mkdir(parents=True, exist_ok=True)
    tf_info = _setup_tensorflow(42)

    train_frame, train_arrays = _load_frame(train_csv, train_graph, image_col, score_col, image_dir, label_threshold, label_rule, max_train_samples, max_objects)
    val_frame, val_arrays = _load_frame(val_csv, val_graph, image_col, score_col, image_dir, label_threshold, label_rule, max_val_samples, max_objects)

    long_cpu_run = (len(train_frame) > 1000 or len(val_frame) > 1000) and epochs > 1
    if not tf_info["visible_gpus"] and long_cpu_run and not getattr(args, "allow_cpu_mid", False):
        raise RuntimeError("GPU is not visible for a mid/full AVA run. Re-run with GPU access or pass --allow_cpu_mid explicitly.")

    train_ds = _make_dataset(train_frame, train_arrays, patch_size, global_size, patch_count, batch_size, True, True)
    val_ds = _make_dataset(val_frame, val_arrays, patch_size, global_size, patch_count, batch_size, False, False)

    model = build_alamp_paper_ava_v2_model(
        config,
        patch_count=patch_count,
        image_size_patch=patch_size,
        image_size_global=global_size,
        max_objects=max_objects,
        backbone_weights=backbone_weights,
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=_classification_metrics(),
        jit_compile=False,
    )

    sample_inputs, _ = next(iter(train_ds.take(1)))
    forward_sample = model(sample_inputs, training=False)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(out_dir / "best.weights.h5"),
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(out_dir / "training_history.csv")),
    ]

    history_obj = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2)
    history = {k: [float(v) for v in vs] for k, vs in history_obj.history.items()}

    final_model_path = out_dir / "final_model.keras"
    trained_sample = model(sample_inputs, training=False)
    model.save(str(final_model_path))

    loaded = tf.keras.models.load_model(str(final_model_path), compile=False, custom_objects=get_v2_custom_objects())
    loaded_sample = loaded(sample_inputs, training=False)
    max_abs_diff = float(tf.reduce_max(tf.abs(trained_sample - loaded_sample)))

    # Compute no_object counts
    train_no_obj = int(np.sum(np.sum(train_arrays[0], axis=1) == 0))
    val_no_obj = int(np.sum(np.sum(val_arrays[0], axis=1) == 0))

    summary = {
        "model_variant": MODEL_VARIANT,
        "official_reproduction": False,
        "paper_comparability_note": "A-LAMP-paper-oriented approximation with graph-lite branch.",
        "graph_artifacts": {
            "train": train_graph,
            "val": val_graph
        },
        "label_rule": label_rule,
        "train_samples": len(train_frame),
        "val_samples": len(val_frame),
        "skipped_images": {
            "train": len(train_frame.attrs.get("skipped_images", [])),
            "val": len(val_frame.attrs.get("skipped_images", []))
        },
        "graph_match_summary": {
            "train": train_frame.attrs.get("graph_match_summary", {}),
            "val": val_frame.attrs.get("graph_match_summary", {})
        },
        "no_object_counts": {
            "train": train_no_obj,
            "val": val_no_obj
        },
        "architecture": getattr(model, "_alamp_paper_ava_config", {}),
        "epochs_completed": len(history.get("loss", [])),
        "best_epoch": int(np.argmin(history.get("val_loss", [])) + 1) if history.get("val_loss") else None,
        "best_val_loss": min(history.get("val_loss", [])) if history.get("val_loss") else None,
        "save_load_max_abs_diff": max_abs_diff,
        "tensorflow_gpus": tf_info["visible_gpus"]
    }
    
    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
