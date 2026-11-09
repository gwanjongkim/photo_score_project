# A-LAMP 논문 지향 MPNet-레이아웃 그래프 융합 실험을 실행한다.
"""Train or smoke-test research-only MPNet + layout graph fusion.

This is a paper-oriented approximation branch for ablation experiments. It does
not modify or replace the practical app-facing A-LAMP/RGNet models.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.models.alamp_paper_fusion import (
    build_mpnet_graphgcn_fusion_model,
    build_mpnet_graphlite_fusion_model,
)
from src.models.alamp_paper_mpnet import get_alamp_paper_mpnet_custom_objects
from src.train.train_alamp_paper_mpnet_ava import (
    _load_jsonl_records,
    _process_image_crops,
)


DEFAULT_MPNET_MODEL = (
    "outputs/alamp_paper_mpnet_selector_compare_20260512/train/"
    "v4_1024_gpu_tuned/v4_1024_gpu_tuned/best_val_auc_model.keras"
)
DEFAULT_TRAIN_PATCH_JSONL = (
    "outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/"
    "train_patch_boxes_1024_v4.jsonl"
)
DEFAULT_VAL_PATCH_JSONL = (
    "outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/"
    "val_patch_boxes_1024_v4.jsonl"
)
DEFAULT_TRAIN_GRAPH_JSONL = (
    "outputs/alamp_object_graph_subset_20260511/graphs_conf010/"
    "train_graphs_1024.jsonl"
)
DEFAULT_VAL_GRAPH_JSONL = (
    "outputs/alamp_object_graph_subset_20260511/graphs_conf010/"
    "val_graphs_1024.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    f"outputs/alamp_paper_fusion_smoke_{datetime.now().strftime('%Y%m%d')}"
)


@dataclass
class PreparedSplit:
    split: str
    patch_jsonl: Path
    patch_records: list[dict[str, Any]]
    image_paths: list[str]
    labels: np.ndarray
    graph_vectors: np.ndarray
    node_features: np.ndarray
    adjacency: np.ndarray
    object_mask: np.ndarray
    graph_match_summary: dict[str, int]
    crops_shape: list[int] | None = None
    mpnet_features: np.ndarray | None = None
    feature_cache: dict[str, Any] | None = None
    skipped_missing_graph: int = 0
    skipped_bad_crop: int = 0

    def shape_report(self) -> dict[str, Any]:
        return {
            "records": int(self.labels.shape[0]),
            "patch_records": len(self.patch_records),
            "crops": self.crops_shape,
            "labels": list(self.labels.shape),
            "graph_vectors": list(self.graph_vectors.shape),
            "node_features": list(self.node_features.shape),
            "adjacency": list(self.adjacency.shape),
            "object_mask": list(self.object_mask.shape),
            "mpnet_features": (
                list(self.mpnet_features.shape)
                if self.mpnet_features is not None
                else None
            ),
            "skipped_missing_graph": int(self.skipped_missing_graph),
            "skipped_bad_crop": int(self.skipped_bad_crop),
            "graph_match_summary": self.graph_match_summary,
            "feature_cache": self.feature_cache,
        }


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")


def _mpnet_checkpoint_policy_reason(path: Path) -> str | None:
    lowered = str(path).lower()
    name = path.name.lower()
    if "best_val" in lowered or "best-val" in lowered:
        return "path contains best_val/best-val"
    if name == "best.weights.h5" and "mpnet" in lowered:
        return "path is the MPNet monitor-selected best.weights.h5 alias"
    if name == "final_model.keras" and "mpnet" in lowered and any(
        marker in lowered
        for marker in ("train", "tuned", "selector_compare", "paper_mpnet")
    ):
        return "path is final_model.keras under an MPNet training/tuned output"
    return None


def _is_val_selected_mpnet_checkpoint(path: Path) -> bool:
    return _mpnet_checkpoint_policy_reason(path) is not None


def _enforce_mpnet_checkpoint_policy(args: argparse.Namespace) -> None:
    policy_reason = _mpnet_checkpoint_policy_reason(args.mpnet_model_path)
    if policy_reason is None:
        return
    message = (
        f"MPNet checkpoint path appears validation-selected or derived from "
        f"validation-selected weights ({policy_reason}). best_val/best-monitor "
        "checkpoints and derived final_model.keras artifacts can bias fusion "
        "validation metrics when the same validation split is reused. Use a "
        "train-only non-selected checkpoint, evaluate on a separate held-out "
        "split, or pass --allow_val_selected_mpnet for development-only runs."
    )
    if args.smoke:
        print(f"WARNING: {message} Smoke metrics are not final.")
        return
    if not args.allow_val_selected_mpnet:
        raise SystemExit(f"ERROR: {message}")
    print(f"WARNING: {message} Continuing because --allow_val_selected_mpnet was set.")


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


def _graph_keys(record: dict[str, Any]) -> list[str]:
    return _image_match_keys(record.get("image_path"))


def _patch_record_keys(record: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for field in ("resolved_image_path", "image_path"):
        value = str(record.get(field) or "")
        keys.extend(_image_match_keys(value))
    return keys


def _load_graph_index(graph_jsonl: Path) -> dict[str, dict[str, Any]]:
    graph_index: dict[str, dict[str, Any]] = {}
    ambiguous_keys: set[str] = set()
    with graph_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            for key in _graph_keys(record):
                if key in ambiguous_keys:
                    continue
                existing = graph_index.get(key)
                if existing is not None and existing.get("image_path") != record.get("image_path"):
                    graph_index.pop(key, None)
                    ambiguous_keys.add(key)
                    continue
                graph_index[key] = record
    return graph_index


def _graph_matches_patch_record(
    graph_record: dict[str, Any], patch_record: dict[str, Any]
) -> bool:
    graph_keys = set(_graph_keys(graph_record))
    patch_keys = set(_patch_record_keys(patch_record))
    return bool(graph_keys.intersection(patch_keys))


def _find_graph_record(
    patch_record: dict[str, Any], graph_index: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    for key in _patch_record_keys(patch_record):
        graph_record = graph_index.get(key)
        if graph_record is not None and _graph_matches_patch_record(
            graph_record, patch_record
        ):
            return graph_record
    return None


def _array_field(
    record: dict[str, Any],
    key: str,
    shape: tuple[int, ...],
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    value = record.get(key)
    if value is None:
        raise ValueError(f"Graph record is missing required field '{key}'.")
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape:
        raise ValueError(
            f"Graph field '{key}' has shape {array.shape}; expected {shape}."
        )
    return array


def _graph_arrays(
    graph_record: dict[str, Any],
    max_objects: int,
    class_id_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    object_mask = _array_field(
        graph_record, "object_mask", (max_objects,), np.float32
    )
    boxes = _array_field(
        graph_record, "boxes_norm_xyxy", (max_objects, 4), np.float32
    )
    centers = _array_field(
        graph_record, "centers_norm_xy", (max_objects, 2), np.float32
    )
    area = _array_field(graph_record, "area_ratio", (max_objects,), np.float32)
    class_ids = _array_field(graph_record, "class_ids", (max_objects,), np.float32)
    confidences = _array_field(
        graph_record, "confidences", (max_objects,), np.float32
    )
    local_edges = _array_field(
        graph_record, "local_edges", (max_objects, max_objects, 3), np.float32
    )
    global_edges = _array_field(
        graph_record, "global_edges", (max_objects, 3), np.float32
    )

    class_norm = np.clip(class_ids / max(class_id_scale, 1.0), 0.0, 1.0)
    graph_vector = np.concatenate(
        [
            object_mask.reshape(-1),
            boxes.reshape(-1),
            centers.reshape(-1),
            area.reshape(-1),
            class_norm.reshape(-1),
            confidences.reshape(-1),
            local_edges.reshape(-1),
            global_edges.reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)

    neighbor_mask = object_mask[:, None] * object_mask[None, :]
    local_edge_sum = np.sum(local_edges * neighbor_mask[:, :, None], axis=1)
    node_features = np.concatenate(
        [
            boxes,
            centers,
            area[:, None],
            class_norm[:, None],
            confidences[:, None],
            global_edges,
            local_edge_sum,
        ],
        axis=1,
    ).astype(np.float32)
    node_features *= object_mask[:, None]

    distance = local_edges[:, :, 0]
    adjacency = (1.0 / (1.0 + np.maximum(distance, 0.0))).astype(np.float32)
    adjacency *= neighbor_mask.astype(np.float32)
    np.fill_diagonal(adjacency, 0.0)

    return graph_vector, node_features, adjacency, object_mask


def _prepare_split(
    split: str,
    patch_jsonl: Path,
    graph_jsonl: Path,
    max_samples: int | None,
    patch_size: int,
    patch_count: int,
    label_threshold: float,
    max_objects: int,
    class_id_scale: float,
) -> PreparedSplit:
    graph_index = _load_graph_index(graph_jsonl)
    patch_records = _load_jsonl_records(str(patch_jsonl), max_samples=None)

    image_paths: list[str] = []
    selected_patch_records: list[dict[str, Any]] = []
    labels: list[list[float]] = []
    graph_vectors: list[np.ndarray] = []
    node_features: list[np.ndarray] = []
    adjacency: list[np.ndarray] = []
    object_masks: list[np.ndarray] = []
    skipped_missing_graph = 0
    skipped_bad_crop = 0
    graph_match_summary = {
        "matched_count": 0,
        "missing_count": 0,
        "skipped_count": 0,
        "total_count": 0,
    }

    for patch_record in patch_records:
        if max_samples is not None and len(labels) >= max_samples:
            break

        graph_match_summary["total_count"] += 1
        graph_record = _find_graph_record(patch_record, graph_index)
        if graph_record is None:
            skipped_missing_graph += 1
            graph_match_summary["missing_count"] += 1
            continue
        graph_match_summary["matched_count"] += 1

        graph_vector, node_feature, adj, mask = _graph_arrays(
            graph_record, max_objects=max_objects, class_id_scale=class_id_scale
        )
        score = float(patch_record.get("mean_score", 0.0))
        # Use strict > to match the MPNet AVA label rule.
        label = 1.0 if score > label_threshold else 0.0

        selected_patch_records.append(patch_record)
        image_paths.append(str(patch_record.get("resolved_image_path") or ""))
        labels.append([label])
        graph_vectors.append(graph_vector)
        node_features.append(node_feature)
        adjacency.append(adj)
        object_masks.append(mask)

    if not labels:
        raise RuntimeError(
            f"No usable {split} examples from {patch_jsonl} and {graph_jsonl}."
        )

    return PreparedSplit(
        split=split,
        patch_jsonl=patch_jsonl,
        patch_records=selected_patch_records,
        image_paths=image_paths,
        labels=np.asarray(labels, dtype=np.float32),
        graph_vectors=np.asarray(graph_vectors, dtype=np.float32),
        node_features=np.asarray(node_features, dtype=np.float32),
        adjacency=np.asarray(adjacency, dtype=np.float32),
        object_mask=np.asarray(object_masks, dtype=np.float32),
        crops_shape=[len(labels), patch_count, patch_size, patch_size, 3],
        skipped_missing_graph=skipped_missing_graph,
        skipped_bad_crop=skipped_bad_crop,
        graph_match_summary={
            **graph_match_summary,
            "skipped_count": int(skipped_missing_graph + skipped_bad_crop),
        },
    )


def _load_mpnet_feature_extractor(model_path: Path) -> tf.keras.Model:
    mpnet_model = tf.keras.models.load_model(
        str(model_path),
        compile=False,
        custom_objects=get_alamp_paper_mpnet_custom_objects(),
    )
    try:
        feature_layer = mpnet_model.get_layer("fused_dense")
    except ValueError as exc:
        raise ValueError(
            "Expected MPNet checkpoint to contain penultimate layer 'fused_dense'."
        ) from exc

    feature_extractor = tf.keras.Model(
        inputs=mpnet_model.input,
        outputs=feature_layer.output,
        name="mpnet_fused_dense_feature_extractor",
    )
    feature_extractor.trainable = False
    return feature_extractor


def _feature_cache_metadata(
    split_data: PreparedSplit,
    mpnet_model_path: Path,
    patch_size: int,
    patch_count: int,
    source_sample_count: int,
    valid_indices: list[int],
    feature_shape: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "source_jsonl_path": str(split_data.patch_jsonl),
        "mpnet_model_path": str(mpnet_model_path),
        "patch_size": int(patch_size),
        "patch_count": int(patch_count),
        "source_sample_count": int(source_sample_count),
        "sample_count": int(feature_shape[0]),
        "feature_shape": list(feature_shape),
        "valid_indices": [int(index) for index in valid_indices],
        "skipped_bad_crop": int(source_sample_count - len(valid_indices)),
    }


def _filter_split_to_feature_indices(
    split_data: PreparedSplit,
    valid_indices: list[int],
    source_sample_count: int,
) -> None:
    if valid_indices == list(range(source_sample_count)):
        return

    index_array = np.asarray(valid_indices, dtype=np.int64)
    split_data.patch_records = [split_data.patch_records[index] for index in valid_indices]
    split_data.image_paths = [split_data.image_paths[index] for index in valid_indices]
    split_data.labels = split_data.labels[index_array]
    split_data.graph_vectors = split_data.graph_vectors[index_array]
    split_data.node_features = split_data.node_features[index_array]
    split_data.adjacency = split_data.adjacency[index_array]
    split_data.object_mask = split_data.object_mask[index_array]
    skipped_bad_crop = source_sample_count - len(valid_indices)
    split_data.skipped_bad_crop += skipped_bad_crop
    split_data.graph_match_summary["skipped_count"] = int(
        split_data.graph_match_summary.get("skipped_count", 0) + skipped_bad_crop
    )


def _load_cached_features(
    split_data: PreparedSplit,
    output_path: Path,
    mpnet_model_path: Path,
    patch_size: int,
    patch_count: int,
    force_reextract: bool,
) -> bool:
    if force_reextract:
        logging.info(
            "%s: --force_reextract_features set; ignoring feature cache %s",
            split_data.split,
            output_path,
        )
        return False
    if not output_path.exists():
        logging.info("%s: feature cache not found at %s", split_data.split, output_path)
        return False

    source_sample_count = len(split_data.patch_records)
    with np.load(output_path, allow_pickle=False) as cache:
        features = np.asarray(cache["mpnet_features"], dtype=np.float32)
        if "metadata" not in cache.files:
            logging.warning("%s: feature cache has no metadata; regenerating", split_data.split)
            return False
        metadata_raw = cache["metadata"]
        metadata = json.loads(str(metadata_raw.item()))

    expected = {
        "source_jsonl_path": str(split_data.patch_jsonl),
        "mpnet_model_path": str(mpnet_model_path),
        "patch_size": int(patch_size),
        "patch_count": int(patch_count),
        "source_sample_count": int(source_sample_count),
    }
    mismatches = {
        key: {"expected": value, "found": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    valid_indices = [int(index) for index in metadata.get("valid_indices", [])]
    if not valid_indices:
        valid_indices = list(range(int(metadata.get("sample_count", features.shape[0]))))
    count_ok = (
        features.shape[0] == int(metadata.get("sample_count", -1))
        and features.shape[0] == len(valid_indices)
        and source_sample_count == int(metadata.get("source_sample_count", -1))
        and list(features.shape) == metadata.get("feature_shape")
    )
    if mismatches or not count_ok:
        logging.warning(
            "%s: feature cache mismatch; regenerating. mismatches=%s, "
            "feature_count=%s, valid_indices=%s, source_records=%s",
            split_data.split,
            mismatches,
            features.shape[0],
            len(valid_indices),
            source_sample_count,
        )
        return False

    _filter_split_to_feature_indices(split_data, valid_indices, source_sample_count)
    split_data.mpnet_features = features
    split_data.crops_shape = [features.shape[0], patch_count, patch_size, patch_size, 3]
    split_data.feature_cache = {
        "status": "reused",
        "path": str(output_path),
        "metadata": metadata,
    }
    logging.info(
        "%s: reused cached MPNet features from %s with shape %s",
        split_data.split,
        output_path,
        features.shape,
    )
    return True


def _extract_features(
    feature_extractor: tf.keras.Model,
    split_data: PreparedSplit,
    feature_batch_size: int,
    output_path: Path,
    mpnet_model_path: Path,
    patch_size: int,
    patch_count: int,
) -> None:
    if feature_batch_size <= 0:
        raise ValueError("--feature_batch_size must be positive.")

    source_sample_count = len(split_data.patch_records)
    feature_chunks: list[np.ndarray] = []
    valid_indices: list[int] = []

    logging.info(
        "%s: extracting MPNet features in chunks of %s images into %s",
        split_data.split,
        feature_batch_size,
        output_path,
    )
    for start in range(0, source_sample_count, feature_batch_size):
        end = min(start + feature_batch_size, source_sample_count)
        crop_chunk: list[np.ndarray] = []
        chunk_indices: list[int] = []
        for index in range(start, end):
            crop_array = _process_image_crops(
                split_data.patch_records[index],
                patch_size=patch_size,
                patch_count=patch_count,
            )
            if crop_array is None:
                logging.warning(
                    "%s: skipping sample %s during feature extraction because crop processing failed",
                    split_data.split,
                    index,
                )
                continue
            crop_chunk.append(crop_array)
            chunk_indices.append(index)

        if not crop_chunk:
            continue

        crop_batch = np.asarray(crop_chunk, dtype=np.float32)
        logging.info(
            "%s: MPNet feature chunk %s:%s, crop_batch_shape=%s",
            split_data.split,
            start,
            end,
            crop_batch.shape,
        )
        features = feature_extractor.predict(
            crop_batch,
            batch_size=min(feature_batch_size, crop_batch.shape[0]),
            verbose=0,
        )
        feature_chunks.append(np.asarray(features, dtype=np.float32))
        valid_indices.extend(chunk_indices)
        del crop_chunk, crop_batch, features
        gc.collect()

    if not feature_chunks:
        raise RuntimeError(f"No MPNet features extracted for {split_data.split}.")

    split_data.mpnet_features = np.concatenate(feature_chunks, axis=0).astype(np.float32)
    _filter_split_to_feature_indices(split_data, valid_indices, source_sample_count)
    split_data.crops_shape = [
        int(split_data.mpnet_features.shape[0]),
        patch_count,
        patch_size,
        patch_size,
        3,
    ]
    metadata = _feature_cache_metadata(
        split_data=split_data,
        mpnet_model_path=mpnet_model_path,
        patch_size=patch_size,
        patch_count=patch_count,
        source_sample_count=source_sample_count,
        valid_indices=valid_indices,
        feature_shape=split_data.mpnet_features.shape,
    )
    np.savez_compressed(
        output_path,
        mpnet_features=split_data.mpnet_features,
        labels=split_data.labels,
        image_paths=np.asarray(split_data.image_paths),
        metadata=json.dumps(metadata, sort_keys=True),
    )
    split_data.feature_cache = {
        "status": "extracted",
        "path": str(output_path),
        "metadata": metadata,
    }
    logging.info(
        "%s: wrote MPNet feature cache %s with shape %s",
        split_data.split,
        output_path,
        split_data.mpnet_features.shape,
    )
    del feature_chunks
    gc.collect()


def _make_dataset(
    split_data: PreparedSplit,
    fusion_mode: str,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    if split_data.mpnet_features is None:
        raise RuntimeError(f"MPNet features were not extracted for {split_data.split}.")

    if fusion_mode == "graphlite":
        inputs = {
            "mpnet_features": split_data.mpnet_features,
            "graph_vector": split_data.graph_vectors,
        }
    elif fusion_mode == "graphgcn":
        inputs = {
            "mpnet_features": split_data.mpnet_features,
            "node_features": split_data.node_features,
            "adjacency": split_data.adjacency,
            "object_mask": split_data.object_mask,
        }
    else:
        raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

    dataset = tf.data.Dataset.from_tensor_slices((inputs, split_data.labels))
    if shuffle:
        dataset = dataset.shuffle(
            min(int(split_data.labels.shape[0]), 256), seed=123, reshuffle_each_iteration=True
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _build_model(fusion_mode: str, train_data: PreparedSplit) -> tf.keras.Model:
    if train_data.mpnet_features is None:
        raise RuntimeError("MPNet features must be available before model build.")

    mpnet_dim = int(train_data.mpnet_features.shape[1])
    if fusion_mode == "graphlite":
        return build_mpnet_graphlite_fusion_model(
            mpnet_feature_dim=mpnet_dim,
            graph_vector_dim=int(train_data.graph_vectors.shape[1]),
        )
    if fusion_mode == "graphgcn":
        return build_mpnet_graphgcn_fusion_model(
            mpnet_feature_dim=mpnet_dim,
            node_feature_dim=int(train_data.node_features.shape[2]),
            max_objects=int(train_data.node_features.shape[1]),
        )
    raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")


def _compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )


def _run_smoke(
    fusion_mode: str,
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> dict[str, Any]:
    batch_inputs, batch_labels = next(iter(dataset))
    predictions = model(batch_inputs, training=False)
    loss_before = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(batch_labels, predictions)
    )
    train_result = model.train_on_batch(
        batch_inputs, batch_labels, return_dict=True
    )
    return {
        "fusion_mode": fusion_mode,
        "batch_input_shapes": {
            key: list(value.shape) for key, value in batch_inputs.items()
        },
        "batch_label_shape": list(batch_labels.shape),
        "prediction_shape": list(predictions.shape),
        "loss_before_train_step": float(loss_before.numpy()),
        "train_on_batch": {
            key: float(value) for key, value in train_result.items()
        },
    }


def _run_training(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    epochs: int,
) -> dict[str, list[float]]:
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        verbose=2,
    )
    return {key: [float(v) for v in values] for key, values in history.history.items()}


def _print_shape_report(train_data: PreparedSplit, val_data: PreparedSplit) -> None:
    print("Tensor shapes:")
    for split_data in (train_data, val_data):
        report = split_data.shape_report()
        print(f"  {split_data.split}:")
        for key, value in report.items():
            print(f"    {key}: {value}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _command_for_report() -> str:
    command = " ".join([sys.executable, *sys.argv])
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        return f"PYTHONPATH={pythonpath} {command}"
    return command


def _write_report(
    out_dir: Path,
    args: argparse.Namespace,
    train_data: PreparedSplit,
    val_data: PreparedSplit,
    mode_results: dict[str, Any],
) -> None:
    shape_report = {
        "train": train_data.shape_report(),
        "val": val_data.shape_report(),
    }
    summary = {
        "status": "smoke_completed" if args.smoke else "training_completed",
        "paper_reproduction_claim": "paper-oriented approximation",
        "files_changed": [
            "src/models/alamp_paper_fusion.py",
            "src/train/train_alamp_paper_mpnet_graph_fusion.py",
            "checklist.md",
            "context-notes.md",
        ],
        "data_files_required": {
            "train_patch_jsonl": str(args.train_patch_jsonl),
            "val_patch_jsonl": str(args.val_patch_jsonl),
            "train_graph_jsonl": str(args.train_graph_jsonl),
            "val_graph_jsonl": str(args.val_graph_jsonl),
        },
        "checkpoint_paths_used": {"mpnet_model": str(args.mpnet_model_path)},
        "feature_extraction": {
            "feature_batch_size_images": int(args.feature_batch_size),
            "force_reextract_features": bool(args.force_reextract_features),
            "train_feature_cache": train_data.feature_cache,
            "val_feature_cache": val_data.feature_cache,
        },
        "mpnet_checkpoint_policy": {
            "val_selected_checkpoint": _is_val_selected_mpnet_checkpoint(
                args.mpnet_model_path
            ),
            "policy_reason": _mpnet_checkpoint_policy_reason(args.mpnet_model_path),
            "allow_val_selected_mpnet": bool(args.allow_val_selected_mpnet),
            "smoke_metrics_are_not_final": bool(args.smoke),
        },
        "tensor_shape_report": shape_report,
        "graphgcn_status": (
            "implemented_from_node_features_and_local_edges"
            if "graphgcn" in mode_results
            else "not_requested_in_this_run"
        ),
        "smoke_command": _command_for_report(),
        "mode_results": mode_results,
        "risks_and_unknowns": [
            "Graph cache comes from object detector outputs, not an official A-LAMP detector reproduction.",
            "GraphLite vectorization is an approximation over fixed graph fields.",
            "Fusion branch is not yet validated by full 1024 training/evaluation.",
            "MPNet features are frozen extracted features; this script does not fine-tune MPNet jointly.",
            "If a validation-selected MPNet checkpoint is explicitly allowed for non-smoke training, validation metrics should not be treated as final.",
        ],
    }
    _write_json(out_dir / "fusion_smoke_summary.json", summary)
    _write_json(out_dir / "tensor_shape_report.json", shape_report)

    report_lines = [
        "# MPNet + Layout Graph Fusion Smoke Report",
        "",
        "This branch is a paper-oriented approximation, not an official or exact A-LAMP reproduction.",
        "",
        "## Files Changed",
        "",
        "- `src/models/alamp_paper_fusion.py`",
        "- `src/train/train_alamp_paper_mpnet_graph_fusion.py`",
        "- `checklist.md`",
        "- `context-notes.md`",
        "",
        "## Data Files Required",
        "",
        f"- train patch JSONL: `{args.train_patch_jsonl}`",
        f"- val patch JSONL: `{args.val_patch_jsonl}`",
        f"- train graph JSONL: `{args.train_graph_jsonl}`",
        f"- val graph JSONL: `{args.val_graph_jsonl}`",
        f"- MPNet checkpoint: `{args.mpnet_model_path}`",
        "",
        "## Feature Extraction",
        "",
        "```json",
        json.dumps(summary["feature_extraction"], indent=2, sort_keys=True),
        "```",
        "",
        "## Tensor Shapes",
        "",
        "```json",
        json.dumps(shape_report, indent=2, sort_keys=True),
        "```",
        "",
        "## GraphGCN Status",
        "",
        summary["graphgcn_status"],
        "",
        "## Smoke Command",
        "",
        "```bash",
        _command_for_report(),
        "```",
        "",
        "## Smoke Result",
        "",
        "```json",
        json.dumps(mode_results, indent=2, sort_keys=True),
        "```",
        "",
        "## Risks And Unknowns",
        "",
    ]
    report_lines.extend(f"- {item}" for item in summary["risks_and_unknowns"])
    (out_dir / "fusion_smoke_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion_mode", choices=["graphlite", "graphgcn", "both"], default="both")
    parser.add_argument("--mpnet_model_path", default=DEFAULT_MPNET_MODEL)
    parser.add_argument("--train_patch_jsonl", default=DEFAULT_TRAIN_PATCH_JSONL)
    parser.add_argument("--val_patch_jsonl", default=DEFAULT_VAL_PATCH_JSONL)
    parser.add_argument("--train_graph_jsonl", default=DEFAULT_TRAIN_GRAPH_JSONL)
    parser.add_argument("--val_graph_jsonl", default=DEFAULT_VAL_GRAPH_JSONL)
    parser.add_argument("--out_dir", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_val_samples", type=int)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--patch_count", type=int, default=5)
    parser.add_argument("--max_objects", type=int, default=4)
    parser.add_argument("--class_id_scale", type=float, default=79.0)
    parser.add_argument("--label_threshold", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--feature_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--allow_val_selected_mpnet", action="store_true")
    parser.add_argument("--force_reextract_features", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    args = _parse_args()

    args.mpnet_model_path = Path(args.mpnet_model_path)
    args.train_patch_jsonl = Path(args.train_patch_jsonl)
    args.val_patch_jsonl = Path(args.val_patch_jsonl)
    args.train_graph_jsonl = Path(args.train_graph_jsonl)
    args.val_graph_jsonl = Path(args.val_graph_jsonl)
    out_dir = Path(args.out_dir)

    _require_file(args.mpnet_model_path, "MPNet checkpoint")
    _require_file(args.train_patch_jsonl, "train patch JSONL")
    _require_file(args.val_patch_jsonl, "val patch JSONL")
    _require_file(args.train_graph_jsonl, "train graph JSONL")
    _require_file(args.val_graph_jsonl, "val graph JSONL")
    _enforce_mpnet_checkpoint_policy(args)

    out_dir.mkdir(parents=True, exist_ok=True)
    features_dir = out_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    if args.smoke:
        args.max_train_samples = args.max_train_samples or 8
        args.max_val_samples = args.max_val_samples or 4

    train_data = _prepare_split(
        split="train",
        patch_jsonl=args.train_patch_jsonl,
        graph_jsonl=args.train_graph_jsonl,
        max_samples=args.max_train_samples,
        patch_size=args.patch_size,
        patch_count=args.patch_count,
        label_threshold=args.label_threshold,
        max_objects=args.max_objects,
        class_id_scale=args.class_id_scale,
    )
    val_data = _prepare_split(
        split="val",
        patch_jsonl=args.val_patch_jsonl,
        graph_jsonl=args.val_graph_jsonl,
        max_samples=args.max_val_samples,
        patch_size=args.patch_size,
        patch_count=args.patch_count,
        label_threshold=args.label_threshold,
        max_objects=args.max_objects,
        class_id_scale=args.class_id_scale,
    )

    train_features_path = features_dir / "train_mpnet_features.npz"
    val_features_path = features_dir / "val_mpnet_features.npz"
    train_features_cached = _load_cached_features(
        train_data,
        output_path=train_features_path,
        mpnet_model_path=args.mpnet_model_path,
        patch_size=args.patch_size,
        patch_count=args.patch_count,
        force_reextract=args.force_reextract_features,
    )
    val_features_cached = _load_cached_features(
        val_data,
        output_path=val_features_path,
        mpnet_model_path=args.mpnet_model_path,
        patch_size=args.patch_size,
        patch_count=args.patch_count,
        force_reextract=args.force_reextract_features,
    )

    feature_extractor = None
    if not train_features_cached or not val_features_cached:
        feature_extractor = _load_mpnet_feature_extractor(args.mpnet_model_path)
    if not train_features_cached:
        _extract_features(
            feature_extractor,
            train_data,
            feature_batch_size=args.feature_batch_size,
            output_path=train_features_path,
            mpnet_model_path=args.mpnet_model_path,
            patch_size=args.patch_size,
            patch_count=args.patch_count,
        )
    if not val_features_cached:
        _extract_features(
            feature_extractor,
            val_data,
            feature_batch_size=args.feature_batch_size,
            output_path=val_features_path,
            mpnet_model_path=args.mpnet_model_path,
            patch_size=args.patch_size,
            patch_count=args.patch_count,
        )
    del feature_extractor
    gc.collect()
    _print_shape_report(train_data, val_data)

    modes = ["graphlite", "graphgcn"] if args.fusion_mode == "both" else [args.fusion_mode]
    mode_results: dict[str, Any] = {}

    for mode in modes:
        model = _build_model(mode, train_data)
        _compile_model(model, learning_rate=args.learning_rate)
        train_dataset = _make_dataset(train_data, mode, args.batch_size, shuffle=not args.smoke)
        val_dataset = _make_dataset(val_data, mode, args.batch_size, shuffle=False)

        if args.smoke:
            mode_results[mode] = _run_smoke(mode, model, train_dataset)
        else:
            history = _run_training(
                model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=args.epochs,
            )
            mode_results[mode] = {"history": history}

        if args.save_model or not args.smoke:
            mode_dir = out_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            model.save(mode_dir / "final_model.keras")
            mode_results[mode]["saved_model"] = str(mode_dir / "final_model.keras")

    _write_report(out_dir, args, train_data, val_data, mode_results)
    print(f"Wrote fusion report to {out_dir / 'fusion_smoke_report.md'}")


if __name__ == "__main__":
    main()
