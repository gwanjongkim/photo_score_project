# 누락된 미학 모델 교차 예측을 TFLite로 생성하고 정렬 준비 상태를 보고한다.
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageFile, ImageOps


ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs/aesthetic_cross_predictions_for_weight_search_20260603"
SEED = 42

EXISTING = {
    ("ava", "nima"): ROOT / "outputs/aesthetic_mobile_paper_benchmark_20260603/full/nima_ava/nima_predictions.csv",
    ("ava", "alamp_adaptive"): ROOT
    / "outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603/full_adaptive/alamp_adaptive_predictions.csv",
    ("aadb", "rgnet"): ROOT / "outputs/aesthetic_mobile_paper_benchmark_20260603/full/rgnet_aadb/rgnet_predictions.csv",
    ("icaa17k", "icaa"): ROOT / "outputs/aesthetic_mobile_paper_benchmark_20260603/full/icaa_icaa17k/icaa_predictions.csv",
}

MODEL_PATHS = {
    "nima": ROOT / "forWeights2/models/aesthetic/nima_mobile_fixed_preproc_fp16.tflite",
    "rgnet": ROOT / "forWeights2/models/aesthetic/rgnet_pil_resize_aadb_fp16.tflite",
    "alamp_adaptive": ROOT / "forWeights2/models/aesthetic/mobile_alamp_v2_fp16.tflite",
    "icaa": ROOT / "forWeights2/models/aesthetic/icaa_dat_tf_native_fp16.tflite",
}

REQUESTED_ASSET_PATHS = {
    "nima": ROOT / "assets/models/nima_mobile_fixed_preproc_fp16.tflite",
    "rgnet": ROOT / "assets/models/rgnet_pil_resize_aadb_fp16.tflite",
    "alamp_adaptive": ROOT / "assets/models/mobile_alamp_v2_fp16.tflite",
    "icaa": ROOT / "assets/models/icaa_dat_tf_native_fp16.tflite",
}

OUTPUT_FILES = {
    ("ava", "rgnet"): OUT_DIR / "ava_rgnet_predictions.csv",
    ("ava", "icaa"): OUT_DIR / "ava_icaa_predictions.csv",
    ("aadb", "nima"): OUT_DIR / "aadb_nima_predictions.csv",
    ("aadb", "alamp_adaptive"): OUT_DIR / "aadb_alamp_adaptive_predictions.csv",
    ("aadb", "icaa"): OUT_DIR / "aadb_icaa_predictions.csv",
}

FAILURE_FILES = {
    key: OUT_DIR / f"{key[0]}_{key[1]}_failures.csv" for key in OUTPUT_FILES
}


@dataclass(frozen=True)
class Row:
    dataset: str
    source_id: str
    join_key: str
    image_path: Path
    gt_score: float


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def join_key(value: object) -> str:
    return Path(str(value)).stem.lower()


def resolve_image_path(value: object) -> Path:
    text = str(value)
    candidates = [
        Path(text),
        ROOT / text,
        ROOT / "data/raw/ava/images" / Path(text).name,
        ROOT / "data/raw/aadb/images" / Path(text).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


def load_split(dataset: str, limit: Optional[int]) -> List[Row]:
    if dataset == "ava":
        df = pd.read_csv(ROOT / "data/processed/ava/test.csv")
        path_col = "image_path"
        gt_col = "mean_score"
    elif dataset == "aadb":
        df = pd.read_csv(ROOT / "data/processed/aadb/test.csv")
        path_col = "image_path"
        gt_col = "score"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if limit is not None:
        df = df.head(limit).copy()
    rows = []
    for _, row in df.iterrows():
        source_id = str(row[path_col])
        rows.append(
            Row(
                dataset=dataset,
                source_id=source_id,
                join_key=join_key(source_id),
                image_path=resolve_image_path(source_id),
                gt_score=float(row[gt_col]),
            )
        )
    return rows


def load_rgb(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.exif_transpose(image)


def resize_array(image: Image.Image, size: Tuple[int, int]) -> np.ndarray:
    return np.array(image.resize(size, Image.BILINEAR)).astype(np.float32)


def normalize_icaa(arr: np.ndarray) -> np.ndarray:
    arr = arr / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (arr - mean) / std


def compute_saliency_v2(image_bgr: np.ndarray) -> np.ndarray:
    img_small = cv2.resize(image_bgr, (64, 64))
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fft = np.fft.fft2(gray)
    log_amplitude = np.log(np.abs(fft) + 1e-6)
    phase = np.angle(fft)
    avg_log_amplitude = cv2.blur(log_amplitude, (3, 3))
    spectral_residual = log_amplitude - avg_log_amplitude
    saliency_map = np.abs(np.fft.ifft2(np.exp(spectral_residual + 1j * phase))) ** 2
    saliency_map = cv2.GaussianBlur(saliency_map, (3, 3), 0)
    h, w = image_bgr.shape[:2]
    saliency_map = cv2.resize(saliency_map, (w, h))
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    center_bias = np.exp(-((x - cx) ** 2 / (w / 2.0) ** 2 + (y - cy) ** 2 / (h / 2.0) ** 2))
    saliency_map = saliency_map * (0.8 + 0.2 * center_bias)
    m_min, m_max = saliency_map.min(), saliency_map.max()
    if m_max > m_min:
        return (saliency_map - m_min) / (m_max - m_min)
    return saliency_map


def generate_candidate_boxes(w: int, h: int, scales: Sequence[float], grid_size: int = 8) -> np.ndarray:
    boxes = []
    short_edge = min(w, h)
    for scale in scales:
        size = int(short_edge * scale)
        if size < 32:
            continue
        step_x = max(1, (w - size) // grid_size)
        step_y = max(1, (h - size) // grid_size)
        for y in range(0, h - size + 1, step_y):
            for x in range(0, w - size + 1, step_x):
                boxes.append([x, y, x + size, y + size])
    return np.array(boxes)


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-6)


def extract_features(image_bgr: np.ndarray, box: np.ndarray) -> Dict[str, np.ndarray]:
    x1, y1, x2, y2 = map(int, box)
    patch = image_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return {"color_hist": np.zeros(32, dtype=np.float32), "edge_hist": np.zeros(8, dtype=np.float32)}
    patch_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    hist_l = cv2.calcHist([patch_lab], [0], None, [16], [0, 256]).flatten()
    hist_a = cv2.calcHist([patch_lab], [1], None, [8], [0, 256]).flatten()
    hist_b = cv2.calcHist([patch_lab], [2], None, [8], [0, 256]).flatten()
    color_hist = np.concatenate([hist_l, hist_a, hist_b])
    color_hist /= color_hist.sum() + 1e-6
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_hist = cv2.calcHist([edges], [0], None, [8], [0, 256]).flatten()
    edge_hist /= edge_hist.sum() + 1e-6
    return {"color_hist": color_hist, "edge_hist": edge_hist}


def select_adaptive_patches(image: Image.Image, patch_count: int = 5) -> Tuple[np.ndarray, List[List[int]]]:
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]
    candidates = generate_candidate_boxes(w, h, [0.25, 0.35, 0.5])
    saliency_map = compute_saliency_v2(image_bgr)
    box_scores = []
    box_feats = []
    box_centers = []
    for box in candidates:
        x1, y1, x2, y2 = map(int, box)
        crop = saliency_map[y1:y2, x1:x2]
        box_scores.append(crop.mean() if crop.size > 0 else 0.0)
        box_feats.append(extract_features(image_bgr, box))
        box_centers.append([(x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h])
    box_scores = np.array(box_scores, dtype=np.float32)
    box_centers = np.array(box_centers, dtype=np.float32)
    selected_indices: List[int] = []
    for _ in range(patch_count):
        if not selected_indices:
            best_idx = int(np.argmax(box_scores))
        else:
            scores = box_scores.copy()
            for idx in selected_indices:
                ious = compute_iou(candidates[idx], candidates)
                scores[ious > 0.5] = -1.0
                curr_feat = box_feats[idx]
                for i, other_feat in enumerate(box_feats):
                    if scores[i] <= 0:
                        continue
                    color_dist = cv2.compareHist(curr_feat["color_hist"], other_feat["color_hist"], cv2.HISTCMP_BHATTACHARYYA)
                    edge_dist = cv2.compareHist(curr_feat["edge_hist"], other_feat["edge_hist"], cv2.HISTCMP_BHATTACHARYYA)
                    scores[i] += 0.5 * (color_dist + edge_dist)
                dist = np.sqrt(np.sum((box_centers - box_centers[idx]) ** 2, axis=1))
                scores += 0.3 * dist
            scores[selected_indices] = -1.0
            best_idx = int(np.argmax(scores))
        selected_indices.append(best_idx)
    selected_boxes = candidates[selected_indices]
    patches = []
    for box in selected_boxes:
        x1, y1, x2, y2 = map(int, box)
        patch_img = image_bgr[y1:y2, x1:x2]
        patch_img = cv2.resize(patch_img, (224, 224), interpolation=cv2.INTER_LINEAR)
        patch_img = cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB)
        patches.append(patch_img.astype(np.float32))
    return np.expand_dims(np.stack(patches), axis=0), selected_boxes.astype(int).tolist()


class TFLiteModel:
    def __init__(self, model: str) -> None:
        self.model = model
        self.path = MODEL_PATHS[model]
        self.interpreter = tf.lite.Interpreter(model_path=str(self.path), num_threads=max(1, min(4, os.cpu_count() or 1)))
        self.interpreter.allocate_tensors()
        self.inputs = self.interpreter.get_input_details()
        self.outputs = self.interpreter.get_output_details()

    def set_single(self, value: np.ndarray) -> None:
        self.interpreter.set_tensor(self.inputs[0]["index"], value.astype(np.float32))

    def set_alamp(self, full: np.ndarray, patches: np.ndarray) -> None:
        for detail in self.inputs:
            shape = tuple(int(x) for x in detail["shape"])
            if len(shape) == 4 and shape[1] == 384:
                self.interpreter.set_tensor(detail["index"], full.astype(np.float32))
            elif len(shape) == 5 or (len(shape) == 4 and shape[1] == 5):
                self.interpreter.set_tensor(detail["index"], patches.astype(np.float32))
            else:
                raise ValueError(f"Unexpected A-LAMP input shape: {detail['name']} {shape}")

    def invoke(self) -> List[np.ndarray]:
        self.interpreter.invoke()
        return [self.interpreter.get_tensor(detail["index"]) for detail in self.outputs]


def predict(model: TFLiteModel, row: Row) -> Dict[str, object]:
    t0 = time.perf_counter()
    image = load_rgb(row.image_path)
    patch_coords = ""
    if model.model == "nima":
        arr = resize_array(image, (224, 224)) / 255.0
        input_value = np.expand_dims(arr, axis=0)
        preprocess_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        model.set_single(input_value)
        output = model.invoke()[0][0].astype(float)
        inference_ms = (time.perf_counter() - t1) * 1000.0
        pred_mean = float(np.sum(output * np.arange(1, 11)))
        result = {
            "pred_raw": pred_mean,
            "pred_scaled_1_10": pred_mean,
            "pred_mean": pred_mean,
        }
        for idx, value in enumerate(output, start=1):
            result[f"pred_distribution_{idx}"] = float(value)
    elif model.model == "rgnet":
        arr = resize_array(image, (256, 256)) / 255.0
        input_value = np.expand_dims(arr, axis=0)
        preprocess_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        model.set_single(input_value)
        pred = float(model.invoke()[0][0][0])
        inference_ms = (time.perf_counter() - t1) * 1000.0
        result = {"pred_raw": pred, "pred_scaled_1_10": pred * 10.0}
    elif model.model == "icaa":
        arr = normalize_icaa(resize_array(image, (224, 224)))
        input_value = np.expand_dims(arr, axis=0)
        preprocess_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        model.set_single(input_value)
        output = model.invoke()[0][0].astype(float)
        inference_ms = (time.perf_counter() - t1) * 1000.0
        mos = float(output[0]) if len(output) > 0 else np.nan
        color = float(output[1]) if len(output) > 1 else np.nan
        result = {
            "pred_raw": color,
            "pred_scaled_1_10": color * 10.0,
            "pred_mos_raw": mos,
            "pred_color_raw": color,
            "pred_color_scaled_1_10": color * 10.0,
        }
    elif model.model == "alamp_adaptive":
        full = np.expand_dims(resize_array(image, (384, 384)), axis=0)
        patches, boxes = select_adaptive_patches(image)
        patch_coords = json.dumps(boxes, separators=(",", ":"))
        preprocess_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        model.set_alamp(full, patches)
        pred = float(model.invoke()[0][0][0])
        inference_ms = (time.perf_counter() - t1) * 1000.0
        result = {
            "pred_raw": pred,
            "pred_scaled_1_10": pred * 10.0,
            "patch_selector": "v2_spectral_residual_pattern_diversity_overlap",
            "patch_coords": patch_coords,
        }
    else:
        raise ValueError(f"Unsupported model: {model.model}")
    result.update({"preprocess_ms": preprocess_ms, "inference_ms": inference_ms, "failure_reason": ""})
    return result


def base_prediction_record(row: Row, model: str) -> Dict[str, object]:
    return {
        "dataset": row.dataset,
        "model": model,
        "source_id": row.source_id,
        "join_key": row.join_key,
        "image_path": str(row.image_path),
        "gt_score": row.gt_score,
    }


def read_existing_success_keys(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(path, usecols=["join_key"])
    except Exception:
        return set()
    return set(df["join_key"].astype(str))


def append_csv(path: Path, records: List[Dict[str, object]]) -> None:
    if not records:
        return
    existing_header = path.exists() and path.stat().st_size > 0
    fieldnames = sorted({key for record in records for key in record.keys()})
    preferred = [
        "dataset",
        "model",
        "source_id",
        "join_key",
        "image_path",
        "gt_score",
        "pred_raw",
        "pred_scaled_1_10",
        "inference_ms",
        "preprocess_ms",
        "failure_reason",
        "pred_mean",
        "pred_mos_raw",
        "pred_color_raw",
        "pred_color_scaled_1_10",
        "patch_selector",
        "patch_coords",
    ]
    fieldnames = [x for x in preferred if x in fieldnames] + [x for x in fieldnames if x not in preferred]
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not existing_header:
            writer.writeheader()
        for record in records:
            writer.writerow(record)


def run_pair(dataset: str, model_name: str, limit: Optional[int], resume: bool, flush_every: int) -> Dict[str, object]:
    out_path = OUTPUT_FILES[(dataset, model_name)]
    fail_path = FAILURE_FILES[(dataset, model_name)]
    rows = load_split(dataset, limit)
    done = read_existing_success_keys(out_path) if resume else set()
    model = TFLiteModel(model_name)
    successes = 0
    failures = 0
    skipped = 0
    buffer: List[Dict[str, object]] = []
    fail_buffer: List[Dict[str, object]] = []
    t_start = time.perf_counter()
    for idx, row in enumerate(rows, start=1):
        if row.join_key in done:
            skipped += 1
            continue
        try:
            if not row.image_path.exists():
                raise FileNotFoundError(str(row.image_path))
            record = base_prediction_record(row, model_name)
            record.update(predict(model, row))
            buffer.append(record)
            successes += 1
        except Exception as exc:
            failures += 1
            fail_buffer.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "source_id": row.source_id,
                    "image_path": str(row.image_path),
                    "failure_reason": str(exc),
                }
            )
        if idx % flush_every == 0:
            append_csv(out_path, buffer)
            append_csv(fail_path, fail_buffer)
            buffer.clear()
            fail_buffer.clear()
            print(f"{dataset}/{model_name}: processed {idx}/{len(rows)} success={successes} fail={failures} skipped={skipped}", flush=True)
    append_csv(out_path, buffer)
    append_csv(fail_path, fail_buffer)
    elapsed = time.perf_counter() - t_start
    return {
        "dataset": dataset,
        "model": model_name,
        "split_rows": len(rows),
        "successes_this_run": successes,
        "failures_this_run": failures,
        "skipped_existing": skipped,
        "elapsed_seconds": elapsed,
        "output_path": rel(out_path),
        "failure_path": rel(fail_path),
    }


def load_prediction_for_alignment(path: Path, model: str, existing_schema: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    if existing_schema:
        image_col = "image"
        pred_col = "pred_score"
        gt_col = "gt_score"
        out = pd.DataFrame(
            {
                "join_key": df[image_col].map(join_key),
                "source_id": df[image_col].astype(str),
                "gt_score": pd.to_numeric(df[gt_col], errors="coerce"),
                f"{model}_pred_raw": pd.to_numeric(df[pred_col], errors="coerce"),
            }
        )
        if model == "nima":
            out[f"{model}_pred_scaled_1_10"] = out[f"{model}_pred_raw"]
        elif model == "rgnet":
            out[f"{model}_pred_scaled_1_10"] = out[f"{model}_pred_raw"] * 10.0
        elif model == "alamp_adaptive":
            out[f"{model}_pred_scaled_1_10"] = out[f"{model}_pred_raw"] * 10.0
        elif model == "icaa":
            out[f"{model}_pred_scaled_1_10"] = out[f"{model}_pred_raw"] * 10.0
        return out.dropna(subset=["join_key"])
    out = pd.DataFrame(
        {
            "join_key": df["join_key"].astype(str),
            "source_id": df["source_id"].astype(str),
            "gt_score": pd.to_numeric(df["gt_score"], errors="coerce"),
            f"{model}_pred_raw": pd.to_numeric(df["pred_raw"], errors="coerce"),
            f"{model}_pred_scaled_1_10": pd.to_numeric(df["pred_scaled_1_10"], errors="coerce"),
        }
    )
    return out.dropna(subset=["join_key"])


def align_dataset(dataset: str) -> Dict[str, object]:
    if dataset == "ava":
        specs = [
            ("nima", EXISTING[("ava", "nima")], True),
            ("rgnet", OUTPUT_FILES[("ava", "rgnet")], False),
            ("alamp_adaptive", EXISTING[("ava", "alamp_adaptive")], True),
            ("icaa", OUTPUT_FILES[("ava", "icaa")], False),
        ]
        output = OUT_DIR / "aligned_ava_complete_primary.csv"
    elif dataset == "aadb":
        specs = [
            ("nima", OUTPUT_FILES[("aadb", "nima")], False),
            ("rgnet", EXISTING[("aadb", "rgnet")], True),
            ("alamp_adaptive", OUTPUT_FILES[("aadb", "alamp_adaptive")], False),
            ("icaa", OUTPUT_FILES[("aadb", "icaa")], False),
        ]
        output = OUT_DIR / "aligned_aadb_complete_primary.csv"
    else:
        raise ValueError(dataset)
    frames = []
    counts = {}
    for model, path, existing_schema in specs:
        if not path.exists():
            counts[f"{model}_rows"] = 0
            continue
        frame = load_prediction_for_alignment(path, model, existing_schema)
        frame = frame.drop_duplicates("join_key", keep="first")
        counts[f"{model}_rows"] = len(frame)
        frames.append((model, frame))
    if len(frames) != len(specs):
        aligned = pd.DataFrame()
    else:
        aligned = frames[0][1]
        for _, frame in frames[1:]:
            aligned = aligned.merge(frame, on="join_key", how="inner", suffixes=("", "_other"))
            if "gt_score_other" in aligned.columns:
                aligned = aligned.drop(columns=["gt_score_other"])
            if "source_id_other" in aligned.columns:
                aligned = aligned.drop(columns=["source_id_other"])
    aligned.to_csv(output, index=False)
    split_rows = len(load_split(dataset, None))
    summary = {
        "dataset": dataset,
        "split_rows": split_rows,
        "aligned_complete_rows": len(aligned),
        "output_path": rel(output),
        **counts,
    }
    for model, path, existing_schema in specs:
        rows = counts.get(f"{model}_rows", 0)
        summary[f"{model}_missing_vs_split"] = max(0, split_rows - rows)
        summary[f"{model}_path"] = rel(path)
        summary[f"{model}_reused_existing"] = existing_schema
    return summary


def summarize_prediction_file(path: Path) -> Dict[str, object]:
    if not path.exists() or path.stat().st_size == 0:
        return {"rows": 0}
    df = pd.read_csv(path)
    summary: Dict[str, object] = {"rows": len(df)}
    for col in ["pred_raw", "pred_scaled_1_10", "inference_ms", "preprocess_ms"]:
        if col in df.columns and len(df):
            values = pd.to_numeric(df[col], errors="coerce")
            summary[f"{col}_min"] = float(values.min())
            summary[f"{col}_max"] = float(values.max())
            summary[f"{col}_mean"] = float(values.mean())
    return summary


def write_inventory(run_summaries: Sequence[Dict[str, object]]) -> None:
    rows = []
    for model, actual_path in MODEL_PATHS.items():
        requested = REQUESTED_ASSET_PATHS[model]
        rows.append(
            {
                "kind": "model",
                "dataset": "",
                "model": model,
                "requested_asset_exists": requested.exists(),
                "requested_asset_path": rel(requested),
                "actual_path": rel(actual_path),
                "actual_exists": actual_path.exists(),
                "note": "using equivalent forWeights2 artifact because requested assets path is absent",
            }
        )
    for (dataset, model), path in EXISTING.items():
        rows.append(
            {
                "kind": "prediction_reused",
                "dataset": dataset,
                "model": model,
                "requested_asset_exists": "",
                "requested_asset_path": "",
                "actual_path": rel(path),
                "actual_exists": path.exists(),
                "note": "primary existing benchmark prediction",
            }
        )
    for (dataset, model), path in OUTPUT_FILES.items():
        summary = summarize_prediction_file(path)
        rows.append(
            {
                "kind": "prediction_generated",
                "dataset": dataset,
                "model": model,
                "requested_asset_exists": "",
                "requested_asset_path": "",
                "actual_path": rel(path),
                "actual_exists": path.exists(),
                "note": json.dumps(summary, sort_keys=True),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "prediction_inventory.csv", index=False)


def write_failure_summary() -> None:
    rows = []
    for (dataset, model), path in FAILURE_FILES.items():
        count = 0
        first = ""
        if path.exists() and path.stat().st_size > 0:
            df = pd.read_csv(path)
            count = len(df)
            first = "" if df.empty else str(df.iloc[0].get("failure_reason", ""))
        rows.append({"dataset": dataset, "model": model, "failure_rows": count, "failure_path": rel(path), "first_failure": first})
    pd.DataFrame(rows).to_csv(OUT_DIR / "failure_summary.csv", index=False)


def write_runtime_summary() -> pd.DataFrame:
    rows = []
    for (dataset, model), path in OUTPUT_FILES.items():
        row: Dict[str, object] = {"dataset": dataset, "model": model, "prediction_path": rel(path)}
        if path.exists() and path.stat().st_size > 0:
            df = pd.read_csv(path)
            row["successful_prediction_rows"] = len(df)
            for col in ["preprocess_ms", "inference_ms"]:
                values = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)
                row[f"{col}_mean"] = float(values.mean()) if len(values) else np.nan
                row[f"{col}_median"] = float(values.median()) if len(values) else np.nan
                row[f"{col}_max"] = float(values.max()) if len(values) else np.nan
            row["estimated_total_seconds_from_row_timings"] = float(
                (
                    pd.to_numeric(df.get("preprocess_ms", pd.Series(dtype=float)), errors="coerce").sum()
                    + pd.to_numeric(df.get("inference_ms", pd.Series(dtype=float)), errors="coerce").sum()
                )
                / 1000.0
            )
        else:
            row["successful_prediction_rows"] = 0
            row["preprocess_ms_mean"] = np.nan
            row["inference_ms_mean"] = np.nan
            row["estimated_total_seconds_from_row_timings"] = np.nan
        rows.append(row)
    runtime = pd.DataFrame(rows)
    runtime.to_csv(OUT_DIR / "runtime_summary.csv", index=False)
    return runtime


def write_report(run_summaries: Sequence[Dict[str, object]], alignment_rows: Sequence[Dict[str, object]]) -> None:
    inventory = pd.read_csv(OUT_DIR / "prediction_inventory.csv")
    failures = pd.read_csv(OUT_DIR / "failure_summary.csv")
    runtime = pd.read_csv(OUT_DIR / "runtime_summary.csv")
    alignments = pd.DataFrame(alignment_rows)
    ava_ready = bool((alignments["dataset"] == "ava").any() and int(alignments.loc[alignments["dataset"] == "ava", "aligned_complete_rows"].iloc[0]) > 0)
    aadb_ready = bool((alignments["dataset"] == "aadb").any() and int(alignments.loc[alignments["dataset"] == "aadb", "aligned_complete_rows"].iloc[0]) > 0)

    def table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_None._"
        cols = [str(c) for c in df.columns]
        lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, row in df.iterrows():
            values = []
            for col in df.columns:
                value = "" if pd.isna(row[col]) else str(row[col])
                values.append(value.replace("\n", " ").replace("|", "\\|"))
            lines.append("|" + "|".join(values) + "|")
        return "\n".join(lines)

    next_prompt = (
        "Run the weight-search step using "
        "`outputs/aesthetic_cross_predictions_for_weight_search_20260603/aligned_ava_complete_primary.csv` "
        "and `outputs/aesthetic_cross_predictions_for_weight_search_20260603/aligned_aadb_complete_primary.csv`, "
        "without using non-primary candidate predictions and while keeping ICAA separately reported."
    )
    report = [
        "# Aesthetic Cross Predictions For Weight Search 20260603",
        "",
        "## Executive Summary",
        (
            "Generated missing same-row primary cross-predictions where requested and created complete "
            "alignment tables for the next weight-search step. This report does not tune weights and does "
            "not change production scoring."
        ),
        "",
        "## Model Artifacts",
        table(inventory[inventory["kind"] == "model"]),
        "",
        "## Generated Missing Predictions",
        table(inventory[inventory["kind"] == "prediction_generated"]),
        "",
        "## Reused Existing Predictions",
        table(inventory[inventory["kind"] == "prediction_reused"]),
        "",
        "## Runtime Summary",
        table(runtime),
        "",
        "## Command Runtime Summary",
        table(pd.DataFrame(run_summaries)),
        "",
        "## Row Counts And Alignment",
        table(alignments),
        "",
        "## Failure Summary",
        table(failures),
        "",
        "## Score Range Sanity Checks",
        table(inventory[inventory["kind"] == "prediction_generated"][["dataset", "model", "actual_path", "note"]]),
        "",
        "## Readiness",
        f"AVA complete alignment ready for weight search: {ava_ready}.",
        "",
        f"AADB complete alignment ready for weight search: {aadb_ready}.",
        "",
        "ICAA should still be treated as a separate color subscore until the next step tests its holistic AVA/AADB contribution.",
        "",
        "## Next Weight-Search Command Or Prompt",
        next_prompt,
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_pairs(values: Sequence[str]) -> List[Tuple[str, str]]:
    if not values:
        return list(OUTPUT_FILES.keys())
    pairs = []
    for value in values:
        dataset, model = value.split(":", 1)
        key = (dataset, model)
        if key not in OUTPUT_FILES:
            raise ValueError(f"Unsupported pair: {value}")
        pairs.append(key)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--flush-every", type=int, default=50)
    parser.add_argument("--align-only", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_summaries = []
    if not args.align_only:
        for dataset, model in parse_pairs(args.pairs):
            print(f"running {dataset}/{model} limit={args.limit} resume={args.resume}", flush=True)
            run_summaries.append(run_pair(dataset, model, args.limit, args.resume, args.flush_every))
    alignment_rows = [align_dataset("ava"), align_dataset("aadb")]
    pd.DataFrame(alignment_rows).to_csv(OUT_DIR / "alignment_summary.csv", index=False)
    write_inventory(run_summaries)
    write_failure_summary()
    write_runtime_summary()
    write_report(run_summaries, alignment_rows)
    ava_rows = int(pd.DataFrame(alignment_rows).set_index("dataset").loc["ava", "aligned_complete_rows"])
    aadb_rows = int(pd.DataFrame(alignment_rows).set_index("dataset").loc["aadb", "aligned_complete_rows"])
    print(f"AVA complete alignment row count: {ava_rows}")
    print(f"AADB complete alignment row count: {aadb_rows}")
    for path in OUTPUT_FILES.values():
        print(f"prediction CSV: {path}")
    print(f"aligned AVA: {OUT_DIR / 'aligned_ava_complete_primary.csv'}")
    print(f"aligned AADB: {OUT_DIR / 'aligned_aadb_complete_primary.csv'}")
    print(f"weight search can now proceed: {ava_rows > 0 or aadb_rows > 0}")


if __name__ == "__main__":
    main()
