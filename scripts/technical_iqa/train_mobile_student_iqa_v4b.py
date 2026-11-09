# 모바일 technical IQA v4b 학생 모델을 순위 상관 개선 및 안전성 유지 전략으로 학습하고 평가하는 스크립트
from __future__ import annotations

import argparse
import csv
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402


DEFAULT_MANIFEST = "outputs/student_training_manifest_50k_20260601/student_training_manifest.csv"
DEFAULT_OUT_DIR = "outputs/mobile_student_iqa_v4b_20260601"
SPLITS = ["train", "val", "test"]
DATASET_ORDER = ["flive", "koniq10k", "spaq"]
TARGET_NORM_COL = "target_norm"
MOS_NORM_COL = "mos_norm"
SAFETY_FLAG_COL = "safety_flag"
SAMPLE_WEIGHT_COL = "sample_weight"
PRED_COL = "pred_0_100"
REQUIRED_COLUMNS = [
    "image_path",
    "dataset",
    "student_split",
    "quality_score_0_100",
    "guarded_target_0_100",
]
V3_BASELINE = {
    "test_srcc": 0.658204,
    "test_mae": 4.543416,
    "test_severe_fp": 28.0,
    "flive_test_srcc": 0.514385,
    "koniq10k_test_srcc": 0.643786,
    "spaq_test_srcc": 0.840971,
}
V4A_BASELINE = {
    "test_srcc": 0.630522,
    "test_mae": 5.410899,
    "test_severe_fp": 13.0,
    "flive_test_srcc": 0.449674,
    "koniq10k_test_srcc": 0.639833,
    "spaq_test_srcc": 0.832751,
}


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mobile Student IQA v4b with ranking and safety strategy.")
    parser.add_argument("--manifest_csv", default=DEFAULT_MANIFEST)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--backbone", choices=["efficientnetv2b0", "efficientnetb0", "mobilenetv3small"], default="efficientnetv2b0")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--head_epochs", type=int, default=5)
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--unfreeze_last_n", type=int, default=40)
    parser.add_argument("--learning_rate_head", type=float, default=3e-4)
    parser.add_argument("--learning_rate_finetune", type=float, default=1e-5)
    parser.add_argument("--huber_delta", type=float, default=0.05)
    parser.add_argument("--dataset_sampling_mode", choices=["balanced", "natural"], default="balanced")
    parser.add_argument("--use_sample_weights", type=parse_bool, default=True)
    parser.add_argument("--low_mos_weight", type=float, default=2.0)
    parser.add_argument("--severe_low_mos_weight", type=float, default=3.0)
    parser.add_argument("--safety_lambda", type=float, default=0.75)
    parser.add_argument("--safety_threshold", type=float, default=0.65)
    parser.add_argument("--target_safety_cutoff", type=float, default=55.0)
    parser.add_argument("--mos_safety_cutoff", type=float, default=50.0)
    parser.add_argument("--ranking_lambda", type=float, default=0.25)
    parser.add_argument("--rank_margin_target", type=float, default=0.10)
    parser.add_argument("--rank_margin_pred", type=float, default=0.02)
    parser.add_argument("--max_rank_pairs_per_batch", type=int, default=256)
    parser.add_argument("--mae_penalty", type=float, default=0.01)
    parser.add_argument("--severe_fp_count_penalty", type=float, default=0.02)
    parser.add_argument("--dataset_collapse_penalty", type=float, default=0.5)
    parser.add_argument("--loss", choices=["huber", "mae"], default="huber")
    parser.add_argument("--backbone_weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--enable_posthoc_calibration", type=parse_bool, default=True)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--debug_history", action="store_true")
    parser.add_argument("--seed", type=int, default=20260601)
    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def require_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"manifest_csv is missing required columns: {missing}")


def apply_optional_caps(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    caps = {"train": args.max_train_samples, "val": args.max_val_samples, "test": args.max_test_samples}
    parts = []
    for split in SPLITS:
        split_df = df[df["student_split"] == split].copy()
        cap = caps[split]
        if cap is not None and len(split_df) > cap:
            split_df = split_df.sample(n=cap, random_state=args.seed)
        parts.append(split_df)
    return pd.concat(parts, ignore_index=True)


def validate_image_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    usable_rows = []
    error_rows = []
    for idx, row in df.iterrows():
        image_path = str(row["image_path"])
        error: Optional[str] = None
        if not image_path or image_path.lower() == "nan":
            error = "empty image path"
        elif not Path(image_path).is_file():
            error = "missing image file"
        else:
            for col in ["quality_score_0_100", "guarded_target_0_100"]:
                try:
                    value = float(row[col])
                except Exception:
                    error = f"non-numeric {col}"
                    break
                if not math.isfinite(value):
                    error = f"non-finite {col}"
                    break
        if error is None:
            try:
                with Image.open(image_path) as img:
                    img.convert("RGB")
            except Exception as exc:
                error = f"image load failed: {exc}"
        if error:
            error_rows.append({
                "row_index": idx,
                "dataset": row.get("dataset", ""),
                "student_split": row.get("student_split", ""),
                "image_path": image_path,
                "error": error,
            })
        else:
            usable_rows.append(row)
    return pd.DataFrame(usable_rows).reset_index(drop=True), pd.DataFrame(error_rows)


def add_training_columns(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out["quality_score_0_100"] = pd.to_numeric(out["quality_score_0_100"], errors="coerce")
    out["guarded_target_0_100"] = pd.to_numeric(out["guarded_target_0_100"], errors="coerce")
    out[TARGET_NORM_COL] = np.clip(out["guarded_target_0_100"].to_numpy(dtype=np.float64) / 100.0, 0.0, 1.0)
    out[MOS_NORM_COL] = np.clip(out["quality_score_0_100"].to_numpy(dtype=np.float64) / 100.0, 0.0, 1.0)
    out[SAFETY_FLAG_COL] = (
        (out["quality_score_0_100"] < args.mos_safety_cutoff)
        | (out["guarded_target_0_100"] < args.target_safety_cutoff)
    ).astype(np.float32)
    weights = np.ones(len(out), dtype=np.float32)
    if args.use_sample_weights:
        # Base weight 1.0
        # If MOS < 50, multiply by low_mos_weight
        # If MOS < 40, multiply by severe_low_mos_weight
        # We do this incrementally (not cumulative multiplication for same condition)
        is_low = out["quality_score_0_100"].to_numpy(dtype=np.float64) < 50.0
        is_severe = out["quality_score_0_100"].to_numpy(dtype=np.float64) < 40.0
        
        weights[is_low] = args.low_mos_weight
        weights[is_severe] = args.severe_low_mos_weight
        
        # Normalize weights per dataset to keep loss scale similar
        for dataset in sorted(out["dataset"].astype(str).unique()):
            mask = out["dataset"].astype(str).to_numpy() == dataset
            mean_weight = float(np.mean(weights[mask])) if mask.any() else 1.0
            if mean_weight > 0:
                weights[mask] /= mean_weight
    out[SAMPLE_WEIGHT_COL] = weights
    return out


def load_rgb_resized(path: str, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32)


def y_vector(row: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(row[TARGET_NORM_COL]), float(row[MOS_NORM_COL]), float(row[SAFETY_FLAG_COL])], dtype=np.float32)


def make_eval_dataset(df: pd.DataFrame, image_size: int, batch_size: int, repeat: bool) -> tf.data.Dataset:
    records = df[["image_path", TARGET_NORM_COL, MOS_NORM_COL, SAFETY_FLAG_COL, SAMPLE_WEIGHT_COL]].to_dict("records")

    def generator() -> Iterable[Tuple[np.ndarray, np.ndarray, np.float32]]:
        for row in records:
            yield load_rgb_resized(str(row["image_path"]), image_size), y_vector(row), np.float32(row[SAMPLE_WEIGHT_COL])

    output_signature = (
        tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(3,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature).batch(batch_size)
    if repeat:
        ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)


def make_predict_dataset(df: pd.DataFrame, image_size: int, batch_size: int) -> tf.data.Dataset:
    records = df[["image_path"]].to_dict("records")

    def generator() -> Iterable[np.ndarray]:
        for row in records:
            yield load_rgb_resized(str(row["image_path"]), image_size)

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
    ).batch(batch_size).repeat()
    return ds.prefetch(tf.data.AUTOTUNE)


class BalancedBatchGenerator:
    def __init__(self, df: pd.DataFrame, image_size: int, batch_size: int, seed: int) -> None:
        if batch_size % len(DATASET_ORDER) != 0:
            raise ValueError("--batch_size must be divisible by 3 when using balanced sampling")
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.per_dataset = batch_size // len(DATASET_ORDER)
        self.rng = np.random.default_rng(seed)
        self.indices: dict[str, np.ndarray] = {}
        self.positions: dict[str, int] = {}
        for dataset in DATASET_ORDER:
            dataset_indices = np.flatnonzero(self.df["dataset"].astype(str).to_numpy() == dataset)
            if len(dataset_indices) == 0:
                raise ValueError(f"Cannot build balanced batches: no rows for dataset {dataset}")
            self.indices[dataset] = self.rng.permutation(dataset_indices)
            self.positions[dataset] = 0

    def next_indices(self, dataset: str) -> np.ndarray:
        selected: list[int] = []
        while len(selected) < self.per_dataset:
            available = self.indices[dataset]
            pos = self.positions[dataset]
            remaining = len(available) - pos
            take = min(self.per_dataset - len(selected), remaining)
            selected.extend(available[pos : pos + take].tolist())
            pos += take
            if pos >= len(available):
                available = self.rng.permutation(available)
                pos = 0
            self.indices[dataset] = available
            self.positions[dataset] = pos
        return np.asarray(selected, dtype=np.int64)

    def __iter__(self) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        while True:
            batch_indices = np.concatenate([self.next_indices(dataset) for dataset in DATASET_ORDER])
            self.rng.shuffle(batch_indices)
            images = []
            y_values = []
            weights = []
            for idx in batch_indices:
                row = self.df.iloc[int(idx)].to_dict()
                images.append(load_rgb_resized(str(row["image_path"]), self.image_size))
                y_values.append(y_vector(row))
                weights.append(np.float32(row[SAMPLE_WEIGHT_COL]))
            yield np.stack(images).astype(np.float32), np.stack(y_values).astype(np.float32), np.asarray(weights, dtype=np.float32)


def make_balanced_train_dataset(df: pd.DataFrame, image_size: int, batch_size: int, seed: int) -> tf.data.Dataset:
    generator = BalancedBatchGenerator(df, image_size, batch_size, seed)
    output_signature = (
        tf.TensorSpec(shape=(batch_size, image_size, image_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size,), dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(lambda: iter(generator), output_signature=output_signature).prefetch(tf.data.AUTOTUNE)


def batch_count(num_rows: int, batch_size: int) -> int:
    if num_rows <= 0:
        raise ValueError("Cannot create steps for an empty dataset.")
    return int(math.ceil(num_rows / float(batch_size)))


def backbone_factory(backbone_name: str) -> Any:
    if backbone_name == "efficientnetv2b0":
        return tf.keras.applications.EfficientNetV2B0
    if backbone_name == "efficientnetb0":
        return tf.keras.applications.EfficientNetB0
    if backbone_name == "mobilenetv3small":
        return tf.keras.applications.MobileNetV3Small
    raise ValueError(f"Unsupported backbone: {backbone_name}")


def create_backbone(backbone_name: str, image_size: int, weights_arg: str | None) -> tuple[tf.keras.Model, bool]:
    app = backbone_factory(backbone_name)
    kwargs = {"input_shape": (image_size, image_size, 3), "include_top": False, "weights": weights_arg, "include_preprocessing": True}
    try:
        return app(**kwargs), True
    except TypeError:
        kwargs.pop("include_preprocessing")
        return app(**kwargs), False


def combined_loss(args: argparse.Namespace) -> Any:
    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # y_true: [batch, 3] (target, mos, safety)
        # y_pred: [batch, 1]
        target = y_true[:, 0:1]
        safety = y_true[:, 2:3]
        
        # 1. Base Loss (Huber or MAE)
        error = target - y_pred
        abs_error = tf.abs(error)
        if args.loss == "huber":
            delta = tf.cast(args.huber_delta, y_pred.dtype)
            base = tf.where(abs_error <= delta, 0.5 * tf.square(error), delta * (abs_error - 0.5 * delta))
        else:
            base = abs_error
        base = tf.squeeze(base, axis=-1)

        # 2. Safety Penalty
        # relu(pred - threshold)^2 for safety-risk samples
        penalty = tf.squeeze(safety * tf.square(tf.nn.relu(y_pred - tf.cast(args.safety_threshold, y_pred.dtype))), axis=-1)
        
        # 3. Pairwise Ranking Loss
        # Create pairs within batch
        ti = tf.transpose(target) # [1, batch]
        tj = target # [batch, 1]
        pi = tf.transpose(y_pred) # [1, batch]
        pj = y_pred # [batch, 1]
        
        target_diff = tj - ti # [batch, batch]
        pred_diff = pj - pi # [batch, batch]
        
        # Mask for pairs with enough target difference
        rank_mask = tf.abs(target_diff) >= args.rank_margin_target
        # Remove self-diagonal
        rank_mask = tf.logical_and(rank_mask, tf.logical_not(tf.eye(tf.shape(target)[0], dtype=tf.bool)))
        
        s = tf.sign(target_diff)
        pair_loss = tf.nn.relu(args.rank_margin_pred - s * pred_diff)
        
        masked_rank_loss = tf.where(rank_mask, pair_loss, tf.zeros_like(pair_loss))
        
        # Limit pairs if requested
        if args.max_rank_pairs_per_batch > 0:
            flat_mask = tf.reshape(rank_mask, [-1])
            indices = tf.where(flat_mask)
            num_valid = tf.shape(indices)[0]
            take = tf.minimum(num_valid, args.max_rank_pairs_per_batch)
            # Just take first N valid pairs in graph mode for efficiency
            selected_indices = indices[:take]
            flat_rank_loss = tf.reshape(masked_rank_loss, [-1])
            selected_losses = tf.gather(flat_rank_loss, selected_indices)
            batch_rank_loss = tf.cond(
                take > 0,
                lambda: tf.reduce_mean(selected_losses),
                lambda: tf.constant(0.0, dtype=y_pred.dtype)
            )
        else:
            num_valid = tf.reduce_sum(tf.cast(rank_mask, tf.float32))
            batch_rank_loss = tf.cond(
                num_valid > 0.0,
                lambda: tf.reduce_sum(masked_rank_loss) / num_valid,
                lambda: tf.constant(0.0, dtype=y_pred.dtype)
            )
        
        # Broadcast batch ranking loss to each sample
        return base + tf.cast(args.safety_lambda, y_pred.dtype) * penalty + tf.cast(args.ranking_lambda, y_pred.dtype) * batch_rank_loss

    return loss_fn


def target_mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.abs(y_true[:, 0:1] - y_pred))


def target_rmse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.sqrt(tf.reduce_mean(tf.square(y_true[:, 0:1] - y_pred)))


def compile_model(model: tf.keras.Model, learning_rate: float, args: argparse.Namespace) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=combined_loss(args),
        metrics=[target_mae, target_rmse],
    )


def build_model(args: argparse.Namespace) -> Tuple[tf.keras.Model, tf.keras.Model, Dict[str, Any]]:
    weights_arg = None if args.backbone_weights == "none" else "imagenet"
    build_info: Dict[str, Any] = {
        "backbone": args.backbone,
        "requested_backbone_weights": args.backbone_weights,
        "backbone_weights_loaded": False,
        "backbone_weights_error": "",
        "include_preprocessing": True,
        "preprocessing_note": "Input generator emits RGB float32 0..255. Keras application preprocessing is included when supported.",
        "output_activation": "sigmoid",
    }
    try:
        backbone, include_preprocessing = create_backbone(args.backbone, args.image_size, weights_arg)
        build_info["backbone_weights_loaded"] = weights_arg == "imagenet"
        build_info["include_preprocessing"] = include_preprocessing
    except Exception as exc:
        if weights_arg != "imagenet":
            raise
        build_info["backbone_weights_error"] = repr(exc)
        backbone, include_preprocessing = create_backbone(args.backbone, args.image_size, None)
        build_info["include_preprocessing"] = include_preprocessing
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(args.image_size, args.image_size, 3), name="image")
    x = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="regression_dense_256")(x)
    x = tf.keras.layers.Dropout(0.3, name="regression_dropout_03")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="regression_dense_64")(x)
    x = tf.keras.layers.Dropout(0.2, name="regression_dropout_02")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="score_norm")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobile_student_iqa_v4b")
    compile_model(model, args.learning_rate_head, args)
    return model, backbone, build_info


def configure_stage_a(backbone: tf.keras.Model) -> Dict[str, int]:
    backbone.trainable = False
    for layer in backbone.layers:
        layer.trainable = False
    return {"backbone_layers": len(backbone.layers), "backbone_trainable_layers": 0, "batch_norm_trainable_layers": 0}


def configure_stage_b(backbone: tf.keras.Model, unfreeze_last_n: int) -> Dict[str, int]:
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    selected_layers = backbone.layers[-max(unfreeze_last_n, 0):] if unfreeze_last_n > 0 else []
    for layer in selected_layers:
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
    return {
        "backbone_layers": len(backbone.layers),
        "requested_unfreeze_last_n": unfreeze_last_n,
        "backbone_trainable_layers": sum(int(layer.trainable) for layer in backbone.layers),
        "batch_norm_trainable_layers": sum(int(layer.trainable) for layer in backbone.layers if isinstance(layer, tf.keras.layers.BatchNormalization)),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    return pearson_corr(rankdata(y_true[mask]), rankdata(y_pred[mask]))


def regression_metrics(df: pd.DataFrame, target_col: str, split: str, dataset: str) -> Dict[str, Any]:
    y_true = df[target_col].to_numpy(dtype=np.float64)
    y_pred = df[PRED_COL].to_numpy(dtype=np.float64)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() == 0:
        return {"split": split, "dataset": dataset, "n": len(df), "finite_n": 0, "mae": float("nan"), "rmse": float("nan"), "srcc": float("nan"), "plcc": float("nan"), "pred_min": float("nan"), "pred_max": float("nan"), "pred_mean": float("nan"), "pred_std": float("nan")}
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    return {
        "split": split,
        "dataset": dataset,
        "n": len(df),
        "finite_n": int(finite.sum()),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "rmse": float(np.sqrt(np.mean(np.square(y_pred - y_true)))),
        "srcc": spearman_corr(y_true, y_pred),
        "plcc": pearson_corr(y_true, y_pred),
        "pred_min": float(np.min(y_pred)),
        "pred_max": float(np.max(y_pred)),
        "pred_mean": float(np.mean(y_pred)),
        "pred_std": float(np.std(y_pred)),
    }


def safety_counts(df: pd.DataFrame) -> Dict[str, int]:
    mos = df["quality_score_0_100"].to_numpy(dtype=np.float64)
    pred = df[PRED_COL].to_numpy(dtype=np.float64)
    return {
        "fp_count": int(((mos < 50.0) & (pred > 65.0)).sum()),
        "severe_fp_count": int(((mos < 40.0) & (pred > 65.0)).sum()),
        "fn_count": int(((mos > 75.0) & (pred < 45.0)).sum()),
    }


def safety_metrics(df: pd.DataFrame, split: str, dataset: str) -> Dict[str, Any]:
    counts = safety_counts(df)
    return {"split": split, "dataset": dataset, "n": len(df), "finite_n": int(np.isfinite(df[PRED_COL].to_numpy(dtype=np.float64)).sum()), **counts}


def predict_split(model: tf.keras.Model, df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    ds = make_predict_dataset(df, args.image_size, args.batch_size)
    pred_norm = model.predict(ds, steps=batch_count(len(df), args.batch_size), verbose=1).reshape(-1)
    out = df.copy()
    out["pred_norm"] = np.clip(pred_norm.astype(np.float64), 0.0, 1.0)
    out[PRED_COL] = out["pred_norm"] * 100.0
    out["abs_error_to_target"] = np.abs(out[PRED_COL].to_numpy(dtype=np.float64) - out["guarded_target_0_100"].to_numpy(dtype=np.float64))
    out["abs_error_to_mos"] = np.abs(out[PRED_COL].to_numpy(dtype=np.float64) - out["quality_score_0_100"].to_numpy(dtype=np.float64))
    return out


class CompositeCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_df: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> None:
        super().__init__()
        self.val_df = val_df.reset_index(drop=True)
        self.args = args
        self.out_dir = out_dir
        self.val_steps = batch_count(len(val_df), args.batch_size)
        self.phase = "head"
        self.global_offset = 0
        self.best_composite = -float("inf")
        self.best_mae = float("inf")
        self.best_srcc = -float("inf")
        self.best_safety = float("inf")
        self.records: list[dict[str, Any]] = []

    def set_phase(self, phase: str, global_offset: int) -> None:
        self.phase = phase
        self.global_offset = global_offset

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:
        logs = logs or {}
        pred_norm = self.model.predict(
            make_predict_dataset(self.val_df, self.args.image_size, self.args.batch_size),
            steps=self.val_steps,
            verbose=0,
        ).reshape(-1)
        pred_df = self.val_df.copy()
        pred_df["pred_norm"] = np.clip(pred_norm.astype(np.float64), 0.0, 1.0)
        pred_df[PRED_COL] = pred_df["pred_norm"] * 100.0
        
        # Overall metrics
        metrics = regression_metrics(pred_df, "guarded_target_0_100", "val", "overall")
        counts = safety_counts(pred_df)
        
        # Dataset-wise SRCC for collapse detection
        ds_srccs = []
        ds_metrics = {}
        for ds in DATASET_ORDER:
            subset = pred_df[pred_df["dataset"] == ds]
            if not subset.empty:
                m = regression_metrics(subset, "guarded_target_0_100", "val", ds)
                ds_srccs.append(float(m["srcc"]))
                ds_metrics[f"val_{ds}_srcc"] = m["srcc"]
            else:
                ds_srccs.append(-1.0)
                ds_metrics[f"val_{ds}_srcc"] = float("nan")
        
        val_dataset_srcc_min = min(ds_srccs) if ds_srccs else -1.0
        
        # Composite score calculation
        # composite = val_srcc - mae_penalty * val_mae - severe_fp_count_penalty * val_severe_fp_count - dataset_collapse_penalty * max(0, 0.50 - val_dataset_srcc_min)
        val_srcc = float(metrics["srcc"])
        val_mae = float(metrics["mae"])
        severe_fp_count = float(counts["severe_fp_count"])
        
        collapse_penalty = self.args.dataset_collapse_penalty * max(0.0, 0.50 - val_dataset_srcc_min)
        composite = val_srcc - self.args.mae_penalty * val_mae - self.args.severe_fp_count_penalty * severe_fp_count - collapse_penalty
        
        saved_best_composite = composite > self.best_composite
        saved_best_mae = val_mae < self.best_mae
        saved_best_srcc = val_srcc > self.best_srcc
        saved_best_safety = severe_fp_count < self.best_safety
        
        if saved_best_composite:
            self.best_composite = composite
            self.model.save(self.out_dir / "best_model.keras", include_optimizer=False)
        if saved_best_mae:
            self.best_mae = val_mae
            self.model.save(self.out_dir / "best_by_val_mae_model.keras", include_optimizer=False)
        if saved_best_srcc:
            self.best_srcc = val_srcc
            self.model.save(self.out_dir / "best_by_val_srcc_model.keras", include_optimizer=False)
        if saved_best_safety:
            self.best_safety = severe_fp_count
            self.model.save(self.out_dir / "best_by_val_safety_model.keras", include_optimizer=False)
            
        global_epoch = self.global_offset + epoch + 1
        record = {
            "global_epoch": global_epoch,
            "phase": self.phase,
            "val_mae": metrics["mae"],
            "val_rmse": metrics["rmse"],
            "val_srcc": metrics["srcc"],
            "val_plcc": metrics["plcc"],
            "val_fp_count": counts["fp_count"],
            "val_severe_fp_count": counts["severe_fp_count"],
            "val_fn_count": counts["fn_count"],
            "val_dataset_srcc_min": val_dataset_srcc_min,
            **ds_metrics,
            "composite_score": composite,
            "saved_best_composite": saved_best_composite,
            "saved_best_mae": saved_best_mae,
            "saved_best_srcc": saved_best_srcc,
            "saved_best_safety": saved_best_safety,
        }
        self.records.append(record)
        for key, value in record.items():
            if isinstance(value, (int, float, np.floating, np.integer, bool)):
                logs[key] = value
        pd.DataFrame(self.records).to_csv(self.out_dir / "checkpoint_metrics.csv", index=False)


def history_frame(history: tf.keras.callbacks.History, phase: str, start_epoch: int) -> pd.DataFrame:
    history_dict = {key: list(values) for key, values in history.history.items()}
    max_len = max((len(values) for values in history_dict.values()), default=0)
    normalized = {key: values + [float("nan")] * (max_len - len(values)) for key, values in history_dict.items()}
    df = pd.DataFrame(normalized)
    if "phase" not in df.columns:
        df.insert(0, "phase", phase)
    if "phase_epoch" not in df.columns:
        df.insert(1, "phase_epoch", np.arange(1, len(df) + 1))
    if "global_epoch" not in df.columns:
        df.insert(2, "global_epoch", np.arange(start_epoch, start_epoch + len(df)))
    return df


def debug_history(history: tf.keras.callbacks.History, phase: str) -> None:
    print(f"[debug_history] phase={phase}")
    print(f"[debug_history] keys={list(history.history.keys())}")
    for key, values in history.history.items():
        print(f"[debug_history] {phase}.{key}={values}")


def severe_fp_cases(pred_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split in ["val", "test"]:
        df = pred_frames[split]
        fp = df[(df["quality_score_0_100"] < 50.0) & (df[PRED_COL] > 65.0)].copy()
        for _, row in fp.iterrows():
            rows.append({
                "fp_type": "severe_fp" if float(row["quality_score_0_100"]) < 40.0 else "fp",
                "dataset": row.get("dataset", ""),
                "split": split,
                "image_path": row.get("image_path", ""),
                "source_group": row.get("source_group", ""),
                "quality_score_0_100": row.get("quality_score_0_100", ""),
                "guarded_target_0_100": row.get("guarded_target_0_100", ""),
                "pred_0_100": row.get(PRED_COL, ""),
                "absolute_error_to_target": row.get("abs_error_to_target", ""),
            })
    columns = ["fp_type", "dataset", "split", "image_path", "source_group", "quality_score_0_100", "guarded_target_0_100", "pred_0_100", "absolute_error_to_target"]
    return pd.DataFrame(rows, columns=columns).sort_values(["fp_type", "absolute_error_to_target"], ascending=[False, False]) if rows else pd.DataFrame(columns=columns)


def markdown_table(df: pd.DataFrame, columns: List[str]) -> str:
    if df.empty:
        return "_No rows._"
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df[columns].iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def final_decision(overall_metrics: pd.DataFrame, dataset_metrics: pd.DataFrame, safety_df: pd.DataFrame, load_errors: pd.DataFrame) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    test_row = overall_metrics[overall_metrics["split"] == "test"].iloc[0]
    test_srcc = float(test_row["srcc"])
    test_mae = float(test_row["mae"])
    finite_ok = int(test_row["finite_n"]) == int(test_row["n"])
    range_ok = 0.0 <= float(test_row["pred_min"]) <= 100.0 and 0.0 <= float(test_row["pred_max"]) <= 100.0
    severe_fp = int(safety_df[(safety_df["split"] == "test") & (safety_df["dataset"] == "overall")]["severe_fp_count"].sum())
    no_dataset_collapse = bool((dataset_metrics["srcc"].fillna(-1.0) >= 0.50).all())
    load_failure_rate = len(load_errors) / max(int(overall_metrics["n"].sum()), 1)
    if test_srcc < 0.85:
        reasons.append(f"Test SRCC {test_srcc:.4f} is below 0.85.")
    if test_mae > 10.0:
        reasons.append(f"Test MAE {test_mae:.4f} is above 10.")
    if severe_fp != 0:
        reasons.append(f"Test severe FP count is {severe_fp}, expected 0.")
    if not no_dataset_collapse:
        reasons.append("At least one test dataset SRCC is below 0.50.")
    if not finite_ok or not range_ok:
        reasons.append("Predictions are non-finite or outside 0..100.")
    if load_failure_rate > 0.001:
        reasons.append(f"Image load failure rate {load_failure_rate:.6f} is not negligible.")
    if not reasons:
        return "READY_FOR_BASELINE_COMPARISON", ["All READY criteria passed."]
    if (not finite_ok) or severe_fp > V3_BASELINE["test_severe_fp"] * 2 or (not range_ok):
        return "DO_NOT_CONTINUE_WITH_THIS_STUDENT", reasons
    return "NEEDS_MORE_TUNING", reasons


def evaluate_posthoc_calibration(val_preds, val_targets, test_preds, test_targets):
    try:
        from sklearn.isotonic import IsotonicRegression
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(val_preds, val_targets)
        cal_test_preds = calibrator.transform(test_preds)
        method = "isotonic"
    except ImportError:
        # Fallback to linear regression
        coeffs = np.polyfit(val_preds, val_targets, 1)
        cal_test_preds = np.polyval(coeffs, test_preds)
        method = "linear"
    
    cal_test_preds = np.clip(cal_test_preds, 0.0, 100.0)
    mae = np.mean(np.abs(cal_test_preds - test_targets))
    rmse = np.sqrt(np.mean(np.square(cal_test_preds - test_targets)))
    # SRCC won't change with monotonic transformation like isotonic
    srcc = spearman_corr(test_targets, cal_test_preds)
    plcc = pearson_corr(test_targets, cal_test_preds)
    
    return cal_test_preds, {
        "method": method,
        "cal_mae": mae,
        "cal_rmse": rmse,
        "cal_srcc": srcc,
        "cal_plcc": plcc
    }


def write_report(
    path: Path,
    args: argparse.Namespace,
    build_info: Dict[str, Any],
    stage_info: Dict[str, Dict[str, int]],
    split_counts: pd.Series,
    train_df: pd.DataFrame,
    load_errors: pd.DataFrame,
    history: pd.DataFrame,
    checkpoint_metrics: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    dataset_metrics: pd.DataFrame,
    mos_metrics: pd.DataFrame,
    safety_df: pd.DataFrame,
    severe_cases: pd.DataFrame,
    calibration_info: Optional[Dict[str, Any]],
    decision: str,
    decision_reasons: List[str],
) -> None:
    metric_cols = ["split", "dataset", "n", "finite_n", "mae", "rmse", "srcc", "plcc", "pred_min", "pred_max", "pred_mean", "pred_std"]
    safety_cols = ["split", "dataset", "n", "finite_n", "fp_count", "severe_fp_count", "fn_count"]
    test_overall = overall_metrics[overall_metrics["split"] == "test"].iloc[0]
    test_safety = safety_df[(safety_df["split"] == "test") & (safety_df["dataset"] == "overall")].iloc[0]
    
    comparison = pd.DataFrame([
        {"metric": "test_srcc", "v3": V3_BASELINE["test_srcc"], "v4a": V4A_BASELINE["test_srcc"], "v4b": float(test_overall["srcc"])},
        {"metric": "test_mae", "v3": V3_BASELINE["test_mae"], "v4a": V4A_BASELINE["test_mae"], "v4b": float(test_overall["mae"])},
        {"metric": "test_severe_fp", "v3": V3_BASELINE["test_severe_fp"], "v4a": V4A_BASELINE["test_severe_fp"], "v4b": float(test_safety["severe_fp_count"])},
    ])
    for dataset, key in [("flive", "flive_test_srcc"), ("koniq10k", "koniq10k_test_srcc"), ("spaq", "spaq_test_srcc")]:
        row = dataset_metrics[dataset_metrics["dataset"] == dataset]
        v4b_value = float(row.iloc[0]["srcc"]) if not row.empty else float("nan")
        comparison.loc[len(comparison)] = {"metric": key, "v3": V3_BASELINE[key], "v4a": V4A_BASELINE[key], "v4b": v4b_value}
    
    train_distribution = train_df["dataset"].value_counts().to_dict()
    balanced_batches_enabled = getattr(args, "balanced_batches", None)
    if balanced_batches_enabled is None:
        balanced_batches_enabled = getattr(args, "dataset_sampling_mode", "balanced") == "balanced"

    lines = [
        "# Mobile Student IQA v4b Training Report",
        "",
        "## 1. Summary",
        f"- Final decision: `{decision}`",
        f"- Input manifest: `{args.manifest_csv}`",
        f"- Output directory: `{args.out_dir}`",
        "",
        "## 2. Dataset",
        f"- Split counts after image validation: `{split_counts.to_dict()}`",
        f"- Image load failures: `{len(load_errors)}`",
        "",
        "## 3. Sampling and Weights",
        f"- dataset_sampling_mode: `{args.dataset_sampling_mode}`",
        f"- balanced_batches_v4a_logic: `{balanced_batches_enabled}`",
        f"- batch_size: `{args.batch_size}`",
        f"- train dataset distribution before balancing: `{train_distribution}`",
        f"- sample weights enabled: `{args.use_sample_weights}`",
        f"- low_mos_weight: `{args.low_mos_weight}`",
        f"- severe_low_mos_weight: `{args.severe_low_mos_weight}`",
        "",
        "## 4. Model",
        f"- Backbone: `{build_info['backbone']}`",
        f"- Input size: `{args.image_size}x{args.image_size}x3`",
        "- Head: `GAP -> Dense(256, relu) -> Dropout(0.3) -> Dense(64, relu) -> Dropout(0.2) -> Dense(1, sigmoid)`",
        f"- Include preprocessing in Keras application: `{build_info['include_preprocessing']}`",
        "",
        "## 5. Training Configuration",
        f"- Stage A head epochs: `{args.head_epochs}`",
        f"- Stage B fine-tune epochs: `{args.finetune_epochs}`",
        f"- Stage B trainable backbone layers: `{stage_info['finetune']['backbone_trainable_layers']}`",
        f"- Training history rows: `{len(history)}`",
        "",
        "## 6. Loss Components",
        f"- base loss: `{args.loss}`",
        f"- huber_delta: `{args.huber_delta}`",
        f"- safety_lambda: `{args.safety_lambda}`",
        f"- safety_threshold: `{args.safety_threshold}` (prediction cutoff)",
        f"- target_safety_cutoff: `{args.target_safety_cutoff}` (guarded target cutoff)",
        f"- mos_safety_cutoff: `{args.mos_safety_cutoff}` (MOS cutoff)",
        "",
        "## 7. Ranking Loss",
        f"- ranking_lambda: `{args.ranking_lambda}`",
        f"- rank_margin_target: `{args.rank_margin_target}`",
        f"- rank_margin_pred: `{args.rank_margin_pred}`",
        f"- max_rank_pairs_per_batch: `{args.max_rank_pairs_per_batch}`",
        "",
        "## 8. Checkpoint Selection",
        "- Composite checkpointing used: `True`",
        f"- mae_penalty: `{args.mae_penalty}`",
        f"- severe_fp_count_penalty: `{args.severe_fp_count_penalty}`",
        f"- dataset_collapse_penalty: `{args.dataset_collapse_penalty}`",
        "",
        "## 9. Training History Summary",
        f"- best validation composite: `{checkpoint_metrics['composite_score'].max() if not checkpoint_metrics.empty else 'n/a'}`",
        f"- best validation SRCC: `{checkpoint_metrics['val_srcc'].max() if not checkpoint_metrics.empty else 'n/a'}`",
        f"- best validation severe FP count: `{checkpoint_metrics['val_severe_fp_count'].min() if not checkpoint_metrics.empty else 'n/a'}`",
        "",
        "## 10. Overall Metrics Against Guarded Target",
        markdown_table(overall_metrics, metric_cols),
        "",
        "## 11. Dataset-wise Metrics Against Guarded Target",
        markdown_table(dataset_metrics, metric_cols),
        "",
        "## 12. MOS Reference Metrics",
        markdown_table(mos_metrics, metric_cols),
        "",
        "## 13. Safety Metrics",
        markdown_table(safety_df, safety_cols),
        "",
        "## 14. Post-hoc Calibration Evaluation",
    ]
    if calibration_info:
        lines.extend([
            f"- method: `{calibration_info['method']}`",
            f"- cal_mae: `{calibration_info['cal_mae']:.6f}`",
            f"- cal_srcc: `{calibration_info['cal_srcc']:.6f}`",
            f"- cal_plcc: `{calibration_info['cal_plcc']:.6f}`",
            "",
            "Note: Calibration is for analysis and does not change the raw saved model.",
        ])
    else:
        lines.append("- Post-hoc calibration disabled or failed.")
        
    lines.extend([
        "",
        "## 15. Severe False Positive Cases",
        f"- Severe/standard FP case rows saved: `{len(severe_cases)}`",
        "",
        "## 16. v3/v4a Comparison",
        markdown_table(comparison, ["metric", "v3", "v4a", "v4b"]),
        "",
        "## 17. Risks",
        "- Ranking loss might increase MAE while improving SRCC.",
        "- EfficientNetV2B0 at 256 might still be resolution-limited for some teacher features.",
        "",
        "## 18. Final Decision",
        f"- `{decision}`",
    ])
    lines.extend([f"- {reason}" for reason in decision_reasons])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.dataset_sampling_mode == "balanced" and args.batch_size % 3 != 0:
        raise ValueError("--batch_size must be divisible by 3 when --dataset_sampling_mode balanced")
    set_reproducibility(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest_csv)
    require_columns(manifest)
    manifest = manifest[manifest["student_split"].isin(SPLITS)].copy()
    manifest = apply_optional_caps(manifest, args)
    usable_df, load_errors = validate_image_rows(manifest)
    usable_df = add_training_columns(usable_df, args)
    load_errors.to_csv(out_dir / "load_errors.csv", index=False)

    train_df = usable_df[usable_df["student_split"] == "train"].reset_index(drop=True)
    val_df = usable_df[usable_df["student_split"] == "val"].reset_index(drop=True)
    test_df = usable_df[usable_df["student_split"] == "test"].reset_index(drop=True)
    train_steps = batch_count(len(train_df), args.batch_size)
    val_steps = batch_count(len(val_df), args.batch_size)

    if args.dataset_sampling_mode == "balanced":
        train_ds = make_balanced_train_dataset(train_df, args.image_size, args.batch_size, args.seed)
    else:
        train_ds = make_eval_dataset(train_df, args.image_size, args.batch_size, repeat=True)
    val_ds = make_eval_dataset(val_df, args.image_size, args.batch_size, repeat=True)

    model, backbone, build_info = build_model(args)
    checkpoint = CompositeCheckpointCallback(val_df, args, out_dir)
    callbacks = [checkpoint, tf.keras.callbacks.ReduceLROnPlateau(monitor="val_target_mae", factor=0.5, patience=3, min_lr=1e-7, mode="min")]

    stage_info = {"head": configure_stage_a(backbone)}
    compile_model(model, args.learning_rate_head, args)
    checkpoint.set_phase("head", 0)
    head_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        initial_epoch=0,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )

    stage_info["finetune"] = configure_stage_b(backbone, args.unfreeze_last_n)
    compile_model(model, args.learning_rate_finetune, args)
    checkpoint.set_phase("finetune", len(head_history.history.get("loss", [])))
    finetune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.finetune_epochs,
        initial_epoch=0,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )

    history = pd.concat([
        history_frame(head_history, "head", 1),
        history_frame(finetune_history, "finetune", len(head_history.history.get("loss", [])) + 1),
    ], ignore_index=True)
    history.to_csv(out_dir / "training_history.csv", index=False)

    model.save(out_dir / "final_model.keras", include_optimizer=False)
    best_path = out_dir / "best_model.keras"
    eval_model = tf.keras.models.load_model(best_path, compile=False) if best_path.exists() else model

    pred_frames = {
        "train": predict_split(eval_model, train_df, args),
        "val": predict_split(eval_model, val_df, args),
        "test": predict_split(eval_model, test_df, args),
    }
    pred_cols = ["image_path", "dataset", "student_split", "source_group", "quality_score_0_100", "guarded_target_0_100", "pred_norm", "pred_0_100", "abs_error_to_target", "abs_error_to_mos"]
    for split, pred_df in pred_frames.items():
        cols = [col for col in pred_cols if col in pred_df.columns]
        pred_df.to_csv(out_dir / f"predictions_{split}.csv", columns=cols, index=False)

    overall_metrics = pd.DataFrame([regression_metrics(pred_frames[split], "guarded_target_0_100", split, "overall") for split in SPLITS])
    dataset_metrics = pd.DataFrame([
        regression_metrics(group, "guarded_target_0_100", "test", str(dataset))
        for dataset, group in pred_frames["test"].groupby("dataset", sort=True)
    ])
    mos_metrics = pd.DataFrame([regression_metrics(pred_frames["test"], "quality_score_0_100", "test", "overall")] + [
        regression_metrics(group, "quality_score_0_100", "test", str(dataset)) for dataset, group in pred_frames["test"].groupby("dataset", sort=True)
    ])
    safety_rows = [safety_metrics(pred_frames[split], split, "overall") for split in ["val", "test"]]
    for split in ["val", "test"]:
        safety_rows.extend(safety_metrics(group, split, str(dataset)) for dataset, group in pred_frames[split].groupby("dataset", sort=True))
    safety_df = pd.DataFrame(safety_rows)
    severe_cases = severe_fp_cases(pred_frames)
    severe_cases.to_csv(out_dir / "severe_false_positive_cases.csv", index=False)

    calibration_info = None
    if args.enable_posthoc_calibration:
        val_p = pred_frames["val"][PRED_COL].to_numpy()
        val_t = pred_frames["val"]["guarded_target_0_100"].to_numpy()
        test_p = pred_frames["test"][PRED_COL].to_numpy()
        test_t = pred_frames["test"]["guarded_target_0_100"].to_numpy()
        cal_preds, calibration_info = evaluate_posthoc_calibration(val_p, val_t, test_p, test_t)
        cal_test_df = pred_frames["test"].copy()
        cal_test_df["pred_0_100_calibrated"] = cal_preds
        cal_test_df.to_csv(out_dir / "predictions_test_calibrated.csv", index=False)
        pd.DataFrame([calibration_info]).to_csv(out_dir / "posthoc_calibration_metrics.csv", index=False)

    decision, reasons = final_decision(overall_metrics, dataset_metrics, safety_df, load_errors)
    write_report(
        out_dir / "report.md",
        args,
        build_info,
        stage_info,
        usable_df["student_split"].value_counts(),
        train_df,
        load_errors,
        history,
        pd.DataFrame(checkpoint.records),
        overall_metrics,
        dataset_metrics,
        mos_metrics,
        safety_df,
        severe_cases,
        calibration_info,
        decision,
        reasons,
    )
    print(f"Training complete. Final decision: {decision}")
    print(f"Report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
