from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.distill.data import make_student_dataset


def predict_frame(
    model: tf.keras.Model,
    frame: pd.DataFrame,
    image_size: int,
    batch_size: int,
    target_col: str,
) -> np.ndarray:
    dataset = make_student_dataset(
        frame=frame,
        image_size=image_size,
        batch_size=batch_size,
        target_col=target_col,
        training=False,
        shuffle_buffer=1,
        augment_horizontal_flip=False,
        cache=False,
    )
    predictions = model.predict(dataset, verbose=0)
    return np.asarray(predictions, dtype=np.float32).reshape(-1)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    if float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mse = float(np.mean(np.square(error)))

    total_variance = float(np.sum(np.square(y_true - np.mean(y_true))))
    explained_error = float(np.sum(np.square(error)))
    r2 = None if total_variance <= 1e-12 else float(1.0 - explained_error / total_variance)

    true_ranks = pd.Series(y_true).rank(method="average").to_numpy(dtype=np.float32)
    pred_ranks = pd.Series(y_pred).rank(method="average").to_numpy(dtype=np.float32)

    return {
        "num_samples": int(y_true.size),
        "mae": mae,
        "rmse": rmse,
        "mse": mse,
        "r2": r2,
        "pearson": _safe_corr(y_true, y_pred),
        "spearman": _safe_corr(true_ranks, pred_ranks),
    }


def sample_pairwise_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    margin: float,
    max_pairs: int,
    seed: int = 42,
) -> dict[str, float | int | None]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    if y_true.size < 2:
        return {
            "pairwise_accuracy": None,
            "pairwise_pairs_scored": 0,
        }

    rng = np.random.default_rng(seed)
    target_pairs = int(max(0, max_pairs))
    if target_pairs == 0:
        return {
            "pairwise_accuracy": None,
            "pairwise_pairs_scored": 0,
        }

    pairs_scored = 0
    correct = 0
    attempts = 0
    max_attempts = max(target_pairs * 20, target_pairs)
    while pairs_scored < target_pairs and attempts < max_attempts:
        left = int(rng.integers(0, y_true.size))
        right = int(rng.integers(0, y_true.size - 1))
        if right >= left:
            right += 1
        attempts += 1

        teacher_delta = float(y_true[left] - y_true[right])
        if abs(teacher_delta) < float(margin):
            continue

        student_delta = float(y_pred[left] - y_pred[right])
        if (teacher_delta > 0.0 and student_delta > 0.0) or (teacher_delta < 0.0 and student_delta < 0.0):
            correct += 1
        pairs_scored += 1

    accuracy = None if pairs_scored == 0 else float(correct / pairs_scored)
    return {
        "pairwise_accuracy": accuracy,
        "pairwise_pairs_scored": int(pairs_scored),
    }


def top_k_overlap(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> dict[str, float | int | None]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    k = int(k)
    if y_true.size == 0 or k <= 0:
        return {"top_k": 0, "top_k_overlap": None}
    actual_k = min(k, y_true.size)
    teacher_top = set(np.argsort(-y_true)[:actual_k].tolist())
    student_top = set(np.argsort(-y_pred)[:actual_k].tolist())
    overlap = len(teacher_top & student_top) / float(actual_k)
    return {
        "top_k": int(actual_k),
        "top_k_overlap": float(overlap),
    }


def build_predictions_frame(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    target_col: str,
) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    out["student_prediction"] = np.asarray(predictions, dtype=np.float32).reshape(-1)
    out["signed_error"] = out["student_prediction"] - out[target_col].astype("float32")
    out["abs_error"] = out["signed_error"].abs()
    out["teacher_rank"] = out[target_col].rank(method="first", ascending=False).astype(int)
    out["student_rank"] = out["student_prediction"].rank(method="first", ascending=False).astype(int)
    ordered_columns = [
        column
        for column in [
            "image_path",
            "split",
            "dataset",
            "relative_path",
            target_col,
            "student_prediction",
            "signed_error",
            "abs_error",
            "teacher_rank",
            "student_rank",
            "aadb_score",
            "nima_mean_score",
            "alamp_score",
            "musiq_score",
            "rgnet_score",
        ]
        if column in out.columns
    ]
    remaining_columns = [column for column in out.columns if column not in ordered_columns]
    return out[ordered_columns + remaining_columns]


def evaluate_model_on_frame(
    model: tf.keras.Model,
    frame: pd.DataFrame,
    image_size: int,
    batch_size: int,
    target_col: str,
    pairwise_margin: float,
    pairwise_max_pairs: int,
    top_k: int,
    seed: int,
) -> tuple[dict[str, float | int | None], pd.DataFrame]:
    predictions = predict_frame(
        model=model,
        frame=frame,
        image_size=image_size,
        batch_size=batch_size,
        target_col=target_col,
    )
    y_true = frame[target_col].astype("float32").to_numpy()
    metrics = compute_regression_metrics(y_true=y_true, y_pred=predictions)
    metrics.update(
        sample_pairwise_accuracy(
            y_true=y_true,
            y_pred=predictions,
            margin=pairwise_margin,
            max_pairs=pairwise_max_pairs,
            seed=seed,
        )
    )
    metrics.update(top_k_overlap(y_true=y_true, y_pred=predictions, k=top_k))
    predictions_frame = build_predictions_frame(frame=frame, predictions=predictions, target_col=target_col)
    return metrics, predictions_frame


def write_evaluation_artifacts(
    output_dir: str | Path,
    split_name: str,
    metrics: dict[str, float | int | None],
    predictions_frame: pd.DataFrame,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / f"{split_name}_metrics.json"
    predictions_path = output_path / f"{split_name}_predictions.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    predictions_frame.to_csv(predictions_path, index=False)
    return metrics_path, predictions_path


class ValidationRankingMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        *,
        frame: pd.DataFrame,
        image_size: int,
        batch_size: int,
        target_col: str,
        pairwise_margin: float,
        pairwise_max_pairs: int,
        top_k: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.frame = frame
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        self.target_col = str(target_col)
        self.pairwise_margin = float(pairwise_margin)
        self.pairwise_max_pairs = int(pairwise_max_pairs)
        self.top_k = int(top_k)
        self.seed = int(seed)
        self.metric_history: list[dict[str, object]] = []

    def on_epoch_end(self, epoch: int, logs: dict[str, object] | None = None) -> None:
        if logs is None:
            logs = {}
        metrics, _ = evaluate_model_on_frame(
            model=self.model,
            frame=self.frame,
            image_size=self.image_size,
            batch_size=self.batch_size,
            target_col=self.target_col,
            pairwise_margin=self.pairwise_margin,
            pairwise_max_pairs=self.pairwise_max_pairs,
            top_k=self.top_k,
            seed=self.seed + int(epoch),
        )
        logs["val_pearson"] = float(metrics["pearson"]) if metrics.get("pearson") is not None else float("-inf")
        logs["val_spearman"] = float(metrics["spearman"]) if metrics.get("spearman") is not None else float("-inf")
        logs["val_pairwise_accuracy"] = (
            float(metrics["pairwise_accuracy"]) if metrics.get("pairwise_accuracy") is not None else 0.0
        )
        logs["val_top_k_overlap"] = float(metrics["top_k_overlap"]) if metrics.get("top_k_overlap") is not None else 0.0
        self.metric_history.append(
            {
                "epoch": int(epoch) + 1,
                **metrics,
            }
        )
