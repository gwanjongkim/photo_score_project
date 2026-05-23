# DistortionGuard-IQA v1 Stage B 모델을 기술 IQA 테스트셋과 hard-FP 세트에서 평가한다.
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.distortionguard import build_distortionguard_iqa_v1


IMAGE_SIZE = 384
MODEL_NAME = "distortionguard_stageB"
DEFAULT_CANDIDATE = "outputs/distortionguard_stageB_authentic_frozen_from_stageA_e10_20260523"
DEFAULT_OUT = "outputs/eval_distortionguard_stageB_authentic_frozen_from_stageA_e10_20260523"
DEFAULT_FLIVE = "data/processed/techiqa_guard/test_flive.csv"
DEFAULT_KONIQ = "data/processed/techiqa_guard/test_koniq.csv"
DEFAULT_SPAQ = "data/processed/techiqa_guard/test_spaq.csv"
DEFAULT_HARD_FP = "data/processed/techiqa_guard/fp_mining_20260522/hard_false_positive_confirmed_v2.csv"
BASELINE_TEST_DIR = Path("outputs/eval_final_topiq_candidates_vs_existing_technical_20260520")
BASELINE_HARD_FP_DIR = Path("outputs/eval_techiqa_guard_v1_hard_fp_confirmed_v2_20260522")


TEST_BASELINE_NAMES = {
    "koniq_mobile": "koniq_mobile",
    "flive_image_mobile": "flive_mobile",
    "existing_avg_technical": "existing_avg",
    "topiq_lite_mixed112_frozen_fp16": "topiq_mixed112",
    "topiq_lite_ranking_lam01_gap05_fp16": "topiq_ranking",
}
HARD_FP_PREDICTION_FILES = {
    "koniq_mobile": "predictions_koniq_mobile.csv",
    "flive_mobile": "predictions_flive_mobile.csv",
    "existing_avg": "predictions_existing_avg.csv",
    "topiq_mixed112": "predictions_topiq_mixed112.csv",
    "topiq_ranking": "predictions_topiq_ranking.csv",
    "techiqa_stage4": "predictions_techiqa_stage4.csv",
}
MODEL_ORDER = {
    MODEL_NAME: 0,
    "koniq_mobile": 1,
    "flive_mobile": 2,
    "existing_avg": 3,
    "topiq_mixed112": 4,
    "topiq_ranking": 5,
    "techiqa_stage4": 6,
}


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(result) or np.isinf(result):
        return None
    return result


def _fmt(value: Any, digits: int = 2) -> str:
    number = _safe_float(value)
    if number is None:
        return "NA"
    return f"{number:.{digits}f}"


def _safe_corr(target: np.ndarray, pred: np.ndarray, method: str) -> float | None:
    if len(target) < 2 or np.std(target) <= 1.0e-12 or np.std(pred) <= 1.0e-12:
        return None
    try:
        if method == "spearman":
            from scipy.stats import spearmanr

            value = spearmanr(target, pred).correlation
        else:
            from scipy.stats import pearsonr

            result = pearsonr(target, pred)
            value = result.statistic if hasattr(result, "statistic") else result[0]
    except Exception:
        return None
    return None if np.isnan(value) else float(value)


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"1", "true", "yes", "y"})


def _resolve_image_path(value: Any, csv_path: Path) -> str:
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    csv_relative = (csv_path.parent / path).resolve()
    if csv_relative.is_file():
        return str(csv_relative)
    return str((REPO_ROOT / path).resolve())


def _load_test_frame(csv_path: Path, target_col: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(csv_path)
    required = {"image_path", target_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    original_count = len(df)
    df = df.copy()
    df["image_path"] = [
        _resolve_image_path(value, csv_path) for value in df["image_path"].astype(str)
    ]
    exists = df["image_path"].map(lambda value: Path(value).is_file())
    premarked_missing = (
        _bool_series(df["missing_image"])
        if "missing_image" in df.columns
        else pd.Series(False, index=df.index)
    )
    target_values = pd.to_numeric(df[target_col], errors="coerce")
    usable = exists & ~premarked_missing & target_values.notna()
    df = df.loc[usable].copy()
    target_values = target_values.loc[usable].astype("float32")
    if target_values.max() > 1.5:
        df["target_100"] = target_values.astype("float32")
        df["target_norm"] = (target_values / 100.0).clip(0.0, 1.0).astype("float32")
    else:
        df["target_norm"] = target_values.clip(0.0, 1.0).astype("float32")
        df["target_100"] = (df["target_norm"] * 100.0).astype("float32")

    summary = {
        "path": str(csv_path),
        "original_count": int(original_count),
        "used_count": int(len(df)),
        "dropped_missing_or_unreadable_path": int((~exists | premarked_missing).sum()),
        "dropped_missing_target": int(target_values.isna().sum()),
        "target_col": target_col,
    }
    if len(df) == 0:
        raise ValueError(f"{csv_path} has no usable rows after filtering.")
    return df.reset_index(drop=True), summary


def _load_image_frame(csv_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns:
        raise ValueError(f"{csv_path} is missing required column: image_path")
    original_count = len(df)
    df = df.copy()
    df["image_path"] = [
        _resolve_image_path(value, csv_path) for value in df["image_path"].astype(str)
    ]
    exists = df["image_path"].map(lambda value: Path(value).is_file())
    premarked_missing = (
        _bool_series(df["missing_image"])
        if "missing_image" in df.columns
        else pd.Series(False, index=df.index)
    )
    usable = exists & ~premarked_missing
    df = df.loc[usable].copy()
    summary = {
        "path": str(csv_path),
        "original_count": int(original_count),
        "used_count": int(len(df)),
        "dropped_missing_or_unreadable_path": int((~exists | premarked_missing).sum()),
    }
    if len(df) == 0:
        raise ValueError(f"{csv_path} has no usable image rows after filtering.")
    return df.reset_index(drop=True), summary


def _decode_resize_with_pad(path: tf.Tensor) -> tf.Tensor:
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)
    image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
    return image


def _make_prediction_dataset(paths: list[str], batch_size: int) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            np.arange(len(paths), dtype=np.int64),
            paths,
        )
    )

    def _map_fn(index: tf.Tensor, path: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return index, _decode_resize_with_pad(path)

    return (
        dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .apply(tf.data.experimental.ignore_errors())
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def _predict_normalized(
    model: tf.keras.Model,
    paths: list[str],
    batch_size: int,
) -> tuple[np.ndarray, float, int]:
    predictions = np.full(len(paths), np.nan, dtype=np.float32)
    dataset = _make_prediction_dataset(paths, batch_size=batch_size)
    start = time.perf_counter()
    predicted_count = 0
    for batch_indices, images in dataset:
        batch_pred = model(images, training=False).numpy().reshape(-1)
        batch_pred = np.clip(batch_pred.astype("float32"), 0.0, 1.0)
        indices = batch_indices.numpy().astype(np.int64)
        predictions[indices] = batch_pred
        predicted_count += len(indices)
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / predicted_count) * 1000.0 if predicted_count else 0.0
    return predictions, avg_ms, int(predicted_count)


def _load_candidate_model(
    candidate_dir: Path,
    training_summary: dict[str, Any],
) -> tuple[tf.keras.Model, dict[str, Any]]:
    final_model_path = candidate_dir / "final_model.keras"
    best_weights_path = candidate_dir / "best.weights.h5"
    load_info: dict[str, Any] = {
        "candidate_dir": str(candidate_dir),
        "final_model": str(final_model_path),
        "best_weights": str(best_weights_path),
        "method": None,
        "safe_mode_false": False,
        "errors": [],
    }

    if final_model_path.is_file():
        try:
            model = tf.keras.models.load_model(str(final_model_path), compile=False)
            load_info["method"] = "final_model.keras"
            return model, load_info
        except Exception as exc:
            load_info["errors"].append(f"load_model: {type(exc).__name__}: {exc}")
        try:
            model = tf.keras.models.load_model(
                str(final_model_path),
                compile=False,
                safe_mode=False,
            )
            load_info["method"] = "final_model.keras_safe_mode_false"
            load_info["safe_mode_false"] = True
            return model, load_info
        except TypeError as exc:
            load_info["errors"].append(f"load_model_safe_mode_unsupported: {exc}")
        except Exception as exc:
            load_info["errors"].append(
                f"load_model_safe_mode_false: {type(exc).__name__}: {exc}"
            )

    if not best_weights_path.is_file():
        raise FileNotFoundError(
            f"Could not load final_model.keras and best weights not found: {best_weights_path}"
        )

    train_args = training_summary.get("args", {})
    stagea_weights = train_args.get("stageA_weights")
    stagea_path = Path(stagea_weights) if stagea_weights else None
    if stagea_path is not None and not stagea_path.is_absolute():
        stagea_path = (REPO_ROOT / stagea_path).resolve()
    model = build_distortionguard_iqa_v1(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        stageA_weights=str(stagea_path) if stagea_path and stagea_path.is_file() else None,
        backbone_trainable=False,
        freeze_batch_norm=True,
        dropout=float(train_args.get("dropout", 0.25)),
        head_units=int(train_args.get("head_units", 256)),
    )
    model.load_weights(str(best_weights_path))
    load_info["method"] = "rebuilt_model_best.weights.h5"
    load_info["stageA_weights_used_for_rebuild"] = str(stagea_path) if stagea_path else None
    return model, load_info


def _test_metrics(
    dataset: str,
    target_100: np.ndarray,
    pred_norm: np.ndarray,
    avg_ms: float,
    source: str,
) -> dict[str, Any]:
    usable = ~np.isnan(pred_norm)
    target = target_100[usable].astype("float32")
    pred = (pred_norm[usable] * 100.0).astype("float32")
    if len(pred) == 0:
        raise RuntimeError(f"{dataset} produced no usable predictions.")
    pred_std = float(np.std(pred))
    target_std = float(np.std(target))
    return {
        "dataset": dataset,
        "model": MODEL_NAME,
        "n": int(len(pred)),
        "mae": float(np.mean(np.abs(target - pred))),
        "rmse": float(np.sqrt(np.mean(np.square(target - pred)))),
        "srcc": _safe_corr(target, pred, method="spearman"),
        "plcc": _safe_corr(target, pred, method="pearson"),
        "bias": float(np.mean(pred - target)),
        "pred_mean": float(np.mean(pred)),
        "pred_std": pred_std,
        "target_mean": float(np.mean(target)),
        "target_std": target_std,
        "std_ratio": None if target_std <= 0.0 else float(pred_std / target_std),
        "avg_ms": float(avg_ms),
        "source": source,
    }


def _hard_fp_summary(
    model_name: str,
    scores_100: np.ndarray,
    source: str,
) -> dict[str, Any]:
    scores = scores_100[~np.isnan(scores_100)].astype("float32")
    if len(scores) == 0:
        raise RuntimeError(f"{model_name} produced no hard-FP predictions.")
    return {
        "model": model_name,
        "n": int(len(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "median": float(np.median(scores)),
        "count_gt_55": int(np.sum(scores > 55.0)),
        "count_gt_65": int(np.sum(scores > 65.0)),
        "source": source,
    }


def _candidate_prediction_columns(
    df: pd.DataFrame,
    pred_norm: np.ndarray,
    avg_ms: float,
) -> pd.DataFrame:
    out = df.copy()
    out["distortionguard_score"] = pred_norm
    out["distortionguard_score_100"] = pred_norm * 100.0
    out["prediction_status"] = np.where(np.isnan(pred_norm), "failed", "ok")
    out["avg_ms"] = avg_ms
    return out


def _load_test_baselines() -> list[dict[str, Any]]:
    summary_path = REPO_ROOT / BASELINE_TEST_DIR / "summary.csv"
    if not summary_path.is_file():
        return []
    baseline_df = pd.read_csv(summary_path)
    rows: list[dict[str, Any]] = []
    for _, row in baseline_df.iterrows():
        local_name = str(row.get("model", ""))
        if local_name not in TEST_BASELINE_NAMES:
            continue
        mapped = TEST_BASELINE_NAMES[local_name]
        rows.append(
            {
                "dataset": str(row.get("dataset")),
                "model": mapped,
                "n": int(row.get("n", 0)),
                "mae": _safe_float(row.get("mae")),
                "rmse": _safe_float(row.get("rmse")),
                "srcc": _safe_float(row.get("srcc")),
                "plcc": _safe_float(row.get("plcc")),
                "bias": _safe_float(row.get("bias")),
                "pred_mean": _safe_float(row.get("pred_mean")),
                "pred_std": _safe_float(row.get("pred_std")),
                "target_mean": _safe_float(row.get("target_mean")),
                "target_std": _safe_float(row.get("target_std")),
                "std_ratio": _safe_float(row.get("std_ratio")),
                "avg_ms": _safe_float(row.get("avg_ms")),
                "source": str(summary_path),
            }
        )
    return rows


def _load_hard_fp_baselines() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_dir = REPO_ROOT / BASELINE_HARD_FP_DIR
    for model_name, filename in HARD_FP_PREDICTION_FILES.items():
        path = base_dir / filename
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        if "y_pred" not in df.columns:
            continue
        scores = pd.to_numeric(df["y_pred"], errors="coerce").to_numpy(dtype="float32")
        rows.append(_hard_fp_summary(model_name, scores, source=str(path)))
    return rows


def _sort_test_summary(df: pd.DataFrame) -> pd.DataFrame:
    dataset_order = {"flive": 0, "koniq": 1, "spaq": 2}
    sortable = df.copy()
    sortable["_dataset_order"] = sortable["dataset"].map(dataset_order).fillna(99)
    sortable["_model_order"] = sortable["model"].map(MODEL_ORDER).fillna(99)
    sortable = sortable.sort_values(["_dataset_order", "_model_order", "model"])
    return sortable.drop(columns=["_dataset_order", "_model_order"])


def _sort_hard_summary(df: pd.DataFrame) -> pd.DataFrame:
    sortable = df.copy()
    sortable["_model_order"] = sortable["model"].map(MODEL_ORDER).fillna(99)
    sortable = sortable.sort_values(["_model_order", "model"])
    return sortable.drop(columns=["_model_order"])


def _mode_collapse(metrics: list[dict[str, Any]]) -> bool:
    for row in metrics:
        pred_std = _safe_float(row.get("pred_std"))
        if pred_std is not None and pred_std < 1.0:
            return True
    return False


def _choose_decision(
    candidate_metrics: list[dict[str, Any]],
    hard_fp_row: dict[str, Any],
) -> tuple[str, str, str]:
    srcc_values = [
        row["srcc"]
        for row in candidate_metrics
        if _safe_float(row.get("srcc")) is not None
    ]
    avg_srcc = float(np.mean(srcc_values)) if srcc_values else None
    collapse = _mode_collapse(candidate_metrics)
    hard_mean = float(hard_fp_row["mean"])
    hard_gt65 = int(hard_fp_row["count_gt_65"])

    if collapse or avg_srcc is None or avg_srcc < 0.55:
        return (
            "FAIL",
            "C. Stop",
            "Do not proceed to export or longer training until the evaluation failure is diagnosed.",
        )
    if hard_mean > 55.0 or hard_gt65 > 0:
        return (
            "PARTIAL PASS",
            "B. Add hard-FP guard",
            "Keep the Stage B candidate for comparison, then add a hard-FP guard before any export work.",
        )
    return (
        "PASS",
        "A. Continue to partial unfreeze Stage B2",
        "Run a bounded partial-unfreeze Stage B2 experiment from this frozen candidate.",
    )


def _best_baseline_by_dataset(summary_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    baselines = summary_df[summary_df["model"] != MODEL_NAME].copy()
    for dataset, group in baselines.groupby("dataset"):
        group = group.dropna(subset=["srcc", "plcc"])
        if group.empty:
            continue
        best = group.sort_values(["srcc", "plcc"], ascending=False).iloc[0]
        rows[str(dataset)] = best.to_dict()
    return rows


def _write_report(
    path: Path,
    status: str,
    decision: str,
    next_step: str,
    training_summary: dict[str, Any],
    stagea_report: dict[str, Any],
    load_info: dict[str, Any],
    test_summary: pd.DataFrame,
    hard_summary: pd.DataFrame,
    top_hard_fp: pd.DataFrame,
) -> None:
    final_metrics = training_summary.get("final_metrics", {})
    best_epoch = training_summary.get("best_epoch")
    best_baselines = _best_baseline_by_dataset(test_summary)
    candidate_test = test_summary[test_summary["model"] == MODEL_NAME]
    candidate_hard = hard_summary[hard_summary["model"] == MODEL_NAME].iloc[0]

    lines: list[str] = []
    lines.append("# DistortionGuard-IQA v1 Stage B Evaluation")
    lines.append("")
    lines.append("## 1. Summary")
    lines.append(status)
    lines.append("")
    lines.append(f"- Candidate: {load_info.get('candidate_dir')}")
    lines.append(f"- Model load method: {load_info.get('method')}")
    if load_info.get("safe_mode_false"):
        lines.append("- Keras safe_mode=False was used for this trusted local artifact.")
    lines.append(f"- Decision: {decision}")
    lines.append("")
    lines.append("## 2. Training Result")
    lines.append(f"- best_epoch: {best_epoch}")
    lines.append(f"- val_srcc: {_fmt(final_metrics.get('val_srcc'), 4)}")
    lines.append(f"- val_plcc: {_fmt(final_metrics.get('val_plcc'), 4)}")
    lines.append(f"- val_mae_100: {_fmt(final_metrics.get('val_mae_100'), 2)}")
    lines.append(f"- val_rmse_100: {_fmt(final_metrics.get('val_rmse_100'), 2)}")
    lines.append(f"- mode_collapse: {final_metrics.get('mode_collapse')}")
    lines.append(
        "- Stage A transfer: "
        f"loaded={stagea_report.get('loaded_layer_count', 'NA')}, "
        f"skipped={stagea_report.get('skipped_layer_count', 'NA')}, "
        f"mismatched={stagea_report.get('mismatched_layer_count', 'NA')}"
    )
    lines.append("")
    lines.append("## 3. Test Metrics")
    lines.append("| Dataset | Model | MAE | RMSE | SRCC | PLCC | Bias | Avg ms |")
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in test_summary.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['model']} | {_fmt(row.get('mae'))} | "
            f"{_fmt(row.get('rmse'))} | {_fmt(row.get('srcc'), 4)} | "
            f"{_fmt(row.get('plcc'), 4)} | {_fmt(row.get('bias'))} | "
            f"{_fmt(row.get('avg_ms'))} |"
        )
    lines.append("")
    lines.append("## 4. Hard-FP v2 Behavior")
    lines.append("| Model | Mean | Median | Count >55 | Count >65 |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    for _, row in hard_summary.iterrows():
        lines.append(
            f"| {row['model']} | {_fmt(row.get('mean'))} | {_fmt(row.get('median'))} | "
            f"{int(row.get('count_gt_55', 0))} | {int(row.get('count_gt_65', 0))} |"
        )
    lines.append("")
    lines.append("Top over-scored DistortionGuard hard-FP images:")
    lines.append("")
    lines.append("| Rank | Filename | Score | Image path |")
    lines.append("| ---: | :--- | ---: | :--- |")
    for rank, (_, row) in enumerate(top_hard_fp.iterrows(), start=1):
        filename = row.get("filename") or Path(str(row["image_path"])).name
        lines.append(
            f"| {rank} | {filename} | {_fmt(row.get('distortionguard_score_100'))} | "
            f"{row['image_path']} |"
        )
    lines.append("")
    lines.append("## 5. Comparison to Existing Models")
    for _, row in candidate_test.iterrows():
        dataset = str(row["dataset"])
        best = best_baselines.get(dataset)
        if best:
            lines.append(
                f"- {dataset}: DistortionGuard SRCC {_fmt(row.get('srcc'), 4)} / "
                f"PLCC {_fmt(row.get('plcc'), 4)} vs best local baseline "
                f"{best['model']} SRCC {_fmt(best.get('srcc'), 4)} / "
                f"PLCC {_fmt(best.get('plcc'), 4)}."
            )
        else:
            lines.append(
                f"- {dataset}: DistortionGuard SRCC {_fmt(row.get('srcc'), 4)} / "
                f"PLCC {_fmt(row.get('plcc'), 4)}; no local baseline summary found."
            )
    existing_avg = hard_summary[hard_summary["model"] == "existing_avg"]
    techiqa_stage4 = hard_summary[hard_summary["model"] == "techiqa_stage4"]
    if not existing_avg.empty:
        lines.append(
            f"- hard-FP v2: DistortionGuard mean {_fmt(candidate_hard.get('mean'))} vs "
            f"existing_avg mean {_fmt(existing_avg.iloc[0].get('mean'))}."
        )
    if not techiqa_stage4.empty:
        lines.append(
            f"- hard-FP v2: DistortionGuard mean {_fmt(candidate_hard.get('mean'))} vs "
            f"TechIQA-Guard Stage 4 mean {_fmt(techiqa_stage4.iloc[0].get('mean'))}."
        )
    if "techiqa_stage4" not in set(test_summary["model"]):
        lines.append(
            "- TechIQA-Guard Stage 4 MOS test-set predictions were not available in the local baseline summaries."
        )
    lines.append("")
    lines.append("## 6. Decision")
    lines.append(decision)
    lines.append("")
    lines.append("## 7. Recommended Next Step")
    lines.append(next_step)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a DistortionGuard-IQA v1 Stage B authentic model."
    )
    parser.add_argument("--candidate_dir", default=DEFAULT_CANDIDATE)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)
    parser.add_argument("--flive_csv", default=DEFAULT_FLIVE)
    parser.add_argument("--koniq_csv", default=DEFAULT_KONIQ)
    parser.add_argument("--spaq_csv", default=DEFAULT_SPAQ)
    parser.add_argument("--hard_fp_csv", default=DEFAULT_HARD_FP)
    parser.add_argument("--target_col", default="normalized_mos")
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")

    _configure_tensorflow()
    candidate_dir = Path(args.candidate_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_summary = _read_json(candidate_dir / "training_summary.json")
    stagea_report = _read_json(candidate_dir / "stageA_weight_load_report.json")
    model, load_info = _load_candidate_model(candidate_dir, training_summary)
    print("Candidate load info:")
    print(json.dumps(load_info, indent=2, sort_keys=True))

    datasets = [
        ("flive", Path(args.flive_csv), out_dir / "predictions_flive.csv"),
        ("koniq", Path(args.koniq_csv), out_dir / "predictions_koniq.csv"),
        ("spaq", Path(args.spaq_csv), out_dir / "predictions_spaq.csv"),
    ]
    candidate_metrics: list[dict[str, Any]] = []
    dataset_summaries: dict[str, Any] = {}

    for dataset_name, csv_path, pred_out in datasets:
        df, data_summary = _load_test_frame(csv_path, target_col=args.target_col)
        dataset_summaries[dataset_name] = data_summary
        pred_norm, avg_ms, predicted_count = _predict_normalized(
            model,
            df["image_path"].astype(str).tolist(),
            batch_size=args.batch_size,
        )
        print(
            f"{dataset_name}: predicted {predicted_count}/{len(df)} images, avg_ms={avg_ms:.2f}"
        )
        pred_df = _candidate_prediction_columns(df, pred_norm, avg_ms)
        pred_df.to_csv(pred_out, index=False)
        candidate_metrics.append(
            _test_metrics(
                dataset=dataset_name,
                target_100=df["target_100"].to_numpy(dtype="float32"),
                pred_norm=pred_norm,
                avg_ms=avg_ms,
                source=str(pred_out),
            )
        )

    hard_df, hard_summary = _load_image_frame(Path(args.hard_fp_csv))
    hard_pred_norm, hard_avg_ms, hard_predicted_count = _predict_normalized(
        model,
        hard_df["image_path"].astype(str).tolist(),
        batch_size=args.batch_size,
    )
    print(
        f"hard_fp_v2: predicted {hard_predicted_count}/{len(hard_df)} images, "
        f"avg_ms={hard_avg_ms:.2f}"
    )
    hard_pred_df = _candidate_prediction_columns(hard_df, hard_pred_norm, hard_avg_ms)
    hard_pred_out = out_dir / "predictions_hard_fp_v2.csv"
    hard_pred_df.to_csv(hard_pred_out, index=False)
    candidate_hard = _hard_fp_summary(
        MODEL_NAME,
        hard_pred_norm * 100.0,
        source=str(hard_pred_out),
    )
    candidate_hard["avg_ms"] = float(hard_avg_ms)

    test_summary = pd.DataFrame(candidate_metrics + _load_test_baselines())
    test_summary = _sort_test_summary(test_summary)
    test_summary.to_csv(out_dir / "summary.csv", index=False)

    hard_summary_df = pd.DataFrame([candidate_hard] + _load_hard_fp_baselines())
    hard_summary_df = _sort_hard_summary(hard_summary_df)
    hard_summary_df.to_csv(out_dir / "hard_fp_v2_summary.csv", index=False)

    status, decision, next_step = _choose_decision(candidate_metrics, candidate_hard)
    top_hard_fp = hard_pred_df.sort_values(
        "distortionguard_score_100",
        ascending=False,
    ).head(10)
    _write_report(
        path=out_dir / "report.md",
        status=status,
        decision=decision,
        next_step=next_step,
        training_summary=training_summary,
        stagea_report=stagea_report,
        load_info=load_info,
        test_summary=test_summary,
        hard_summary=hard_summary_df,
        top_hard_fp=top_hard_fp,
    )

    print("Dataset summaries:")
    print(json.dumps(dataset_summaries, indent=2, sort_keys=True))
    print("Hard-FP input summary:")
    print(json.dumps(hard_summary, indent=2, sort_keys=True))
    print("Evaluation complete.")
    print(f"summary_csv: {out_dir / 'summary.csv'}")
    print(f"hard_fp_summary_csv: {out_dir / 'hard_fp_v2_summary.csv'}")
    print(f"report: {out_dir / 'report.md'}")
    print(f"status: {status}")
    print(f"decision: {decision}")


if __name__ == "__main__":
    main()
