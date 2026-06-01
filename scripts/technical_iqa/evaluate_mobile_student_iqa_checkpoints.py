# 모바일 Student IQA v4b 체크포인트들을 동일한 검증/테스트 분할에서 재평가하는 스크립트
from __future__ import annotations

import argparse
import math
import os
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402


DEFAULT_MANIFEST = "outputs/student_training_manifest_50k_20260601/student_training_manifest.csv"
DEFAULT_CHECKPOINT_DIR = "outputs/mobile_student_iqa_v4b_20260601"
DEFAULT_OUT_DIR = "outputs/mobile_student_iqa_v4b_checkpoint_reeval_20260601"
SPLITS = ["val", "test"]
CHECKPOINTS = [
    "best_model.keras",
    "best_by_val_mae_model.keras",
    "best_by_val_srcc_model.keras",
    "best_by_val_safety_model.keras",
    "final_model.keras",
]
DATASETS = ["flive", "koniq10k", "spaq"]
PRED_COL = "pred_0_100"
REQUIRED_COLUMNS = [
    "image_path",
    "dataset",
    "student_split",
    "quality_score_0_100",
    "guarded_target_0_100",
]
BASELINES = [
    {
        "checkpoint": "v3_headline",
        "test_srcc": 0.658204,
        "test_mae": 4.543416,
        "test_severe_fp": 28,
        "flive_test_srcc": 0.514385,
        "koniq10k_test_srcc": 0.643786,
        "spaq_test_srcc": 0.840971,
    },
    {
        "checkpoint": "v4a_headline",
        "test_srcc": 0.630522,
        "test_mae": 5.410899,
        "test_severe_fp": 13,
        "flive_test_srcc": 0.449674,
        "koniq10k_test_srcc": 0.639833,
        "spaq_test_srcc": 0.832751,
    },
    {
        "checkpoint": "v4b_report_headline",
        "test_srcc": 0.647898,
        "test_mae": 5.108165,
        "test_severe_fp": 12,
        "flive_test_srcc": 0.469730,
        "koniq10k_test_srcc": 0.654136,
        "spaq_test_srcc": 0.853897,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate saved Mobile Student IQA v4b checkpoints.")
    parser.add_argument("--manifest_csv", default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--default_image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--save_predictions", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def require_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"manifest_csv is missing required columns: {missing}")


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
            error_rows.append(
                {
                    "row_index": idx,
                    "dataset": row.get("dataset", ""),
                    "student_split": row.get("student_split", ""),
                    "image_path": image_path,
                    "error": error,
                }
            )
        else:
            usable_rows.append(row)
    return pd.DataFrame(usable_rows).reset_index(drop=True), pd.DataFrame(error_rows)


def load_rgb_resized(path: str, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32)


def make_predict_dataset(df: pd.DataFrame, image_size: int, batch_size: int) -> tf.data.Dataset:
    records = df[["image_path"]].to_dict("records")

    def generator() -> Iterable[np.ndarray]:
        for row in records:
            yield load_rgb_resized(str(row["image_path"]), image_size)

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
    )
    return ds.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)


def batch_count(num_rows: int, batch_size: int) -> int:
    if num_rows <= 0:
        raise ValueError("Cannot evaluate an empty split.")
    return int(math.ceil(num_rows / float(batch_size)))


def infer_image_size(model: tf.keras.Model, default_image_size: int) -> int:
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    if len(shape) >= 3 and shape[1] is not None and shape[2] is not None and int(shape[1]) == int(shape[2]):
        return int(shape[1])
    return int(default_image_size)


def target_mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.abs(y_true[:, 0:1] - y_pred))


def target_rmse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.sqrt(tf.reduce_mean(tf.square(y_true[:, 0:1] - y_pred)))


def load_checkpoint(path: Path) -> Tuple[Optional[tf.keras.Model], Optional[str]]:
    if not path.exists():
        return None, "checkpoint file missing"
    try:
        return tf.keras.models.load_model(path, compile=False), None
    except Exception as exc:
        custom_objects = {"target_mae": target_mae, "target_rmse": target_rmse}
        try:
            return tf.keras.models.load_model(path, compile=False, custom_objects=custom_objects), None
        except Exception:
            return None, f"{exc!r}\n{traceback.format_exc(limit=5)}"


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


def regression_metrics(df: pd.DataFrame, target_col: str, split: str, dataset: str, checkpoint: str) -> Dict[str, Any]:
    y_true = df[target_col].to_numpy(dtype=np.float64)
    y_pred = df[PRED_COL].to_numpy(dtype=np.float64)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() == 0:
        return {
            "checkpoint": checkpoint,
            "split": split,
            "dataset": dataset,
            "n": len(df),
            "finite_n": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "srcc": float("nan"),
            "plcc": float("nan"),
            "pred_min": float("nan"),
            "pred_max": float("nan"),
            "pred_mean": float("nan"),
            "pred_std": float("nan"),
        }
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    return {
        "checkpoint": checkpoint,
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


def safety_metrics(df: pd.DataFrame, split: str, dataset: str, checkpoint: str) -> Dict[str, Any]:
    mos = df["quality_score_0_100"].to_numpy(dtype=np.float64)
    pred = df[PRED_COL].to_numpy(dtype=np.float64)
    finite = np.isfinite(pred)
    return {
        "checkpoint": checkpoint,
        "split": split,
        "dataset": dataset,
        "n": len(df),
        "finite_n": int(finite.sum()),
        "fp_count": int(((mos < 50.0) & (pred > 65.0)).sum()),
        "severe_fp_count": int(((mos < 40.0) & (pred > 65.0)).sum()),
        "fn_count": int(((mos > 75.0) & (pred < 45.0)).sum()),
    }


def predict_split(model: tf.keras.Model, df: pd.DataFrame, image_size: int, batch_size: int) -> pd.DataFrame:
    ds = make_predict_dataset(df, image_size, batch_size)
    pred_norm = model.predict(ds, steps=batch_count(len(df), batch_size), verbose=1).reshape(-1)[: len(df)]
    out = df.copy()
    out["pred_norm"] = np.clip(pred_norm.astype(np.float64), 0.0, 1.0)
    out[PRED_COL] = out["pred_norm"] * 100.0
    out["abs_error_to_target"] = np.abs(out[PRED_COL].to_numpy(dtype=np.float64) - out["guarded_target_0_100"].to_numpy(dtype=np.float64))
    out["abs_error_to_mos"] = np.abs(out[PRED_COL].to_numpy(dtype=np.float64) - out["quality_score_0_100"].to_numpy(dtype=np.float64))
    return out


def evaluate_posthoc_calibration(
    val_preds: np.ndarray,
    val_targets: np.ndarray,
    test_preds: np.ndarray,
) -> Tuple[np.ndarray, str]:
    try:
        from sklearn.isotonic import IsotonicRegression

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(val_preds, val_targets)
        return np.clip(calibrator.transform(test_preds), 0.0, 100.0), "isotonic"
    except Exception:
        coeffs = np.polyfit(val_preds, val_targets, 1)
        return np.clip(np.polyval(coeffs, test_preds), 0.0, 100.0), "linear"


def calibration_metrics(raw_test_df: pd.DataFrame, calibrated_preds: np.ndarray, checkpoint: str, method: str) -> List[Dict[str, Any]]:
    rows = []
    cal_df = raw_test_df.copy()
    cal_df[PRED_COL] = calibrated_preds
    for dataset, group in [("overall", cal_df)] + [(str(ds), g) for ds, g in cal_df.groupby("dataset", sort=True)]:
        metrics = regression_metrics(group, "guarded_target_0_100", "test", dataset, checkpoint)
        safety = safety_metrics(group, "test", dataset, checkpoint)
        rows.append(
            {
                "checkpoint": checkpoint,
                "method": method,
                "split": "test",
                "dataset": dataset,
                "n": metrics["n"],
                "finite_n": metrics["finite_n"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "srcc": metrics["srcc"],
                "plcc": metrics["plcc"],
                "fp_count": safety["fp_count"],
                "severe_fp_count": safety["severe_fp_count"],
                "fn_count": safety["fn_count"],
            }
        )
    return rows


def checkpoint_summary_row(
    checkpoint: str,
    split_metrics: List[Dict[str, Any]],
    dataset_metrics: List[Dict[str, Any]],
    safety_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    test = next(row for row in split_metrics if row["checkpoint"] == checkpoint and row["split"] == "test")
    val = next(row for row in split_metrics if row["checkpoint"] == checkpoint and row["split"] == "val")
    safety = next(row for row in safety_rows if row["checkpoint"] == checkpoint and row["split"] == "test" and row["dataset"] == "overall")
    ds_rows = [row for row in dataset_metrics if row["checkpoint"] == checkpoint and row["split"] == "test"]
    ds_srccs = {row["dataset"]: row["srcc"] for row in ds_rows}
    dataset_srcc_min = min(ds_srccs.values()) if ds_srccs else float("nan")
    no_dataset_below_050 = bool(np.isfinite(dataset_srcc_min) and dataset_srcc_min >= 0.50)
    readiness = bool(
        np.isfinite(test["srcc"])
        and test["srcc"] >= 0.85
        and np.isfinite(test["mae"])
        and test["mae"] <= 10.0
        and int(safety["severe_fp_count"]) == 0
        and no_dataset_below_050
    )
    return {
        "checkpoint": checkpoint,
        "val_mae": val["mae"],
        "val_srcc": val["srcc"],
        "test_mae": test["mae"],
        "test_rmse": test["rmse"],
        "test_srcc": test["srcc"],
        "test_plcc": test["plcc"],
        "test_pred_min": test["pred_min"],
        "test_pred_max": test["pred_max"],
        "test_pred_mean": test["pred_mean"],
        "test_pred_std": test["pred_std"],
        "test_fp": safety["fp_count"],
        "test_severe_fp": safety["severe_fp_count"],
        "test_fn": safety["fn_count"],
        "flive_test_srcc": ds_srccs.get("flive", float("nan")),
        "koniq10k_test_srcc": ds_srccs.get("koniq10k", float("nan")),
        "spaq_test_srcc": ds_srccs.get("spaq", float("nan")),
        "test_dataset_srcc_min": dataset_srcc_min,
        "no_dataset_test_srcc_below_0_50": no_dataset_below_050,
        "raw_readiness_satisfied": readiness,
    }


def choose_best_raw(summary: pd.DataFrame) -> Optional[pd.Series]:
    if summary.empty:
        return None
    ranking = summary.copy()
    ranking["_score"] = (
        ranking["test_srcc"].fillna(-1.0) * 100.0
        + ranking["test_dataset_srcc_min"].fillna(-1.0) * 20.0
        - ranking["test_mae"].fillna(999.0)
        - ranking["test_severe_fp"].fillna(999.0) * 0.5
    )
    return ranking.sort_values(["raw_readiness_satisfied", "_score"], ascending=[False, False]).iloc[0]


def choose_best_calibrated(calibration_df: pd.DataFrame) -> Optional[pd.Series]:
    overall = calibration_df[calibration_df["dataset"] == "overall"].copy()
    if overall.empty:
        return None
    ds_min = (
        calibration_df[calibration_df["dataset"].isin(DATASETS)]
        .groupby("checkpoint")["srcc"]
        .min()
        .rename("dataset_srcc_min")
    )
    overall = overall.merge(ds_min, on="checkpoint", how="left")
    overall["_score"] = (
        overall["srcc"].fillna(-1.0) * 100.0
        + overall["dataset_srcc_min"].fillna(-1.0) * 20.0
        - overall["mae"].fillna(999.0)
        - overall["severe_fp_count"].fillna(999.0) * 0.5
    )
    return overall.sort_values("_score", ascending=False).iloc[0]


def final_decision(summary: pd.DataFrame, load_failures: pd.DataFrame, best: Optional[pd.Series]) -> Tuple[str, List[str]]:
    if summary.empty:
        if len(load_failures) > 0:
            return "EVALUATION_FAILED", ["No checkpoint could be loaded and evaluated."]
        return "NO_USABLE_CHECKPOINT", ["No checkpoint rows were produced."]
    ready = summary[summary["raw_readiness_satisfied"]]
    if not ready.empty:
        return "CHECKPOINT_READY_FOR_BASELINE_COMPARISON", [f"{ready.iloc[0]['checkpoint']} satisfies all raw readiness criteria."]
    if best is None:
        return "EVALUATION_FAILED", ["Unable to rank evaluated checkpoints."]

    improves_tradeoff = bool(
        best["test_severe_fp"] <= 12
        or best["test_srcc"] >= 0.658204
        or best["test_mae"] <= 4.543416
        or best["test_dataset_srcc_min"] >= 0.50
    )
    if improves_tradeoff:
        return (
            "BEST_CHECKPOINT_STILL_NEEDS_TUNING",
            [
                f"{best['checkpoint']} is the best raw tradeoff but misses readiness.",
                f"test_srcc={best['test_srcc']:.6f}, test_mae={best['test_mae']:.6f}, severe_fp={int(best['test_severe_fp'])}, dataset_srcc_min={best['test_dataset_srcc_min']:.6f}.",
            ],
        )
    return (
        "NO_USABLE_CHECKPOINT",
        [
            "All evaluated raw checkpoints are worse than prior headline baselines or remain unsafe.",
            f"Best ranked raw checkpoint was {best['checkpoint']}.",
        ],
    )


def markdown_table(df: pd.DataFrame, columns: List[str], max_rows: Optional[int] = None) -> str:
    if df.empty:
        return "_No rows._"
    view = df.loc[:, [col for col in columns if col in df.columns]].copy()
    if max_rows is not None:
        view = view.head(max_rows)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6f}")
    headers = [str(col) for col in view.columns]
    rows = [[str(value) for value in row] for row in view.to_numpy()]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]
    header_line = "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |" for row in rows]
    return "\n".join([header_line, separator] + body)


def write_recommendation(path: Path, best_raw: Optional[pd.Series], best_calibrated: Optional[pd.Series], decision: str) -> None:
    lines = ["# Best Checkpoint Recommendation", ""]
    if best_raw is None:
        lines.extend(["- Best raw checkpoint: `n/a`", f"- Decision: `{decision}`"])
    else:
        lines.extend(
            [
                f"- Best raw checkpoint: `{best_raw['checkpoint']}`",
                f"- Raw test SRCC: `{best_raw['test_srcc']:.6f}`",
                f"- Raw test MAE: `{best_raw['test_mae']:.6f}`",
                f"- Raw severe FP: `{int(best_raw['test_severe_fp'])}`",
                f"- Raw minimum dataset SRCC: `{best_raw['test_dataset_srcc_min']:.6f}`",
            ]
        )
    if best_calibrated is not None:
        lines.extend(
            [
                "",
                f"- Best calibrated checkpoint: `{best_calibrated['checkpoint']}`",
                f"- Calibrated method: `{best_calibrated['method']}`",
                f"- Calibrated test SRCC: `{best_calibrated['srcc']:.6f}`",
                f"- Calibrated test MAE: `{best_calibrated['mae']:.6f}`",
                f"- Calibrated severe FP: `{int(best_calibrated['severe_fp_count'])}`",
            ]
        )
    lines.extend(["", f"- Final decision: `{decision}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(
    path: Path,
    args: argparse.Namespace,
    split_counts: Dict[str, int],
    image_errors: pd.DataFrame,
    load_failures: pd.DataFrame,
    summary: pd.DataFrame,
    dataset_metrics: pd.DataFrame,
    mos_metrics: pd.DataFrame,
    safety_metrics_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    best_raw: Optional[pd.Series],
    best_calibrated: Optional[pd.Series],
    decision: str,
    reasons: List[str],
) -> None:
    baseline_df = pd.DataFrame(BASELINES)
    comparison_cols = ["checkpoint", "test_srcc", "test_mae", "test_severe_fp", "flive_test_srcc", "koniq10k_test_srcc", "spaq_test_srcc"]
    comparison_df = pd.concat([baseline_df, summary[comparison_cols]], ignore_index=True) if not summary.empty else baseline_df
    lines = [
        "# Mobile Student IQA v4b Checkpoint Re-evaluation Report",
        "",
        "## 1. Executive Summary",
        f"- Final decision: `{decision}`",
        f"- Manifest: `{args.manifest_csv}`",
        f"- Checkpoint directory: `{args.checkpoint_dir}`",
        f"- Output directory: `{args.out_dir}`",
        f"- Validated split counts: `{split_counts}`",
        f"- Image validation failures: `{len(image_errors)}`",
    ]
    if best_raw is not None:
        lines.extend(
            [
                f"- Best raw checkpoint: `{best_raw['checkpoint']}`",
                f"- Best raw metrics: SRCC `{best_raw['test_srcc']:.6f}`, MAE `{best_raw['test_mae']:.6f}`, severe FP `{int(best_raw['test_severe_fp'])}`, min dataset SRCC `{best_raw['test_dataset_srcc_min']:.6f}`",
            ]
        )
    if best_calibrated is not None:
        lines.append(
            f"- Best calibrated checkpoint: `{best_calibrated['checkpoint']}` ({best_calibrated['method']}), SRCC `{best_calibrated['srcc']:.6f}`, MAE `{best_calibrated['mae']:.6f}`, severe FP `{int(best_calibrated['severe_fp_count'])}`"
        )
    lines.extend(
        [
            "",
            "## 2. Checkpoints Evaluated",
            markdown_table(summary, ["checkpoint", "val_mae", "val_srcc", "test_mae", "test_srcc", "test_severe_fp", "raw_readiness_satisfied"]),
            "",
        ]
    )
    if not load_failures.empty:
        lines.extend(["Load failures:", markdown_table(load_failures, ["checkpoint", "path", "error"], max_rows=10), ""])
    lines.extend(
        [
            "## 3. Raw Test Metrics",
            markdown_table(summary, ["checkpoint", "test_mae", "test_rmse", "test_srcc", "test_plcc", "test_pred_min", "test_pred_max", "test_pred_mean", "test_pred_std"]),
            "",
            "## 4. Dataset-wise Raw Metrics",
            markdown_table(dataset_metrics, ["checkpoint", "split", "dataset", "n", "mae", "rmse", "srcc", "plcc", "pred_min", "pred_max", "pred_mean", "pred_std"]),
            "",
            "## 5. Safety Metrics",
            markdown_table(safety_metrics_df, ["checkpoint", "split", "dataset", "n", "fp_count", "severe_fp_count", "fn_count"]),
            "",
            "## 6. Post-hoc Calibration Metrics",
            markdown_table(calibration_df, ["checkpoint", "method", "dataset", "mae", "rmse", "srcc", "plcc", "fp_count", "severe_fp_count", "fn_count"]),
            "",
            "## 7. Comparison with v3/v4a/v4b Report",
            markdown_table(comparison_df, comparison_cols),
            "",
            "## 8. Best Checkpoint Recommendation",
        ]
    )
    if best_raw is None:
        lines.append("- No raw checkpoint could be recommended.")
    else:
        lines.append(f"- Recommend `{best_raw['checkpoint']}` as the best raw checkpoint for any further baseline comparison work, but only if the final decision permits it.")
    if best_calibrated is not None:
        lines.append(f"- Calibration analysis favors `{best_calibrated['checkpoint']}`; this does not modify the raw model.")
    lines.extend(
        [
            "",
            "## 9. Readiness Decision",
            f"- `{decision}`",
        ]
    )
    lines.extend([f"- {reason}" for reason in reasons])
    lines.extend(
        [
            "",
            "## 10. Risks",
            "- Post-hoc calibration is analysis-only and must not be treated as a saved model improvement.",
            "- The v4b split is reused from the manifest after image validation; any future manifest change would invalidate direct comparison.",
            "- Readiness is judged on raw checkpoint predictions only.",
            "",
            "## 11. Next Step",
        ]
    )
    if decision == "CHECKPOINT_READY_FOR_BASELINE_COMPARISON":
        lines.append("- Use the ready raw checkpoint for the next baseline comparison stage.")
    elif decision == "BEST_CHECKPOINT_STILL_NEEDS_TUNING":
        lines.append("- Keep v4b in tuning; use this report to decide whether the best raw checkpoint is worth a targeted follow-up run.")
    elif decision == "NO_USABLE_CHECKPOINT":
        lines.append("- Do not continue from these checkpoints; revisit training configuration before any export or mobile work.")
    else:
        lines.append("- Fix the evaluation failure before drawing checkpoint conclusions.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest_csv)
    require_columns(manifest)
    manifest = manifest[manifest["student_split"].isin(SPLITS)].copy()
    usable_df, image_errors = validate_image_rows(manifest)
    usable_df["quality_score_0_100"] = pd.to_numeric(usable_df["quality_score_0_100"], errors="coerce")
    usable_df["guarded_target_0_100"] = pd.to_numeric(usable_df["guarded_target_0_100"], errors="coerce")
    image_errors.to_csv(out_dir / "image_validation_failures.csv", index=False)

    split_frames = {
        split: usable_df[usable_df["student_split"] == split].reset_index(drop=True)
        for split in SPLITS
    }
    split_counts = {split: len(df) for split, df in split_frames.items()}

    split_metric_rows: List[Dict[str, Any]] = []
    dataset_metric_rows: List[Dict[str, Any]] = []
    mos_metric_rows: List[Dict[str, Any]] = []
    safety_rows: List[Dict[str, Any]] = []
    calibration_rows: List[Dict[str, Any]] = []
    load_failures: List[Dict[str, Any]] = []
    loaded_rows: List[Dict[str, Any]] = []

    checkpoint_dir = Path(args.checkpoint_dir)
    for checkpoint_file in CHECKPOINTS:
        checkpoint_path = checkpoint_dir / checkpoint_file
        checkpoint_name = checkpoint_path.stem
        print(f"\n=== Evaluating {checkpoint_file} ===")
        model, error = load_checkpoint(checkpoint_path)
        if model is None:
            load_failures.append({"checkpoint": checkpoint_name, "path": str(checkpoint_path), "error": error})
            print(f"FAILED to load {checkpoint_file}: {error}")
            continue

        image_size = infer_image_size(model, args.default_image_size)
        loaded_rows.append({"checkpoint": checkpoint_name, "path": str(checkpoint_path), "image_size": image_size})
        predictions: Dict[str, pd.DataFrame] = {}
        for split in SPLITS:
            predictions[split] = predict_split(model, split_frames[split], image_size, args.batch_size)
            split_metric_rows.append(regression_metrics(predictions[split], "guarded_target_0_100", split, "overall", checkpoint_name))
            safety_rows.append(safety_metrics(predictions[split], split, "overall", checkpoint_name))
            for dataset, group in predictions[split].groupby("dataset", sort=True):
                safety_rows.append(safety_metrics(group, split, str(dataset), checkpoint_name))

        for dataset, group in predictions["test"].groupby("dataset", sort=True):
            dataset_metric_rows.append(regression_metrics(group, "guarded_target_0_100", "test", str(dataset), checkpoint_name))

        mos_metric_rows.append(regression_metrics(predictions["test"], "quality_score_0_100", "test", "overall", checkpoint_name))
        for dataset, group in predictions["test"].groupby("dataset", sort=True):
            mos_metric_rows.append(regression_metrics(group, "quality_score_0_100", "test", str(dataset), checkpoint_name))

        val_preds = predictions["val"][PRED_COL].to_numpy(dtype=np.float64)
        val_targets = predictions["val"]["guarded_target_0_100"].to_numpy(dtype=np.float64)
        test_preds = predictions["test"][PRED_COL].to_numpy(dtype=np.float64)
        calibrated_preds, method = evaluate_posthoc_calibration(val_preds, val_targets, test_preds)
        calibration_rows.extend(calibration_metrics(predictions["test"], calibrated_preds, checkpoint_name, method))

        if args.save_predictions:
            pred_cols = [
                "image_path",
                "dataset",
                "student_split",
                "source_group",
                "quality_score_0_100",
                "guarded_target_0_100",
                "pred_norm",
                "pred_0_100",
                "abs_error_to_target",
                "abs_error_to_mos",
            ]
            predictions["test"].to_csv(out_dir / f"predictions_test_{checkpoint_name}.csv", columns=[c for c in pred_cols if c in predictions["test"].columns], index=False)
            calibrated_df = predictions["test"].copy()
            calibrated_df["pred_0_100_calibrated"] = calibrated_preds
            calibrated_df.to_csv(out_dir / f"predictions_test_{checkpoint_name}_calibrated.csv", index=False)

        del model
        tf.keras.backend.clear_session()

    split_metrics_df = pd.DataFrame(split_metric_rows)
    dataset_metrics_df = pd.DataFrame(dataset_metric_rows)
    mos_metrics_df = pd.DataFrame(mos_metric_rows)
    safety_metrics_df = pd.DataFrame(safety_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    load_failures_df = pd.DataFrame(load_failures)
    loaded_df = pd.DataFrame(loaded_rows)

    summary_rows = []
    for checkpoint in loaded_df["checkpoint"].tolist() if not loaded_df.empty else []:
        summary_rows.append(checkpoint_summary_row(checkpoint, split_metric_rows, dataset_metric_rows, safety_rows))
    summary_df = pd.DataFrame(summary_rows)

    baseline_df = pd.DataFrame(BASELINES)
    if not summary_df.empty:
        summary_with_baselines = pd.concat([baseline_df, summary_df], ignore_index=True, sort=False)
    else:
        summary_with_baselines = baseline_df

    best_raw = choose_best_raw(summary_df)
    best_calibrated = choose_best_calibrated(calibration_df)
    decision, reasons = final_decision(summary_df, load_failures_df, best_raw)

    summary_with_baselines.to_csv(out_dir / "checkpoint_reeval_summary.csv", index=False)
    dataset_metrics_df.to_csv(out_dir / "checkpoint_dataset_metrics.csv", index=False)
    mos_metrics_df.to_csv(out_dir / "checkpoint_mos_reference_metrics.csv", index=False)
    safety_metrics_df.to_csv(out_dir / "checkpoint_safety_metrics.csv", index=False)
    calibration_df.to_csv(out_dir / "checkpoint_calibration_metrics.csv", index=False)
    load_failures_df.to_csv(out_dir / "checkpoint_load_failures.csv", index=False)
    loaded_df.to_csv(out_dir / "checkpoint_load_successes.csv", index=False)

    write_recommendation(out_dir / "best_checkpoint_recommendation.md", best_raw, best_calibrated, decision)
    write_report(
        out_dir / "report.md",
        args,
        split_counts,
        image_errors,
        load_failures_df,
        summary_df,
        dataset_metrics_df,
        mos_metrics_df,
        safety_metrics_df,
        calibration_df,
        best_raw,
        best_calibrated,
        decision,
        reasons,
    )
    print(f"\nFinal decision: {decision}")
    print(f"Report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
