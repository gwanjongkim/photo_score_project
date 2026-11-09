# 50k MDIQA student controlled B0 학습과 공식 Track B 평가를 실행한다.
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = ROOT / "scripts" / "technical_iqa" / "train_mobile_student_iqa_50k_smoke.py"
DEFAULT_MANIFEST = ROOT / "outputs" / "mdiqa_distillation_50k_full_targets_20260608" / "final_50k_mdiqa_distillation_manifest.csv"
DEFAULT_OFFICIAL = ROOT / "outputs" / "official_protocol_benchmark_manifest_20260604" / "official_benchmark_test_manifest.csv"
DEFAULT_OUT_DIR = ROOT / "outputs" / "mobile_student_iqa_50k_controlled_b0_20260608"
SOURCE_REPORTS = [
    ROOT / "outputs" / "mdiqa_distillation_50k_full_targets_20260608" / "report.md",
    ROOT / "outputs" / "mobile_student_iqa_50k_gpu_visibility_fix_20260608" / "gpu_preflight_report.md",
    ROOT / "outputs" / "mobile_student_iqa_50k_smoke_gpu_fixed_venv_tf_20260608" / "report.md",
]

BASELINE_14K_OFFICIAL_SRCC = 0.7436
MDIQA_TEACHER_OFFICIAL_SRCC = 0.8227
TOPIQ_FLIVE_LOG_ONLY_NOTE = "topiq_existing_flive_57_43 was log-only and not used as a direct checkpoint baseline."
PRED_COL = "pred_0_100"
SPLITS = ["train", "val", "test"]
PHASE18_W5LITE_OFFICIAL_SRCC = 0.530639
PHASE16_UNWEIGHTED_B0_OFFICIAL_SRCC = 0.703290
W5V2_TARGET_SHARES = {"flive": 0.30, "koniq10k": 0.35, "spaq": 0.35}


def load_smoke_module() -> Any:
    spec = importlib.util.spec_from_file_location("train_mobile_student_iqa_50k_smoke", SMOKE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import smoke trainer from {SMOKE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = load_smoke_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one controlled B0 Mobile Student IQA training pass on 50k MDIQA targets.")
    parser.add_argument("--manifest_csv", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--official_manifest_csv", default=str(DEFAULT_OFFICIAL))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--backbone", choices=["efficientnetv2b0", "efficientnetb0", "mobilenetv3small"], default="efficientnetv2b0")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--head_epochs", type=int, default=5)
    parser.add_argument("--finetune_epochs", type=int, default=10)
    parser.add_argument("--unfreeze_last_n", type=int, default=16)
    parser.add_argument("--learning_rate_head", type=float, default=3e-4)
    parser.add_argument("--learning_rate_finetune", type=float, default=1e-5)
    parser.add_argument("--loss", choices=["huber", "mae", "mse"], default="huber")
    parser.add_argument("--huber_delta", type=float, default=0.05)
    parser.add_argument("--backbone_weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--dataset_weight_flive", type=float, default=1.0)
    parser.add_argument("--dataset_weight_koniq10k", type=float, default=1.0)
    parser.add_argument("--dataset_weight_spaq", type=float, default=1.0)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--skip_official_eval", action="store_true")
    parser.add_argument("--seed", type=int, default=20260608)
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


def safety_counts(df: pd.DataFrame, mos_col: str = "quality_score_0_100") -> dict[str, int]:
    mos = pd.to_numeric(df[mos_col], errors="coerce").to_numpy(dtype=np.float64)
    pred = pd.to_numeric(df[PRED_COL], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(mos) & np.isfinite(pred)
    mos = mos[finite]
    pred = pred[finite]
    return {
        "fp_count": int(((mos < 40.0) & (pred > 60.0)).sum()),
        "severe_fp_count": int(((mos < 40.0) & (pred > 65.0)).sum()),
        "fn_count": int(((mos > 75.0) & (pred < 45.0)).sum()),
    }


def metric_tables(pred_frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics, dataset_metrics, prediction_summary = smoke.metric_tables(pred_frames)
    safety_rows = []
    for split, df in pred_frames.items():
        row = {"split": split, "dataset": "overall", "n": len(df)}
        row.update(safety_counts(df))
        safety_rows.append(row)
        for dataset, group in df.groupby("dataset", sort=True):
            row = {"split": split, "dataset": str(dataset), "n": len(group)}
            row.update(safety_counts(group))
            safety_rows.append(row)
    safety_df = pd.DataFrame(safety_rows)
    return metrics, dataset_metrics, prediction_summary, safety_df


def prepare_official_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"dataset", "content_type", "image_path", "image_exists", "mos_0_100"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"official manifest missing columns: {missing}")
    df = df[df["content_type"].astype(str).eq("image")].copy()
    df["dataset"] = df["dataset"].astype(str).str.lower()
    df["image_path"] = df["image_path"].astype(str).map(lambda value: str((ROOT / value).resolve()) if not Path(value).is_absolute() else value)
    df["selected_teacher_target_0_100"] = pd.to_numeric(df["mos_0_100"], errors="coerce")
    df["target_norm"] = df["selected_teacher_target_0_100"] / 100.0
    df["student_split"] = "official_track_b"
    return df


def validate_official_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    renamed = df.rename(columns={"mos_0_100": "quality_score_0_100"}).copy()
    return smoke.validate_rows(renamed)


def predict_official(model: tf.keras.Model, official_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    pred_df = smoke.predict_split(model, official_df, args)
    pred_df["mos_0_100"] = pred_df["selected_teacher_target_0_100"]
    pred_df["abs_error_to_mos_0_100"] = pred_df["abs_error_to_target_0_100"]
    return pred_df


def official_metric_tables(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = [smoke.regression_metrics(pred_df, "official_track_b", "overall")]
    dataset_rows = []
    safety_rows = []
    overall_safety = {"split": "official_track_b", "dataset": "overall", "n": len(pred_df)}
    overall_safety.update(safety_counts(pred_df, "mos_0_100"))
    safety_rows.append(overall_safety)
    for dataset, group in pred_df.groupby("dataset", sort=True):
        metric = smoke.regression_metrics(group, "official_track_b", str(dataset))
        rows.append(metric)
        dataset_rows.append(metric)
        safety = {"split": "official_track_b", "dataset": str(dataset), "n": len(group)}
        safety.update(safety_counts(group, "mos_0_100"))
        safety_rows.append(safety)
    return pd.DataFrame(rows), pd.DataFrame(dataset_rows), pd.DataFrame(safety_rows)


def is_weighted_smoke(args: argparse.Namespace) -> bool:
    weights = smoke.dataset_weight_map(args)
    return any(not math.isclose(value, 1.0) for value in weights.values())


def is_w5v2_smoke(args: argparse.Namespace) -> bool:
    weights = smoke.dataset_weight_map(args)
    expected = {"flive": 0.443, "koniq10k": 2.433, "spaq": 1.963}
    return all(math.isclose(weights[key], expected[key], rel_tol=0.0, abs_tol=1e-6) for key in expected)


def effective_contribution_audit(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        frame = frames[split]
        total_count = len(frame)
        weighted_total = float(frame["dataset_sample_weight"].sum()) if len(frame) else 0.0
        for dataset, group in frame.groupby("dataset", sort=True):
            configured_weight = float(group["dataset_sample_weight"].iloc[0]) if len(group) else float("nan")
            weighted_count = float(len(group) * configured_weight)
            effective_share = float(weighted_count / weighted_total) if weighted_total else float("nan")
            target_share = W5V2_TARGET_SHARES.get(str(dataset), float("nan"))
            rows.append(
                {
                    "split": split,
                    "dataset": str(dataset),
                    "raw_count": int(len(group)),
                    "raw_share": float(len(group) / total_count) if total_count else float("nan"),
                    "configured_sample_weight": configured_weight,
                    "effective_weighted_count": weighted_count,
                    "effective_share": effective_share,
                    "target_effective_share": target_share,
                    "abs_delta_from_target": abs(effective_share - target_share) if math.isfinite(effective_share) and math.isfinite(target_share) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def train_contribution_close(contribution_audit: pd.DataFrame, tolerance: float = 0.03) -> bool:
    if contribution_audit.empty:
        return False
    train_rows = contribution_audit[contribution_audit["split"].eq("train")]
    if len(train_rows) != len(W5V2_TARGET_SHARES):
        return False
    return bool((train_rows["abs_delta_from_target"] <= tolerance).all())


class BestValMAECheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, out_dir: Path) -> None:
        super().__init__()
        self.out_dir = out_dir
        self.best_val_mae = float("inf")
        self.records: list[dict[str, Any]] = []
        self.phase = "head"
        self.offset = 0

    def set_phase(self, phase: str, offset: int) -> None:
        self.phase = phase
        self.offset = offset

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs or {}
        val_mae = float(logs.get("val_mae", float("nan")))
        saved = bool(math.isfinite(val_mae) and val_mae < self.best_val_mae)
        if saved:
            self.best_val_mae = val_mae
            self.model.save(self.out_dir / "best_by_val_mae_model.keras", include_optimizer=False)
        record = {
            "phase": self.phase,
            "phase_epoch": epoch + 1,
            "global_epoch": self.offset + epoch + 1,
            "val_mae": val_mae,
            "val_loss": float(logs.get("val_loss", float("nan"))),
            "saved_best_by_val_mae": saved,
            "best_val_mae": self.best_val_mae,
        }
        self.records.append(record)
        for key, value in record.items():
            if isinstance(value, (int, float, bool, np.integer, np.floating)):
                logs[key] = value
        pd.DataFrame(self.records).to_csv(self.out_dir / "checkpoint_metrics.csv", index=False)


def final_decision(
    training_ok: bool,
    env_info: dict[str, Any],
    failures: pd.DataFrame,
    history: pd.DataFrame,
    collapse: dict[str, Any],
    official_metrics: pd.DataFrame,
    official_safety: pd.DataFrame,
) -> str:
    if not training_ok:
        return "STUDENT_50K_B0_FAILED"
    if env_info.get("gpu_count", 0) != 1:
        return "STUDENT_50K_B0_NEEDS_PIPELINE_FIX"
    if len(failures) > 0:
        return "STUDENT_50K_B0_NEEDS_PIPELINE_FIX"
    if history.empty or not np.isfinite(history.select_dtypes(include=[np.number]).to_numpy()).all():
        return "STUDENT_50K_B0_NEEDS_PIPELINE_FIX"
    if collapse.get("nan_inf_count", 0) or collapse.get("pred_outside_0_100") or collapse.get("pred_std_too_low"):
        return "STUDENT_50K_B0_NEEDS_TUNING"
    if official_metrics.empty:
        return "STUDENT_50K_B0_READY_FOR_CHECKPOINT_EVAL"
    official_overall = official_metrics[(official_metrics["split"].eq("official_track_b")) & (official_metrics["dataset"].eq("overall"))]
    if official_overall.empty:
        return "STUDENT_50K_B0_READY_FOR_CHECKPOINT_EVAL"
    srcc = float(official_overall.iloc[0]["srcc"])
    official_safety_row = official_safety[official_safety["dataset"].eq("overall")]
    severe_fp = int(official_safety_row.iloc[0]["severe_fp_count"]) if len(official_safety_row) else 999999
    improved = math.isfinite(srcc) and srcc >= BASELINE_14K_OFFICIAL_SRCC + 0.01
    gap_remaining = math.isfinite(srcc) and srcc < MDIQA_TEACHER_OFFICIAL_SRCC - 0.02
    severe_fp_ok = severe_fp <= 12
    if improved and severe_fp_ok and gap_remaining:
        return "STUDENT_50K_B0_READY_FOR_B1_B2_COMPARISON"
    return "STUDENT_50K_B0_NEEDS_TUNING"


def weighted_smoke_final_decision(
    training_ok: bool,
    env_info: dict[str, Any],
    failures: pd.DataFrame,
    history: pd.DataFrame,
    collapse: dict[str, Any],
    official_metrics: pd.DataFrame,
) -> str:
    if not training_ok:
        return "WEIGHTED_B0_SMOKE_FAILED"
    if env_info.get("gpu_count", 0) < 1:
        return "WEIGHTED_B0_SMOKE_NEEDS_PIPELINE_FIX"
    if len(failures) > 0:
        return "WEIGHTED_B0_SMOKE_NEEDS_PIPELINE_FIX"
    if history.empty or not np.isfinite(history.select_dtypes(include=[np.number]).to_numpy()).all():
        return "WEIGHTED_B0_SMOKE_NEEDS_PIPELINE_FIX"
    if collapse.get("nan_inf_count", 0) or collapse.get("pred_outside_0_100"):
        return "WEIGHTED_B0_SMOKE_NEEDS_PIPELINE_FIX"
    if collapse.get("pred_std_too_low"):
        return "WEIGHTED_B0_SMOKE_NEEDS_WEIGHT_TUNING"
    if not official_metrics.empty:
        official_overall = official_metrics[(official_metrics["split"].eq("official_track_b")) & (official_metrics["dataset"].eq("overall"))]
        if len(official_overall):
            srcc = float(official_overall.iloc[0]["srcc"])
            if math.isfinite(srcc) and srcc < 0.68:
                return "WEIGHTED_B0_SMOKE_NEEDS_WEIGHT_TUNING"
    return "WEIGHTED_B0_SMOKE_READY_FOR_CONTROLLED_RERUN"


def w5v2_smoke_final_decision(
    training_ok: bool,
    env_info: dict[str, Any],
    failures: pd.DataFrame,
    history: pd.DataFrame,
    collapse: dict[str, Any],
    contribution_audit: pd.DataFrame,
    official_metrics: pd.DataFrame,
) -> str:
    if not training_ok:
        return "WEIGHTED_B0_W5V2_SMOKE_FAILED"
    if env_info.get("gpu_count", 0) != 1:
        return "WEIGHTED_B0_W5V2_SMOKE_NEEDS_PIPELINE_FIX"
    if len(failures) > 0:
        return "WEIGHTED_B0_W5V2_SMOKE_NEEDS_PIPELINE_FIX"
    if history.empty or not np.isfinite(history.select_dtypes(include=[np.number]).to_numpy()).all():
        return "WEIGHTED_B0_W5V2_SMOKE_NEEDS_PIPELINE_FIX"
    if collapse.get("nan_inf_count", 0) or collapse.get("pred_outside_0_100"):
        return "WEIGHTED_B0_W5V2_SMOKE_NEEDS_PIPELINE_FIX"
    if collapse.get("pred_std_too_low"):
        return "WEIGHTED_B0_W5V2_SMOKE_NEEDS_WEIGHT_TUNING"
    if not train_contribution_close(contribution_audit):
        return "WEIGHTED_B0_W5V2_SMOKE_NEEDS_WEIGHT_TUNING"
    if official_metrics.empty:
        return "WEIGHTED_B0_W5V2_SMOKE_NEEDS_PIPELINE_FIX"
    official_overall = official_metrics[(official_metrics["split"].eq("official_track_b")) & (official_metrics["dataset"].eq("overall"))]
    if official_overall.empty:
        return "WEIGHTED_B0_W5V2_SMOKE_NEEDS_PIPELINE_FIX"
    srcc = float(official_overall.iloc[0]["srcc"])
    if not math.isfinite(srcc):
        return "WEIGHTED_B0_W5V2_SMOKE_NEEDS_PIPELINE_FIX"
    if srcc >= PHASE16_UNWEIGHTED_B0_OFFICIAL_SRCC - 0.02 and srcc > PHASE18_W5LITE_OFFICIAL_SRCC + 0.03:
        return "WEIGHTED_B0_W5V2_SMOKE_READY_FOR_CONTROLLED_RERUN"
    if srcc > PHASE18_W5LITE_OFFICIAL_SRCC + 0.03:
        return "WEIGHTED_B0_W5V2_SMOKE_NEEDS_WEIGHT_TUNING"
    return "WEIGHTED_B0_W5V2_SMOKE_NEEDS_LABEL_SCALING_OR_RANKING"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    return smoke.markdown_table(df, [col for col in columns if col in df.columns])


def write_next_actions(out_dir: Path, decision: str) -> None:
    if decision == "WEIGHTED_B0_W5V2_SMOKE_READY_FOR_CONTROLLED_RERUN":
        rows = [{"priority": 1, "action": "Run a separate controlled Weighted B0 rerun with W5-v2 inverse-frequency weights.", "status": "recommended"}]
    elif decision == "WEIGHTED_B0_W5V2_SMOKE_NEEDS_WEIGHT_TUNING":
        rows = [{"priority": 1, "action": "Tune inverse-frequency target shares before controlled rerun; keep B1/B2 paused.", "status": "required"}]
    elif decision == "WEIGHTED_B0_W5V2_SMOKE_NEEDS_LABEL_SCALING_OR_RANKING":
        rows = [{"priority": 1, "action": "Investigate KonIQ label scaling or ranking loss before more dataset-weight sweeps.", "status": "required"}]
    elif decision == "WEIGHTED_B0_W5V2_SMOKE_NEEDS_PIPELINE_FIX":
        rows = [{"priority": 1, "action": "Fix GPU, sample-weight audit, dataloader, metric, or reporting issue before rerun.", "status": "blocked"}]
    elif decision == "WEIGHTED_B0_W5V2_SMOKE_FAILED":
        rows = [{"priority": 1, "action": "Inspect failure_cases.csv and runtime logs before retry.", "status": "blocked"}]
    elif decision == "WEIGHTED_B0_SMOKE_READY_FOR_CONTROLLED_RERUN":
        rows = [{"priority": 1, "action": "Run a separate controlled Weighted B0 rerun with the same W5-lite dataset weights.", "status": "recommended"}]
    elif decision == "WEIGHTED_B0_SMOKE_NEEDS_WEIGHT_TUNING":
        rows = [{"priority": 1, "action": "Tune dataset weights before controlled rerun; keep B1/B2 paused.", "status": "required"}]
    elif decision == "WEIGHTED_B0_SMOKE_NEEDS_PIPELINE_FIX":
        rows = [{"priority": 1, "action": "Fix GPU, dataloader, sample-weight, metric, or reporting issue before rerun.", "status": "blocked"}]
    elif decision == "WEIGHTED_B0_SMOKE_FAILED":
        rows = [{"priority": 1, "action": "Inspect failure_cases.csv and runtime logs before retry.", "status": "blocked"}]
    else:
        rows = []
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "next_actions.csv", index=False)
        return
    if decision == "STUDENT_50K_B0_READY_FOR_B1_B2_COMPARISON":
        rows = [{"priority": 1, "action": "Prepare a separate B1/B2 comparison plan; do not start it from this run.", "status": "recommended"}]
    elif decision == "STUDENT_50K_B0_READY_FOR_CHECKPOINT_EVAL":
        rows = [{"priority": 1, "action": "Run separate checkpoint evaluation on official benchmark before backbone changes.", "status": "recommended"}]
    elif decision == "STUDENT_50K_B0_NEEDS_TUNING":
        rows = [{"priority": 1, "action": "Analyze official benchmark errors, severe FP cases, and scale compression before any B1/B2 experiment.", "status": "required"}]
    elif decision == "STUDENT_50K_B0_NEEDS_PIPELINE_FIX":
        rows = [{"priority": 1, "action": "Fix GPU, split, dataloader, or reporting issue before rerun.", "status": "blocked"}]
    else:
        rows = [{"priority": 1, "action": "Inspect failure_cases.csv and runtime logs before retry.", "status": "blocked"}]
    pd.DataFrame(rows).to_csv(out_dir / "next_actions.csv", index=False)


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    env_info: dict[str, Any],
    build_info: dict[str, Any],
    split_meta: dict[str, Any],
    history: pd.DataFrame,
    checkpoint_metrics: pd.DataFrame,
    metrics: pd.DataFrame,
    dataset_metrics: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    safety_metrics: pd.DataFrame,
    official_metrics: pd.DataFrame,
    official_dataset_metrics: pd.DataFrame,
    official_safety: pd.DataFrame,
    failures: pd.DataFrame,
    official_failures: pd.DataFrame,
    contribution_audit: pd.DataFrame,
    collapse: dict[str, Any],
    decision: str,
    runtime_seconds: float,
    finetune_info: dict[str, Any],
) -> None:
    final_train_loss = float(history["loss"].iloc[-1]) if not history.empty and "loss" in history else float("nan")
    final_val_loss = float(history["val_loss"].iloc[-1]) if not history.empty and "val_loss" in history else float("nan")
    best_row = checkpoint_metrics[checkpoint_metrics["saved_best_by_val_mae"].eq(True)].tail(1)
    best_summary = best_row.iloc[0].to_dict() if len(best_row) else {}
    metric_cols = ["split", "dataset", "n", "finite_n", "mae_0_100", "rmse_0_100", "srcc", "plcc", "pred_min", "pred_max", "pred_mean", "pred_std", "target_min", "target_max", "target_mean", "target_std", "nan_inf_count"]
    safety_cols = ["split", "dataset", "n", "fp_count", "severe_fp_count", "fn_count"]
    official_overall = official_metrics[(official_metrics["split"].eq("official_track_b")) & (official_metrics["dataset"].eq("overall"))] if not official_metrics.empty else pd.DataFrame()
    official_srcc = float(official_overall.iloc[0]["srcc"]) if len(official_overall) else float("nan")
    comparison = pd.DataFrame(
        [
            {"system": "unweighted 50k B0 smoke", "official_srcc": float("nan"), "delta_vs_14k": float("nan"), "note": "prior smoke had no official Track B table"},
            {"system": "Phase 18 W5-lite smoke", "official_srcc": PHASE18_W5LITE_OFFICIAL_SRCC, "delta_vs_14k": PHASE18_W5LITE_OFFICIAL_SRCC - BASELINE_14K_OFFICIAL_SRCC, "note": "raw sample weights"},
            {"system": "unweighted 50k B0 controlled", "official_srcc": PHASE16_UNWEIGHTED_B0_OFFICIAL_SRCC, "delta_vs_14k": PHASE16_UNWEIGHTED_B0_OFFICIAL_SRCC - BASELINE_14K_OFFICIAL_SRCC, "note": "Phase 16"},
            {"system": "14k student official", "official_srcc": BASELINE_14K_OFFICIAL_SRCC, "delta_vs_14k": 0.0, "note": "previous student"},
            {"system": "MDIQA teacher official", "official_srcc": MDIQA_TEACHER_OFFICIAL_SRCC, "delta_vs_14k": MDIQA_TEACHER_OFFICIAL_SRCC - BASELINE_14K_OFFICIAL_SRCC, "note": "teacher ceiling reference"},
            {"system": "W5-v2 inverse-frequency smoke" if is_w5v2_smoke(args) else "weighted 50k B0 smoke best_by_val_mae" if is_weighted_smoke(args) else "50k B0 best_by_val_mae", "official_srcc": official_srcc, "delta_vs_14k": official_srcc - BASELINE_14K_OFFICIAL_SRCC if math.isfinite(official_srcc) else float("nan"), "note": "this run"},
        ]
    )
    contribution_cols = ["split", "dataset", "raw_count", "raw_share", "configured_sample_weight", "effective_weighted_count", "effective_share", "target_effective_share", "abs_delta_from_target"]
    if is_w5v2_smoke(args):
        lines = [
            "# Mobile Student IQA 50k Weighted B0 Smoke W5-v2 Inverse-Frequency Report",
            "",
            "## 1. Executive Summary",
            f"- Final decision: `{decision}`.",
            "- Run type: inverse-frequency corrected weighted B0 smoke only; no B1/B2, no full controlled training, no TFLite export, no Flutter edits, no production `technical_score` edits.",
            f"- Output directory: `{out_dir}`.",
            f"- Runtime seconds: `{runtime_seconds:.2f}`.",
            f"- Train/val/test counts: `{split_meta.get('usable_counts')}`.",
            f"- Final train loss: `{final_train_loss:.6f}`.",
            f"- Final val loss: `{final_val_loss:.6f}`.",
            "",
            "## 2. Sources Inspected",
            f"- `{args.manifest_csv}`",
            f"- `{args.official_manifest_csv}`",
            "- `scripts/technical_iqa/train_mobile_student_iqa_50k_controlled.py`",
            "- `scripts/technical_iqa/train_mobile_student_iqa_50k_smoke.py`",
            "- `outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/report.md`",
            "- `outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609/report.md`",
            "",
            "## 3. Why W5-lite Failed",
            "- W5-lite used raw per-sample weights `0.30/0.35/0.35`.",
            "- Because sample weights multiply each sample loss, FLIVE still dominated due to higher raw sample count.",
            "- Phase 18 effective train contribution remained about FLIVE `64.4%`, KonIQ-10k `15.9%`, SPAQ `19.7%`.",
            f"- Phase 18 official Track B SRCC was `{PHASE18_W5LITE_OFFICIAL_SRCC}`.",
            "",
            "## 4. W5-v2 Weighting Policy",
            "- Full 50k observed shares: FLIVE `0.67782`, KonIQ-10k `0.14386`, SPAQ `0.17832`.",
            "- Target effective shares: FLIVE `0.30`, KonIQ-10k `0.35`, SPAQ `0.35`.",
            f"- Applied per-sample weights: `{smoke.dataset_weight_map(args)}`.",
            "- Validation, test prediction, and official benchmark evaluation are unweighted.",
            "- Target values and model architecture are unchanged.",
            "",
            "## 5. Effective Contribution Audit",
            markdown_table(contribution_audit, contribution_cols),
            "",
            "## 6. Code Changes",
            "- Added `dataset_weight_effective_contribution.csv` audit for train/val/test manifests.",
            "- Added W5-v2-specific report sections and decision labels.",
            "- Reused Phase 18 train-only sample-weight path; no architecture or target-value change.",
            "",
            "## 7. Environment / GPU Preflight",
            f"- TensorFlow: `{env_info.get('tf_version')}`.",
            f"- CUDA build: `{env_info.get('built_cuda')}`.",
            f"- GPUs visible: `{env_info.get('gpus')}`.",
            f"- GPU count: `{env_info.get('gpu_count')}`.",
            f"- Correct environment: `.venv_tf`.",
            "",
            "## 8. Smoke Training Configuration",
            f"- Backbone: `{build_info.get('backbone')}`.",
            f"- Image size: `{args.image_size}`.",
            f"- Batch size: `{args.batch_size}`.",
            f"- Head epochs: `{args.head_epochs}`.",
            f"- Finetune epochs: `{args.finetune_epochs}`.",
            f"- Unfreeze last N: `{args.unfreeze_last_n}`.",
            f"- Loss: `{args.loss}`.",
            f"- Target: `teacher_target_0_100 / 100.0`.",
            f"- ImageNet weights requested: `{args.backbone_weights}`.",
            f"- ImageNet weights loaded: `{build_info.get('backbone_weights_loaded')}`.",
            f"- Finetune trainable backbone layers: `{finetune_info.get('backbone_trainable_layers')}`.",
            "",
            "## 9. Training History",
            markdown_table(history, list(history.columns)),
            "",
            "## 10. Prediction and Metric Summary",
            markdown_table(prediction_summary, metric_cols),
            "",
            "## 11. Dataset-wise Metrics",
            markdown_table(dataset_metrics, metric_cols),
            "",
            "## 12. Official Benchmark Evaluation",
            f"- Official evaluation failures: `{len(official_failures)}`.",
            markdown_table(official_metrics, metric_cols),
            "",
            markdown_table(official_dataset_metrics, metric_cols),
            "",
            markdown_table(official_safety, safety_cols),
            "",
            "## 13. Comparison Against W5-lite, Unweighted B0, and 14k Student",
            markdown_table(comparison, ["system", "official_srcc", "delta_vs_14k", "note"]),
            f"- MDIQA teacher official SRCC: `{MDIQA_TEACHER_OFFICIAL_SRCC}`.",
            "",
            "## 14. Collapse / Safety Checks",
            f"- Training/input failure rows: `{len(failures)}`.",
            f"- Official failure rows: `{len(official_failures)}`.",
            f"- Collapse check: `{collapse}`.",
            "",
            markdown_table(safety_metrics, safety_cols),
            "",
            "## 15. Final Decision",
            decision,
            "",
            "## 16. Recommended Next Step",
        ]
        if decision == "WEIGHTED_B0_W5V2_SMOKE_READY_FOR_CONTROLLED_RERUN":
            lines.append("- Run controlled W5-v2 B0 rerun as a separate explicitly approved task.")
        elif decision == "WEIGHTED_B0_W5V2_SMOKE_NEEDS_LABEL_SCALING_OR_RANKING":
            lines.append("- Keep B1/B2 paused and investigate KonIQ label scaling or ranking loss.")
        elif decision == "WEIGHTED_B0_W5V2_SMOKE_NEEDS_WEIGHT_TUNING":
            lines.append("- Tune target effective shares before controlled rerun.")
        elif decision == "WEIGHTED_B0_W5V2_SMOKE_NEEDS_PIPELINE_FIX":
            lines.append("- Fix pipeline or audit issue, then rerun W5-v2 smoke.")
        else:
            lines.append("- Inspect failure cases and runtime logs before retry.")
        (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    if is_weighted_smoke(args):
        lines = [
            "# Mobile Student IQA 50k Weighted B0 Smoke W5-lite Report",
            "",
            "## 1. Executive Summary",
            f"- Final decision: `{decision}`.",
            "- Run type: weighted B0 smoke only; no B1/B2, no TFLite export, no Flutter edits, no production `technical_score` edits.",
            f"- Output directory: `{out_dir}`.",
            f"- Runtime seconds: `{runtime_seconds:.2f}`.",
            f"- Train/val/test counts: `{split_meta.get('usable_counts')}`.",
            f"- Final train loss: `{final_train_loss:.6f}`.",
            f"- Final val loss: `{final_val_loss:.6f}`.",
            "",
            "## 2. Sources Inspected",
            f"- `{args.manifest_csv}`",
            f"- `{args.official_manifest_csv}`",
            "- `scripts/technical_iqa/train_mobile_student_iqa_50k_controlled.py`",
            "- `scripts/technical_iqa/train_mobile_student_iqa_50k_smoke.py`",
            "- `outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/recommendation.json`",
            "- `outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/report.md`",
            "",
            "## 3. Weighting Policy",
            f"- Dataset weights: `{smoke.dataset_weight_map(args)}`.",
            "- Training dataset emits Keras sample weights from `dataset_sample_weight`.",
            "- Validation, test prediction, and official benchmark evaluation are unweighted.",
            "- Target values, model architecture, and official MOS labels are unchanged.",
            "",
            "## 4. Code Changes",
            "- Added dataset-weight CLI args to 50k smoke and controlled runners with default `1.0` values.",
            "- Added `dataset_sample_weight` to split manifests for auditability.",
            "- Applied sample weights only when building the training dataset.",
            "",
            "## 5. Environment / GPU Preflight",
            f"- TensorFlow: `{env_info.get('tf_version')}`.",
            f"- CUDA build: `{env_info.get('built_cuda')}`.",
            f"- GPUs visible: `{env_info.get('gpus')}`.",
            f"- GPU count: `{env_info.get('gpu_count')}`.",
            f"- Correct environment: `.venv_tf`.",
            "",
            "## 6. Smoke Training Configuration",
            f"- Backbone: `{build_info.get('backbone')}`.",
            f"- Image size: `{args.image_size}`.",
            f"- Batch size: `{args.batch_size}`.",
            f"- Head epochs: `{args.head_epochs}`.",
            f"- Finetune epochs: `{args.finetune_epochs}`.",
            f"- Unfreeze last N: `{args.unfreeze_last_n}`.",
            f"- Loss: `{args.loss}`.",
            f"- Target: `teacher_target_0_100 / 100.0`.",
            f"- ImageNet weights requested: `{args.backbone_weights}`.",
            f"- ImageNet weights loaded: `{build_info.get('backbone_weights_loaded')}`.",
            f"- Finetune trainable backbone layers: `{finetune_info.get('backbone_trainable_layers')}`.",
            "",
            "## 7. Training History",
            markdown_table(history, list(history.columns)),
            "",
            "## 8. Prediction and Metric Summary",
            markdown_table(prediction_summary, metric_cols),
            "",
            "## 9. Dataset-wise Metrics",
            markdown_table(dataset_metrics, metric_cols),
            "",
            "## 10. Official Benchmark Evaluation",
            f"- Official evaluation failures: `{len(official_failures)}`.",
            markdown_table(official_metrics, metric_cols),
            "",
            markdown_table(official_dataset_metrics, metric_cols),
            "",
            markdown_table(official_safety, safety_cols),
            "",
            "## 11. Comparison Against Unweighted B0 and 14k Student",
            markdown_table(comparison, ["system", "official_srcc", "delta_vs_14k", "note"]),
            f"- MDIQA teacher official SRCC: `{MDIQA_TEACHER_OFFICIAL_SRCC}`.",
            "",
            "## 12. Collapse / Safety Checks",
            f"- Training/input failure rows: `{len(failures)}`.",
            f"- Official failure rows: `{len(official_failures)}`.",
            f"- Collapse check: `{collapse}`.",
            "",
            markdown_table(safety_metrics, safety_cols),
            "",
            "## 13. Final Decision",
            decision,
            "",
            "## 14. Recommended Next Step",
        ]
        if decision == "WEIGHTED_B0_SMOKE_READY_FOR_CONTROLLED_RERUN":
            lines.append("- Run controlled Weighted B0 rerun with the same W5-lite weights before any B1/B2 work.")
        elif decision == "WEIGHTED_B0_SMOKE_NEEDS_WEIGHT_TUNING":
            lines.append("- Tune W5-lite weights and rerun smoke; keep B1/B2 paused.")
        elif decision == "WEIGHTED_B0_SMOKE_NEEDS_PIPELINE_FIX":
            lines.append("- Fix pipeline or sample-weight issue, then rerun the same weighted smoke.")
        else:
            lines.append("- Inspect failure cases and runtime logs before retry.")
        (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines = [
        "# Mobile Student IQA 50k Controlled B0 Training Report",
        "",
        "## 1. Executive Summary",
        f"- Final decision: `{decision}`.",
        "- Run type: controlled B0 baseline; not final production training.",
        f"- Output directory: `{out_dir}`.",
        f"- Runtime seconds: `{runtime_seconds:.2f}`.",
        f"- Train/val/test counts: `{split_meta.get('usable_counts')}`.",
        f"- Final train loss: `{final_train_loss:.6f}`.",
        f"- Final val loss: `{final_val_loss:.6f}`.",
        "",
        "## 2. Sources Inspected",
        f"- `{args.manifest_csv}`",
        f"- `{args.official_manifest_csv}`",
        *[f"- `{path}`" for path in SOURCE_REPORTS],
        "- `scripts/technical_iqa/train_mobile_student_iqa_50k_smoke.py`",
        "- `scripts/technical_iqa/train_mobile_student_iqa_14105_smoke.py`",
        "- `scripts/technical_iqa/train_mobile_student_iqa_v4a.py`",
        "- `scripts/technical_iqa/train_mobile_student_iqa_v4b.py`",
        "- `scripts/technical_iqa/evaluate_mobile_student_iqa_checkpoints.py`",
        "- `scripts/technical_iqa/evaluate_student_iqa_14105_checkpoints.py`",
        "- `scripts/technical_iqa/run_mdiqa_official_image_only_benchmark.py`",
        "",
        "## 3. Manifest and Split Construction",
        f"- Input rows: `{split_meta.get('input_rows')}`.",
        f"- Full split counts before caps: `{split_meta.get('full_split_counts')}`.",
        f"- Capped counts before validation: `{split_meta.get('capped_counts_before_validation')}`.",
        f"- Usable counts: `{split_meta.get('usable_counts')}`.",
        f"- Duplicate image paths in input: `{split_meta.get('duplicate_image_paths_input')}`.",
        f"- Image-path overlap across splits: `{split_meta.get('overlap_image_paths_across_splits')}`.",
        f"- Source-group overlap across splits: `{split_meta.get('overlap_source_groups_across_splits')}`.",
        f"- Official Track B rows in train: `{split_meta.get('train_official_test_rows')}`.",
        "",
        "## 4. Environment / GPU Preflight",
        f"- TensorFlow: `{env_info.get('tf_version')}`.",
        f"- CUDA build: `{env_info.get('built_cuda')}`.",
        f"- GPUs visible: `{env_info.get('gpus')}`.",
        f"- GPU count: `{env_info.get('gpu_count')}`.",
        f"- Correct environment: `.venv_tf`.",
        "",
        "## 5. Training Configuration",
        f"- Backbone: `{build_info.get('backbone')}`.",
        f"- Image size: `{args.image_size}`.",
        f"- Batch size: `{args.batch_size}`.",
        f"- Head epochs: `{args.head_epochs}`.",
        f"- Finetune epochs: `{args.finetune_epochs}`.",
        f"- Unfreeze last N: `{args.unfreeze_last_n}`.",
        f"- Loss: `{args.loss}`.",
        f"- Target: `teacher_target_0_100 / 100.0`.",
        f"- ImageNet weights requested: `{args.backbone_weights}`.",
        f"- ImageNet weights loaded: `{build_info.get('backbone_weights_loaded')}`.",
        f"- Finetune trainable backbone layers: `{finetune_info.get('backbone_trainable_layers')}`.",
        "",
        "## 6. Training History",
        markdown_table(history, list(history.columns)),
        "",
        "## 7. Best Checkpoint Summary",
        f"- Best checkpoint path: `{out_dir / 'best_by_val_mae_model.keras'}`.",
        f"- Best checkpoint row: `{best_summary}`.",
        f"- Final checkpoint path: `{out_dir / 'final_model.keras'}`.",
        "",
        "## 8. Prediction and Metric Summary",
        markdown_table(prediction_summary, metric_cols),
        "",
        "## 9. Dataset-wise Metrics",
        markdown_table(dataset_metrics, metric_cols),
        "",
        "## 10. Official Benchmark Evaluation",
        f"- Official evaluation failures: `{len(official_failures)}`.",
        markdown_table(official_metrics, metric_cols),
        "",
        markdown_table(official_dataset_metrics, metric_cols),
        "",
        markdown_table(official_safety, safety_cols),
        "",
        "## 11. Comparison Against 14k Student and MDIQA Teacher",
        markdown_table(comparison, ["system", "official_srcc", "delta_vs_14k", "note"]),
        f"- TOPIQ+FLIVE log-only note: {TOPIQ_FLIVE_LOG_ONLY_NOTE}",
        "",
        "## 12. Failure Cases",
        f"- Training/input failure rows: `{len(failures)}`.",
        f"- Official failure rows: `{len(official_failures)}`.",
        "",
        "## 13. Collapse / Safety Checks",
        f"- Collapse check: `{collapse}`.",
        "",
        markdown_table(safety_metrics, safety_cols),
        "",
        "## 14. Final Decision",
        decision,
        "",
        "## 15. Recommended Next Step",
    ]
    if decision == "STUDENT_50K_B0_READY_FOR_B1_B2_COMPARISON":
        lines.append("- Prepare B1/B2 comparison as a separate explicitly approved experiment.")
    elif decision == "STUDENT_50K_B0_READY_FOR_CHECKPOINT_EVAL":
        lines.append("- Run separate checkpoint evaluation before any backbone comparison.")
    elif decision == "STUDENT_50K_B0_NEEDS_TUNING":
        lines.append("- Analyze official benchmark errors and scale compression before B1/B2.")
    elif decision == "STUDENT_50K_B0_NEEDS_PIPELINE_FIX":
        lines.append("- Fix pipeline issue before rerun.")
    else:
        lines.append("- Inspect failure cases and runtime logs before retry.")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    set_reproducibility(args.seed)

    gpus = tf.config.list_physical_devices("GPU")
    env_info = {
        "tf_version": tf.__version__,
        "built_cuda": tf.test.is_built_with_cuda(),
        "gpus": [str(gpu) for gpu in gpus],
        "gpu_count": len(gpus),
    }
    if len(gpus) != 1:
        pd.DataFrame([{"row_index": "", "split": "", "dataset": "", "image_path": "", "error": f"expected exactly 1 TensorFlow GPU, got {len(gpus)}"}]).to_csv(out_dir / "failure_cases.csv", index=False)
        write_next_actions(out_dir, "STUDENT_50K_B0_NEEDS_PIPELINE_FIX")
        raise RuntimeError(f"expected exactly 1 TensorFlow GPU, got {len(gpus)}")

    frames, failures, split_meta = smoke.prepare_manifest(args, out_dir)
    contribution_audit = effective_contribution_audit(frames)
    contribution_audit.to_csv(out_dir / "dataset_weight_effective_contribution.csv", index=False)
    if len(failures) > 0:
        failures.to_csv(out_dir / "failure_cases.csv", index=False)
        write_next_actions(out_dir, "STUDENT_50K_B0_NEEDS_PIPELINE_FIX")
        raise RuntimeError(f"image validation failures: {len(failures)}")

    training_ok = False
    history = pd.DataFrame()
    checkpoint_metrics = pd.DataFrame()
    metrics = pd.DataFrame()
    dataset_metrics = pd.DataFrame()
    prediction_summary = pd.DataFrame()
    safety_metrics_df = pd.DataFrame()
    official_metrics = pd.DataFrame()
    official_dataset_metrics = pd.DataFrame()
    official_safety = pd.DataFrame()
    official_failures = pd.DataFrame(columns=smoke.FAILURE_COLUMNS)
    collapse: dict[str, Any] = {"not_run": True}
    build_info: dict[str, Any] = {}
    finetune_info: dict[str, Any] = {}

    try:
        train_ds = smoke.make_dataset(frames["train"], args.image_size, args.batch_size, shuffle=True, repeat=True, seed=args.seed, sample_weight_col="dataset_sample_weight")
        val_ds = smoke.make_dataset(frames["val"], args.image_size, args.batch_size, shuffle=False, repeat=True, seed=args.seed)
        model, backbone, build_info = smoke.build_model(args)
        checkpoint = BestValMAECheckpoint(out_dir)
        callbacks = [checkpoint]

        checkpoint.set_phase("head", 0)
        head_history = model.fit(
            train_ds,
            validation_data=val_ds,
            steps_per_epoch=smoke.batch_count(len(frames["train"]), args.batch_size),
            validation_steps=smoke.batch_count(len(frames["val"]), args.batch_size),
            epochs=args.head_epochs,
            callbacks=callbacks,
            verbose=2,
        )

        finetune_info = smoke.configure_finetune(backbone, args.unfreeze_last_n)
        smoke.compile_model(model, args.learning_rate_finetune, args.loss, args.huber_delta)
        checkpoint.set_phase("finetune", len(head_history.history.get("loss", [])))
        finetune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            steps_per_epoch=smoke.batch_count(len(frames["train"]), args.batch_size),
            validation_steps=smoke.batch_count(len(frames["val"]), args.batch_size),
            epochs=args.finetune_epochs,
            callbacks=callbacks,
            verbose=2,
        )

        history = pd.concat(
            [
                smoke.history_frame(head_history, "head", 1),
                smoke.history_frame(finetune_history, "finetune", args.head_epochs + 1),
            ],
            ignore_index=True,
        )
        checkpoint_metrics = pd.DataFrame(checkpoint.records)
        model.save(out_dir / "final_model.keras", include_optimizer=False)
        best_path = out_dir / "best_by_val_mae_model.keras"
        eval_model = tf.keras.models.load_model(best_path, compile=False) if best_path.exists() else model

        pred_frames = {split: smoke.predict_split(eval_model, frames[split], args) for split in SPLITS}
        prediction_cols = ["dataset", "student_split", "image_path", "quality_score_0_100", smoke.SELECTED_TARGET_COL, smoke.TARGET_NORM_COL, "pred_norm", PRED_COL, "abs_error_to_target_0_100", "target_source"]
        for split, pred_df in pred_frames.items():
            pred_df.to_csv(out_dir / f"predictions_{split}.csv", columns=[col for col in prediction_cols if col in pred_df.columns], index=False)
        metrics, dataset_metrics, prediction_summary, safety_metrics_df = metric_tables(pred_frames)
        collapse = smoke.collapse_checks(metrics)

        if not args.skip_official_eval:
            official_df = prepare_official_manifest(Path(args.official_manifest_csv))
            official_usable, official_failures = validate_official_rows(official_df)
            official_failures.to_csv(out_dir / "official_benchmark_failures.csv", index=False)
            official_pred = predict_official(eval_model, official_usable, args)
            official_pred.to_csv(out_dir / "official_benchmark_predictions.csv", index=False)
            official_metrics, official_dataset_metrics, official_safety = official_metric_tables(official_pred)
        training_ok = True
    except Exception as exc:  # noqa: BLE001
        failures = pd.concat(
            [
                failures,
                pd.DataFrame([{"row_index": "", "split": "", "dataset": "", "image_path": "", "error": f"training failed: {exc!r}"}], columns=smoke.FAILURE_COLUMNS),
            ],
            ignore_index=True,
        )
        failures.to_csv(out_dir / "failure_cases.csv", index=False)

    runtime_seconds = time.time() - start
    history.to_csv(out_dir / "history.csv", index=False)
    checkpoint_metrics.to_csv(out_dir / "checkpoint_metrics.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    dataset_metrics.to_csv(out_dir / "datasetwise_metrics.csv", index=False)
    prediction_summary.to_csv(out_dir / "prediction_summary.csv", index=False)
    safety_metrics_df.to_csv(out_dir / "safety_metrics.csv", index=False)
    official_metrics.to_csv(out_dir / "official_benchmark_metrics.csv", index=False)
    official_dataset_metrics.to_csv(out_dir / "official_benchmark_datasetwise_metrics.csv", index=False)
    official_safety.to_csv(out_dir / "official_benchmark_safety_metrics.csv", index=False)
    contribution_audit.to_csv(out_dir / "dataset_weight_effective_contribution.csv", index=False)
    failures.to_csv(out_dir / "failure_cases.csv", index=False)

    if is_w5v2_smoke(args):
        decision = w5v2_smoke_final_decision(training_ok, env_info, failures, history, collapse, contribution_audit, official_metrics)
    elif is_weighted_smoke(args):
        decision = weighted_smoke_final_decision(training_ok, env_info, failures, history, collapse, official_metrics)
    else:
        decision = final_decision(training_ok, env_info, failures, history, collapse, official_metrics, official_safety)
    write_next_actions(out_dir, decision)
    write_report(
        out_dir,
        args,
        env_info,
        build_info,
        split_meta,
        history,
        checkpoint_metrics,
        metrics,
        dataset_metrics,
        prediction_summary,
        safety_metrics_df,
        official_metrics,
        official_dataset_metrics,
        official_safety,
        failures,
        official_failures,
        contribution_audit,
        collapse,
        decision,
        runtime_seconds,
        finetune_info,
    )
    print(f"Report: {out_dir / 'report.md'}")
    print(f"Decision: {decision}")
    print(f"Counts: {split_meta.get('usable_counts')}")
    print(f"Runtime seconds: {runtime_seconds:.2f}")


if __name__ == "__main__":
    main()
