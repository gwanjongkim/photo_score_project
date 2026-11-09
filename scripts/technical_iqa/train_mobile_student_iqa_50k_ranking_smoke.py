# 50k MDIQA 학생 모델의 within-dataset pairwise ranking smoke 학습을 실행한다.
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = ROOT / "scripts" / "technical_iqa" / "train_mobile_student_iqa_50k_smoke.py"
CONTROLLED_SCRIPT = ROOT / "scripts" / "technical_iqa" / "train_mobile_student_iqa_50k_controlled.py"
DEFAULT_MANIFEST = ROOT / "outputs" / "mdiqa_distillation_50k_full_targets_20260608" / "final_50k_mdiqa_distillation_manifest.csv"
DEFAULT_OFFICIAL = ROOT / "outputs" / "official_protocol_benchmark_manifest_20260604" / "official_benchmark_test_manifest.csv"
DEFAULT_OUT_DIR = ROOT / "outputs" / "mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609"
PHASE16_OFFICIAL_SRCC = 0.703290
PHASE19_W5V2_OFFICIAL_SRCC = 0.516337
BASELINE_14K_OFFICIAL_SRCC = 0.743600
MDIQA_TEACHER_OFFICIAL_SRCC = 0.822700
DATASET_ID_MAP = {"flive": 0, "koniq10k": 1, "spaq": 2}
SPLITS = ["train", "val", "test"]
PRED_COL = "pred_0_100"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = load_module("train_mobile_student_iqa_50k_smoke", SMOKE_SCRIPT)
controlled = load_module("train_mobile_student_iqa_50k_controlled", CONTROLLED_SCRIPT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run B0 50k ranking smoke with Huber + within-dataset pairwise logistic loss.")
    parser.add_argument("--manifest_csv", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--official_manifest_csv", default=str(DEFAULT_OFFICIAL))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--backbone", choices=["efficientnetv2b0", "efficientnetb0", "mobilenetv3small"], default="efficientnetv2b0")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--head_epochs", type=int, default=1)
    parser.add_argument("--finetune_epochs", type=int, default=2)
    parser.add_argument("--unfreeze_last_n", type=int, default=16)
    parser.add_argument("--learning_rate_head", type=float, default=3e-4)
    parser.add_argument("--learning_rate_finetune", type=float, default=1e-5)
    parser.add_argument("--loss", choices=["huber"], default="huber")
    parser.add_argument("--huber_delta", type=float, default=0.05)
    parser.add_argument("--backbone_weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--dataset_weight_flive", type=float, default=1.0)
    parser.add_argument("--dataset_weight_koniq10k", type=float, default=1.0)
    parser.add_argument("--dataset_weight_spaq", type=float, default=1.0)
    parser.add_argument("--max_train_samples", type=int, default=4000)
    parser.add_argument("--max_val_samples", type=int, default=1000)
    parser.add_argument("--max_test_samples", type=int, default=1000)
    parser.add_argument("--enable_within_dataset_pairwise_rank_loss", action="store_true")
    parser.add_argument("--rank_margin", type=float, default=0.05)
    parser.add_argument("--lambda_rank", type=float, default=0.10)
    parser.add_argument("--rank_loss_type", choices=["logistic"], default="logistic")
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


def add_dataset_ids(frames: dict[str, pd.DataFrame]) -> None:
    for frame in frames.values():
        frame["dataset_id"] = frame["dataset"].astype(str).str.lower().map(DATASET_ID_MAP).astype(np.int32)


def save_manifests(frames: dict[str, pd.DataFrame], out_dir: Path) -> None:
    for split in SPLITS:
        frames[split].to_csv(out_dir / f"{split}_manifest.csv", index=False)


def make_ranking_dataset(df: pd.DataFrame, image_size: int, batch_size: int, shuffle: bool, repeat: bool, seed: int) -> tf.data.Dataset:
    records = df[["image_path", smoke.TARGET_NORM_COL, "dataset_id", "dataset_sample_weight"]].to_dict("records")

    def generator() -> Iterable[tuple[np.ndarray, dict[str, np.ndarray]]]:
        for row in records:
            yield smoke.load_rgb_resized(str(row["image_path"]), image_size), {
                "target": np.asarray([float(row[smoke.TARGET_NORM_COL])], dtype=np.float32),
                "dataset_id": np.asarray(int(row["dataset_id"]), dtype=np.int32),
                "sample_weight": np.asarray(float(row["dataset_sample_weight"]), dtype=np.float32),
            }

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
            {
                "target": tf.TensorSpec(shape=(1,), dtype=tf.float32),
                "dataset_id": tf.TensorSpec(shape=(), dtype=tf.int32),
                "sample_weight": tf.TensorSpec(shape=(), dtype=tf.float32),
            },
        ),
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(records), 1024), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if repeat:
        ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)


class RankingModel(tf.keras.Model):
    def __init__(self, base_model: tf.keras.Model, rank_margin: float, lambda_rank: float, huber_delta: float) -> None:
        super().__init__(name="mobile_student_iqa_50k_ranking_smoke")
        self.base_model = base_model
        self.rank_margin = float(rank_margin)
        self.lambda_rank = float(lambda_rank)
        self.huber_delta = float(huber_delta)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.huber_loss_tracker = tf.keras.metrics.Mean(name="huber_loss")
        self.rank_loss_tracker = tf.keras.metrics.Mean(name="rank_loss")
        self.valid_pair_count_tracker = tf.keras.metrics.Mean(name="valid_pair_count")
        self.valid_pair_rate_tracker = tf.keras.metrics.Mean(name="valid_pair_rate")
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name="mae")
        self.rmse_tracker = tf.keras.metrics.RootMeanSquaredError(name="rmse")

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        return [
            self.total_loss_tracker,
            self.huber_loss_tracker,
            self.rank_loss_tracker,
            self.valid_pair_count_tracker,
            self.valid_pair_rate_tracker,
            self.mae_tracker,
            self.rmse_tracker,
        ]

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.base_model(inputs, training=training)

    def huber_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor | None = None) -> tf.Tensor:
        err = y_true - y_pred
        abs_err = tf.abs(err)
        quadratic = 0.5 * tf.square(err)
        linear = self.huber_delta * (abs_err - 0.5 * self.huber_delta)
        per_sample = tf.where(abs_err <= self.huber_delta, quadratic, linear)
        per_sample = tf.reshape(per_sample, [-1])
        if sample_weight is not None:
            weights = tf.reshape(tf.cast(sample_weight, tf.float32), [-1])
            return tf.math.divide_no_nan(tf.reduce_sum(per_sample * weights), tf.reduce_sum(weights))
        return tf.reduce_mean(per_sample)

    def pairwise_rank_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor, dataset_id: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        y = tf.reshape(y_true, [-1])
        p = tf.reshape(y_pred, [-1])
        d = tf.reshape(dataset_id, [-1])
        y_diff = y[:, None] - y[None, :]
        p_diff = p[:, None] - p[None, :]
        same_dataset = tf.equal(d[:, None], d[None, :])
        target_far = tf.abs(y_diff) >= self.rank_margin
        not_diag = tf.logical_not(tf.eye(tf.shape(y)[0], dtype=tf.bool))
        mask = tf.logical_and(tf.logical_and(same_dataset, target_far), not_diag)
        sign = tf.sign(y_diff)
        pair_losses = tf.nn.softplus(-sign * p_diff)
        valid_losses = tf.boolean_mask(pair_losses, mask)
        valid_pair_count = tf.cast(tf.size(valid_losses), tf.float32)
        possible_pair_count = tf.cast(tf.size(pair_losses) - tf.shape(y)[0], tf.float32)
        rank_loss = tf.cond(valid_pair_count > 0.0, lambda: tf.reduce_mean(valid_losses), lambda: tf.constant(0.0, dtype=tf.float32))
        valid_pair_rate = tf.math.divide_no_nan(valid_pair_count, possible_pair_count)
        return rank_loss, valid_pair_count, valid_pair_rate

    def train_step(self, data: Any) -> dict[str, tf.Tensor]:
        x, y_info = data
        y_true = tf.cast(y_info["target"], tf.float32)
        dataset_id = tf.cast(y_info["dataset_id"], tf.int32)
        sample_weight = tf.cast(y_info["sample_weight"], tf.float32)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            huber = self.huber_loss(y_true, y_pred, sample_weight)
            rank_loss, valid_pair_count, valid_pair_rate = self.pairwise_rank_loss(y_true, y_pred, dataset_id)
            total = huber + self.lambda_rank * rank_loss
        gradients = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.total_loss_tracker.update_state(total)
        self.huber_loss_tracker.update_state(huber)
        self.rank_loss_tracker.update_state(rank_loss)
        self.valid_pair_count_tracker.update_state(valid_pair_count)
        self.valid_pair_rate_tracker.update_state(valid_pair_rate)
        self.mae_tracker.update_state(y_true, y_pred)
        self.rmse_tracker.update_state(y_true, y_pred)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data: Any) -> dict[str, tf.Tensor]:
        x, y_true = data
        y_true = tf.cast(y_true, tf.float32)
        y_pred = self(x, training=False)
        huber = self.huber_loss(y_true, y_pred)
        self.total_loss_tracker.update_state(huber)
        self.huber_loss_tracker.update_state(huber)
        self.rank_loss_tracker.update_state(0.0)
        self.valid_pair_count_tracker.update_state(0.0)
        self.valid_pair_rate_tracker.update_state(0.0)
        self.mae_tracker.update_state(y_true, y_pred)
        self.rmse_tracker.update_state(y_true, y_pred)
        return {metric.name: metric.result() for metric in self.metrics}


class BestValMAECheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, out_dir: Path, base_model: tf.keras.Model) -> None:
        super().__init__()
        self.out_dir = out_dir
        self.base_model = base_model
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
            self.base_model.save(self.out_dir / "best_by_val_mae_model.keras", include_optimizer=False)
        record = {
            "phase": self.phase,
            "phase_epoch": epoch + 1,
            "global_epoch": self.offset + epoch + 1,
            "val_mae": val_mae,
            "val_total_loss": float(logs.get("val_total_loss", float("nan"))),
            "val_huber_loss": float(logs.get("val_huber_loss", float("nan"))),
            "saved_best_by_val_mae": saved,
            "best_val_mae": self.best_val_mae,
        }
        self.records.append(record)
        pd.DataFrame(self.records).to_csv(self.out_dir / "checkpoint_metrics.csv", index=False)


def history_frame(history: tf.keras.callbacks.History, phase: str, start_epoch: int) -> pd.DataFrame:
    history_dict = {key: list(values) for key, values in history.history.items()}
    max_len = max((len(values) for values in history_dict.values()), default=0)
    normalized = {key: values + [float("nan")] * (max_len - len(values)) for key, values in history_dict.items()}
    df = pd.DataFrame(normalized)
    df.insert(0, "phase", phase)
    df.insert(1, "phase_epoch", np.arange(1, len(df) + 1))
    df.insert(2, "global_epoch", np.arange(start_epoch, start_epoch + len(df)))
    return df


def ranking_pair_stats(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase, group in history.groupby("phase", sort=False):
        rows.append(
            {
                "phase": phase,
                "epochs": len(group),
                "mean_valid_pair_count": float(group["valid_pair_count"].mean()) if "valid_pair_count" in group else float("nan"),
                "min_valid_pair_count": float(group["valid_pair_count"].min()) if "valid_pair_count" in group else float("nan"),
                "mean_valid_pair_rate": float(group["valid_pair_rate"].mean()) if "valid_pair_rate" in group else float("nan"),
                "final_rank_loss": float(group["rank_loss"].iloc[-1]) if "rank_loss" in group and len(group) else float("nan"),
                "final_total_loss": float(group["total_loss"].iloc[-1]) if "total_loss" in group and len(group) else float("nan"),
                "final_huber_loss": float(group["huber_loss"].iloc[-1]) if "huber_loss" in group and len(group) else float("nan"),
            }
        )
    rows.append(
        {
            "phase": "overall",
            "epochs": len(history),
            "mean_valid_pair_count": float(history["valid_pair_count"].mean()) if "valid_pair_count" in history else float("nan"),
            "min_valid_pair_count": float(history["valid_pair_count"].min()) if "valid_pair_count" in history else float("nan"),
            "mean_valid_pair_rate": float(history["valid_pair_rate"].mean()) if "valid_pair_rate" in history else float("nan"),
            "final_rank_loss": float(history["rank_loss"].iloc[-1]) if "rank_loss" in history and len(history) else float("nan"),
            "final_total_loss": float(history["total_loss"].iloc[-1]) if "total_loss" in history and len(history) else float("nan"),
            "final_huber_loss": float(history["huber_loss"].iloc[-1]) if "huber_loss" in history and len(history) else float("nan"),
        }
    )
    return pd.DataFrame(rows)


def final_decision(
    training_ok: bool,
    env_info: dict[str, Any],
    failures: pd.DataFrame,
    history: pd.DataFrame,
    collapse: dict[str, Any],
    pair_stats: pd.DataFrame,
    official_metrics: pd.DataFrame,
) -> str:
    if not training_ok:
        return "RANKING_B0_SMOKE_FAILED"
    if env_info.get("gpu_count", 0) != 1:
        return "RANKING_B0_SMOKE_NEEDS_PIPELINE_FIX"
    if len(failures) > 0:
        return "RANKING_B0_SMOKE_NEEDS_PIPELINE_FIX"
    if history.empty or not np.isfinite(history.select_dtypes(include=[np.number]).to_numpy()).all():
        return "RANKING_B0_SMOKE_NEEDS_PIPELINE_FIX"
    if collapse.get("nan_inf_count", 0) or collapse.get("pred_outside_0_100") or collapse.get("pred_std_too_low"):
        return "RANKING_B0_SMOKE_NEEDS_PIPELINE_FIX"
    overall_pairs = pair_stats[pair_stats["phase"].eq("overall")]
    if overall_pairs.empty or float(overall_pairs.iloc[0]["mean_valid_pair_count"]) <= 0:
        return "RANKING_B0_SMOKE_NEEDS_PIPELINE_FIX"
    if official_metrics.empty:
        return "RANKING_B0_SMOKE_NEEDS_PIPELINE_FIX"
    official_overall = official_metrics[(official_metrics["split"].eq("official_track_b")) & (official_metrics["dataset"].eq("overall"))]
    if official_overall.empty:
        return "RANKING_B0_SMOKE_NEEDS_PIPELINE_FIX"
    srcc = float(official_overall.iloc[0]["srcc"])
    if not math.isfinite(srcc):
        return "RANKING_B0_SMOKE_NEEDS_PIPELINE_FIX"
    if srcc >= PHASE16_OFFICIAL_SRCC - 0.02 and srcc > PHASE19_W5V2_OFFICIAL_SRCC + 0.03:
        return "RANKING_B0_SMOKE_READY_FOR_CONTROLLED_RERUN"
    if srcc > PHASE19_W5V2_OFFICIAL_SRCC + 0.03:
        return "RANKING_B0_SMOKE_NEEDS_LAMBDA_MARGIN_TUNING"
    return "RANKING_B0_SMOKE_NEEDS_LABEL_SCALING"


def write_next_actions(out_dir: Path, decision: str) -> None:
    if decision == "RANKING_B0_SMOKE_READY_FOR_CONTROLLED_RERUN":
        rows = [{"priority": 1, "action": "Prepare a separate controlled B0 rerun with the same ranking loss settings.", "status": "recommended"}]
    elif decision == "RANKING_B0_SMOKE_NEEDS_LAMBDA_MARGIN_TUNING":
        rows = [{"priority": 1, "action": "Tune exactly one ranking setting next: lambda_rank or rank_margin, not B1/B2.", "status": "required"}]
    elif decision == "RANKING_B0_SMOKE_NEEDS_LABEL_SCALING":
        rows = [{"priority": 1, "action": "Test a separate KonIQ label-scaling smoke before more ranking-loss variants.", "status": "required"}]
    elif decision == "RANKING_B0_SMOKE_NEEDS_PIPELINE_FIX":
        rows = [{"priority": 1, "action": "Fix ranking dataset, pair mask, loss, or official evaluation before retry.", "status": "blocked"}]
    else:
        rows = [{"priority": 1, "action": "Inspect failure_cases.csv and runtime logs before retry.", "status": "blocked"}]
    pd.DataFrame(rows).to_csv(out_dir / "next_actions.csv", index=False)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    return smoke.markdown_table(df, [col for col in columns if col in df.columns])


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    env_info: dict[str, Any],
    build_info: dict[str, Any],
    split_meta: dict[str, Any],
    history: pd.DataFrame,
    pair_stats: pd.DataFrame,
    metrics: pd.DataFrame,
    dataset_metrics: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    safety_metrics: pd.DataFrame,
    official_metrics: pd.DataFrame,
    official_dataset_metrics: pd.DataFrame,
    official_safety: pd.DataFrame,
    failures: pd.DataFrame,
    official_failures: pd.DataFrame,
    collapse: dict[str, Any],
    decision: str,
    runtime_seconds: float,
    finetune_info: dict[str, Any],
) -> None:
    metric_cols = ["split", "dataset", "n", "finite_n", "mae_0_100", "rmse_0_100", "srcc", "plcc", "pred_min", "pred_max", "pred_mean", "pred_std", "target_min", "target_max", "target_mean", "target_std", "nan_inf_count"]
    safety_cols = ["split", "dataset", "n", "fp_count", "severe_fp_count", "fn_count"]
    pair_cols = ["phase", "epochs", "mean_valid_pair_count", "min_valid_pair_count", "mean_valid_pair_rate", "final_rank_loss", "final_total_loss", "final_huber_loss"]
    official_overall = official_metrics[(official_metrics["split"].eq("official_track_b")) & (official_metrics["dataset"].eq("overall"))] if not official_metrics.empty else pd.DataFrame()
    official_srcc = float(official_overall.iloc[0]["srcc"]) if len(official_overall) else float("nan")
    comparison = pd.DataFrame(
        [
            {"system": "Phase 16 unweighted B0 controlled", "official_srcc": PHASE16_OFFICIAL_SRCC, "delta_vs_14k": PHASE16_OFFICIAL_SRCC - BASELINE_14K_OFFICIAL_SRCC, "note": "controlled reference"},
            {"system": "Phase 19 W5-v2 weighted smoke", "official_srcc": PHASE19_W5V2_OFFICIAL_SRCC, "delta_vs_14k": PHASE19_W5V2_OFFICIAL_SRCC - BASELINE_14K_OFFICIAL_SRCC, "note": "weighted smoke reference"},
            {"system": "14k student official", "official_srcc": BASELINE_14K_OFFICIAL_SRCC, "delta_vs_14k": 0.0, "note": "student baseline"},
            {"system": "MDIQA teacher official", "official_srcc": MDIQA_TEACHER_OFFICIAL_SRCC, "delta_vs_14k": MDIQA_TEACHER_OFFICIAL_SRCC - BASELINE_14K_OFFICIAL_SRCC, "note": "teacher reference"},
            {"system": "Phase 21 ranking B0 smoke", "official_srcc": official_srcc, "delta_vs_14k": official_srcc - BASELINE_14K_OFFICIAL_SRCC if math.isfinite(official_srcc) else float("nan"), "note": "this run"},
        ]
    )
    final = history.tail(1).iloc[0].to_dict() if len(history) else {}
    lines = [
        "# Mobile Student IQA 50k Ranking B0 Smoke R2 Report",
        "",
        "## 1. Executive Summary",
        f"- Final decision: `{decision}`.",
        "- Run type: one bounded B0 ranking smoke only; no B1/B2, no full controlled training, no TFLite export, no Flutter edits, no production `technical_score` edits.",
        f"- Output directory: `{out_dir}`.",
        f"- Runtime seconds: `{runtime_seconds:.2f}`.",
        f"- Train/val/test counts: `{split_meta.get('usable_counts')}`.",
        f"- Final total/huber/rank loss: `{final.get('total_loss')}` / `{final.get('huber_loss')}` / `{final.get('rank_loss')}`.",
        "",
        "## 2. Sources Inspected",
        "- `scripts/technical_iqa/train_mobile_student_iqa_50k_smoke.py`",
        "- `scripts/technical_iqa/train_mobile_student_iqa_50k_controlled.py`",
        "- `outputs/mobile_student_iqa_50k_label_scaling_ranking_plan_20260609/report.md`",
        "- `outputs/mobile_student_iqa_50k_controlled_b0_20260608/report.md`",
        "- `outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/report.md`",
        "",
        "## 3. Implementation Summary",
        "- Created focused ranking smoke runner to avoid changing existing default smoke/controlled behavior.",
        "- Training dataset emits image tensor plus target, dataset id, and neutral sample weight.",
        "- Custom `train_step` computes normalized Huber loss plus within-dataset pairwise logistic rank loss.",
        "- Validation, prediction, safety metrics, and official Track B evaluation stay unweighted.",
        "",
        "## 4. Environment / GPU Preflight",
        f"- TensorFlow: `{env_info.get('tf_version')}`.",
        f"- CUDA build: `{env_info.get('built_cuda')}`.",
        f"- GPUs visible: `{env_info.get('gpus')}`.",
        f"- GPU count: `{env_info.get('gpu_count')}`.",
        f"- Correct environment: `.venv_tf`.",
        "",
        "## 5. Ranking Loss Configuration",
        f"- Enabled: `{args.enable_within_dataset_pairwise_rank_loss}`.",
        f"- Rank loss type: `{args.rank_loss_type}`.",
        f"- Rank margin: `{args.rank_margin}`.",
        f"- Lambda rank: `{args.lambda_rank}`.",
        "- Pair mask: same dataset, target abs diff >= margin, diagonal excluded.",
        "- Dataset sample weights: neutral/default `1.0`.",
        "",
        "## 6. Smoke Training Configuration",
        f"- Backbone: `{build_info.get('backbone')}`.",
        f"- Image size: `{args.image_size}`.",
        f"- Batch size: `{args.batch_size}`.",
        f"- Head epochs: `{args.head_epochs}`.",
        f"- Finetune epochs: `{args.finetune_epochs}`.",
        f"- Unfreeze last N: `{args.unfreeze_last_n}`.",
        f"- ImageNet weights requested: `{args.backbone_weights}`.",
        f"- ImageNet weights loaded: `{build_info.get('backbone_weights_loaded')}`.",
        f"- Finetune trainable backbone layers: `{finetune_info.get('backbone_trainable_layers')}`.",
        "",
        "## 7. Training History",
        markdown_table(history, list(history.columns)),
        "",
        "## 8. Ranking Pair Statistics",
        markdown_table(pair_stats, pair_cols),
        "",
        "## 9. Prediction and Metric Summary",
        markdown_table(prediction_summary, metric_cols),
        "",
        "## 10. Dataset-wise Metrics",
        markdown_table(dataset_metrics, metric_cols),
        "",
        "## 11. Official Benchmark Evaluation",
        f"- Official evaluation failures: `{len(official_failures)}`.",
        markdown_table(official_metrics, metric_cols),
        "",
        markdown_table(official_dataset_metrics, metric_cols),
        "",
        markdown_table(official_safety, safety_cols),
        "",
        "## 12. Comparison Against Phase 16, Phase 19, and 14k Student",
        markdown_table(comparison, ["system", "official_srcc", "delta_vs_14k", "note"]),
        "",
        "## 13. Collapse / Safety Checks",
        f"- Training/input failure rows: `{len(failures)}`.",
        f"- Official failure rows: `{len(official_failures)}`.",
        f"- Collapse check: `{collapse}`.",
        "",
        markdown_table(safety_metrics, safety_cols),
        "",
        "## 14. Final Decision",
        decision,
        "",
        "## 15. Recommended Next Step",
    ]
    if decision == "RANKING_B0_SMOKE_READY_FOR_CONTROLLED_RERUN":
        lines.append("- Prepare controlled ranking B0 rerun as a separate explicitly approved task.")
    elif decision == "RANKING_B0_SMOKE_NEEDS_LAMBDA_MARGIN_TUNING":
        lines.append("- Tune one ranking setting next; do not run B1/B2.")
    elif decision == "RANKING_B0_SMOKE_NEEDS_LABEL_SCALING":
        lines.append("- Test KonIQ label scaling as a separate smoke before more ranking variants.")
    elif decision == "RANKING_B0_SMOKE_NEEDS_PIPELINE_FIX":
        lines.append("- Fix ranking pipeline issue and rerun the same smoke.")
    else:
        lines.append("- Inspect failure cases and runtime logs before retry.")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    set_reproducibility(args.seed)
    if not args.enable_within_dataset_pairwise_rank_loss:
        raise ValueError("--enable_within_dataset_pairwise_rank_loss is required for Phase 21")

    gpus = tf.config.list_physical_devices("GPU")
    env_info = {
        "tf_version": tf.__version__,
        "built_cuda": tf.test.is_built_with_cuda(),
        "gpus": [str(gpu) for gpu in gpus],
        "gpu_count": len(gpus),
    }
    if len(gpus) != 1:
        pd.DataFrame([{"row_index": "", "split": "", "dataset": "", "image_path": "", "error": f"expected exactly 1 TensorFlow GPU, got {len(gpus)}"}]).to_csv(out_dir / "failure_cases.csv", index=False)
        raise RuntimeError(f"expected exactly 1 TensorFlow GPU, got {len(gpus)}")

    frames, failures, split_meta = smoke.prepare_manifest(args, out_dir)
    add_dataset_ids(frames)
    save_manifests(frames, out_dir)
    if len(failures) > 0:
        failures.to_csv(out_dir / "failure_cases.csv", index=False)
        raise RuntimeError(f"image validation failures: {len(failures)}")

    training_ok = False
    history = pd.DataFrame()
    pair_stats = pd.DataFrame()
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
        train_ds = make_ranking_dataset(frames["train"], args.image_size, args.batch_size, shuffle=True, repeat=True, seed=args.seed)
        val_ds = smoke.make_dataset(frames["val"], args.image_size, args.batch_size, shuffle=False, repeat=True, seed=args.seed)
        base_model, backbone, build_info = smoke.build_model(args)
        ranking_model = RankingModel(base_model, args.rank_margin, args.lambda_rank, args.huber_delta)
        ranking_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate_head))
        checkpoint = BestValMAECheckpoint(out_dir, base_model)
        callbacks = [checkpoint]

        checkpoint.set_phase("head", 0)
        head_history = ranking_model.fit(
            train_ds,
            validation_data=val_ds,
            steps_per_epoch=smoke.batch_count(len(frames["train"]), args.batch_size),
            validation_steps=smoke.batch_count(len(frames["val"]), args.batch_size),
            epochs=args.head_epochs,
            callbacks=callbacks,
            verbose=2,
        )

        finetune_info = smoke.configure_finetune(backbone, args.unfreeze_last_n)
        ranking_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate_finetune))
        checkpoint.set_phase("finetune", len(head_history.history.get("total_loss", [])))
        finetune_history = ranking_model.fit(
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
                history_frame(head_history, "head", 1),
                history_frame(finetune_history, "finetune", args.head_epochs + 1),
            ],
            ignore_index=True,
        )
        pair_stats = ranking_pair_stats(history)
        checkpoint_metrics = pd.DataFrame(checkpoint.records)
        base_model.save(out_dir / "final_model.keras", include_optimizer=False)
        best_path = out_dir / "best_by_val_mae_model.keras"
        eval_model = tf.keras.models.load_model(best_path, compile=False) if best_path.exists() else base_model

        pred_frames = {split: smoke.predict_split(eval_model, frames[split], args) for split in SPLITS}
        prediction_cols = ["dataset", "dataset_id", "student_split", "image_path", "quality_score_0_100", smoke.SELECTED_TARGET_COL, smoke.TARGET_NORM_COL, "pred_norm", PRED_COL, "abs_error_to_target_0_100", "target_source"]
        for split, pred_df in pred_frames.items():
            pred_df.to_csv(out_dir / f"predictions_{split}.csv", columns=[col for col in prediction_cols if col in pred_df.columns], index=False)
        metrics, dataset_metrics, prediction_summary, safety_metrics_df = controlled.metric_tables(pred_frames)
        collapse = smoke.collapse_checks(metrics)

        if not args.skip_official_eval:
            official_df = controlled.prepare_official_manifest(Path(args.official_manifest_csv))
            official_usable, official_failures = controlled.validate_official_rows(official_df)
            official_failures.to_csv(out_dir / "official_benchmark_failures.csv", index=False)
            official_pred = controlled.predict_official(eval_model, official_usable, args)
            official_pred.to_csv(out_dir / "official_benchmark_predictions.csv", index=False)
            official_metrics, official_dataset_metrics, official_safety = controlled.official_metric_tables(official_pred)
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
    pair_stats.to_csv(out_dir / "ranking_pair_stats.csv", index=False)
    checkpoint_metrics.to_csv(out_dir / "checkpoint_metrics.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    dataset_metrics.to_csv(out_dir / "datasetwise_metrics.csv", index=False)
    prediction_summary.to_csv(out_dir / "prediction_summary.csv", index=False)
    safety_metrics_df.to_csv(out_dir / "safety_metrics.csv", index=False)
    official_metrics.to_csv(out_dir / "official_benchmark_metrics.csv", index=False)
    official_dataset_metrics.to_csv(out_dir / "official_benchmark_datasetwise_metrics.csv", index=False)
    official_safety.to_csv(out_dir / "official_benchmark_safety_metrics.csv", index=False)
    failures.to_csv(out_dir / "failure_cases.csv", index=False)

    decision = final_decision(training_ok, env_info, failures, history, collapse, pair_stats, official_metrics)
    write_next_actions(out_dir, decision)
    write_report(
        out_dir,
        args,
        env_info,
        build_info,
        split_meta,
        history,
        pair_stats,
        metrics,
        dataset_metrics,
        prediction_summary,
        safety_metrics_df,
        official_metrics,
        official_dataset_metrics,
        official_safety,
        failures,
        official_failures,
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
