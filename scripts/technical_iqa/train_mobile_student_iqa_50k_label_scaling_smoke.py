# 50k MDIQA 학생 모델의 KonIQ 라벨 스케일링 smoke 학습을 실행한다.
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
CONTROLLED_SCRIPT = ROOT / "scripts" / "technical_iqa" / "train_mobile_student_iqa_50k_controlled.py"
DEFAULT_MANIFEST = ROOT / "outputs" / "mdiqa_distillation_50k_full_targets_20260608" / "final_50k_mdiqa_distillation_manifest.csv"
DEFAULT_OFFICIAL = ROOT / "outputs" / "official_protocol_benchmark_manifest_20260604" / "official_benchmark_test_manifest.csv"
DEFAULT_OUT_DIR = ROOT / "outputs" / "mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609"

PHASE16_OFFICIAL_SRCC = 0.703290
PHASE19_W5V2_OFFICIAL_SRCC = 0.516337
PHASE21_RANKING_OFFICIAL_SRCC = 0.480260
BASELINE_14K_OFFICIAL_SRCC = 0.743600
MDIQA_TEACHER_OFFICIAL_SRCC = 0.822700
PRODUCTION_KONIQ_FLIVE_OFFICIAL_SRCC = 0.7168146787631084
PRODUCTION_REFERENCE = ROOT / "outputs" / "official_protocol_technical_signal_benchmark_20260604" / "official_signal_metrics_overall.csv"
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
    parser = argparse.ArgumentParser(description="Run final S2 KonIQ x2 label-scaling B0 smoke on 50k MDIQA targets.")
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
    parser.add_argument("--target_scaling_policy", choices=["s2_koniq_x2"], required=True)
    parser.add_argument("--koniq_scale_factor", type=float, required=True)
    parser.add_argument("--disable_rank_loss", action="store_true")
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


def add_target_scaling_columns(full: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if not math.isclose(float(args.koniq_scale_factor), 2.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Phase 23 requires --koniq_scale_factor 2.0")
    out = full.copy()
    original = pd.to_numeric(out[smoke.SELECTED_TARGET_COL], errors="coerce").astype(float)
    dataset = out["dataset"].astype(str).str.lower()
    koniq_mask = dataset.eq("koniq10k")
    mean_k = float(original[koniq_mask].mean())
    raw_scaled = original.copy()
    raw_scaled.loc[koniq_mask] = mean_k + float(args.koniq_scale_factor) * (original.loc[koniq_mask] - mean_k)
    clipped = raw_scaled.clip(0.0, 100.0)
    out["teacher_target_0_100_original"] = original
    out["teacher_target_0_100_scaled"] = clipped
    out["target_norm_scaled"] = clipped / 100.0
    out["target_scaling_policy"] = args.target_scaling_policy
    out["target_scale_factor"] = float(args.koniq_scale_factor)
    out["target_delta"] = clipped - original
    out["target_clipped_by_scaling"] = (np.abs(clipped - raw_scaled) > 1e-12).astype(bool)
    out["koniq_target_mean_for_scaling"] = mean_k
    out[smoke.SELECTED_TARGET_COL] = clipped
    out[smoke.TARGET_NORM_COL] = clipped / 100.0
    return out


def target_scaling_audit(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_frames = []
    for split, frame in frames.items():
        split_frame = frame.copy()
        split_frame["audit_split"] = split
        all_frames.append(split_frame)
    df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    rows = []
    for dataset, group in df.groupby("dataset", sort=True):
        original = pd.to_numeric(group["teacher_target_0_100_original"], errors="coerce").to_numpy(dtype=np.float64)
        scaled = pd.to_numeric(group["teacher_target_0_100_scaled"], errors="coerce").to_numpy(dtype=np.float64)
        delta = pd.to_numeric(group["target_delta"], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(original) & np.isfinite(scaled) & np.isfinite(delta)
        original = original[finite]
        scaled = scaled[finite]
        delta = delta[finite]
        clipped = group.loc[finite, "target_clipped_by_scaling"].astype(bool).to_numpy()
        original_std = float(np.std(original)) if len(original) else float("nan")
        scaled_std = float(np.std(scaled)) if len(scaled) else float("nan")
        rows.append(
            {
                "dataset": str(dataset),
                "count": int(len(group)),
                "finite_count": int(len(original)),
                "original_min": float(np.min(original)) if len(original) else float("nan"),
                "original_max": float(np.max(original)) if len(original) else float("nan"),
                "original_mean": float(np.mean(original)) if len(original) else float("nan"),
                "original_std": original_std,
                "scaled_min": float(np.min(scaled)) if len(scaled) else float("nan"),
                "scaled_max": float(np.max(scaled)) if len(scaled) else float("nan"),
                "scaled_mean": float(np.mean(scaled)) if len(scaled) else float("nan"),
                "scaled_std": scaled_std,
                "std_ratio_scaled_over_original": float(scaled_std / original_std) if original_std > 0 else float("nan"),
                "delta_mean": float(np.mean(delta)) if len(delta) else float("nan"),
                "delta_std": float(np.std(delta)) if len(delta) else float("nan"),
                "max_abs_delta": float(np.max(np.abs(delta))) if len(delta) else float("nan"),
                "clipped_count": int(clipped.sum()) if len(clipped) else 0,
                "clipped_ratio": float(clipped.mean()) if len(clipped) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def target_scaling_valid(audit: pd.DataFrame) -> bool:
    if audit.empty:
        return False
    rows = {str(row.dataset): row for row in audit.itertuples(index=False)}
    required = {"flive", "koniq10k", "spaq"}
    if set(rows) != required:
        return False
    koniq_ratio = float(rows["koniq10k"].std_ratio_scaled_over_original)
    koniq_ok = math.isfinite(koniq_ratio) and 1.80 <= koniq_ratio <= 2.05
    unchanged_ok = all(
        abs(float(rows[dataset].max_abs_delta)) <= 1e-9 and int(rows[dataset].clipped_count) == 0
        for dataset in ["flive", "spaq"]
    )
    mean_ok = abs(float(rows["koniq10k"].delta_mean)) <= 0.01
    return bool(koniq_ok and unchanged_ok and mean_ok)


def prepare_manifest(args: argparse.Namespace, out_dir: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any], pd.DataFrame]:
    raw = pd.read_csv(args.manifest_csv)
    smoke.require_columns(raw)
    full = smoke.add_target_columns(raw)
    full = add_target_scaling_columns(full, args)
    official_test_rows = full[full["is_official_test"].map(smoke.as_bool)].copy()
    trainable = full[~full["is_official_test"].map(smoke.as_bool)].copy()
    split_full = smoke.split_groups(trainable, args.seed)
    capped = {
        "train": smoke.balanced_cap(split_full[split_full["student_split"].eq("train")], args.max_train_samples, args.seed + 1),
        "val": smoke.balanced_cap(split_full[split_full["student_split"].eq("val")], args.max_val_samples, args.seed + 2),
        "test": smoke.balanced_cap(split_full[split_full["student_split"].eq("test")], args.max_test_samples, args.seed + 3),
    }
    capped_all = pd.concat(capped.values(), ignore_index=True)
    usable, failures = smoke.validate_rows(capped_all)
    frames = {split: usable[usable["student_split"].eq(split)].reset_index(drop=True) for split in SPLITS}
    smoke.add_dataset_sample_weights(frames, args)
    for split in SPLITS:
        frames[split].to_csv(out_dir / f"{split}_manifest.csv", index=False)
    failures.to_csv(out_dir / "failure_cases.csv", index=False)
    audit = target_scaling_audit(frames)
    audit.to_csv(out_dir / "target_scaling_audit.csv", index=False)
    meta = {
        "input_rows": len(raw),
        "official_test_rows_excluded_before_split": len(official_test_rows),
        "duplicate_image_paths_input": int(raw["image_path"].duplicated().sum()),
        "full_split_counts": split_full["student_split"].value_counts().to_dict(),
        "capped_counts_before_validation": {split: len(df) for split, df in capped.items()},
        "usable_counts": {split: len(frames[split]) for split in SPLITS},
        "overlap_image_paths_across_splits": smoke.overlap_count(frames, "image_path"),
        "overlap_source_groups_across_splits": smoke.overlap_count(frames, "source_group"),
        "train_official_test_rows": int(frames["train"]["is_official_test"].map(smoke.as_bool).sum()) if len(frames["train"]) else 0,
        "target_scaling_valid": target_scaling_valid(audit),
    }
    return frames, failures, meta, audit


def final_decision(
    training_ok: bool,
    env_info: dict[str, Any],
    failures: pd.DataFrame,
    official_failures: pd.DataFrame,
    history: pd.DataFrame,
    collapse: dict[str, Any],
    target_audit_ok: bool,
    official_metrics: pd.DataFrame,
    official_safety: pd.DataFrame,
) -> str:
    if not training_ok:
        return "LABEL_SCALED_B0_SMOKE_FAILED"
    if env_info.get("gpu_count", 0) != 1:
        return "LABEL_SCALED_B0_SMOKE_NEEDS_PIPELINE_FIX"
    if not target_audit_ok:
        return "LABEL_SCALED_B0_SMOKE_NEEDS_PIPELINE_FIX"
    if len(failures) > 0 or len(official_failures) > 0:
        return "LABEL_SCALED_B0_SMOKE_NEEDS_PIPELINE_FIX"
    if history.empty or not np.isfinite(history.select_dtypes(include=[np.number]).to_numpy()).all():
        return "LABEL_SCALED_B0_SMOKE_NEEDS_PIPELINE_FIX"
    if collapse.get("nan_inf_count", 0) or collapse.get("pred_outside_0_100") or collapse.get("pred_std_too_low"):
        return "LABEL_SCALED_B0_SMOKE_NEEDS_PIPELINE_FIX"
    official_overall = official_metrics[(official_metrics["split"].eq("official_track_b")) & (official_metrics["dataset"].eq("overall"))] if not official_metrics.empty else pd.DataFrame()
    if official_overall.empty:
        return "LABEL_SCALED_B0_SMOKE_NEEDS_PIPELINE_FIX"
    srcc = float(official_overall.iloc[0]["srcc"])
    if not math.isfinite(srcc):
        return "LABEL_SCALED_B0_SMOKE_NEEDS_PIPELINE_FIX"
    safety_row = official_safety[official_safety["dataset"].eq("overall")] if not official_safety.empty else pd.DataFrame()
    severe_fp = int(safety_row.iloc[0]["severe_fp_count"]) if len(safety_row) else 999999
    safety_ok = severe_fp <= 12
    if safety_ok and srcc >= BASELINE_14K_OFFICIAL_SRCC:
        return "LABEL_SCALED_B0_SMOKE_READY_FOR_CONTROLLED_RERUN"
    if safety_ok and srcc >= PHASE16_OFFICIAL_SRCC + 0.02:
        return "LABEL_SCALED_B0_SMOKE_NEEDS_SCALE_TUNING"
    return "LABEL_SCALED_B0_SMOKE_NO_GO_RESEARCH_REFERENCE_ONLY"


def write_next_actions(out_dir: Path, decision: str) -> None:
    if decision == "LABEL_SCALED_B0_SMOKE_READY_FOR_CONTROLLED_RERUN":
        rows = [{"priority": 1, "action": "Consider one controlled B0 rerun with the same S2 KonIQ x2 label-scaling policy; keep B1/B2 paused until controlled evidence exists.", "status": "recommended"}]
    elif decision == "LABEL_SCALED_B0_SMOKE_NEEDS_SCALE_TUNING":
        rows = [{"priority": 1, "action": "If the line remains open, test exactly one adjacent KonIQ scale factor such as x1.5 or x2.5 before controlled training.", "status": "optional"}]
    elif decision == "LABEL_SCALED_B0_SMOKE_NO_GO_RESEARCH_REFERENCE_ONLY":
        rows = [{"priority": 1, "action": "Terminate the MDIQA mobile student smoke line; keep existing KonIQ+FLIVE production stack and treat MDIQA teacher as research reference.", "status": "recommended"}]
    elif decision == "LABEL_SCALED_B0_SMOKE_NEEDS_PIPELINE_FIX":
        rows = [{"priority": 1, "action": "Fix target scaling, GPU, manifest, prediction, or official evaluation issue before interpreting Phase 23.", "status": "blocked"}]
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
    target_audit: pd.DataFrame,
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
    collapse: dict[str, Any],
    decision: str,
    runtime_seconds: float,
    finetune_info: dict[str, Any],
) -> None:
    metric_cols = ["split", "dataset", "n", "finite_n", "mae_0_100", "rmse_0_100", "srcc", "plcc", "pred_min", "pred_max", "pred_mean", "pred_std", "target_min", "target_max", "target_mean", "target_std", "nan_inf_count"]
    safety_cols = ["split", "dataset", "n", "fp_count", "severe_fp_count", "fn_count"]
    audit_cols = ["dataset", "count", "original_min", "original_max", "original_mean", "original_std", "scaled_min", "scaled_max", "scaled_mean", "scaled_std", "std_ratio_scaled_over_original", "delta_mean", "delta_std", "clipped_count", "clipped_ratio"]
    official_overall = official_metrics[(official_metrics["split"].eq("official_track_b")) & (official_metrics["dataset"].eq("overall"))] if not official_metrics.empty else pd.DataFrame()
    official_srcc = float(official_overall.iloc[0]["srcc"]) if len(official_overall) else float("nan")
    final_row = history.tail(1).iloc[0].to_dict() if len(history) else {}
    comparison = pd.DataFrame(
        [
            {"system": "Phase 16 unweighted 50k B0 controlled", "official_srcc": PHASE16_OFFICIAL_SRCC, "delta_vs_this": official_srcc - PHASE16_OFFICIAL_SRCC if math.isfinite(official_srcc) else float("nan"), "note": "controlled reference"},
            {"system": "Phase 19 W5-v2 weighted smoke", "official_srcc": PHASE19_W5V2_OFFICIAL_SRCC, "delta_vs_this": official_srcc - PHASE19_W5V2_OFFICIAL_SRCC if math.isfinite(official_srcc) else float("nan"), "note": "weighted smoke reference"},
            {"system": "Phase 21 ranking B0 smoke", "official_srcc": PHASE21_RANKING_OFFICIAL_SRCC, "delta_vs_this": official_srcc - PHASE21_RANKING_OFFICIAL_SRCC if math.isfinite(official_srcc) else float("nan"), "note": "ranking reference"},
            {"system": "14k student official", "official_srcc": BASELINE_14K_OFFICIAL_SRCC, "delta_vs_this": official_srcc - BASELINE_14K_OFFICIAL_SRCC if math.isfinite(official_srcc) else float("nan"), "note": "student baseline"},
            {"system": "MDIQA teacher official", "official_srcc": MDIQA_TEACHER_OFFICIAL_SRCC, "delta_vs_this": official_srcc - MDIQA_TEACHER_OFFICIAL_SRCC if math.isfinite(official_srcc) else float("nan"), "note": "teacher reference"},
            {"system": "production KonIQ+FLIVE existing", "official_srcc": PRODUCTION_KONIQ_FLIVE_OFFICIAL_SRCC, "delta_vs_this": official_srcc - PRODUCTION_KONIQ_FLIVE_OFFICIAL_SRCC if math.isfinite(official_srcc) else float("nan"), "note": str(PRODUCTION_REFERENCE)},
            {"system": "Phase 23 S2 KonIQ x2 label-scaled B0 smoke", "official_srcc": official_srcc, "delta_vs_this": 0.0, "note": "this run"},
        ]
    )
    lines = [
        "# Mobile Student IQA 50k Label-Scaled B0 Smoke S2 KonIQ x2 Report",
        "",
        "## 1. Executive Summary",
        f"- Final decision: `{decision}`.",
        "- Run type: final bounded B0 label-scaling smoke only; no B1/B2, no full controlled training, no TFLite export, no Flutter edits, no production `technical_score` edits, no ranking loss, no extra label-scaling variants.",
        f"- Output directory: `{out_dir}`.",
        f"- Runtime seconds: `{runtime_seconds:.2f}`.",
        f"- Train/val/test counts: `{split_meta.get('usable_counts')}`.",
        f"- Official Track B SRCC: `{official_srcc:.6f}`.",
        "",
        "## 2. Sources Inspected",
        f"- `{args.manifest_csv}`",
        f"- `{args.official_manifest_csv}`",
        "- `scripts/technical_iqa/train_mobile_student_iqa_50k_smoke.py`",
        "- `scripts/technical_iqa/train_mobile_student_iqa_50k_controlled.py`",
        "- `scripts/technical_iqa/train_mobile_student_iqa_50k_ranking_smoke.py`",
        "- `outputs/mobile_student_iqa_50k_controlled_b0_20260608/report.md`",
        "- `outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/report.md`",
        "- `outputs/mobile_student_iqa_50k_ranking_b0_smoke_r2_lambda010_margin005_20260609/report.md`",
        "- `outputs/mdiqa_student_pre_final_audit_20260609/report.md`",
        f"- `{PRODUCTION_REFERENCE}`",
        "",
        "## 3. Why This Is the Final Smoke",
        "- Phase 16 proved the 50k B0 line can train, but official SRCC stayed below the 14k student.",
        "- Phase 19 W5-v2 weighting and Phase 21 ranking both regressed, leaving KonIQ target compression as the last narrow bottleneck to test.",
        "- This run tests exactly one target scaling policy; a weak result terminates the MDIQA mobile student smoke line.",
        "",
        "## 4. Target Scaling Policy",
        f"- Policy: `{args.target_scaling_policy}`.",
        f"- KonIQ scale factor: `{args.koniq_scale_factor}`.",
        "- KonIQ rows use `mean_k + 2.0 * (teacher_target_0_100 - mean_k)`, clipped to `[0, 100]`.",
        "- FLIVE and SPAQ targets remain unchanged.",
        "- Training uses `target_norm = teacher_target_0_100_scaled / 100.0`.",
        "- Official benchmark uses official MOS labels, not scaled teacher targets.",
        "",
        "## 5. Target Scaling Audit",
        f"- Target scaling audit valid: `{split_meta.get('target_scaling_valid')}`.",
        markdown_table(target_audit, audit_cols),
        "",
        "## 6. Implementation Summary",
        "- Added focused phase runner instead of changing existing smoke/controlled/ranking defaults.",
        "- Reused smoke helper Module for manifest splits, image loading, model creation, prediction, metrics, and collapse checks.",
        "- Reused controlled helper Module for official Track B image-only evaluation and safety metrics.",
        "- Scaling columns are saved in each split manifest for auditability.",
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
        f"- ImageNet weights requested: `{args.backbone_weights}`.",
        f"- ImageNet weights loaded: `{build_info.get('backbone_weights_loaded')}`.",
        f"- Backbone weights error: `{build_info.get('backbone_weights_error')}`.",
        f"- Include preprocessing: `{build_info.get('include_preprocessing')}`.",
        f"- Finetune trainable backbone layers: `{finetune_info.get('backbone_trainable_layers')}`.",
        f"- Dataset sample weights: `{smoke.dataset_weight_map(args)}`.",
        f"- Ranking loss disabled: `{args.disable_rank_loss}`.",
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
        "## 13. Comparison Against Phase 16, Phase 19, Phase 21, 14k Student, and MDIQA Teacher",
        markdown_table(comparison, ["system", "official_srcc", "delta_vs_this", "note"]),
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
    if decision == "LABEL_SCALED_B0_SMOKE_READY_FOR_CONTROLLED_RERUN":
        lines.append("- Consider one controlled S2 KonIQ x2 B0 rerun; do not start B1/B2 from this smoke alone.")
    elif decision == "LABEL_SCALED_B0_SMOKE_NEEDS_SCALE_TUNING":
        lines.append("- If continuing, test exactly one adjacent scale factor; otherwise keep production KonIQ+FLIVE.")
    elif decision == "LABEL_SCALED_B0_SMOKE_NO_GO_RESEARCH_REFERENCE_ONLY":
        lines.append("- Terminate the MDIQA mobile student smoke line; keep existing KonIQ+FLIVE and archive MDIQA teacher as research reference.")
    elif decision == "LABEL_SCALED_B0_SMOKE_NEEDS_PIPELINE_FIX":
        lines.append("- Fix the scaling/evaluation pipeline before making the MDIQA student-line decision.")
    else:
        lines.append("- Inspect failure cases and runtime logs before retry.")
    lines.extend(
        [
            "",
            "## Checkpoint Summary",
            f"- Best checkpoint path: `{out_dir / 'best_by_val_mae_model.keras'}`.",
            f"- Final checkpoint path: `{out_dir / 'final_model.keras'}`.",
            f"- Checkpoint metric rows: `{len(checkpoint_metrics)}`.",
            f"- Final loss row: `{final_row}`.",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.disable_rank_loss:
        raise ValueError("Phase 23 requires --disable_rank_loss")
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
        raise RuntimeError(f"expected exactly 1 TensorFlow GPU, got {len(gpus)}")

    frames, failures, split_meta, target_audit = prepare_manifest(args, out_dir)
    if len(failures) > 0:
        failures.to_csv(out_dir / "failure_cases.csv", index=False)
        write_next_actions(out_dir, "LABEL_SCALED_B0_SMOKE_NEEDS_PIPELINE_FIX")
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
        checkpoint = controlled.BestValMAECheckpoint(out_dir)

        checkpoint.set_phase("head", 0)
        head_history = model.fit(
            train_ds,
            validation_data=val_ds,
            steps_per_epoch=smoke.batch_count(len(frames["train"]), args.batch_size),
            validation_steps=smoke.batch_count(len(frames["val"]), args.batch_size),
            epochs=args.head_epochs,
            callbacks=[checkpoint],
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
            callbacks=[checkpoint],
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
        prediction_cols = [
            "dataset",
            "student_split",
            "image_path",
            "quality_score_0_100",
            "teacher_target_0_100_original",
            "teacher_target_0_100_scaled",
            "target_norm_scaled",
            smoke.SELECTED_TARGET_COL,
            smoke.TARGET_NORM_COL,
            "target_scaling_policy",
            "target_scale_factor",
            "target_delta",
            "target_clipped_by_scaling",
            "pred_norm",
            PRED_COL,
            "abs_error_to_target_0_100",
            "target_source",
        ]
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
    checkpoint_metrics.to_csv(out_dir / "checkpoint_metrics.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    dataset_metrics.to_csv(out_dir / "datasetwise_metrics.csv", index=False)
    prediction_summary.to_csv(out_dir / "prediction_summary.csv", index=False)
    safety_metrics_df.to_csv(out_dir / "safety_metrics.csv", index=False)
    target_audit.to_csv(out_dir / "target_scaling_audit.csv", index=False)
    official_metrics.to_csv(out_dir / "official_benchmark_metrics.csv", index=False)
    official_dataset_metrics.to_csv(out_dir / "official_benchmark_datasetwise_metrics.csv", index=False)
    official_safety.to_csv(out_dir / "official_benchmark_safety_metrics.csv", index=False)
    failures.to_csv(out_dir / "failure_cases.csv", index=False)

    decision = final_decision(
        training_ok,
        env_info,
        failures,
        official_failures,
        history,
        collapse,
        bool(split_meta.get("target_scaling_valid")),
        official_metrics,
        official_safety,
    )
    write_next_actions(out_dir, decision)
    write_report(
        out_dir,
        args,
        env_info,
        build_info,
        split_meta,
        target_audit,
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
