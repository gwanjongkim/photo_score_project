# C6 후보를 기존 기술 IQA 공통 행 벤치마크에 추가해 비교한다.
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "outputs/technical_iqa_signal_comparison_20260604"
OUT_DIR = ROOT / "outputs/technical_iqa_signal_comparison_with_c6_20260604"
PRED_CSV = IN_DIR / "signal_predictions_common.csv"

DATASET_ORDER = ["flive", "koniq10k", "spaq"]
C6_VARIANTS = {
    "c6_topiq_koniq_cap": (0.7, 8.0),
    "c6_60_40_cap8": (0.6, 8.0),
    "c6_80_20_cap8": (0.8, 8.0),
    "c6_70_30_cap5": (0.7, 5.0),
    "c6_70_30_cap10": (0.7, 10.0),
}
COMPARE_SIGNALS = [
    "production_existing",
    "topiq_existing",
    "koniq_existing",
    "flive_existing",
    "student_14105",
    "topiq_existing_flive_57_43",
    "topiq_existing_flive_50_50",
    "topiq_existing_flive_grid_s06",
    "student_14105_flive_50_50",
    *C6_VARIANTS.keys(),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add C6 variants to existing technical IQA common-row comparison.")
    parser.add_argument("--input_csv", default=str(PRED_CSV))
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    return parser.parse_args()


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def corr(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < 2:
        return float("nan")
    x = a[finite].astype(np.float64)
    y = b[finite].astype(np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def srcc(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < 2:
        return float("nan")
    return corr(rankdata(a[finite]), rankdata(b[finite]))


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    src = df[columns].head(limit) if limit else df[columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in src.iterrows():
        vals = []
        for value in row.tolist():
            vals.append(fmt(value, 6) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def metric_row(df: pd.DataFrame, signal: str, dataset: str) -> dict[str, Any]:
    pred_col = f"{signal}_0_100"
    target = df["mos_0_100"].to_numpy(dtype=np.float64)
    pred = df[pred_col].to_numpy(dtype=np.float64)
    finite = np.isfinite(target) & np.isfinite(pred)
    target = target[finite]
    pred = pred[finite]
    row: dict[str, Any] = {"signal": signal, "dataset": dataset, "n": int(len(target))}
    if len(target) == 0:
        for key in ["mae", "rmse", "srcc", "plcc", "pred_min", "pred_max", "pred_mean", "pred_std", "target_min", "target_max", "target_mean", "target_std"]:
            row[key] = float("nan")
        row.update({"fp_count": 0, "severe_fp_count": 0, "fn_count": 0})
        return row
    row.update(
        {
            "mae": float(np.mean(np.abs(pred - target))),
            "rmse": float(np.sqrt(np.mean(np.square(pred - target)))),
            "srcc": srcc(target, pred),
            "plcc": corr(target, pred),
            "pred_min": float(np.min(pred)),
            "pred_max": float(np.max(pred)),
            "pred_mean": float(np.mean(pred)),
            "pred_std": float(np.std(pred)),
            "target_min": float(np.min(target)),
            "target_max": float(np.max(target)),
            "target_mean": float(np.mean(target)),
            "target_std": float(np.std(target)),
            "fp_count": int(((target < 40.0) & (pred > 60.0)).sum()),
            "severe_fp_count": int(((target < 40.0) & (pred > 65.0)).sum()),
            "fn_count": int(((target > 75.0) & (pred < 45.0)).sum()),
        }
    )
    return row


def compute_metrics(df: pd.DataFrame, signals: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = [metric_row(df, s, "overall") for s in signals]
    dataset_rows = []
    for signal in signals:
        for dataset in DATASET_ORDER:
            dataset_rows.append(metric_row(df[df["dataset"].astype(str).eq(dataset)], signal, dataset))
    return pd.DataFrame(overall), pd.DataFrame(dataset_rows)


def add_c6(df: pd.DataFrame) -> pd.DataFrame:
    topiq = df["topiq_existing_0_100"].to_numpy(dtype=np.float64)
    koniq = df["koniq_existing_0_100"].to_numpy(dtype=np.float64)
    for name, (topiq_weight, cap) in C6_VARIANTS.items():
        raw = topiq_weight * topiq + (1.0 - topiq_weight) * koniq
        cap_value = koniq + cap
        capped = np.minimum(raw, cap_value)
        df[f"{name}_raw_0_100"] = raw
        df[f"{name}_cap_value_0_100"] = cap_value
        df[f"{name}_cap_activated"] = raw > cap_value
        df[f"{name}_cap_clip_amount_0_100"] = np.maximum(raw - cap_value, 0.0)
        df[f"{name}_0_100"] = np.clip(capped, 0.0, 100.0)
    return df


def cap_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for signal in C6_VARIANTS:
        for dataset in ["overall", *DATASET_ORDER]:
            sub = df if dataset == "overall" else df[df["dataset"].astype(str).eq(dataset)]
            active = sub[f"{signal}_cap_activated"].astype(bool)
            clip = sub[f"{signal}_cap_clip_amount_0_100"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "signal": signal,
                    "dataset": dataset,
                    "n": int(len(sub)),
                    "cap_activation_count": int(active.sum()),
                    "cap_activation_pct": float(active.mean()) if len(sub) else float("nan"),
                    "mean_clip_amount_0_100": float(clip[clip > 0].mean()) if np.any(clip > 0) else 0.0,
                    "max_clip_amount_0_100": float(clip.max()) if len(clip) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def safety_cases(df: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    rows = []
    for signal in signals:
        pred = df[f"{signal}_0_100"]
        masks = {
            "fp": (df["mos_0_100"] < 40.0) & (pred > 60.0),
            "severe_fp": (df["mos_0_100"] < 40.0) & (pred > 65.0),
            "fn": (df["mos_0_100"] > 75.0) & (pred < 45.0),
        }
        for case_type, mask in masks.items():
            for _, row in df[mask].iterrows():
                rows.append({"signal": signal, "case_type": case_type, "dataset": row["dataset"], "image_path": row["image_path"], "mos_0_100": row["mos_0_100"], "pred_0_100": row[f"{signal}_0_100"]})
    return pd.DataFrame(rows)


def choose_decision(overall: pd.DataFrame, datasetwise: pd.DataFrame) -> dict[str, Any]:
    selected = overall[overall["signal"].eq("topiq_existing_flive_57_43")].iloc[0].to_dict()
    c6 = overall[overall["signal"].eq("c6_topiq_koniq_cap")].iloc[0].to_dict()
    selected_ds = datasetwise[datasetwise["signal"].eq("topiq_existing_flive_57_43")]
    c6_ds = datasetwise[datasetwise["signal"].eq("c6_topiq_koniq_cap")]
    deltas = []
    dangerous = False
    for dataset in DATASET_ORDER:
        s = selected_ds[selected_ds["dataset"].eq(dataset)].iloc[0]
        c = c6_ds[c6_ds["dataset"].eq(dataset)].iloc[0]
        delta = float(c["srcc"]) - float(s["srcc"])
        deltas.append({"dataset": dataset, "c6_srcc_delta_vs_topiq_flive": delta})
        if delta < -0.03:
            dangerous = True
    if int(c6["n"]) < 1000:
        decision = "C6_NEEDS_MORE_COVERAGE"
    elif float(c6["srcc"]) >= float(selected["srcc"]) - 0.005 and int(c6["severe_fp_count"]) <= int(selected["severe_fp_count"]) and not dangerous:
        decision = "C6_READY_FOR_LOG_ONLY_PROFILING"
    elif float(c6["srcc"]) > 0.78 and int(c6["severe_fp_count"]) <= int(selected["severe_fp_count"]):
        decision = "C6_NEEDS_RATIO_OR_CAP_TUNING"
    else:
        decision = "TOPIQ_FLIVE_REMAINS_BEST_LOG_ONLY_CANDIDATE"
    return {
        "final_decision": decision,
        "c6_signal": "c6_topiq_koniq_cap",
        "comparison_signal": "topiq_existing_flive_57_43",
        "c6_overall_srcc": float(c6["srcc"]),
        "c6_overall_mae": float(c6["mae"]),
        "topiq_flive_overall_srcc": float(selected["srcc"]),
        "topiq_flive_overall_mae": float(selected["mae"]),
        "c6_severe_fp": int(c6["severe_fp_count"]),
        "topiq_flive_severe_fp": int(selected["severe_fp_count"]),
        "dataset_deltas": deltas,
        "log_only_candidate": decision == "C6_READY_FOR_LOG_ONLY_PROFILING",
        "production_replacement": False,
    }


def write_report(out_dir: Path, coverage: pd.DataFrame, overall: pd.DataFrame, datasetwise: pd.DataFrame, cap: pd.DataFrame, ranking: pd.DataFrame, safety: pd.DataFrame, rec: dict[str, Any]) -> None:
    metric_cols = ["signal", "dataset", "n", "mae", "rmse", "srcc", "plcc", "fp_count", "severe_fp_count", "fn_count"]
    c6_dataset = datasetwise[datasetwise["signal"].isin(["c6_topiq_koniq_cap", "topiq_existing_flive_57_43", "topiq_existing"])]
    lines = [
        "# Technical IQA Signal Comparison with C6 Report",
        "",
        "## 1. Executive Summary",
        f"- Final decision: `{rec['final_decision']}`.",
        "- Benchmark type: offline WSL CSV computation only.",
        "- Candidate status: log-only only; not production replacement.",
        f"- C6 overall SRCC/MAE: `{fmt(rec['c6_overall_srcc'])}` / `{fmt(rec['c6_overall_mae'])}`.",
        f"- TOPIQ+FLIVE 57/43 overall SRCC/MAE: `{fmt(rec['topiq_flive_overall_srcc'])}` / `{fmt(rec['topiq_flive_overall_mae'])}`.",
        "",
        "## 2. Sources Inspected",
        "- `outputs/technical_iqa_signal_comparison_20260604/`",
        "- `scripts/technical_iqa/benchmark_technical_iqa_signals_20260604.py`",
        "- `outputs/technical_iqa_topiq_koniq_cap_benchmark_20260524/`",
        "- `outputs/technical_iqa_topiq_koniq_cap_ranking_audit_20260524/`",
        "- `outputs/c6_android_wsl_parity_20260527/`",
        "- `outputs/c6_android_wsl_parity_exif_fixed_20260529/`",
        "- `outputs/c6_topiq_preprocess_backend_ablation_20260529/`",
        "- `outputs/topiq_mixed112_artifact_recovery_20260527/`",
        "- `outputs/final_topiq_lite_mixed112_export_20260517/`",
        "- `exports/tflite/`",
        "",
        "## 3. Benchmark Scope and Coverage",
        markdown_table(coverage, ["dataset", "n", "topiq_non_null", "koniq_non_null", "c6_non_null", "coverage_pct"]),
        "",
        "## 4. C6 Formula and Inputs",
        "- Main formula: `min(0.7 * topiq_existing + 0.3 * koniq_existing, koniq_existing + 8)`.",
        "- Inputs: `topiq_existing_0_100`, `koniq_existing_0_100` from current 1413-row common predictions.",
        "- Sensitivity variants are experimental.",
        "",
        "## 5. C6 Cap Diagnostics",
        markdown_table(cap, ["signal", "dataset", "n", "cap_activation_count", "cap_activation_pct", "mean_clip_amount_0_100", "max_clip_amount_0_100"]),
        "",
        "## 6. Overall Metrics",
        markdown_table(overall, metric_cols),
        "",
        "## 7. Dataset-wise Metrics",
        markdown_table(c6_dataset.sort_values(["dataset", "signal"]), metric_cols),
        "",
        "## 8. Comparison Against TOPIQ+FLIVE",
        f"- C6 SRCC delta vs TOPIQ+FLIVE 57/43: `{fmt(rec['c6_overall_srcc'] - rec['topiq_flive_overall_srcc'])}`.",
        f"- C6 MAE delta vs TOPIQ+FLIVE 57/43: `{fmt(rec['c6_overall_mae'] - rec['topiq_flive_overall_mae'])}`.",
        f"- Dataset deltas: `{json.dumps(rec['dataset_deltas'])}`.",
        "",
        "## 9. Safety Cases",
        f"- Safety rows: `{len(safety)}`.",
        f"- C6 severe FP: `{rec['c6_severe_fp']}`.",
        f"- TOPIQ+FLIVE severe FP: `{rec['topiq_flive_severe_fp']}`.",
        "",
        "## 10. Final Ranking",
        markdown_table(ranking, ["rank", "signal", "overall_srcc", "overall_mae", "min_dataset_srcc", "fp_count", "severe_fp_count", "composite"]),
        "",
        "## 11. Final Decision",
        str(rec["final_decision"]),
        "",
        "## 12. Recommended Next Step",
        "- Keep production unchanged. If desired, carry only the stronger log-only candidate forward; C6 remains offline/log-only evidence, not production replacement.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)
    required = {"topiq_existing_0_100", "koniq_existing_0_100", "mos_0_100", "dataset", "image_path"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    df = add_c6(df)
    signals = [s for s in COMPARE_SIGNALS if f"{s}_0_100" in df.columns]
    overall, datasetwise = compute_metrics(df, signals)
    cap = cap_diagnostics(df)
    safety = safety_cases(df, signals)
    rows = []
    for signal in signals:
        ov = overall[overall["signal"].eq(signal)].iloc[0]
        ds = datasetwise[datasetwise["signal"].eq(signal)]
        min_srcc = float(ds["srcc"].min())
        severe = int(ds["severe_fp_count"].sum())
        fp = int(ds["fp_count"].sum())
        rows.append(
            {
                "signal": signal,
                "overall_srcc": float(ov["srcc"]),
                "overall_mae": float(ov["mae"]),
                "min_dataset_srcc": min_srcc,
                "fp_count": fp,
                "severe_fp_count": severe,
                "composite": float(ov["srcc"]) + 0.25 * min_srcc - 0.01 * severe - 0.002 * fp,
            }
        )
    ranking = pd.DataFrame(rows).sort_values(["composite", "overall_srcc"], ascending=[False, False]).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    rec = choose_decision(overall, datasetwise)
    coverage_rows = []
    for dataset in ["overall", *DATASET_ORDER]:
        sub = df if dataset == "overall" else df[df["dataset"].astype(str).eq(dataset)]
        n = len(sub)
        c6_non_null = int(sub["c6_topiq_koniq_cap_0_100"].notna().sum())
        coverage_rows.append(
            {
                "dataset": dataset,
                "n": n,
                "topiq_non_null": int(sub["topiq_existing_0_100"].notna().sum()),
                "koniq_non_null": int(sub["koniq_existing_0_100"].notna().sum()),
                "c6_non_null": c6_non_null,
                "coverage_pct": float(c6_non_null / n) if n else float("nan"),
            }
        )
    coverage = pd.DataFrame(coverage_rows)

    keep_cols = ["dataset", "split", "split_original", "image_path", "mos_0_100"] + [f"{s}_0_100" for s in signals]
    df[keep_cols].to_csv(out_dir / "signal_predictions_common_with_c6.csv", index=False)
    overall.to_csv(out_dir / "signal_metrics_overall_with_c6.csv", index=False)
    datasetwise.to_csv(out_dir / "signal_metrics_datasetwise_with_c6.csv", index=False)
    cap.to_csv(out_dir / "c6_cap_diagnostics.csv", index=False)
    overall[overall["signal"].isin(C6_VARIANTS)].to_csv(out_dir / "c6_variant_metrics.csv", index=False)
    ranking.to_csv(out_dir / "candidate_ranking_with_c6.csv", index=False)
    safety.to_csv(out_dir / "safety_cases_with_c6.csv", index=False)
    (out_dir / "c6_recommendation.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(
        [
            {"priority": 1, "action": "Keep production technical_score unchanged."},
            {"priority": 2, "action": "Prefer TOPIQ+FLIVE log-only candidate unless C6 parity/generalization and ratio/cap evidence improves."},
            {"priority": 3, "action": "Do not ship C6 as replacement; C6 remains offline/log-only evidence only."},
        ]
    ).to_csv(out_dir / "next_actions.csv", index=False)
    write_report(out_dir, coverage, overall, datasetwise, cap, ranking, safety, rec)
    print(f"Report: {out_dir / 'report.md'}")
    print(f"Decision: {rec['final_decision']}")
    print(f"C6 SRCC: {rec['c6_overall_srcc']:.6f}")


if __name__ == "__main__":
    main()
