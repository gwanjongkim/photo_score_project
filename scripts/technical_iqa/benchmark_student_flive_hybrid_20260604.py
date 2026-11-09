# 학생 IQA와 FLIVE 컴포넌트의 오프라인 하이브리드 후보를 벤치마크한다.
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "outputs/mobile_student_iqa_14105_full_controlled_20260604/best_by_val_mae_model.keras"
DEFAULT_TEST_MANIFEST = ROOT / "outputs/student_distillation_manifest_14105_20260603/test_manifest.csv"
DEFAULT_OUT_DIR = ROOT / "outputs/student_flive_hybrid_benchmark_20260604"
FUSION_FILES = {
    "flive": ROOT / "outputs/technical_iqa_combo_benchmark_20260524/fusion_predictions_flive.csv",
    "koniq10k": ROOT / "outputs/technical_iqa_combo_benchmark_20260524/fusion_predictions_koniq.csv",
    "spaq": ROOT / "outputs/technical_iqa_combo_benchmark_20260524/fusion_predictions_spaq.csv",
}
SOURCES_INSPECTED = [
    "outputs/mobile_student_iqa_14105_checkpoint_eval_20260604/report.md",
    "outputs/mobile_student_iqa_14105_tflite_export_20260604/report.md",
    "outputs/student_distillation_manifest_14105_20260603/test_manifest.csv",
    "outputs/technical_iqa_combo_benchmark_20260524/fusion_predictions_flive.csv",
    "outputs/technical_iqa_combo_benchmark_20260524/fusion_predictions_koniq.csv",
    "outputs/technical_iqa_combo_benchmark_20260524/fusion_predictions_spaq.csv",
    "outputs/research_protocol_production_technical_eval_20260530/report.md",
    "src/infer/select_best_shots.py",
]
DATASET_ORDER = ["flive", "koniq10k", "spaq"]
FIXED_RATIOS = {
    "hybrid_4_3": 4.0 / 7.0,
    "hybrid_5_5": 0.5,
    "hybrid_6_4": 0.6,
    "hybrid_4_6": 0.4,
}
SIGNAL_ORDER = ["production_existing", "student", "flive", "hybrid_4_3", "hybrid_5_5", "hybrid_6_4", "hybrid_4_6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark student + FLIVE hybrid candidates on cached common rows.")
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL))
    parser.add_argument("--test_manifest", default=str(DEFAULT_TEST_MANIFEST))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--batch_size", type=int, default=32)
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


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
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
    return corrcoef(rankdata(a[finite]), rankdata(b[finite]))


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
            if isinstance(value, float):
                vals.append(fmt(value, 6))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def load_fusion_components() -> pd.DataFrame:
    parts = []
    for dataset, path in FUSION_FILES.items():
        df = pd.read_csv(path)
        need = ["image_path", "target_mos_100", "koniq_score", "flive_score"]
        missing = [col for col in need if col not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns {missing}")
        out = df[need].copy()
        out["dataset"] = dataset
        out["basename"] = out["image_path"].map(lambda p: Path(str(p)).name)
        out = out.drop_duplicates(["dataset", "basename"], keep="first")
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def build_common_rows(test_manifest: Path, fusion: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(test_manifest)
    manifest["basename"] = manifest["image_path"].map(lambda p: Path(str(p)).name)
    manifest_counts = manifest.groupby("dataset").size().reset_index(name="test_manifest_rows")
    fusion_counts = fusion.groupby("dataset").size().reset_index(name="cached_fusion_rows")
    merged = manifest.merge(
        fusion[["dataset", "basename", "koniq_score", "flive_score"]],
        on=["dataset", "basename"],
        how="left",
        indicator=True,
    )
    merged["has_mos"] = pd.to_numeric(merged["quality_score_0_100"], errors="coerce").notna()
    merged["has_fusion"] = merged["_merge"].eq("both") & pd.to_numeric(merged["koniq_score"], errors="coerce").notna() & pd.to_numeric(merged["flive_score"], errors="coerce").notna()
    common = merged[merged["has_mos"] & merged["has_fusion"]].copy()
    common["mos_0_100"] = pd.to_numeric(common["quality_score_0_100"], errors="coerce")
    common["koniq_component_0_100"] = pd.to_numeric(common["koniq_score"], errors="coerce").clip(0.0, 100.0)
    common["flive_0_100"] = pd.to_numeric(common["flive_score"], errors="coerce").clip(0.0, 100.0)
    common["production_existing_0_100"] = ((4.0 * common["koniq_component_0_100"]) + (3.0 * common["flive_0_100"])) / 7.0
    common = common.drop(columns=["_merge"])
    common_counts = common.groupby("dataset").size().reset_index(name="common_rows")
    coverage = manifest_counts.merge(fusion_counts, on="dataset", how="outer").merge(common_counts, on="dataset", how="outer")
    coverage = coverage.fillna(0)
    coverage["test_manifest_rows"] = coverage["test_manifest_rows"].astype(int)
    coverage["cached_fusion_rows"] = coverage["cached_fusion_rows"].astype(int)
    coverage["common_rows"] = coverage["common_rows"].astype(int)
    coverage["common_pct_of_test"] = np.where(
        coverage["test_manifest_rows"] > 0,
        coverage["common_rows"] / coverage["test_manifest_rows"],
        np.nan,
    )
    total = pd.DataFrame(
        [
            {
                "dataset": "overall",
                "test_manifest_rows": int(len(manifest)),
                "cached_fusion_rows": int(len(fusion)),
                "common_rows": int(len(common)),
                "common_pct_of_test": float(len(common) / len(manifest)) if len(manifest) else float("nan"),
            }
        ]
    )
    return common.reset_index(drop=True), pd.concat([total, coverage], ignore_index=True)


def load_rgb_resized(path: str, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32)


def predict_student(model_path: Path, df: pd.DataFrame, batch_size: int) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    model = tf.keras.models.load_model(model_path, compile=False)
    image_size = int(model.input_shape[1])
    images = []
    failures = []
    keep_indices = []
    for idx, row in df.iterrows():
        path = str(row["image_path"])
        try:
            images.append(load_rgb_resized(path, image_size))
            keep_indices.append(idx)
        except Exception as exc:  # noqa: BLE001
            failures.append({"dataset": row.get("dataset"), "image_path": path, "error": str(exc)})
    if not images:
        raise RuntimeError("no images loaded for student inference")
    arr = np.stack(images).astype(np.float32)
    t0 = time.perf_counter()
    pred = model.predict(arr, batch_size=batch_size, verbose=0).reshape(-1).astype(np.float64) * 100.0
    runtime = time.perf_counter() - t0
    out = df.loc[keep_indices].copy().reset_index(drop=True)
    failure_df = pd.DataFrame(failures)
    info = {
        "image_size": image_size,
        "image_load_failures": int(len(failures)),
        "student_runtime_seconds": runtime,
        "student_avg_ms_per_image": float(runtime * 1000.0 / len(out)),
    }
    return np.clip(pred, 0.0, 100.0), out, info


def add_signals(df: pd.DataFrame, student_pred: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["student_0_100"] = np.clip(student_pred, 0.0, 100.0)
    out["flive_0_100"] = out["flive_0_100"].astype(float).clip(0.0, 100.0)
    out["production_existing_0_100"] = out["production_existing_0_100"].astype(float).clip(0.0, 100.0)
    for name, weight in FIXED_RATIOS.items():
        out[f"{name}_0_100"] = np.clip((weight * out["student_0_100"]) + ((1.0 - weight) * out["flive_0_100"]), 0.0, 100.0)
    for i in range(2, 9):
        weight = i / 10.0
        out[f"hybrid_grid_s{int(weight * 10):02d}_0_100"] = np.clip((weight * out["student_0_100"]) + ((1.0 - weight) * out["flive_0_100"]), 0.0, 100.0)
    return out


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
            "plcc": corrcoef(target, pred),
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


def all_metrics(df: pd.DataFrame, signals: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows = [metric_row(df, signal, "overall") for signal in signals]
    dataset_rows = []
    for signal in signals:
        for dataset in DATASET_ORDER:
            sub = df[df["dataset"].astype(str).eq(dataset)]
            if len(sub):
                dataset_rows.append(metric_row(sub, signal, dataset))
    return pd.DataFrame(overall_rows), pd.DataFrame(dataset_rows)


def grid_metrics(df: pd.DataFrame) -> pd.DataFrame:
    signals = [f"hybrid_grid_s{i:02d}" for i in range(2, 9)] + list(FIXED_RATIOS)
    rows = []
    for signal in signals:
        overall = metric_row(df, signal, "overall")
        ds_rows = [metric_row(df[df["dataset"].astype(str).eq(ds)], signal, ds) for ds in DATASET_ORDER if len(df[df["dataset"].astype(str).eq(ds)])]
        min_dataset_srcc = min([r["srcc"] for r in ds_rows if math.isfinite(float(r["srcc"]))], default=float("nan"))
        severe_fp_total = sum(int(r["severe_fp_count"]) for r in ds_rows)
        fp_total = sum(int(r["fp_count"]) for r in ds_rows)
        composite = overall["srcc"] + 0.25 * min_dataset_srcc - 0.01 * severe_fp_total - 0.002 * fp_total
        weight = float(signal.split("_s")[-1]) / 10.0 if signal.startswith("hybrid_grid") else FIXED_RATIOS[signal]
        rows.append(
            {
                "signal": signal,
                "student_weight": weight,
                "flive_weight": 1.0 - weight,
                "overall_n": overall["n"],
                "overall_mae": overall["mae"],
                "overall_srcc": overall["srcc"],
                "overall_plcc": overall["plcc"],
                "min_dataset_srcc": min_dataset_srcc,
                "fp_count": fp_total,
                "severe_fp_count": severe_fp_total,
                "fn_count": sum(int(r["fn_count"]) for r in ds_rows),
                "composite": composite,
            }
        )
    return pd.DataFrame(rows).sort_values(["composite", "overall_srcc"], ascending=[False, False]).reset_index(drop=True)


def collect_safety_cases(df: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    rows = []
    for signal in signals:
        pred = df[f"{signal}_0_100"]
        fp = (df["mos_0_100"] < 40.0) & (pred > 60.0)
        severe = (df["mos_0_100"] < 40.0) & (pred > 65.0)
        fn = (df["mos_0_100"] > 75.0) & (pred < 45.0)
        for flag, mask in [("fp", fp), ("severe_fp", severe), ("fn", fn)]:
            for _, row in df[mask].iterrows():
                rows.append(
                    {
                        "signal": signal,
                        "case_type": flag,
                        "dataset": row["dataset"],
                        "image_path": row["image_path"],
                        "mos_0_100": row["mos_0_100"],
                        "pred_0_100": row[f"{signal}_0_100"],
                        "student_0_100": row["student_0_100"],
                        "flive_0_100": row["flive_0_100"],
                        "production_existing_0_100": row["production_existing_0_100"],
                    }
                )
    return pd.DataFrame(rows)


def choose_recommendation(
    candidate_metrics: pd.DataFrame,
    dataset_metrics: pd.DataFrame,
    grid: pd.DataFrame,
    coverage: pd.DataFrame,
    image_load_failures: int,
) -> dict[str, Any]:
    prod = candidate_metrics[candidate_metrics["signal"].eq("production_existing")].iloc[0].to_dict()
    best_srcc = grid.sort_values("overall_srcc", ascending=False).iloc[0].to_dict()
    best_mae = grid.sort_values("overall_mae", ascending=True).iloc[0].to_dict()
    best_composite = grid.iloc[0].to_dict()
    prod_ds = dataset_metrics[dataset_metrics["signal"].eq("production_existing")]
    selected_name = str(best_composite["signal"])
    selected_weight = float(best_composite["student_weight"])
    for fixed_name, fixed_weight in FIXED_RATIOS.items():
        if abs(selected_weight - fixed_weight) < 1e-9:
            selected_name = fixed_name
            break
    selected_ds = dataset_metrics[dataset_metrics["signal"].eq(selected_name)]
    flive_prod = prod_ds[prod_ds["dataset"].eq("flive")]
    flive_sel = selected_ds[selected_ds["dataset"].eq("flive")]
    severe_prod = int(prod_ds["severe_fp_count"].sum())
    severe_sel = int(selected_ds["severe_fp_count"].sum())
    common_total = int(coverage[coverage["dataset"].eq("overall")]["common_rows"].iloc[0])
    flive_constraint_pass = False
    if not flive_prod.empty and not flive_sel.empty:
        flive_constraint_pass = (
            float(flive_sel.iloc[0]["srcc"]) >= float(flive_prod.iloc[0]["srcc"]) - 0.03
            and severe_sel <= severe_prod
            and float(best_composite["overall_srcc"]) >= float(prod["srcc"]) - 0.01
        )
    stable_datasets_pass = True
    stability_rows = []
    for dataset in ["koniq10k", "spaq"]:
        prod_row = prod_ds[prod_ds["dataset"].eq(dataset)]
        sel_row = selected_ds[selected_ds["dataset"].eq(dataset)]
        if prod_row.empty or sel_row.empty:
            stable_datasets_pass = False
            continue
        delta = float(sel_row.iloc[0]["srcc"]) - float(prod_row.iloc[0]["srcc"])
        stability_rows.append({"dataset": dataset, "srcc_delta_vs_production": delta})
        if delta < -0.03:
            stable_datasets_pass = False
    if common_total < 300 or image_load_failures > 0:
        decision = "HYBRID_STUDENT_FLIVE_NEEDS_MORE_COVERAGE"
    elif float(best_composite["overall_srcc"]) < float(prod["srcc"]) and float(best_composite["overall_mae"]) > float(prod["mae"]):
        decision = "HYBRID_STUDENT_FLIVE_NOT_BETTER_THAN_PRODUCTION"
    elif not flive_constraint_pass or not stable_datasets_pass:
        decision = "HYBRID_STUDENT_FLIVE_NEEDS_RATIO_TUNING"
    else:
        decision = "HYBRID_STUDENT_FLIVE_READY_FOR_LOG_ONLY_PROFILING"
    return {
        "final_decision": decision,
        "selected_ratio": selected_name,
        "selected_student_weight": selected_weight,
        "selected_flive_weight": 1.0 - selected_weight,
        "best_ratio_by_overall_srcc": str(best_srcc["signal"]),
        "best_ratio_by_overall_mae": str(best_mae["signal"]),
        "best_ratio_by_composite": str(best_composite["signal"]),
        "production_overall_srcc": float(prod["srcc"]),
        "production_overall_mae": float(prod["mae"]),
        "selected_overall_srcc": float(best_composite["overall_srcc"]),
        "selected_overall_mae": float(best_composite["overall_mae"]),
        "selected_composite": float(best_composite["composite"]),
        "flive_constraint_pass": bool(flive_constraint_pass),
        "koniq_spaq_stability_pass": bool(stable_datasets_pass),
        "koniq_spaq_stability": stability_rows,
        "common_rows": common_total,
        "image_load_failures": int(image_load_failures),
        "log_only_candidate": decision == "HYBRID_STUDENT_FLIVE_READY_FOR_LOG_ONLY_PROFILING",
        "production_replacement": False,
    }


def write_report(
    out_dir: Path,
    coverage: pd.DataFrame,
    candidate_metrics: pd.DataFrame,
    dataset_metrics: pd.DataFrame,
    grid: pd.DataFrame,
    safety: pd.DataFrame,
    recommendation: dict[str, Any],
    runtime_info: dict[str, Any],
) -> None:
    selected = recommendation["selected_ratio"]
    if recommendation["final_decision"] == "HYBRID_STUDENT_FLIVE_READY_FOR_LOG_ONLY_PROFILING":
        next_step = "- Profile selected hybrid as mobile log-only telemetry only; do not replace production `technical_score`."
    elif recommendation["final_decision"] == "HYBRID_STUDENT_FLIVE_NEEDS_RATIO_TUNING":
        next_step = "- Do not profile yet; increase KonIQ coverage and tune ratio/gating because selected hybrid regresses KonIQ/SPAQ stability."
    else:
        next_step = "- Keep existing production technical score and resolve benchmark coverage/quality gaps before any log-only candidate."
    selected_dataset = dataset_metrics[dataset_metrics["signal"].isin(["production_existing", "student", "flive", selected])]
    metric_cols = ["signal", "dataset", "n", "mae", "rmse", "srcc", "plcc", "fp_count", "severe_fp_count", "fn_count"]
    report = [
        "# Student + FLIVE Hybrid Benchmark Report",
        "",
        "## 1. Executive Summary",
        f"- Final decision: `{recommendation['final_decision']}`.",
        "- Benchmark type: WSL offline only.",
        "- Candidate status: log-only if accepted; not production replacement.",
        f"- Common rows: `{recommendation['common_rows']}`.",
        f"- Selected ratio: `{selected}` with student weight `{fmt(recommendation['selected_student_weight'], 4)}` and FLIVE weight `{fmt(recommendation['selected_flive_weight'], 4)}`.",
        f"- Runtime seconds: `{fmt(runtime_info['runtime_seconds'], 2)}`.",
        "",
        "## 2. Sources Inspected",
        *[f"- `{path}`" for path in SOURCES_INSPECTED],
        "",
        "## 3. Signal Coverage",
        markdown_table(coverage, ["dataset", "test_manifest_rows", "cached_fusion_rows", "common_rows", "common_pct_of_test"]),
        "",
        "## 4. Existing Production vs Student vs FLIVE",
        markdown_table(candidate_metrics[candidate_metrics["signal"].isin(["production_existing", "student", "flive"])], metric_cols),
        "",
        "## 5. Hybrid Ratio Candidates",
        markdown_table(candidate_metrics[candidate_metrics["signal"].str.startswith("hybrid")], metric_cols),
        "",
        "## 6. Dataset-wise Results",
        markdown_table(selected_dataset.sort_values(["dataset", "signal"]), metric_cols),
        "",
        "## 7. Safety Cases",
        f"- Safety case rows: `{len(safety)}`.",
        f"- Selected severe FP total: `{int(dataset_metrics[dataset_metrics['signal'].eq(selected)]['severe_fp_count'].sum())}`.",
        f"- Production severe FP total: `{int(dataset_metrics[dataset_metrics['signal'].eq('production_existing')]['severe_fp_count'].sum())}`.",
        "",
        "## 8. Best Ratio Selection",
        f"- Best by overall SRCC: `{recommendation['best_ratio_by_overall_srcc']}`.",
        f"- Best by overall MAE: `{recommendation['best_ratio_by_overall_mae']}`.",
        f"- Best by composite: `{recommendation['best_ratio_by_composite']}`.",
        f"- FLIVE constraint pass: `{recommendation['flive_constraint_pass']}`.",
        f"- KonIQ/SPAQ stability pass: `{recommendation['koniq_spaq_stability_pass']}`.",
        markdown_table(grid.head(10), ["signal", "student_weight", "flive_weight", "overall_mae", "overall_srcc", "min_dataset_srcc", "fp_count", "severe_fp_count", "composite"]),
        "",
        "## 9. FLIVE Weakness Analysis",
        "- Student alone remains weaker on FLIVE than FLIVE component and production image-level blend on this common-row benchmark.",
        "- Hybrid improves KonIQ/SPAQ exposure by injecting student signal, but selected candidate must not regress FLIVE beyond constraint.",
        "- This benchmark uses cached image-level FLIVE component, not Flutter production runtime.",
        "",
        "## 10. Final Decision",
        str(recommendation["final_decision"]),
        "",
        "## 11. Recommended Next Step",
        next_step,
    ]
    (out_dir / "report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fusion = load_fusion_components()
    common, coverage = build_common_rows(Path(args.test_manifest), fusion)
    if common.empty:
        raise RuntimeError("no common rows with MOS, student manifest, and cached fusion components")

    student_pred, common_loaded, runtime_info = predict_student(Path(args.model_path), common, int(args.batch_size))
    predictions = add_signals(common_loaded, student_pred)

    candidate_metrics, dataset_metrics = all_metrics(predictions, SIGNAL_ORDER)
    grid = grid_metrics(predictions)
    safety = collect_safety_cases(predictions, SIGNAL_ORDER)
    recommendation = choose_recommendation(
        candidate_metrics=candidate_metrics,
        dataset_metrics=dataset_metrics,
        grid=grid,
        coverage=coverage,
        image_load_failures=int(runtime_info["image_load_failures"]),
    )
    runtime_info["runtime_seconds"] = time.perf_counter() - t0

    keep_cols = [
        "dataset",
        "split",
        "split_original",
        "image_path",
        "mos_0_100",
        "koniq_component_0_100",
        "flive_0_100",
        "production_existing_0_100",
        "student_0_100",
        "hybrid_4_3_0_100",
        "hybrid_5_5_0_100",
        "hybrid_6_4_0_100",
        "hybrid_4_6_0_100",
    ]
    predictions[keep_cols].to_csv(out_dir / "hybrid_candidate_predictions.csv", index=False)
    candidate_metrics.to_csv(out_dir / "hybrid_candidate_metrics.csv", index=False)
    dataset_metrics.to_csv(out_dir / "datasetwise_hybrid_metrics.csv", index=False)
    grid.to_csv(out_dir / "hybrid_ratio_grid_metrics.csv", index=False)
    coverage.to_csv(out_dir / "coverage_audit.csv", index=False)
    safety.to_csv(out_dir / "safety_cases.csv", index=False)
    (out_dir / "hybrid_recommendation.json").write_text(json.dumps(recommendation, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(
        [
            {
                "priority": 1,
                "action": "Do not profile hybrid yet; keep production technical_score unchanged.",
            },
            {
                "priority": 2,
                "action": "Increase KonIQ common-row coverage or run controlled student inference on cached production benchmark rows before any runtime decision.",
            },
            {
                "priority": 3,
                "action": "Review FLIVE-heavy disagreement rows and severe/standard FP cases before ratio tuning.",
            },
        ]
    ).to_csv(out_dir / "next_actions.csv", index=False)
    write_report(out_dir, coverage, candidate_metrics, dataset_metrics, grid, safety, recommendation, runtime_info)

    print(f"Report: {out_dir / 'report.md'}")
    print(f"Decision: {recommendation['final_decision']}")
    print(f"Common rows: {recommendation['common_rows']}")
    print(f"Selected: {recommendation['selected_ratio']}")


if __name__ == "__main__":
    main()
