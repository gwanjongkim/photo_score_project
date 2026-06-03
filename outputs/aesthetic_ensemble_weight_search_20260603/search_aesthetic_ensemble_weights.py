# 미학 모델 앙상블 가중치 탐색 결과를 기존 예측 CSV에서 계산한다.
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs/aesthetic_ensemble_weight_search_20260603"
SEED = 42


@dataclass(frozen=True)
class PredictionSpec:
    dataset: str
    model: str
    path: Path
    id_col: str = "image"
    gt_col: str = "gt_score"
    pred_col: str = "pred_score"
    key_mode: str = "stem"
    primary: bool = True
    note: str = ""


PRIMARY_SPECS: List[PredictionSpec] = [
    PredictionSpec(
        "AVA",
        "nima",
        ROOT / "outputs/aesthetic_mobile_paper_benchmark_20260603/full/nima_ava/nima_predictions.csv",
        note="primary 20260603 full NIMA AVA",
    ),
    PredictionSpec(
        "AVA",
        "alamp_adaptive",
        ROOT
        / "outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603/full_adaptive/alamp_adaptive_predictions.csv",
        note="primary 20260603 full adaptive A-LAMP AVA",
    ),
    PredictionSpec(
        "AADB",
        "rgnet",
        ROOT / "outputs/aesthetic_mobile_paper_benchmark_20260603/full/rgnet_aadb/rgnet_predictions.csv",
        note="primary 20260603 full RGNet AADB",
    ),
    PredictionSpec(
        "ICAA17K",
        "icaa",
        ROOT / "outputs/aesthetic_mobile_paper_benchmark_20260603/full/icaa_icaa17k/icaa_predictions.csv",
        key_mode="full",
        note="primary 20260603 full ICAA17K color prediction",
    ),
]

NON_PRIMARY_CANDIDATES: List[PredictionSpec] = [
    PredictionSpec(
        "AADB",
        "nima",
        ROOT / "outputs/distill_full_aadb/teacher_labels_train.csv",
        id_col="image_path",
        gt_col="aadb_score",
        pred_col="nima_mean_score",
        primary=False,
        note="teacher-label cache, not a benchmark prediction CSV; not used for primary optimization",
    ),
    PredictionSpec(
        "AADB",
        "alamp",
        ROOT / "outputs/distill_full_aadb/teacher_labels_train.csv",
        id_col="image_path",
        gt_col="aadb_score",
        pred_col="alamp_score",
        primary=False,
        note="teacher-label cache, non-adaptive A-LAMP score; not used for primary optimization",
    ),
    PredictionSpec(
        "AADB",
        "rgnet",
        ROOT / "outputs/distill_full_aadb/teacher_labels_train.csv",
        id_col="image_path",
        gt_col="aadb_score",
        pred_col="rgnet_score",
        primary=False,
        note="teacher-label cache, not used because primary RGNet AADB exists",
    ),
    PredictionSpec(
        "AVA",
        "rgnet",
        ROOT / "outputs/paper_level_stage1_benchmark_alignment_20260523/rgnet_pil_preprocess_predictions.csv",
        id_col="image_path",
        gt_col="true_score",
        pred_col="prediction",
        primary=False,
        note="older protocol candidate, not mixed with 20260603 adaptive A-LAMP",
    ),
    PredictionSpec(
        "ICAA17K",
        "icaa",
        ROOT / "outputs/icaa_performance_reverification_20260525/icaa_predictions.csv",
        id_col="ID",
        gt_col="gt_color",
        pred_col="pred_color",
        key_mode="full",
        primary=False,
        note="older ICAA reverification candidate; primary 20260603 ICAA17K exists",
    ),
]

REQUESTED = {
    "AVA": ["nima", "rgnet", "alamp_adaptive", "icaa"],
    "AADB": ["nima", "rgnet", "alamp_adaptive", "icaa"],
    "ICAA17K": ["icaa"],
}
GROUP_A = ["nima", "rgnet", "alamp_adaptive"]
GROUP_B = ["nima", "rgnet", "alamp_adaptive", "icaa"]


def key_from_path(value: object, mode: str = "stem") -> str:
    text = str(value)
    if mode == "full":
        return text.strip().lower()
    name = Path(text).name
    return Path(name).stem.lower()


def read_prediction(spec: PredictionSpec) -> pd.DataFrame:
    raw = pd.read_csv(spec.path)
    missing = [c for c in [spec.id_col, spec.gt_col, spec.pred_col] if c not in raw.columns]
    if missing:
        raise ValueError(f"{spec.path} missing columns: {missing}")
    out = raw[[spec.id_col, spec.gt_col, spec.pred_col]].copy()
    out.columns = ["source_id", "gt_score", f"{spec.model}_raw"]
    out["join_key"] = out["source_id"].map(lambda value: key_from_path(value, spec.key_mode))
    out["gt_score"] = pd.to_numeric(out["gt_score"], errors="coerce")
    out[f"{spec.model}_raw"] = pd.to_numeric(out[f"{spec.model}_raw"], errors="coerce")
    out = out.dropna(subset=["join_key", "gt_score", f"{spec.model}_raw"])
    out = out.drop_duplicates("join_key", keep="first")
    out["dataset"] = spec.dataset
    return out


def deterministic_fold(keys: Sequence[str]) -> np.ndarray:
    folds = []
    for key in keys:
        digest = hashlib.sha256(f"{SEED}:{key}".encode("utf-8")).hexdigest()
        value = int(digest[:8], 16) / 0xFFFFFFFF
        folds.append("val" if value < 0.5 else "test")
    return np.array(folds, dtype=object)


def fit_linear(gt: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    if len(gt) < 2 or float(np.nanstd(pred)) == 0.0:
        return 1.0, 0.0
    slope, intercept = np.polyfit(pred, gt, deg=1)
    return float(slope), float(intercept)


def srcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return math.nan
    a = pd.Series(y_true).rank(method="average").to_numpy()
    b = pd.Series(y_pred).rank(method="average").to_numpy()
    return plcc(a, b)


def plcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or float(np.nanstd(y_true)) == 0.0 or float(np.nanstd(y_pred)) == 0.0:
        return math.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else math.nan
    mse = float(np.mean((y_true - y_pred) ** 2)) if len(y_true) else math.nan
    acc = math.nan
    if len(y_true):
        acc = float(np.mean((y_true >= 5.0) == (y_pred >= 5.0)))
    return {
        "srcc": srcc(y_true, y_pred),
        "plcc": plcc(y_true, y_pred),
        "mae": mae,
        "mse": mse,
        "binary_accuracy": acc,
    }


def to_scale_1_10(dataset: str, values: np.ndarray) -> np.ndarray:
    if dataset == "AADB" and np.nanmax(values) <= 1.5:
        return values * 9.0 + 1.0
    if np.nanmax(values) <= 1.5:
        return values * 10.0
    return values


def build_aligned_tables(specs: Sequence[PredictionSpec]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, dict]]]:
    by_dataset: Dict[str, List[PredictionSpec]] = {}
    for spec in specs:
        if spec.path.exists():
            by_dataset.setdefault(spec.dataset, []).append(spec)

    tables: Dict[str, pd.DataFrame] = {}
    params: Dict[str, Dict[str, dict]] = {}
    for dataset, ds_specs in by_dataset.items():
        frames = []
        for spec in ds_specs:
            frame = read_prediction(spec)
            frames.append(frame[["join_key", "source_id", "gt_score", f"{spec.model}_raw"]])
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on="join_key", how="inner", suffixes=("", "_other"))
            if "gt_score_other" in merged.columns:
                gt_delta = (merged["gt_score"] - merged["gt_score_other"]).abs().max()
                if pd.notna(gt_delta) and gt_delta > 1e-5:
                    merged["gt_score"] = merged[["gt_score", "gt_score_other"]].mean(axis=1)
                merged = merged.drop(columns=["gt_score_other"])
            if "source_id_other" in merged.columns:
                merged = merged.drop(columns=["source_id_other"])
        merged["dataset"] = dataset
        merged["fold"] = deterministic_fold(merged["join_key"].astype(str).tolist())
        gt_1_10 = to_scale_1_10(dataset, merged["gt_score"].to_numpy(dtype=float))
        merged["gt_score_1_10"] = np.clip(gt_1_10, 1.0, 10.0)
        params[dataset] = {}
        val_mask = merged["fold"].to_numpy() == "val"
        for spec in ds_specs:
            raw_col = f"{spec.model}_raw"
            scaled = to_scale_1_10(dataset, merged[raw_col].to_numpy(dtype=float))
            slope, intercept = fit_linear(merged.loc[val_mask, "gt_score_1_10"].to_numpy(), scaled[val_mask])
            calibrated = np.clip(slope * scaled + intercept, 1.0, 10.0)
            merged[f"{spec.model}_scaled_1_10"] = np.clip(scaled, 1.0, 10.0)
            merged[f"{spec.model}_calibrated_1_10"] = calibrated
            params[dataset][spec.model] = {
                "path": str(spec.path.relative_to(ROOT)),
                "raw_min": float(np.nanmin(merged[raw_col])),
                "raw_max": float(np.nanmax(merged[raw_col])),
                "scaled_min": float(np.nanmin(scaled)),
                "scaled_max": float(np.nanmax(scaled)),
                "linear_calibration": {"slope": slope, "intercept": intercept},
                "fit_fold": "val",
                "primary": spec.primary,
                "note": spec.note,
            }
        tables[dataset] = merged
    return tables, params


def grid_weights(n: int, step: float, cap_last: Optional[float] = None) -> Iterable[Tuple[float, ...]]:
    units = int(round(1.0 / step))
    current: List[int] = []

    def rec(left: int, depth: int) -> Iterable[Tuple[float, ...]]:
        if depth == n - 1:
            weights = current + [left]
            values = tuple(round(w * step, 10) for w in weights)
            if cap_last is None or values[-1] <= cap_last + 1e-12:
                yield values
            return
        for value in range(left + 1):
            current.append(value)
            yield from rec(left - value, depth + 1)
            current.pop()

    yield from rec(units, 0)


def fine_weights_around(base: Tuple[float, ...], cap_last: Optional[float]) -> Iterable[Tuple[float, ...]]:
    n = len(base)
    seen = set()
    for weights in grid_weights(n, 0.01, cap_last=cap_last):
        if max(abs(a - b) for a, b in zip(weights, base)) <= 0.05 + 1e-12:
            if weights not in seen:
                seen.add(weights)
                yield weights


def evaluate_weights(
    tables: Dict[str, pd.DataFrame],
    models: Sequence[str],
    weights: Sequence[float],
    fold: str,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], List[str]]:
    per_dataset = {}
    missing = []
    objective_parts = []
    if models == GROUP_A:
        objective_weights = {"AVA": 0.6, "AADB": 0.4}
    else:
        objective_weights = {"AVA": 0.5, "AADB": 0.4, "ICAA17K": 0.1}
    available_objective_total = 0.0
    for dataset, obj_weight in objective_weights.items():
        table = tables.get(dataset)
        cols = [f"{m}_calibrated_1_10" for m in models]
        if table is None or any(c not in table.columns for c in cols):
            missing.append(dataset)
            continue
        subset = table[table["fold"] == fold]
        if len(subset) < 2:
            missing.append(dataset)
            continue
        pred = np.zeros(len(subset), dtype=float)
        for model, weight in zip(models, weights):
            pred += weight * subset[f"{model}_calibrated_1_10"].to_numpy(dtype=float)
        ds_metrics = metrics(subset["gt_score_1_10"].to_numpy(dtype=float), pred)
        per_dataset[dataset] = ds_metrics
        if not math.isnan(ds_metrics["srcc"]):
            objective_parts.append(obj_weight * ds_metrics["srcc"])
            available_objective_total += obj_weight
    objective = math.nan
    if available_objective_total > 0:
        objective = float(sum(objective_parts) / available_objective_total)
    aggregate = {"objective_srcc": objective, "available_objective_weight": available_objective_total}
    return per_dataset, aggregate, missing


def run_search(tables: Dict[str, pd.DataFrame], group_name: str, models: Sequence[str], cap_icaa: Optional[float]) -> pd.DataFrame:
    n = len(models)
    coarse_rows = []
    for weights in grid_weights(n, 0.05, cap_last=cap_icaa if models[-1] == "icaa" else None):
        per_dataset, aggregate, missing = evaluate_weights(tables, models, weights, "val")
        if math.isnan(aggregate["objective_srcc"]):
            continue
        row = {
            "group": group_name,
            "search_stage": "coarse",
            "icaa_cap": cap_icaa if cap_icaa is not None else "",
            "fold": "val",
            "objective_srcc": aggregate["objective_srcc"],
            "available_objective_weight": aggregate["available_objective_weight"],
            "missing_objective_datasets": ",".join(missing),
        }
        for model, weight in zip(models, weights):
            row[f"w_{model}"] = weight
        for dataset, ds_metrics in per_dataset.items():
            for name, value in ds_metrics.items():
                row[f"{dataset.lower()}_{name}"] = value
        coarse_rows.append(row)
    if not coarse_rows:
        return pd.DataFrame()
    coarse = pd.DataFrame(coarse_rows).sort_values(
        ["objective_srcc", "ava_srcc", "aadb_srcc", "icaa17k_srcc"], ascending=False, na_position="last"
    )
    top_bases = [
        tuple(float(row[f"w_{m}"]) for m in models)
        for _, row in coarse.head(20).iterrows()
    ]
    fine_rows = []
    seen = set()
    for base in top_bases:
        for weights in fine_weights_around(base, cap_icaa if models[-1] == "icaa" else None):
            if weights in seen:
                continue
            seen.add(weights)
            per_dataset, aggregate, missing = evaluate_weights(tables, models, weights, "val")
            if math.isnan(aggregate["objective_srcc"]):
                continue
            row = {
                "group": group_name,
                "search_stage": "fine",
                "icaa_cap": cap_icaa if cap_icaa is not None else "",
                "fold": "val",
                "objective_srcc": aggregate["objective_srcc"],
                "available_objective_weight": aggregate["available_objective_weight"],
                "missing_objective_datasets": ",".join(missing),
            }
            for model, weight in zip(models, weights):
                row[f"w_{model}"] = weight
            for dataset, ds_metrics in per_dataset.items():
                for name, value in ds_metrics.items():
                    row[f"{dataset.lower()}_{name}"] = value
            fine_rows.append(row)
    fine = pd.DataFrame(fine_rows)
    return pd.concat([coarse, fine], ignore_index=True).sort_values(
        ["objective_srcc", "search_stage"], ascending=[False, True], na_position="last"
    )


def bootstrap_top(
    tables: Dict[str, pd.DataFrame],
    rows: pd.DataFrame,
    group_models: Dict[str, Sequence[str]],
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(SEED)
    out_rows = []
    for _, row in rows.head(10).iterrows():
        models = group_models[row["group"]]
        weights = [float(row.get(f"w_{m}", 0.0)) for m in models]
        for dataset, table in tables.items():
            cols = [f"{m}_calibrated_1_10" for m in models]
            if any(c not in table.columns for c in cols):
                continue
            subset = table[table["fold"] == "test"]
            if len(subset) < 2:
                continue
            pred = np.zeros(len(subset), dtype=float)
            for model, weight in zip(models, weights):
                pred += weight * subset[f"{model}_calibrated_1_10"].to_numpy(dtype=float)
            gt = subset["gt_score_1_10"].to_numpy(dtype=float)
            values = []
            for _ in range(n_bootstrap):
                idx = rng.integers(0, len(gt), size=len(gt))
                values.append(srcc(gt[idx], pred[idx]))
            values = np.array(values, dtype=float)
            out_rows.append(
                {
                    "group": row["group"],
                    "icaa_cap": row["icaa_cap"],
                    "dataset": dataset,
                    "test_srcc": srcc(gt, pred),
                    "bootstrap_mean_srcc": float(np.nanmean(values)),
                    "bootstrap_ci95_low": float(np.nanpercentile(values, 2.5)),
                    "bootstrap_ci95_high": float(np.nanpercentile(values, 97.5)),
                    **{f"w_{m}": w for m, w in zip(models, weights)},
                }
            )
    return pd.DataFrame(out_rows)


def availability_rows(specs: Sequence[PredictionSpec], tables: Dict[str, pd.DataFrame]) -> List[dict]:
    rows = []
    indexed = {(s.dataset, s.model): s for s in specs}
    for dataset, models in REQUESTED.items():
        for model in models:
            spec = indexed.get((dataset, model))
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "available_primary": bool(spec and spec.path.exists()),
                    "rows_in_aligned_available_table": len(tables[dataset]) if dataset in tables else 0,
                    "path": str(spec.path.relative_to(ROOT)) if spec else "",
                    "note": spec.note if spec else "missing primary prediction CSV",
                }
            )
    for spec in NON_PRIMARY_CANDIDATES:
        rows.append(
            {
                "dataset": spec.dataset,
                "model": spec.model,
                "available_primary": False,
                "rows_in_aligned_available_table": "",
                "path": str(spec.path.relative_to(ROOT)),
                "note": f"non-primary candidate exists: {spec.note}" if spec.path.exists() else "non-primary candidate missing",
            }
        )
    return rows


def md_table(rows: List[dict], columns: Sequence[str]) -> str:
    if not rows:
        return "_None._"
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        vals = [str(row.get(c, "")).replace("\n", " ") for c in columns]
        lines.append("|" + "|".join(vals) + "|")
    return "\n".join(lines)


def best_rows_for_report(results: pd.DataFrame, group: str) -> List[dict]:
    if results.empty:
        return []
    subset = results[(results["group"] == group) & (results["search_stage"] == "fine")]
    if subset.empty:
        subset = results[results["group"] == group]
    return subset.head(5).fillna("").to_dict("records")


def write_report(
    tables: Dict[str, pd.DataFrame],
    params: Dict[str, Dict[str, dict]],
    availability: List[dict],
    results: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    scale_rows = []
    for dataset, model_params in params.items():
        for model, p in model_params.items():
            scale_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "raw_range": f"{p['raw_min']:.6g}..{p['raw_max']:.6g}",
                    "scaled_range": f"{p['scaled_min']:.6g}..{p['scaled_max']:.6g}",
                    "calibration": f"y={p['linear_calibration']['slope']:.6g}x+{p['linear_calibration']['intercept']:.6g}",
                    "fit_fold": p["fit_fold"],
                }
            )

    selected_rows = []
    for group in ["without_icaa", "with_icaa_unconstrained", "with_icaa_cap_020", "with_icaa_cap_025"]:
        selected_rows.extend(best_rows_for_report(results, group)[:1])

    metric_rows = []
    for row in selected_rows:
        models = GROUP_A if row["group"] == "without_icaa" else GROUP_B
        weights = [float(row.get(f"w_{m}", 0.0) or 0.0) for m in models]
        per_dataset, _, _ = evaluate_weights(tables, models, weights, "test")
        for dataset, ds_metrics in per_dataset.items():
            metric_rows.append(
                {
                    "group": row["group"],
                    "dataset": dataset,
                    "srcc": f"{ds_metrics['srcc']:.6f}",
                    "plcc": f"{ds_metrics['plcc']:.6f}",
                    "mae": f"{ds_metrics['mae']:.6f}",
                    "binary_accuracy": f"{ds_metrics['binary_accuracy']:.6f}",
                }
            )

    missing_group_a = all(
        any(f"{model}_calibrated_1_10" not in table.columns for model in GROUP_A)
        for table in tables.values()
    )
    missing_group_b = all(
        any(f"{model}_calibrated_1_10" not in table.columns for model in GROUP_B)
        for table in tables.values()
    )

    recommendation = (
        "No production ratio should be changed from this run because no complete primary same-dataset "
        "alignment exists for the requested holistic ensemble. Keep ICAA separate as a color subscore."
    )
    if not results.empty:
        recommendation = (
            "Use the listed validation-selected weights only as benchmark candidates. Because the available "
            "primary tables are incomplete across datasets, keep ICAA separate unless a future same-row "
            "holistic validation shows stable SRCC gains."
        )

    report = [
        "# Aesthetic Ensemble Weight Search 20260603",
        "",
        "## Data Availability",
        md_table(availability, ["dataset", "model", "available_primary", "rows_in_aligned_available_table", "path", "note"]),
        "",
        "## Model Score Scales",
        md_table(scale_rows, ["dataset", "model", "raw_range", "scaled_range", "calibration", "fit_fold"]),
        "",
        "## Best Weights Without ICAA",
        md_table(best_rows_for_report(results, "without_icaa"), list(results.columns) if not results.empty else ["status"]),
        "",
        "## Best Weights With ICAA Unconstrained",
        md_table(best_rows_for_report(results, "with_icaa_unconstrained"), list(results.columns) if not results.empty else ["status"]),
        "",
        "## Best Weights With ICAA Capped At 0.20",
        md_table(best_rows_for_report(results, "with_icaa_cap_020"), list(results.columns) if not results.empty else ["status"]),
        "",
        "## Best Weights With ICAA Capped At 0.25",
        md_table(best_rows_for_report(results, "with_icaa_cap_025"), list(results.columns) if not results.empty else ["status"]),
        "",
        "## Per-Dataset Test Metrics For Selected Weight Sets",
        md_table(metric_rows, ["group", "dataset", "srcc", "plcc", "mae", "binary_accuracy"]),
        "",
        "## Bootstrap Robustness",
        md_table(bootstrap.fillna("").to_dict("records"), list(bootstrap.columns) if not bootstrap.empty else ["status"]),
        "",
        "## ICAA Holistic-Aesthetic Decision",
        (
            "ICAA should remain a separate color subscore in this run. The primary files do not provide "
            "same-row NIMA, RGNet, adaptive A-LAMP, and ICAA predictions on AVA or AADB, so an observed "
            "holistic improvement cannot be established without substituting protocols."
        ),
        "",
        "## Final Recommendation For Production",
        recommendation,
        "",
        "Safest holistic weight ratio: unavailable from primary 20260603 same-row evidence.",
        "",
        "Optional color-aware ratio: unavailable from primary 20260603 same-row evidence.",
        "",
        "Keep ICAA separate: yes.",
        "",
        "## Caveats",
        "- Do not tune on official test if no validation split exists.",
        "- NIMA AVA split is local full, not official paper split.",
        "- A-LAMP is paper-inspired adaptive, not full paper protocol.",
        "- This run did not rerun inference and did not use non-primary protocol-mixed CSVs for optimization.",
        "- Group A complete primary alignment missing: " + str(missing_group_a),
        "- Group B complete primary alignment missing: " + str(missing_group_b),
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables, params = build_aligned_tables(PRIMARY_SPECS)
    output_names = {"AVA": "aligned_ava_predictions.csv", "AADB": "aligned_aadb_predictions.csv", "ICAA17K": "aligned_icaa_predictions.csv"}
    for dataset, table in tables.items():
        table.to_csv(OUT_DIR / output_names[dataset], index=False)
    (OUT_DIR / "calibration_params.json").write_text(json.dumps(params, indent=2, sort_keys=True), encoding="utf-8")

    all_results = []
    group_models = {
        "without_icaa": GROUP_A,
        "with_icaa_unconstrained": GROUP_B,
        "with_icaa_cap_020": GROUP_B,
        "with_icaa_cap_025": GROUP_B,
    }
    searches = [
        ("without_icaa", GROUP_A, None),
        ("with_icaa_unconstrained", GROUP_B, None),
        ("with_icaa_cap_020", GROUP_B, 0.20),
        ("with_icaa_cap_025", GROUP_B, 0.25),
    ]
    for name, models, cap in searches:
        result = run_search(tables, name, models, cap)
        if not result.empty:
            all_results.append(result)
    results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame(
        columns=[
            "group",
            "search_stage",
            "icaa_cap",
            "fold",
            "objective_srcc",
            "available_objective_weight",
            "missing_objective_datasets",
            "w_nima",
            "w_rgnet",
            "w_alamp_adaptive",
            "w_icaa",
        ]
    )
    results.to_csv(OUT_DIR / "weight_search_results_all.csv", index=False)

    without = best_rows_for_report(results, "without_icaa")
    pd.DataFrame(without).to_csv(OUT_DIR / "top_weights_without_icaa.csv", index=False)
    with_rows = []
    for group in ["with_icaa_unconstrained", "with_icaa_cap_020", "with_icaa_cap_025"]:
        with_rows.extend(best_rows_for_report(results, group))
    pd.DataFrame(with_rows).to_csv(OUT_DIR / "top_weights_with_icaa.csv", index=False)

    bootstrap = bootstrap_top(tables, results[results["search_stage"] == "fine"], group_models)
    bootstrap.to_csv(OUT_DIR / "bootstrap_top10.csv", index=False)

    availability = availability_rows(PRIMARY_SPECS, tables)
    with (OUT_DIR / "data_availability.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(availability[0].keys()))
        writer.writeheader()
        writer.writerows(availability)
    write_report(tables, params, availability, results, bootstrap)

    print("best weights without ICAA:", "unavailable" if not without else without[0])
    print("best weights with ICAA:", "unavailable" if not with_rows else with_rows[0])
    print("include ICAA in final aesthetic score:", "no; keep separate color subscore")
    print("report:", OUT_DIR / "report.md")
    print("results:", OUT_DIR / "weight_search_results_all.csv")
    print("aligned AVA:", OUT_DIR / "aligned_ava_predictions.csv")
    print("aligned AADB:", OUT_DIR / "aligned_aadb_predictions.csv")
    print("aligned ICAA:", OUT_DIR / "aligned_icaa_predictions.csv")


if __name__ == "__main__":
    main()
