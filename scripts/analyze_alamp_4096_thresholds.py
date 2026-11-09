# A-LAMP 4096 비교 예측 CSV의 임계값별 균형 지표를 계산한다.
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


THRESHOLDS = [round(value / 100.0, 2) for value in range(5, 96)]


def _read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    labels: list[float] = []
    scores: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels.append(float(row["y_true"]))
            if "y_score" in row and row["y_score"] != "":
                scores.append(float(row["y_score"]))
            else:
                scores.append(float(row["y_pred"]))
    if not labels:
        raise ValueError(f"No predictions found in {path}")
    return np.asarray(labels, dtype=np.float32), np.asarray(scores, dtype=np.float32)


def _binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    labels = y_true.astype(np.int32)
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=np.float64)
    sorted_scores = y_score[order]
    start = 0
    while start < len(sorted_scores):
        stop = start + 1
        while stop < len(sorted_scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    auc = (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return float(auc)


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    labels = y_true.astype(np.int32)
    positive_count = int(np.sum(labels == 1))
    if positive_count == 0:
        return None
    order = np.argsort(-y_score)
    sorted_labels = labels[order]
    cumulative_positive = np.cumsum(sorted_labels)
    ranks = np.arange(1, len(sorted_labels) + 1)
    precision_at_k = cumulative_positive / ranks
    return float(np.sum(precision_at_k * sorted_labels) / positive_count)


def _metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    truth = y_true >= 0.5
    pred = y_score >= threshold
    tp = int(np.sum(pred & truth))
    tn = int(np.sum(~pred & ~truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / max(len(y_true), 1)
    fpr = fp / max(fp + tn, 1)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "false_positive_rate": float(fpr),
        "predicted_positive_ratio": float(np.mean(pred)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _sweep(y_true: np.ndarray, y_score: np.ndarray) -> list[dict[str, Any]]:
    return [_metrics_at_threshold(y_true, y_score, threshold) for threshold in THRESHOLDS]


def _write_sweep(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _best(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    return dict(max(rows, key=lambda row: (row[metric], row["threshold"])))


def _threshold_independent(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    positive_count = int(np.sum(y_true >= 0.5))
    return {
        "sample_count": int(len(y_true)),
        "positive_count": positive_count,
        "negative_count": int(len(y_true) - positive_count),
        "actual_positive_ratio": float(positive_count / max(len(y_true), 1)),
        "roc_auc": _binary_roc_auc(y_true, y_score),
        "average_precision": _average_precision(y_true, y_score),
        "prediction_min": float(np.min(y_score)),
        "prediction_max": float(np.max(y_score)),
        "prediction_mean": float(np.mean(y_score)),
        "prediction_std": float(np.std(y_score)),
    }


def _model_summary(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    sweep_rows = _sweep(y_true, y_score)
    return {
        "threshold_independent": _threshold_independent(y_true, y_score),
        "at_0_5": _metrics_at_threshold(y_true, y_score, 0.5),
        "best_balanced_accuracy": _best(sweep_rows, "balanced_accuracy"),
        "best_f1": _best(sweep_rows, "f1"),
        "best_accuracy": _best(sweep_rows, "accuracy"),
        "sweep_rows": sweep_rows,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    mp = summary["models"]["multipatch_only"]
    db = summary["models"]["dual_branch_gcn"]
    subset = summary["matched_subset"]
    judgment = summary["final_judgment"]
    lines = [
        "# A-LAMP 4096 Fair Comparison: Multi-Patch-only vs Dual-Branch GCN",
        "",
        "## 1. Summary",
        "",
        f"- Multi-Patch-only ROC-AUC: `{_fmt(mp['threshold_independent']['roc_auc'])}`.",
        f"- Dual-Branch GCN ROC-AUC: `{_fmt(db['threshold_independent']['roc_auc'])}`.",
        f"- Multi-Patch-only best balanced accuracy: `{_fmt(mp['best_balanced_accuracy']['balanced_accuracy'])}` at threshold `{_fmt(mp['best_balanced_accuracy']['threshold'])}`.",
        f"- Dual-Branch GCN best balanced accuracy: `{_fmt(db['best_balanced_accuracy']['balanced_accuracy'])}` at threshold `{_fmt(db['best_balanced_accuracy']['threshold'])}`.",
        f"- Judgment: {judgment['recommendation']}",
        "",
        "## 2. Matched 4096 Test Subset",
        "",
        f"- Graph JSONL: `{subset['graph_jsonl']}`.",
        f"- Graph records: `{subset['graph_records']}`.",
        f"- Matched test image IDs: `{subset['matched_image_ids']}`.",
        f"- ID basis: {subset['id_basis']}.",
        f"- Filtered JSONL: `{subset['multipatch_subset_jsonl']}`.",
        "",
        "## 3. Actual Positive Ratio",
        "",
        f"- Positive labels: `{mp['threshold_independent']['positive_count']}`.",
        f"- Negative labels: `{mp['threshold_independent']['negative_count']}`.",
        f"- Actual positive ratio: `{_fmt(mp['threshold_independent']['actual_positive_ratio'])}`.",
        "",
        "## 4. Multi-Patch-only Results at Threshold 0.5",
        "",
        _format_threshold_block(mp["at_0_5"]),
        "",
        "## 5. Dual-Branch GCN Results at Threshold 0.5",
        "",
        _format_threshold_block(db["at_0_5"]),
        "",
        "## 6. Balanced Metrics",
        "",
        "| Model | ROC-AUC | AP | Balanced Acc @0.5 | Specificity @0.5 | FPR @0.5 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Multi-Patch-only | {_fmt(mp['threshold_independent']['roc_auc'])} | {_fmt(mp['threshold_independent']['average_precision'])} | {_fmt(mp['at_0_5']['balanced_accuracy'])} | {_fmt(mp['at_0_5']['specificity'])} | {_fmt(mp['at_0_5']['false_positive_rate'])} |",
        f"| Dual-Branch GCN | {_fmt(db['threshold_independent']['roc_auc'])} | {_fmt(db['threshold_independent']['average_precision'])} | {_fmt(db['at_0_5']['balanced_accuracy'])} | {_fmt(db['at_0_5']['specificity'])} | {_fmt(db['at_0_5']['false_positive_rate'])} |",
        "",
        "## 7. Threshold Sweep Results",
        "",
        f"- Multi-Patch sweep CSV: `{summary['artifacts']['threshold_sweep_multipatch_csv']}`.",
        f"- Dual-Branch sweep CSV: `{summary['artifacts']['threshold_sweep_dual_branch_csv']}`.",
        "",
        "## 8. Best Threshold Comparison",
        "",
        "| Model | Best balanced acc | Threshold | Best F1 | F1 threshold | Best accuracy | Accuracy threshold |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Multi-Patch-only | {_fmt(mp['best_balanced_accuracy']['balanced_accuracy'])} | {_fmt(mp['best_balanced_accuracy']['threshold'])} | {_fmt(mp['best_f1']['f1'])} | {_fmt(mp['best_f1']['threshold'])} | {_fmt(mp['best_accuracy']['accuracy'])} | {_fmt(mp['best_accuracy']['threshold'])} |",
        f"| Dual-Branch GCN | {_fmt(db['best_balanced_accuracy']['balanced_accuracy'])} | {_fmt(db['best_balanced_accuracy']['threshold'])} | {_fmt(db['best_f1']['f1'])} | {_fmt(db['best_f1']['threshold'])} | {_fmt(db['best_accuracy']['accuracy'])} | {_fmt(db['best_accuracy']['threshold'])} |",
        "",
        "## 9. Positive Bias Analysis",
        "",
        f"- Multi-Patch predicted positive ratio at 0.5: `{_fmt(mp['at_0_5']['predicted_positive_ratio'])}`.",
        f"- Dual-Branch predicted positive ratio at 0.5: `{_fmt(db['at_0_5']['predicted_positive_ratio'])}`.",
        f"- Actual positive ratio: `{_fmt(mp['threshold_independent']['actual_positive_ratio'])}`.",
        f"- Dual-Branch specificity at 0.5 is `{_fmt(db['at_0_5']['specificity'])}`, so its F1 is inflated by predicting nearly everything positive.",
        "",
        "## 10. Final Judgment",
        "",
        f"- {judgment['details']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_threshold_block(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"- Accuracy: `{_fmt(row['accuracy'])}`.",
            f"- Balanced accuracy: `{_fmt(row['balanced_accuracy'])}`.",
            f"- F1: `{_fmt(row['f1'])}`.",
            f"- Precision: `{_fmt(row['precision'])}`.",
            f"- Recall: `{_fmt(row['recall'])}`.",
            f"- Specificity: `{_fmt(row['specificity'])}`.",
            f"- FPR: `{_fmt(row['false_positive_rate'])}`.",
            f"- Predicted positive ratio: `{_fmt(row['predicted_positive_ratio'])}`.",
            f"- Confusion matrix: `tn={row['tn']}`, `fp={row['fp']}`, `fn={row['fn']}`, `tp={row['tp']}`.",
        ]
    )


def _same_labels(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.all(a == b))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multipatch_predictions", required=True)
    parser.add_argument("--dual_branch_predictions", required=True)
    parser.add_argument("--matched_subset_summary", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mp_labels, mp_scores = _read_predictions(Path(args.multipatch_predictions))
    db_labels, db_scores = _read_predictions(Path(args.dual_branch_predictions))
    if not _same_labels(mp_labels, db_labels):
        raise ValueError("Prediction files do not have the same labels/order.")

    mp_summary = _model_summary(mp_labels, mp_scores)
    db_summary = _model_summary(db_labels, db_scores)
    mp_sweep = mp_summary.pop("sweep_rows")
    db_sweep = db_summary.pop("sweep_rows")

    mp_sweep_csv = out_dir / "threshold_sweep_multipatch.csv"
    db_sweep_csv = out_dir / "threshold_sweep_dual_branch.csv"
    _write_sweep(mp_sweep_csv, mp_sweep)
    _write_sweep(db_sweep_csv, db_sweep)

    matched_subset = json.loads(Path(args.matched_subset_summary).read_text(encoding="utf-8"))
    matched_subset_for_report = {
        "graph_jsonl": matched_subset["graph_jsonl"],
        "graph_records": matched_subset["graph_records"],
        "matched_image_ids": matched_subset["matched_patch_records"],
        "id_basis": matched_subset["id_basis"],
        "multipatch_subset_jsonl": matched_subset["subset_jsonl"],
        "unmatched_graph_ids": matched_subset["unmatched_graph_ids"],
    }

    mp_ba = mp_summary["best_balanced_accuracy"]["balanced_accuracy"]
    db_ba = db_summary["best_balanced_accuracy"]["balanced_accuracy"]
    mp_auc = mp_summary["threshold_independent"]["roc_auc"]
    db_auc = db_summary["threshold_independent"]["roc_auc"]
    if db_auc is not None and mp_auc is not None and db_auc > mp_auc and db_ba > mp_ba:
        recommendation = "Dual-Branch GCN is promising on this subset."
        details = (
            "Dual-Branch GCN beats Multi-Patch-only on both ROC-AUC and best balanced accuracy, "
            "so follow-up graph work may be justified."
        )
    else:
        recommendation = "Stop this GCN branch for now; improve the Multi-Patch teacher instead."
        details = (
            "Dual-Branch GCN does not beat Multi-Patch-only on ROC-AUC or best balanced accuracy. "
            "Its threshold-0.5 F1 is driven by very high recall and severe positive bias, not better ranking."
        )

    summary = {
        "status": "completed",
        "matched_subset": matched_subset_for_report,
        "models": {
            "multipatch_only": mp_summary,
            "dual_branch_gcn": db_summary,
        },
        "artifacts": {
            "threshold_sweep_multipatch_csv": str(mp_sweep_csv),
            "threshold_sweep_dual_branch_csv": str(db_sweep_csv),
            "summary_json": str(out_dir / "summary.json"),
            "report_md": str(out_dir / "report.md"),
        },
        "final_judgment": {
            "recommendation": recommendation,
            "details": details,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(out_dir / "report.md", summary)
    print(json.dumps(summary["final_judgment"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
