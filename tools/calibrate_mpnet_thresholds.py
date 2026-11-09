# MPNet 예측 CSV의 이진 분류 임계값을 보정하는 도구
"""Calibrate MPNet binary thresholds from validation and test prediction CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate MPNet thresholds from prediction CSVs.")
    parser.add_argument("--val_predictions", required=True, help="Validation predictions CSV.")
    parser.add_argument("--test_predictions", required=True, help="Test predictions CSV.")
    parser.add_argument("--output_dir", required=True, help="Directory for calibration outputs.")
    return parser.parse_args()


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "true" not in df.columns or "pred" not in df.columns:
        raise ValueError(f"{path} must contain 'true' and 'pred' columns.")
    labels = df["true"].astype(int).to_numpy()
    probabilities = df["pred"].astype(float).to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{path} contains non-binary labels.")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError(f"{path} contains probabilities outside [0, 1].")
    return labels, probabilities


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    recall = recall_score(labels, predictions, zero_division=0)
    precision = precision_score(labels, predictions, zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "youden_j": float(recall + specificity - 1.0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "positive_predictions": int(tp + fp),
        "negative_predictions": int(tn + fn),
    }


def threshold_sweep(
    val_labels: np.ndarray,
    val_probabilities: np.ndarray,
    test_labels: np.ndarray,
    test_probabilities: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for threshold_index in range(1, 100):
        threshold = threshold_index / 100.0
        row: dict[str, Any] = {"threshold": round(threshold, 2)}
        val_metrics = calculate_metrics(val_labels, val_probabilities, threshold)
        test_metrics = calculate_metrics(test_labels, test_probabilities, threshold)
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        row.update({f"test_{key}": value for key, value in test_metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def row_at_threshold(sweep_df: pd.DataFrame, threshold: float) -> dict[str, Any]:
    matches = sweep_df[np.isclose(sweep_df["threshold"], threshold)]
    if matches.empty:
        raise ValueError(f"No sweep row found for threshold {threshold:.2f}")
    return matches.iloc[0].to_dict()


def best_validation_row(sweep_df: pd.DataFrame, metric: str) -> dict[str, Any]:
    sort_columns = [f"val_{metric}", "val_balanced_accuracy", "val_f1", "val_specificity", "threshold"]
    ascending = [False, False, False, False, False]
    return sweep_df.sort_values(sort_columns, ascending=ascending).iloc[0].to_dict()


def lowest_validation_specificity_row(sweep_df: pd.DataFrame, minimum_specificity: float) -> dict[str, Any] | None:
    matches = sweep_df[sweep_df["val_specificity"] >= minimum_specificity]
    if matches.empty:
        return None
    return matches.sort_values("threshold", ascending=True).iloc[0].to_dict()


def compact_selection(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    selection: dict[str, Any] = {"threshold": float(row["threshold"])}
    for split in ("val", "test"):
        selection[split] = {
            "accuracy": float(row[f"{split}_accuracy"]),
            "balanced_accuracy": float(row[f"{split}_balanced_accuracy"]),
            "precision": float(row[f"{split}_precision"]),
            "recall": float(row[f"{split}_recall"]),
            "specificity": float(row[f"{split}_specificity"]),
            "f1": float(row[f"{split}_f1"]),
            "youden_j": float(row[f"{split}_youden_j"]),
            "confusion_matrix": {
                "tn": int(row[f"{split}_tn"]),
                "fp": int(row[f"{split}_fp"]),
                "fn": int(row[f"{split}_fn"]),
                "tp": int(row[f"{split}_tp"]),
            },
        }
    return selection


def build_summary(
    sweep_df: pd.DataFrame,
    val_predictions: Path,
    test_predictions: Path,
    val_count: int,
    test_count: int,
) -> dict[str, Any]:
    selections = {
        "threshold_0_50_baseline": compact_selection(row_at_threshold(sweep_df, 0.50)),
        "best_validation_f1": compact_selection(best_validation_row(sweep_df, "f1")),
        "best_validation_balanced_accuracy": compact_selection(best_validation_row(sweep_df, "balanced_accuracy")),
        "best_validation_youden_j": compact_selection(best_validation_row(sweep_df, "youden_j")),
        "lowest_validation_specificity_ge_0_30": compact_selection(
            lowest_validation_specificity_row(sweep_df, 0.30)
        ),
        "lowest_validation_specificity_ge_0_50": compact_selection(
            lowest_validation_specificity_row(sweep_df, 0.50)
        ),
        "lowest_validation_specificity_ge_0_70": compact_selection(
            lowest_validation_specificity_row(sweep_df, 0.70)
        ),
    }
    return {
        "inputs": {
            "val_predictions": str(val_predictions),
            "test_predictions": str(test_predictions),
            "val_rows": val_count,
            "test_rows": test_count,
        },
        "threshold_range": "0.01..0.99",
        "threshold_step": 0.01,
        "selection_split": "validation",
        "selections": selections,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# MPNet Threshold Calibration",
        "",
        "## Inputs",
        f"- Validation predictions: `{summary['inputs']['val_predictions']}`",
        f"- Test predictions: `{summary['inputs']['test_predictions']}`",
        f"- Rows: validation {summary['inputs']['val_rows']}, test {summary['inputs']['test_rows']}",
        "",
        "## Selected Thresholds",
        "| Selection | Threshold | Val Balanced Acc | Val Specificity | Val Recall | Val F1 | Test Balanced Acc | Test Specificity | Test Recall | Test F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, selection in summary["selections"].items():
        if selection is None:
            lines.append(f"| {name} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        val = selection["val"]
        test = selection["test"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{selection['threshold']:.2f}",
                    f"{val['balanced_accuracy']:.6f}",
                    f"{val['specificity']:.6f}",
                    f"{val['recall']:.6f}",
                    f"{val['f1']:.6f}",
                    f"{test['balanced_accuracy']:.6f}",
                    f"{test['specificity']:.6f}",
                    f"{test['recall']:.6f}",
                    f"{test['f1']:.6f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    val_predictions = Path(args.val_predictions)
    test_predictions = Path(args.test_predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_labels, val_probabilities = load_predictions(val_predictions)
    test_labels, test_probabilities = load_predictions(test_predictions)
    sweep_df = threshold_sweep(val_labels, val_probabilities, test_labels, test_probabilities)
    sweep_df.to_csv(output_dir / "threshold_sweep.csv", index=False)

    summary = build_summary(
        sweep_df=sweep_df,
        val_predictions=val_predictions,
        test_predictions=test_predictions,
        val_count=len(val_labels),
        test_count=len(test_labels),
    )
    (output_dir / "threshold_calibration_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir / "threshold_calibration_report.md", summary)
    print(f"Wrote threshold calibration outputs to {output_dir}")


if __name__ == "__main__":
    main()

