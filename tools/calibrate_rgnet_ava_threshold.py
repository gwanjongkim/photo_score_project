# AVA 예측 CSV로 RGNet 분류 임계값을 보정하는 도구
"""Calibrate RGNet AVA binary thresholds from saved prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


LABEL_CANDIDATES = (
    "label",
    "target",
    "y_true",
    "true_label",
    "ground_truth",
    "gt_label",
    "class_label",
)
PROB_CANDIDATES = (
    "prediction_prob",
    "probability",
    "positive_probability",
    "positive_prob",
    "y_prob",
    "pred_prob",
    "prob",
    "prediction",
    "pred",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate RGNet AVA classification thresholds from saved CSV predictions."
    )
    parser.add_argument(
        "--val_csv",
        default="outputs/rgnet_paper_ava_classification_20260510/eval/lse_r4/val_predictions.csv",
        help="Validation prediction CSV.",
    )
    parser.add_argument(
        "--test_csv",
        default="outputs/rgnet_paper_ava_classification_20260510/eval/lse_r4/test_predictions.csv",
        help="Test prediction CSV.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/rgnet_paper_ava_classification_20260510/threshold_calibration",
        help="Directory for calibration outputs.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")
        return list(reader.fieldnames), list(reader)


def lower_map(headers: list[str]) -> dict[str, str]:
    return {header.strip().lower(): header for header in headers}


def is_binary_column(rows: list[dict[str, str]], column: str) -> bool:
    values = {row[column].strip() for row in rows if row.get(column, "").strip() != ""}
    return values <= {"0", "1", "0.0", "1.0"} and bool(values)


def is_unit_float_column(rows: list[dict[str, str]], column: str) -> bool:
    values: list[float] = []
    for row in rows:
        raw_value = row.get(column, "").strip()
        if raw_value == "":
            return False
        try:
            value = float(raw_value)
        except ValueError:
            return False
        if value < 0.0 or value > 1.0:
            return False
        values.append(value)
    return bool(values)


def detect_label_column(headers: list[str], rows: list[dict[str, str]]) -> str:
    by_lower = lower_map(headers)
    for candidate in LABEL_CANDIDATES:
        if candidate in by_lower:
            return by_lower[candidate]

    for header in headers:
        lowered = header.strip().lower()
        if "pred" in lowered or "prob" in lowered:
            continue
        if is_binary_column(rows, header):
            return header

    raise ValueError(f"Could not identify label column from headers: {headers}")


def detect_probability_column(
    headers: list[str], rows: list[dict[str, str]], label_column: str
) -> str:
    by_lower = lower_map(headers)
    for candidate in PROB_CANDIDATES:
        if candidate in by_lower:
            return by_lower[candidate]

    scored_candidates: list[tuple[int, str]] = []
    for header in headers:
        if header == label_column:
            continue
        lowered = header.strip().lower()
        if lowered.endswith("label") or lowered in {"image_path", "path", "filename"}:
            continue
        if not is_unit_float_column(rows, header):
            continue
        score = 0
        if "prob" in lowered:
            score += 3
        if "pred" in lowered or "score" in lowered:
            score += 2
        if not is_binary_column(rows, header):
            score += 1
        scored_candidates.append((score, header))

    if scored_candidates:
        scored_candidates.sort(reverse=True)
        return scored_candidates[0][1]

    raise ValueError(f"Could not identify prediction probability column from headers: {headers}")


def load_predictions(path: Path) -> dict[str, Any]:
    headers, rows = read_rows(path)
    label_column = detect_label_column(headers, rows)
    probability_column = detect_probability_column(headers, rows, label_column)

    labels: list[int] = []
    probabilities: list[float] = []
    for row_index, row in enumerate(rows):
        try:
            label = int(float(row[label_column]))
            probability = float(row[probability_column])
        except ValueError as exc:
            raise ValueError(f"{path} row {row_index} has non-numeric label/probability") from exc
        if label not in {0, 1}:
            raise ValueError(f"{path} row {row_index} has non-binary label: {label}")
        if probability < 0.0 or probability > 1.0:
            raise ValueError(f"{path} row {row_index} has probability outside [0, 1]: {probability}")
        labels.append(label)
        probabilities.append(probability)

    return {
        "path": str(path),
        "headers": headers,
        "label_column": label_column,
        "probability_column": probability_column,
        "labels": labels,
        "probabilities": probabilities,
    }


def compute_metrics(labels: list[int], probabilities: list[float], threshold: float) -> dict[str, Any]:
    tp = tn = fp = fn = 0
    for label, probability in zip(labels, probabilities):
        prediction = 1 if probability >= threshold else 0
        if label == 1 and prediction == 1:
            tp += 1
        elif label == 0 and prediction == 0:
            tn += 1
        elif label == 0 and prediction == 1:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    youden_j = recall + specificity - 1.0

    return {
        "threshold": round(threshold, 2),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "specificity": specificity,
        "youden_j": youden_j,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "positive_predictions": tp + fp,
        "negative_predictions": tn + fn,
    }


def sweep_thresholds(labels: list[int], probabilities: list[float]) -> list[dict[str, Any]]:
    return [compute_metrics(labels, probabilities, threshold / 100.0) for threshold in range(1, 100)]


def write_sweep(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "specificity",
        "youden_j",
        "tn",
        "fp",
        "fn",
        "tp",
        "positive_predictions",
        "negative_predictions",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_best(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            row[metric],
            row["balanced_accuracy"],
            row["f1"],
            row["specificity"],
            row["threshold"],
        ),
    )


def row_for_threshold(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    rounded = round(threshold, 2)
    for row in rows:
        if row["threshold"] == rounded:
            return row
    raise ValueError(f"No sweep row found for threshold {rounded}")


def compact_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "threshold": row["threshold"],
        "accuracy": row["accuracy"],
        "precision": row["precision"],
        "recall": row["recall"],
        "f1": row["f1"],
        "balanced_accuracy": row["balanced_accuracy"],
        "specificity": row["specificity"],
        "youden_j": row["youden_j"],
        "confusion_matrix": {
            "tn": row["tn"],
            "fp": row["fp"],
            "fn": row["fn"],
            "tp": row["tp"],
        },
        "positive_predictions": row["positive_predictions"],
        "negative_predictions": row["negative_predictions"],
    }


def format_float(value: float) -> str:
    return f"{value:.6f}"


def build_summary(
    val_data: dict[str, Any],
    test_data: dict[str, Any],
    val_sweep: list[dict[str, Any]],
    test_sweep: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_val = row_for_threshold(val_sweep, 0.5)
    baseline_test = row_for_threshold(test_sweep, 0.5)
    best_f1_val = select_best(val_sweep, "f1")
    best_balanced_val = select_best(val_sweep, "balanced_accuracy")
    best_youden_val = select_best(val_sweep, "youden_j")

    selections = {
        "threshold_0_50_baseline": {
            "threshold": 0.5,
            "validation": compact_metrics(baseline_val),
            "test": compact_metrics(baseline_test),
        },
        "best_validation_f1": {
            "threshold": best_f1_val["threshold"],
            "validation": compact_metrics(best_f1_val),
            "test": compact_metrics(row_for_threshold(test_sweep, best_f1_val["threshold"])),
        },
        "best_validation_balanced_accuracy": {
            "threshold": best_balanced_val["threshold"],
            "validation": compact_metrics(best_balanced_val),
            "test": compact_metrics(row_for_threshold(test_sweep, best_balanced_val["threshold"])),
        },
        "best_validation_youden_j": {
            "threshold": best_youden_val["threshold"],
            "validation": compact_metrics(best_youden_val),
            "test": compact_metrics(row_for_threshold(test_sweep, best_youden_val["threshold"])),
        },
    }

    baseline_f1 = baseline_val["f1"]
    best_f1 = best_f1_val["f1"]
    baseline_balanced = baseline_val["balanced_accuracy"]
    best_balanced = best_balanced_val["balanced_accuracy"]

    higher_threshold_rows = [
        item
        for item in selections.values()
        if item["threshold"] > 0.5
    ]
    false_positive_reduction = {
        name: {
            "threshold": item["threshold"],
            "validation_fp_delta_vs_0_50": item["validation"]["confusion_matrix"]["fp"]
            - selections["threshold_0_50_baseline"]["validation"]["confusion_matrix"]["fp"],
            "test_fp_delta_vs_0_50": item["test"]["confusion_matrix"]["fp"]
            - selections["threshold_0_50_baseline"]["test"]["confusion_matrix"]["fp"],
        }
        for name, item in selections.items()
        if item["threshold"] > 0.5
    }

    return {
        "calibration_method": {
            "threshold_range": "0.01..0.99",
            "threshold_step": 0.01,
            "selection_split": "validation",
            "selected_thresholds": [
                "threshold_0_50_baseline",
                "best_validation_f1",
                "best_validation_balanced_accuracy",
                "best_validation_youden_j",
            ],
        },
        "inputs": {
            "validation_csv": val_data["path"],
            "test_csv": test_data["path"],
            "validation_label_column": val_data["label_column"],
            "validation_probability_column": val_data["probability_column"],
            "test_label_column": test_data["label_column"],
            "test_probability_column": test_data["probability_column"],
            "validation_rows": len(val_data["labels"]),
            "test_rows": len(test_data["labels"]),
        },
        "selections": selections,
        "answers": {
            "threshold_0_50_already_good": (
                "Yes, threshold 0.50 is a strong baseline, but validation F1 improves by "
                f"{best_f1 - baseline_f1:.6f} at threshold {best_f1_val['threshold']:.2f}."
            ),
            "higher_threshold_reduces_false_positives": (
                "Yes, selected thresholds above 0.50 reduce false positives relative to 0.50."
                if higher_threshold_rows
                else "No selected validation-optimal threshold was above 0.50."
            ),
            "false_positive_reduction": false_positive_reduction,
            "best_validation_f1_threshold": best_f1_val["threshold"],
            "best_validation_balanced_accuracy_threshold": best_balanced_val["threshold"],
            "validation_balanced_accuracy_delta_vs_0_50": best_balanced - baseline_balanced,
        },
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    selections = summary["selections"]
    baseline = selections["threshold_0_50_baseline"]
    best_f1 = selections["best_validation_f1"]
    best_balanced = selections["best_validation_balanced_accuracy"]
    best_youden = selections["best_validation_youden_j"]

    lines = [
        "# RGNet AVA Threshold Calibration",
        "",
        "## Scope",
        "This calibration uses existing full AVA lse_r4 validation and test prediction CSVs only.",
        "No model training or evaluation rerun was performed by this script.",
        "",
        "## Inputs",
        f"- Validation CSV: `{summary['inputs']['validation_csv']}`",
        f"- Test CSV: `{summary['inputs']['test_csv']}`",
        f"- Label columns: validation `{summary['inputs']['validation_label_column']}`, test `{summary['inputs']['test_label_column']}`",
        f"- Probability columns: validation `{summary['inputs']['validation_probability_column']}`, test `{summary['inputs']['test_probability_column']}`",
        f"- Rows: validation {summary['inputs']['validation_rows']}, test {summary['inputs']['test_rows']}",
        "",
        "## Selected Thresholds",
        "| Selection | Threshold | Val F1 | Val Balanced Acc | Val Precision | Val Recall | Val Specificity | Val FP | Test F1 | Test Balanced Acc | Test Precision | Test Recall | Test Specificity | Test FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    table_rows = [
        ("0.50 baseline", baseline),
        ("Best validation F1", best_f1),
        ("Best validation balanced accuracy", best_balanced),
        ("Best validation Youden J", best_youden),
    ]
    for label, item in table_rows:
        val = item["validation"]
        test = item["test"]
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    format_float(item["threshold"]),
                    format_float(val["f1"]),
                    format_float(val["balanced_accuracy"]),
                    format_float(val["precision"]),
                    format_float(val["recall"]),
                    format_float(val["specificity"]),
                    str(val["confusion_matrix"]["fp"]),
                    format_float(test["f1"]),
                    format_float(test["balanced_accuracy"]),
                    format_float(test["precision"]),
                    format_float(test["recall"]),
                    format_float(test["specificity"]),
                    str(test["confusion_matrix"]["fp"]),
                ]
            )
            + " |"
        )

    baseline_val = baseline["validation"]
    baseline_test = baseline["test"]
    best_f1_val = best_f1["validation"]
    best_f1_test = best_f1["test"]
    best_balanced_val = best_balanced["validation"]
    best_balanced_test = best_balanced["test"]

    lines.extend(
        [
            "",
            "## Answers",
            f"- Is threshold 0.5 already good? {summary['answers']['threshold_0_50_already_good']}",
            f"- Does a higher threshold reduce false positives? {summary['answers']['higher_threshold_reduces_false_positives']}",
            f"- Best validation F1 threshold: {best_f1['threshold']:.2f}, with validation F1 {format_float(best_f1_val['f1'])} and test F1 {format_float(best_f1_test['f1'])}.",
            f"- Best validation balanced accuracy threshold: {best_balanced['threshold']:.2f}, with validation balanced accuracy {format_float(best_balanced_val['balanced_accuracy'])} and test balanced accuracy {format_float(best_balanced_test['balanced_accuracy'])}.",
            f"- Applying best validation F1 to test changes F1 from {format_float(baseline_test['f1'])} to {format_float(best_f1_test['f1'])}, and test false positives from {baseline_test['confusion_matrix']['fp']} to {best_f1_test['confusion_matrix']['fp']}.",
            f"- Applying best validation balanced accuracy to test changes balanced accuracy from {format_float(baseline_test['balanced_accuracy'])} to {format_float(best_balanced_test['balanced_accuracy'])}, and test false positives from {baseline_test['confusion_matrix']['fp']} to {best_balanced_test['confusion_matrix']['fp']}.",
            "",
            "## Confusion Matrices",
            f"- 0.50 validation: {baseline_val['confusion_matrix']}",
            f"- 0.50 test: {baseline_test['confusion_matrix']}",
            f"- Best validation F1 applied to test: {best_f1_test['confusion_matrix']}",
            f"- Best validation balanced accuracy applied to test: {best_balanced_test['confusion_matrix']}",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    val_csv = Path(args.val_csv)
    test_csv = Path(args.test_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_data = load_predictions(val_csv)
    test_data = load_predictions(test_csv)
    val_sweep = sweep_thresholds(val_data["labels"], val_data["probabilities"])
    test_sweep = sweep_thresholds(test_data["labels"], test_data["probabilities"])

    write_sweep(output_dir / "threshold_sweep_val.csv", val_sweep)
    write_sweep(output_dir / "threshold_sweep_test.csv", test_sweep)

    summary = build_summary(val_data, test_data, val_sweep, test_sweep)
    summary_path = output_dir / "threshold_calibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(output_dir / "threshold_calibration_report.md", summary)

    print(f"Wrote calibration outputs to {output_dir}")
    print(
        "Best validation F1 threshold:",
        f"{summary['selections']['best_validation_f1']['threshold']:.2f}",
    )
    print(
        "Best validation balanced accuracy threshold:",
        f"{summary['selections']['best_validation_balanced_accuracy']['threshold']:.2f}",
    )


if __name__ == "__main__":
    main()

