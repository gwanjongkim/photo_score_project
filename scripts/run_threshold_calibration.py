# MPNet 임계값 보정 스크립트 (Threshold Calibration Script)
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
import json
from pathlib import Path

def calculate_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall,
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "youden_j": recall + specificity - 1,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "pos_preds": int(tp + fp),
        "neg_preds": int(tn + fn)
    }

def run_calibration():
    base_dir = Path("outputs/alamp_paper_mpnet_selector_compare_20260512/eval/v4_1024_gpu/v4_1024_gpu")
    test_df = pd.read_csv(base_dir / "test_predictions.csv")
    val_df = pd.read_csv(base_dir / "val_predictions.csv")
    
    thresholds = np.arange(0.01, 1.0, 0.01)
    sweep_data = []
    
    for t in thresholds:
        val_m = calculate_metrics(val_df["true"], val_df["pred"], t)
        test_m = calculate_metrics(test_df["true"], test_df["pred"], t)
        
        row = {"threshold": t}
        for k, v in val_m.items():
            if k != "threshold": row[f"val_{k}"] = v
        for k, v in test_m.items():
            if k != "threshold": row[f"test_{k}"] = v
        sweep_data.append(row)
        
    sweep_df = pd.DataFrame(sweep_data)
    sweep_df.to_csv(base_dir / "threshold_sweep.csv", index=False)
    
    # Selection based on validation
    selections = {}
    
    # 1. Baseline
    selections["threshold_0_50_baseline"] = float(sweep_df.loc[np.isclose(sweep_df["threshold"], 0.5), "threshold"].iloc[0])
    
    # 2. Best F1
    selections["best_validation_f1"] = float(sweep_df.loc[sweep_df["val_f1"].idxmax(), "threshold"])
    
    # 3. Best Balanced Acc
    selections["best_validation_balanced_accuracy"] = float(sweep_df.loc[sweep_df["val_balanced_accuracy"].idxmax(), "threshold"])
    
    # 4. Best Youden J
    selections["best_validation_youden_j"] = float(sweep_df.loc[sweep_df["val_youden_j"].idxmax(), "threshold"])
    
    # 5. Specificity >= 0.3
    spec_30 = sweep_df[sweep_df["val_specificity"] >= 0.3]
    selections["spec_0_30"] = float(spec_30["threshold"].min()) if not spec_30.empty else None
    
    # 6. Specificity >= 0.5
    spec_50 = sweep_df[sweep_df["val_specificity"] >= 0.5]
    selections["spec_0_50"] = float(spec_50["threshold"].min()) if not spec_50.empty else None
    
    # 7. Specificity >= 0.7
    spec_70 = sweep_df[sweep_df["val_specificity"] >= 0.7]
    selections["spec_0_70"] = float(spec_70["threshold"].min()) if not spec_70.empty else None
    
    summary = {"selections": {}}
    for label, t in selections.items():
        if t is None: continue
        metrics = sweep_df[np.isclose(sweep_df["threshold"], t)].iloc[0].to_dict()
        summary["selections"][label] = metrics
        
    with open(base_dir / "threshold_calibration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("Calibration sweep complete.")

if __name__ == "__main__":
    run_calibration()
