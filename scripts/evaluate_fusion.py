import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    balanced_accuracy_score
)

# Custom objects
from src.models.alamp_paper_fusion import get_alamp_paper_fusion_custom_objects
from src.models.alamp_paper_mpnet import get_alamp_paper_mpnet_custom_objects
from src.train.train_alamp_paper_mpnet_graph_fusion import _prepare_split, _make_dataset
from tools.calibrate_mpnet_thresholds import threshold_sweep, build_summary

def get_custom_objects():
    objs = get_alamp_paper_mpnet_custom_objects()
    objs.update(get_alamp_paper_fusion_custom_objects())
    return objs

def evaluate(model_path, fusion_mode, output_dir, val_patch_jsonl, val_graph_jsonl, mpnet_features_npz):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    val_data = _prepare_split(
        split="val",
        patch_jsonl=Path(val_patch_jsonl),
        graph_jsonl=Path(val_graph_jsonl),
        max_samples=None,
        patch_size=224,
        patch_count=5,
        label_threshold=5.0,
        max_objects=4,
        class_id_scale=79.0,
    )
    
    # load mpnet features
    npz = np.load(mpnet_features_npz)
    val_data.mpnet_features = npz["mpnet_features"]
    
    val_dataset = _make_dataset(val_data, fusion_mode, batch_size=16, shuffle=False)
    
    # 2. Load model
    model = tf.keras.models.load_model(model_path, compile=False, custom_objects=get_custom_objects())
    
    # 3. Predict
    y_true = []
    y_pred_probs = []
    
    start_time = time.time()
    for batch_inputs, batch_labels in val_dataset:
        probs = model(batch_inputs, training=False).numpy()
        y_true.extend(batch_labels.numpy().flatten())
        y_pred_probs.extend(probs.flatten())
    elapsed = time.time() - start_time
    
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    # 4. Save val_predictions.csv
    df = pd.DataFrame({"true": y_true, "pred": y_pred_probs})
    df.to_csv(output_dir / "val_predictions.csv", index=False)
    
    # 5. Sweep and Summary
    sweep_df = threshold_sweep(
        val_labels=y_true, val_probabilities=y_pred_probs,
        test_labels=y_true, test_probabilities=y_pred_probs # Pass val as test to satisfy API
    )
    sweep_df.to_csv(output_dir / "threshold_sweep.csv", index=False)
    
    # Build calibration summary
    calibration_summary = build_summary(
        sweep_df, 
        val_predictions=output_dir / "val_predictions.csv", 
        test_predictions=output_dir / "val_predictions.csv",
        val_count=len(y_true),
        test_count=len(y_true)
    )
    
    with open(output_dir / "threshold_calibration_summary.json", "w") as f:
        json.dump(calibration_summary, f, indent=2)
        
    # 6. Default Evaluation Summary (threshold 0.5)
    y_pred = (y_pred_probs > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    
    bce = float(tf.keras.losses.binary_crossentropy(y_true.reshape(-1, 1), y_pred_probs.reshape(-1, 1)).numpy().mean())
    
    eval_summary = {
        "fusion_mode": fusion_mode,
        "metrics_at_0_5": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_pred_probs)),
            "average_precision": float(average_precision_score(y_true, y_pred_probs)),
            "bce_loss": bce,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        },
        "seconds_per_image": float(elapsed / len(y_true)) if len(y_true) > 0 else 0.0,
        "total_samples": len(y_true)
    }
    
    with open(output_dir / "evaluation_summary.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--fusion_mode", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--val_patch_jsonl", required=True)
    parser.add_argument("--val_graph_jsonl", required=True)
    parser.add_argument("--mpnet_features_npz", required=True)
    args = parser.parse_args()
    
    evaluate(args.model_path, args.fusion_mode, args.output_dir, args.val_patch_jsonl, args.val_graph_jsonl, args.mpnet_features_npz)
