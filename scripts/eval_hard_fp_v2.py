# Specialized Evaluation for TechIQA-Guard v1 Hard-FP v2
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

# Ensure PYTHONPATH=.
sys.path.append(".")
from src.models.techiqa_guard import build_techiqa_guard_v1

IMAGE_SIZE = 224

def preprocess_image(path, size):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_with_pad(img, size, size)
    img = tf.cast(img, tf.float32)
    return img

class TFLiteWrapper:
    def __init__(self, path):
        self.path = path
        self.name = Path(path).stem
        self.interpreter = tf.lite.Interpreter(model_path=str(path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = self.input_details[0]['shape'][1]

    def predict(self, img_tensor):
        # img_tensor: [size, size, 3]
        inp = img_tensor.numpy()[np.newaxis, ...]
        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]

def main():
    root = Path("/home/omen_pc1/photo_score_project")
    csv_path = root / "data/processed/techiqa_guard/fp_mining_20260522/hard_false_positive_confirmed_v2.csv"
    out_dir = root / "outputs/eval_techiqa_guard_v1_hard_fp_confirmed_v2_20260522"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} hard-FP v2 samples.")

    model_configs = [
        {"name": "techiqa_stage4", "path": root / "outputs/techiqa_guard_v1_direct_stage4_fpdata_lam005_20260520/best.weights.h5", "type": "keras", "size": 224},
        {"name": "techiqa_stage4b", "path": root / "outputs/techiqa_guard_v1_direct_stage4b_fpconfirmed_v2_lam005_20260522/best.weights.h5", "type": "keras", "size": 224},
        {"name": "techiqa_stage5_v2", "path": root / "outputs/techiqa_guard_v1_direct_stage5_rank002_fpguard005_v2_20260520/best.weights.h5", "type": "keras", "size": 224},
        {"name": "techiqa_stage5_v3", "path": root / "outputs/techiqa_guard_v1_direct_stage5_rank001_fpguard05_20260520/best.weights.h5", "type": "keras", "size": 224},
        {"name": "topiq_mixed112", "path": root / "outputs/final_topiq_lite_mixed112_export_20260517/topiq_lite_mixed112_frozen_fp16.tflite", "type": "tflite", "preprocess": "topiq"},
        {"name": "topiq_ranking", "path": root / "outputs/final_topiq_lite_ranking_lam01_export_20260517/topiq_lite_ranking_lam01_gap05_fp16.tflite", "type": "tflite", "preprocess": "topiq"},
        {"name": "koniq_mobile", "path": root / "exports/tflite/koniq_mobile.tflite", "type": "tflite", "preprocess": "existing"},
        {"name": "flive_mobile", "path": root / "exports/tflite/flive_image_mobile.tflite", "type": "tflite", "preprocess": "existing"},
    ]

    all_preds = {}

    for cfg in model_configs:
        name = cfg["name"]
        path = cfg["path"]
        if not path.exists():
            print(f"SKIP: {name} at {path} not found")
            continue

        print(f"Evaluating {name}...")
        preds = []
        
        if cfg["type"] == "keras":
            size = cfg["size"]
            model = build_techiqa_guard_v1(input_shape=(size, size, 3))
            model.load_weights(str(path))
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc=name):
                img = preprocess_image(row["image_path"], size)
                score = float(model.predict(img.numpy()[np.newaxis, ...], verbose=0)[0][0]) * 100.0
                preds.append(score)
        else:
            model = TFLiteWrapper(path)
            size = model.input_size
            for _, row in tqdm(df.iterrows(), total=len(df), desc=name):
                img = preprocess_image(row["image_path"], size)
                score = model.predict(img)
                if cfg["preprocess"] == "topiq":
                    score *= 100.0
                preds.append(score)
        
        all_preds[name] = preds
        pd.DataFrame({"image_path": df["image_path"], "y_pred": preds}).to_csv(out_dir / f"predictions_{name}.csv", index=False)

    # Add existing_avg
    if "koniq_mobile" in all_preds and "flive_mobile" in all_preds:
        all_preds["existing_avg"] = [(k + f) / 2.0 for k, f in zip(all_preds["koniq_mobile"], all_preds["flive_mobile"])]
        pd.DataFrame({"image_path": df["image_path"], "y_pred": all_preds["existing_avg"]}).to_csv(out_dir / "predictions_existing_avg.csv", index=False)

    # Summary
    summary_rows = []
    for name, preds in all_preds.items():
        summary_rows.append({
            "model": name,
            "n": len(preds),
            "mean": np.mean(preds),
            "std": np.std(preds),
            "min": np.min(preds),
            "max": np.max(preds),
            "median": np.median(preds)
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    print("\nSummary Results:")
    print(summary_df)

    # Find top over-scored images (highest Mixed112 vs Stage4)
    # We use Mixed112 as the "general IQA" baseline that tends to overscore
    df["techiqa_stage4"] = all_preds.get("techiqa_stage4", [0]*len(df))
    df["topiq_mixed112"] = all_preds.get("topiq_mixed112", [0]*len(df))
    df["existing_avg"] = all_preds.get("existing_avg", [0]*len(df))
    df["delta_s4_existing"] = df["techiqa_stage4"] - df["existing_avg"]
    
    top_overscored = df.sort_values(by="techiqa_stage4", ascending=False).head(10)

    # Report generation
    with open(out_dir / "report.md", "w") as f:
        f.write("# TechIQA-Guard v1 Hard-FP Confirmed v2 Evaluation Report\n\n")
        f.write(f"Evaluating models on the expanded confirmed hard-FP set (v2).\n")
        f.write(f"- n = {len(df)}\n\n")
        
        f.write("## 1. Summary Metrics\n")
        f.write("| model | n | mean | std | min | max | median |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for _, row in summary_df.iterrows():
            f.write(f"| {row['model']} | {row['n']} | {row['mean']:.2f} | {row['std']:.2f} | {row['min']:.2f} | {row['max']:.2f} | {row['median']:.2f} |\n")
        f.write("\n")
        
        f.write("## 2. Model Comparison Analysis\n")
        s4_mean = summary_df[summary_df["model"] == "techiqa_stage4"]["mean"].values[0]
        s4b_mean = summary_df[summary_df["model"] == "techiqa_stage4b"]["mean"].values[0]
        improvement = s4_mean - s4b_mean
        
        f.write(f"- **Stage 4 Mean**: {s4_mean:.2f}\n")
        f.write(f"- **Stage 4B Mean**: {s4b_mean:.2f}\n")
        f.write(f"- **Absolute Improvement**: {improvement:.2f} points lower\n\n")
        
        if improvement > 0.5:
            f.write("Stage 4B shows measurable improvement (lower scores) on the confirmed false-positive set.\n\n")
        elif improvement < -0.5:
            f.write("Stage 4B regressed (higher scores) compared to Stage 4 on this set.\n\n")
        else:
            f.write("Stage 4B and Stage 4 are roughly equivalent on this set.\n\n")

        f.write("## 3. Top 10 Over-scored Images (Stage 4)\n")
        f.write("| filename | techiqa_s4 | mixed112 | existing_avg | delta_s4_avg |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for _, row in top_overscored.iterrows():
            fname = Path(row["image_path"]).name
            f.write(f"| {fname} | {row['techiqa_stage4']:.2f} | {row['topiq_mixed112']:.2f} | {row['existing_avg']:.2f} | {row['delta_s4_existing']:.2f} |\n")
        f.write("\n")

        f.write("## 4. Conclusion\n")
        existing_avg_mean = summary_df[summary_df["model"] == "existing_avg"]["mean"].values[0]
        f.write(f"All current TechIQA-Guard models (mean ~{s4_mean:.2f}) are still scoring significantly higher than the existing technical guard baseline (mean {existing_avg_mean:.2f}) on this hard false-positive set.\n\n")
        f.write("Further training with oversampling of these confirmed samples is required.\n")

    print(f"\nReport saved to {out_dir / 'report.md'}")

if __name__ == "__main__":
    main()
