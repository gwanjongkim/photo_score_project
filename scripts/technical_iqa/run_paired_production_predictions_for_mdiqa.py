import os
import sys
import csv
import json
import argparse
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# Production-related imports
# We use .venv_gpu for this script
import tensorflow as tf
from src.infer.predict_quality_bundle import load_bundle_models, predict_bundle_for_image, add_bundle_arguments
from src.infer.select_best_shots import DEFAULT_SELECTOR_WEIGHTS, normalize_unit_score, score_component

# Paths
OUTPUT_DIR = Path("outputs/paired_production_mdiqa_significance_20260603")
PAIRED_MANIFEST = OUTPUT_DIR / "paired_eval_manifest.csv"
PAIRED_PREDICTIONS = OUTPUT_DIR / "paired_predictions.csv"
PRODUCTION_FAILURES = OUTPUT_DIR / "production_failures.csv"
SUMMARY_CSV = OUTPUT_DIR / "production_prediction_summary.csv"

def run_predictions(smoke=False, limit=None):
    if not PAIRED_MANIFEST.exists():
        print(f"Error: {PAIRED_MANIFEST} not found.")
        return

    df = pd.read_csv(PAIRED_MANIFEST)
    if smoke:
        print("Running in SMOKE mode (40 images total).")
        df_koniq = df[df["dataset"] == "koniq10k"].head(20)
        df_spaq = df[df["dataset"] == "spaq"].head(20)
        df = pd.concat([df_koniq, df_spaq])
    elif limit:
        print(f"Running with LIMIT mode ({limit} images per dataset).")
        df_list = []
        for ds in df["dataset"].unique():
            df_list.append(df[df["dataset"] == ds].head(limit))
        df = pd.concat(df_list)

    # Load bundle config
    with open("configs/stage5_reference.json", "r") as f:
        config = json.load(f)
    
    # Setup args for QualityBundle
    parser = argparse.ArgumentParser()
    add_bundle_arguments(parser)
    # Map bundle_args to argparse format
    bundle_args = config["bundle_args"]
    args_list = []
    for k, v in bundle_args.items():
        if v is not None:
            args_list.extend([f"--{k}", str(v)])
    args = parser.parse_args(args_list)

    print("Loading Production Bundle Models...")
    models = load_bundle_models(args)
    
    # Resume support
    processed_paths = set()
    if PAIRED_PREDICTIONS.exists():
        try:
            existing_df = pd.read_csv(PAIRED_PREDICTIONS)
            processed_paths = set(existing_df["image_path"].tolist())
            print(f"Found existing predictions. Resuming from {len(processed_paths)} images.")
        except Exception as e:
            print(f"Warning: Could not load existing predictions for resume: {e}")

    predictions = []
    if processed_paths and PAIRED_PREDICTIONS.exists():
        predictions = existing_df.to_dict(orient="records")

    failures = []
    
    total = len(df)
    new_count = 0
    for i, (idx, row) in enumerate(df.iterrows()):
        img_path = row["image_path"]
        if img_path in processed_paths:
            continue

        print(f"[{i+1}/{total}] Predicting: {Path(img_path).name}")
        
        try:
            # Predict
            res = predict_bundle_for_image(Path(img_path), args=args, models=models)
            
            # Map result to record for score_component
            record = {}
            for key in ["koniq_score", "flive_image_score", "flive_patch_mean", "flive_patch_min", "musiq_score"]:
                if key in res:
                    record[key] = res[key]
            
            # Compute technical_score using production weights
            weights = DEFAULT_SELECTOR_WEIGHTS["technical"]
            tech_score, tech_contrib, tech_missing = score_component(record, weights, "technical")
            
            # Convert tech_score (0..1) to 0..100 scale for comparison
            prod_pred_0_100 = tech_score * 100.0 if tech_score is not None else None
            
            pred_row = {
                "dataset": row["dataset"],
                "image_path": row["image_path"],
                "quality_score_0_100": row["quality_score_0_100"],
                "mdiqa_variant": row["mdiqa_variant"] if "mdiqa_variant" in row else "mdiqa",
                "mdiqa_pred_0_100": row["mdiqa_pred_0_100"],
                "production_pred_0_100": prod_pred_0_100,
                "koniq_mobile": record.get("koniq_score"),
                "flive_image_mobile": record.get("flive_image_score"),
                "flive_patch_mean": record.get("flive_patch_mean"),
                "flive_patch_min": record.get("flive_patch_min"),
                "musiq_aadb": record.get("musiq_score"),
                "production_success": True,
                "production_error": ""
            }
            predictions.append(pred_row)
            
        except Exception as e:
            print(f"  Error: {e}")
            failures.append({
                "dataset": row["dataset"],
                "image_path": row["image_path"],
                "error": str(e)
            })
            predictions.append({
                "dataset": row["dataset"],
                "image_path": row["image_path"],
                "quality_score_0_100": row["quality_score_0_100"],
                "mdiqa_pred_0_100": row["mdiqa_pred_0_100"],
                "production_pred_0_100": None,
                "production_success": False,
                "production_error": str(e)
            })

    # Save results
    pd.DataFrame(predictions).to_csv(PAIRED_PREDICTIONS, index=False)
    print(f"Saved predictions to {PAIRED_PREDICTIONS}")
    
    if failures:
        pd.DataFrame(failures).to_csv(PRODUCTION_FAILURES, index=False)
        print(f"Saved failures to {PRODUCTION_FAILURES}")
    
    # Summary
    success_df = pd.DataFrame(predictions)
    summary_rows = []
    for ds in success_df["dataset"].unique():
        ds_df = success_df[(success_df["dataset"] == ds) & success_df["production_success"]]
        if len(ds_df) > 0:
            summary_rows.append({
                "dataset": ds,
                "count": len(ds_df),
                "prod_min": ds_df["production_pred_0_100"].min(),
                "prod_max": ds_df["production_pred_0_100"].max(),
                "prod_mean": ds_df["production_pred_0_100"].mean(),
                "mos_mean": ds_df["quality_score_0_100"].mean(),
            })
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)
    print(f"Saved summary to {SUMMARY_CSV}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_predictions(smoke=args.smoke, limit=args.limit)
