# Score Real Mobile / Test Weight images with Production Bundle

import argparse
import json
import pandas as pd
from pathlib import Path
import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.getcwd())

from src.infer.predict_quality_bundle import add_bundle_arguments, load_bundle_models, predict_bundle_for_image
from src.infer.select_best_shots import score_component, DEFAULT_SELECTOR_WEIGHTS

def main():
    ROOT = Path(os.getcwd())
    CONFIG_PATH = ROOT / "configs/stage5_reference.json"
    INPUT_DIR = ROOT / "data/Test_weight"
    OUTPUT_CSV = ROOT / "outputs/hybrid_teacher_unknown_fallback_audit_20260603/test_weight_technical_scores.csv"
    
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    
    parser = argparse.ArgumentParser()
    add_bundle_arguments(parser)
    bundle_args = config["bundle_args"]
    args_list = []
    for k, v in bundle_args.items():
        if v is not None:
            val = str(v)
            if val.startswith("checkpoints/") or val.startswith("configs/"):
                val = str(ROOT / val)
            args_list.extend([f"--{k}", val])
    
    args = parser.parse_args(args_list)
    models = load_bundle_models(args)
    
    image_paths = list(INPUT_DIR.rglob("*.jpg")) + list(INPUT_DIR.rglob("*.JPG"))
    
    results = []
    for img_path in image_paths:
        try:
            print(f"Scoring {img_path.name}...")
            res = predict_bundle_for_image(img_path, args=args, models=models)
            
            record = {}
            for key in ["koniq_score", "flive_image_score", "flive_patch_mean", "flive_patch_min", "musiq_score"]:
                if key in res:
                    record[key] = res[key]
            
            weights = DEFAULT_SELECTOR_WEIGHTS["technical"]
            tech_score, _, _ = score_component(record, weights, "technical")
            
            results.append({
                "image_path": str(img_path),
                "filename": img_path.name,
                "production_pred_0_100": tech_score * 100.0 if tech_score is not None else None,
                "koniq_mobile": record.get("koniq_score"),
                "flive_image_mobile": record.get("flive_image_score"),
                "flive_patch_mean": record.get("flive_patch_mean"),
                "flive_patch_min": record.get("flive_patch_min"),
                "musiq_aadb": record.get("musiq_score"),
                "label_name": img_path.parent.name # Bad, Soso, Good
            })
        except Exception as e:
            print(f"Failed to score {img_path.name}: {e}")
            
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} scores to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
