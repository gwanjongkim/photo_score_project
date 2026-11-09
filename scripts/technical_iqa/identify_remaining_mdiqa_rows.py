import pandas as pd
import numpy as np
from pathlib import Path
import os

ROOT = Path("/home/omen_pc1/photo_score_project")
OUT_DIR = ROOT / "outputs/mdiqa_distillation_50k_full_targets_20260608"

# 1. Inputs
POLICY_B_MANIFEST = ROOT / "outputs/mdiqa_distillation_50k_manifest_audit_20260608/candidate_50k_manifest_preview.csv"
CACHE_1 = ROOT / "outputs/mdiqa_correct_protocol_full_benchmark_20260601/predictions.csv"
CACHE_2 = ROOT / "outputs/official_protocol_mdiqa_signal_benchmark_20260604/predictions.csv"
SMOKE_PREDS = ROOT / "outputs/mdiqa_distillation_50k_smoke_targets_20260608/smoke_5k_mdiqa_predictions.csv"

def norm_path(p):
    p = str(p)
    if p.startswith('/home/omen_pc1/photo_score_project/'):
        return p.replace('/home/omen_pc1/photo_score_project/', '')
    return p

def main():
    print("Identifying remaining missing rows for MDIQA inference...")
    
    # Load Policy B 50k rows
    df_50k = pd.read_csv(POLICY_B_MANIFEST)
    df_50k['norm_path'] = df_50k['image_path'].apply(norm_path)
    print(f"Total Policy B rows: {len(df_50k)}")

    # Load all existing predictions
    print("Loading existing prediction caches...")
    pred_maps = {} # norm_path -> pred_0_100
    
    for path in [CACHE_1, CACHE_2, SMOKE_PREDS]:
        if path.exists():
            df_p = pd.read_csv(path)
            # Use 'pred_0_100' column
            for _, row in df_p.iterrows():
                np = norm_path(row['image_path'])
                pred_maps[np] = row['pred_0_100']
        else:
            print(f"Warning: Cache not found: {path}")

    print(f"Total unique predictions in caches: {len(pred_maps)}")

    # Check which of the 50k are missing
    df_50k['has_pred'] = df_50k['norm_path'].apply(lambda x: x in pred_maps)
    
    missing_df = df_50k[~df_50k['has_pred']].copy()
    print(f"Remaining missing rows: {len(missing_df)}")

    # Save missing manifest
    missing_df.to_csv(OUT_DIR / "remaining_missing_manifest.csv", index=False)
    
    # Cache merge audit
    merge_audit = [
        {"source": "Policy B 50k", "count": len(df_50k)},
        {"source": "Found in cache", "count": df_50k['has_pred'].sum()},
        {"source": "Missing", "count": len(missing_df)}
    ]
    pd.DataFrame(merge_audit).to_csv(OUT_DIR / "final_50k_cache_merge_audit.csv", index=False)

if __name__ == "__main__":
    main()
