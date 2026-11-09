import pandas as pd
import os
from pathlib import Path

# Paths
MDIQA_PREDS = "outputs/mdiqa_correct_protocol_full_benchmark_20260601/predictions.csv"
OUTPUT_DIR = Path("outputs/paired_production_mdiqa_significance_20260603")
PAIRED_MANIFEST = OUTPUT_DIR / "paired_eval_manifest.csv"
MANIFEST_AUDIT = OUTPUT_DIR / "manifest_audit.csv"

def build_paired_manifest():
    if not os.path.exists(MDIQA_PREDS):
        print(f"Error: {MDIQA_PREDS} not found.")
        return

    df = pd.read_csv(MDIQA_PREDS)
    
    # Filter for KonIQ and SPAQ
    df = df[df["dataset"].isin(["koniq10k", "spaq"])].copy()
    
    # Audit stats
    total_rows = len(df)
    missing_images = df[~df["image_path"].apply(os.path.exists)]
    duplicates = df[df.duplicated("image_path")]
    
    # Drop invalid
    df = df[df["image_path"].apply(os.path.exists)].copy()
    df = df.drop_duplicates("image_path")
    
    # Rename columns for clarity
    df = df.rename(columns={
        "pred_raw": "mdiqa_raw",
        "pred_0_100": "mdiqa_pred_0_100"
    })
    
    # Save manifest
    df.to_csv(PAIRED_MANIFEST, index=False)
    print(f"Saved {len(df)} rows to {PAIRED_MANIFEST}")
    
    # Audit summary
    audit_rows = [
        {"metric": "total_rows_filtered", "value": total_rows},
        {"metric": "missing_image_count", "value": len(missing_images)},
        {"metric": "duplicate_image_path_count", "value": len(duplicates)},
        {"metric": "final_paired_rows", "value": len(df)},
        {"metric": "koniq10k_rows", "value": len(df[df["dataset"] == "koniq10k"])},
        {"metric": "spaq_rows", "value": len(df[df["dataset"] == "spaq"])},
        {"metric": "mdiqa_pred_min", "value": df["mdiqa_pred_0_100"].min()},
        {"metric": "mdiqa_pred_max", "value": df["mdiqa_pred_0_100"].max()},
        {"metric": "mos_min", "value": df["quality_score_0_100"].min()},
        {"metric": "mos_max", "value": df["quality_score_0_100"].max()},
    ]
    pd.DataFrame(audit_rows).to_csv(MANIFEST_AUDIT, index=False)
    print(f"Saved audit to {MANIFEST_AUDIT}")

if __name__ == "__main__":
    build_paired_manifest()
