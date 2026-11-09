import pandas as pd
import numpy as np
from pathlib import Path
import json

OUT_DIR = Path("outputs/mdiqa_distillation_50k_smoke_targets_20260608")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Data Sources
POLICY_B_MANIFEST = "outputs/mdiqa_distillation_50k_manifest_audit_20260608/candidate_50k_manifest_preview.csv"
OFFICIAL_TEST_MANIFEST = "outputs/official_protocol_benchmark_manifest_20260604/official_benchmark_test_manifest.csv"

def build_smoke_manifest():
    print("Loading Policy B manifest...")
    df_b = pd.read_csv(POLICY_B_MANIFEST)
    
    # Exclude already cached rows
    df_missing = df_b[~df_b['is_cached']].copy()
    
    # Ensure no official test leakage
    print("Checking for leakage...")
    df_off_test = pd.read_csv(OFFICIAL_TEST_MANIFEST)
    off_paths = set(df_off_test['image_path'])
    df_missing = df_missing[~df_missing['image_path'].isin(off_paths)]
    
    # Sample 5k (Proportional to missing ratio)
    koniq_pool = df_missing[df_missing['dataset'] == 'koniq10k']
    spaq_pool = df_missing[df_missing['dataset'] == 'spaq']
    flive_pool = df_missing[df_missing['dataset'] == 'flive']
    
    # Target: 5k total. Let's do 10% of each dataset's missing pool roughly
    koniq_sample = koniq_pool.sample(n=min(500, len(koniq_pool)), random_state=42)
    spaq_sample = spaq_pool.sample(n=min(1000, len(spaq_pool)), random_state=42)
    
    remaining = 5000 - len(koniq_sample) - len(spaq_sample)
    flive_sample = flive_pool.sample(n=min(remaining, len(flive_pool)), random_state=42)
    
    smoke_df = pd.concat([koniq_sample, spaq_sample, flive_sample]).sample(frac=1, random_state=42)
    
    out_path = OUT_DIR / "smoke_5k_manifest.csv"
    smoke_df.to_csv(out_path, index=False)
    
    print(f"Created 5k smoke manifest with {len(smoke_df)} rows at {out_path}")
    print(smoke_df['dataset'].value_counts())
    
    leakage_audit = [
        {"split": "Official Test", "overlap_count": len(set(smoke_df['image_path']).intersection(off_paths))}
    ]
    pd.DataFrame(leakage_audit).to_csv(OUT_DIR / "smoke_5k_leakage_audit.csv", index=False)
    
    return out_path

if __name__ == "__main__":
    build_smoke_manifest()
