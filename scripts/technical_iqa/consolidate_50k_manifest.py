import pandas as pd
import numpy as np
from pathlib import Path
import os
import json

ROOT = Path("/home/omen_pc1/photo_score_project")
OUT_DIR = ROOT / "outputs/mdiqa_distillation_50k_full_targets_20260608"

# Inputs
POLICY_B_MANIFEST = ROOT / "outputs/mdiqa_distillation_50k_manifest_audit_20260608/candidate_50k_manifest_preview.csv"
CACHE_1 = ROOT / "outputs/mdiqa_correct_protocol_full_benchmark_20260601/predictions.csv"
CACHE_2 = ROOT / "outputs/official_protocol_mdiqa_signal_benchmark_20260604/predictions.csv"
SMOKE_PREDS = ROOT / "outputs/mdiqa_distillation_50k_smoke_targets_20260608/smoke_5k_mdiqa_predictions.csv"
REMAINING_PREDS = OUT_DIR / "remaining_mdiqa_predictions.csv"

# Outputs
FINAL_MANIFEST = OUT_DIR / "final_50k_mdiqa_distillation_manifest.csv"
FINAL_DISTRIBUTION = OUT_DIR / "final_50k_distribution_summary.csv"
FINAL_FAILURES = OUT_DIR / "final_50k_failure_cases.csv"
FINAL_LEAKAGE = OUT_DIR / "final_50k_leakage_audit.csv"

def norm_path(p):
    p = str(p)
    if p.startswith('/home/omen_pc1/photo_score_project/'):
        return p.replace('/home/omen_pc1/photo_score_project/', '')
    return p

def main():
    print("Consolidating 50k manifest...")
    df_50k = pd.read_csv(POLICY_B_MANIFEST)
    df_50k['norm_path'] = df_50k['image_path'].apply(norm_path)
    
    # Load predictions
    pred_maps = {} # norm_path -> pred_0_100
    source_maps = {} # norm_path -> source cache
    
    caches = {
        "cache_1": CACHE_1,
        "cache_2": CACHE_2,
        "smoke_5k": SMOKE_PREDS,
        "remaining_38k": REMAINING_PREDS
    }
    
    for name, path in caches.items():
        if path.exists():
            df_p = pd.read_csv(path)
            for _, row in df_p.iterrows():
                np_path = norm_path(row['image_path'])
                if 'pred_0_100' in row:
                    pred_maps[np_path] = row['pred_0_100']
                    source_maps[np_path] = name
        else:
            print(f"Warning: Cache not found: {path}")

    # Merge
    df_50k['teacher_target_0_100'] = df_50k['norm_path'].map(pred_maps)
    df_50k['target_source'] = df_50k['norm_path'].map(source_maps)
    
    # Validate
    failures = []
    
    # 1. Exactly 50,000 rows
    row_count = len(df_50k)
    print(f"Total rows: {row_count}")
    if row_count != 50000:
        failures.append({'error': f'Expected 50000 rows, got {row_count}'})
        
    # 2. FLIVE patches = 0
    flive_patches = df_50k[(df_50k['dataset'] == 'flive') & (df_50k['content_type'] == 'patch')]
    if len(flive_patches) > 0:
        failures.append({'error': f'Found {len(flive_patches)} FLIVE patches'})

    # 3. Duplicate paths = 0
    dups = row_count - df_50k['norm_path'].nunique()
    if dups > 0:
        failures.append({'error': f'Found {dups} duplicate paths'})
        
    # 4. Missing targets = 0
    missing = df_50k['teacher_target_0_100'].isna().sum()
    if missing > 0:
        failures.append({'error': f'Found {missing} missing targets'})

    # 5. NaN/Inf = 0
    inf_count = np.isinf(df_50k['teacher_target_0_100']).sum()
    if inf_count > 0:
        failures.append({'error': f'Found {inf_count} Inf targets'})

    # 6. Target range 0~100
    if df_50k['teacher_target_0_100'].min() < 0 or df_50k['teacher_target_0_100'].max() > 100:
        failures.append({'error': 'Targets out of 0~100 range'})

    # Save failures
    pd.DataFrame(failures).to_csv(FINAL_FAILURES, index=False)

    # Save final manifest
    df_50k.to_csv(FINAL_MANIFEST, index=False)
    
    # Distribution Summary
    dist = df_50k.groupby('dataset')['teacher_target_0_100'].agg(['count', 'min', 'max', 'mean', 'std']).reset_index()
    dist.to_csv(FINAL_DISTRIBUTION, index=False)
    print("\nDataset Distribution:")
    print(dist)

    # Source Summary
    print("\nTarget Sources:")
    print(df_50k['target_source'].value_counts())

    # Leakage Audit
    OFFICIAL_TEST_MANIFEST = ROOT / "outputs/official_protocol_benchmark_manifest_20260604/official_benchmark_test_manifest.csv"
    df_off_test = pd.read_csv(OFFICIAL_TEST_MANIFEST)
    df_off_test['norm_path'] = df_off_test['image_path'].apply(norm_path)
    official_test_paths = set(df_off_test['norm_path'])
    
    leakage = len(set(df_50k['norm_path']).intersection(official_test_paths))
    pd.DataFrame([{"split": "Official Test", "overlap_count": leakage}]).to_csv(FINAL_LEAKAGE, index=False)
    if leakage > 0:
        print(f"WARNING: {leakage} official test rows leaked into final manifest!")

if __name__ == "__main__":
    main()
