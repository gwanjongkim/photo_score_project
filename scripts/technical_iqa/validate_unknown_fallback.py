# Unknown / Real-Mobile Fallback Validation Script

import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path('outputs/hybrid_teacher_unknown_fallback_audit_20260603')
OUTPUT_MANIFEST = OUTPUT_DIR / 'unknown_fallback_dryrun_manifest.csv'
OUTPUT_VALIDATION = OUTPUT_DIR / 'unknown_fallback_validation.csv'
OUTPUT_DIST = OUTPUT_DIR / 'unknown_fallback_distribution_summary.csv'
OUTPUT_SAFETY = OUTPUT_DIR / 'unknown_fallback_safety_cases.csv'

# 1. Load Manifest
df = pd.read_csv(OUTPUT_MANIFEST)

# 2. Add Label Info for safety checks
df_c6 = pd.read_csv('data/debug/c6_parity_20260527/android_c6_scores.csv')
labels_map = {}
search_dirs = ['data/Test_weight']

def find_label(fname):
    for sdir in search_dirs:
        matches = list(Path(sdir).rglob(fname))
        if matches:
            return matches[0].parent.name
    return None

df['label_name'] = df['image_path'].apply(lambda x: find_label(Path(x).name))

# 3. Target Distribution
stats = df['selected_teacher_target_0_100'].agg(['count', 'min', 'max', 'mean', 'std']).to_frame().T
quantiles = df['selected_teacher_target_0_100'].quantile([0.05, 0.50, 0.95]).to_frame().T
quantiles.columns = ['p05', 'p50', 'p95']
dist_summary = pd.concat([stats, quantiles], axis=1)

# 4. Clipping and Boundaries
df['is_clipped'] = (df['selected_teacher_target_0_100'] <= 0) | (df['selected_teacher_target_0_100'] >= 100)
dist_summary['clipping_count'] = df['is_clipped'].sum()
dist_summary['near_0'] = (df['selected_teacher_target_0_100'] < 5).sum()
dist_summary['near_100'] = (df['selected_teacher_target_0_100'] > 95).sum()

# 5. Safety Checks (Approximate using label_name)
# Bad -> MOS proxy 20
# Soso -> MOS proxy 50
# Good -> MOS proxy 80
def get_mos_proxy(label):
    if label == 'Bad': return 20.0
    if label == 'Soso': return 50.0
    if label == 'Good': return 80.0
    return None

df['mos_proxy'] = df['label_name'].apply(get_mos_proxy)
df_safety = df.dropna(subset=['mos_proxy']).copy()

if not df_safety.empty:
    df_safety['is_fp'] = (df_safety['mos_proxy'] < 40) & (df_safety['selected_teacher_target_0_100'] > 60)
    df_safety['is_severe_fp'] = (df_safety['mos_proxy'] < 40) & (df_safety['selected_teacher_target_0_100'] > 65)
    df_safety['is_fn'] = (df_safety['mos_proxy'] > 75) & (df_safety['selected_teacher_target_0_100'] < 45)
    
    safety_results = df_safety[['is_fp', 'is_severe_fp', 'is_fn']].sum().to_frame().T
    safety_results.to_csv(OUTPUT_SAFETY, index=False)
    
    # MOS alignment (proxy)
    srcc = df_safety['mos_proxy'].corr(df_safety['selected_teacher_target_0_100'], method='spearman')
    plcc = df_safety['mos_proxy'].corr(df_safety['selected_teacher_target_0_100'], method='pearson')
    mae = (df_safety['mos_proxy'] - df_safety['selected_teacher_target_0_100']).abs().mean()
    validation = pd.DataFrame([{'n_labeled': len(df_safety), 'srcc_proxy': srcc, 'plcc_proxy': plcc, 'mae_proxy': mae}])
    validation.to_csv(OUTPUT_VALIDATION, index=False)
else:
    pd.DataFrame().to_csv(OUTPUT_SAFETY)
    pd.DataFrame().to_csv(OUTPUT_VALIDATION)

dist_summary.to_csv(OUTPUT_DIST, index=False)
print("Validation complete.")
