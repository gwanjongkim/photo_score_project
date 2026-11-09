# Build Unknown / Real-Mobile Fallback Dry-run Manifest

import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path('outputs/hybrid_teacher_unknown_fallback_audit_20260603')
OUTPUT_MANIFEST = OUTPUT_DIR / 'unknown_fallback_dryrun_manifest.csv'

# 1. Load C6 Parity data (Real Mobile)
df_c6 = pd.read_csv('data/debug/c6_parity_20260527/android_c6_scores.csv')
search_dirs = ['data', 'forWeights', 'forWeights2']

def find_path(fname):
    for sdir in search_dirs:
        matches = list(Path(sdir).rglob(fname))
        if matches:
            return str(matches[0])
    return None

df_c6['image_path'] = df_c6['filename'].apply(find_path)

# 2. Build manifest rows
rows = []
for _, r in df_c6.iterrows():
    if r['image_path'] is None:
        continue
    rows.append({
        'image_path': r['image_path'],
        'source_file': 'data/debug/c6_parity_20260527/android_c6_scores.csv',
        'source_category': 'real_mobile',
        'dataset': 'real_mobile_c6',
        'content_type': 'image',
        'production_pred_0_100': r['existing_technical_score_100'],
        'quality_score_0_100': None, # No academic MOS
        'selected_teacher_target_0_100': r['existing_technical_score_100'],
        'selected_teacher_source': 'production_safety_fallback',
        'target_available': True,
        'target_error': '',
        'is_real_mobile': True,
        'is_proxy': False
    })

# 3. Add Test_weight (Product Gold) - Placeholder for production scores
# Since we don't have technical scores for Test_weight, we will use a proxy if needed
# OR we can just use the C6 set which is 50 rows. 50 is better than 0.
# I'll stick to the 50 C6 rows for now as they are confirmed scored.

df_dry = pd.DataFrame(rows)
df_dry.to_csv(OUTPUT_MANIFEST, index=False)
print(f"Wrote {len(df_dry)} rows to {OUTPUT_MANIFEST}")
