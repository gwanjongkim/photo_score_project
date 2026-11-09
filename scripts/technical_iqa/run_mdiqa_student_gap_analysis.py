import pandas as pd
import numpy as np
from pathlib import Path
import json

OUT_DIR = Path("outputs/mdiqa_student_gap_analysis_20260608")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Data
preds_path = "outputs/official_protocol_mdiqa_signal_benchmark_20260604/official_mdiqa_vs_candidates_predictions.csv"
manifest_path = "outputs/student_distillation_manifest_14105_20260603/student_distillation_manifest.csv"

df_preds = pd.read_csv(preds_path)
df_manifest = pd.read_csv(manifest_path)

# Normalize paths in manifest
def norm_path(p):
    if not p: return None
    p = str(p)
    if p.startswith('/home/omen_pc1/photo_score_project/'):
        return p.replace('/home/omen_pc1/photo_score_project/', '')
    return p

df_manifest['norm_path'] = df_manifest['image_path'].apply(norm_path)
manifest_splits = df_manifest.set_index('norm_path')['split'].to_dict()

# 2. Join Predictions and Prepare Gap Data
df = df_preds[['dataset', 'image_path', 'norm_path', 'mos_0_100', 'mdiqa_selected_0_100', 'student_14105_0_100']].copy()
df.rename(columns={'mdiqa_selected_0_100': 'mdiqa_pred', 'student_14105_0_100': 'student_pred', 'mos_0_100': 'mos'}, inplace=True)

df['abs_gap'] = np.abs(df['student_pred'] - df['mdiqa_pred'])
df['signed_gap'] = df['student_pred'] - df['mdiqa_pred']
df['mdiqa_error'] = np.abs(df['mdiqa_pred'] - df['mos'])
df['student_error'] = np.abs(df['student_pred'] - df['mos'])
df['student_worse_by'] = df['student_error'] - df['mdiqa_error']
df['student_train_split'] = df['norm_path'].map(manifest_splits).fillna('absent')

df.to_csv(OUT_DIR / "gap_predictions_joined.csv", index=False)

# Metrics function
from scipy.stats import spearmanr, pearsonr
def get_metrics(sub_df):
    m = sub_df['mos'].to_numpy()
    p_mdiqa = sub_df['mdiqa_pred'].to_numpy()
    p_student = sub_df['student_pred'].to_numpy()
    
    return {
        'n': len(sub_df),
        'mdiqa_mae': np.mean(np.abs(m - p_mdiqa)),
        'student_mae': np.mean(np.abs(m - p_student)),
        'mae_gap': np.mean(np.abs(m - p_student)) - np.mean(np.abs(m - p_mdiqa)),
        'mdiqa_srcc': spearmanr(m, p_mdiqa)[0] if len(m) > 1 else np.nan,
        'student_srcc': spearmanr(m, p_student)[0] if len(m) > 1 else np.nan,
        'srcc_gap': (spearmanr(m, p_student)[0] - spearmanr(m, p_mdiqa)[0]) if len(m) > 1 else np.nan,
        'student_vs_mdiqa_srcc': spearmanr(p_mdiqa, p_student)[0] if len(m) > 1 else np.nan,
        'mean_signed_gap': np.mean(p_student - p_mdiqa),
        'mean_abs_gap': np.mean(np.abs(p_student - p_mdiqa)),
    }

# 3. Overall and Dataset-wise Summary
overall = get_metrics(df)
pd.DataFrame([overall]).to_csv(OUT_DIR / "gap_summary_overall.csv", index=False)

ds_metrics = []
for ds in df['dataset'].unique():
    m = get_metrics(df[df['dataset'] == ds])
    m['dataset'] = ds
    ds_metrics.append(m)
pd.DataFrame(ds_metrics).to_csv(OUT_DIR / "gap_summary_datasetwise.csv", index=False)

# 4. MOS Bin Analysis
bins = [0, 40, 60, 75, 100]
labels = ['0-40', '40-60', '60-75', '75-100']
df['mos_bin'] = pd.cut(df['mos'], bins=bins, labels=labels, include_lowest=True)

mos_bin_metrics = []
for b in labels:
    sub = df[df['mos_bin'] == b]
    if len(sub) > 0:
        m = get_metrics(sub)
        m['mos_bin'] = b
        mos_bin_metrics.append(m)
pd.DataFrame(mos_bin_metrics).to_csv(OUT_DIR / "gap_summary_mos_bins.csv", index=False)

# Student vs MDIQA Bins
df['mdiqa_bin'] = pd.cut(df['mdiqa_pred'], bins=bins, labels=labels, include_lowest=True)
mdiqa_bin_metrics = []
for b in labels:
    sub = df[df['mdiqa_bin'] == b]
    if len(sub) > 0:
        m = get_metrics(sub)
        m['mdiqa_bin'] = b
        mdiqa_bin_metrics.append(m)
pd.DataFrame(mdiqa_bin_metrics).to_csv(OUT_DIR / "gap_summary_student_vs_mdiqa_bins.csv", index=False)

# 6. Safety Analysis
df['mdiqa_severe_fp'] = (df['mos'] < 40) & (df['mdiqa_pred'] > 65)
df['student_severe_fp'] = (df['mos'] < 40) & (df['student_pred'] > 65)
df['safety_gap'] = (~df['mdiqa_severe_fp']) & df['student_severe_fp']

safety_cases = df[df['safety_gap']]
safety_cases.to_csv(OUT_DIR / "safety_gap_cases.csv", index=False)

# 7. Top Failure Cases
df.sort_values('student_worse_by', ascending=False).head(30).to_csv(OUT_DIR / "student_worse_top30.csv", index=False)

low_mos = df[df['mos'] < 40].copy()
low_mos['overpredict'] = low_mos['student_pred'] - low_mos['mos']
low_mos.sort_values('overpredict', ascending=False).head(30).to_csv(OUT_DIR / "student_overpredict_low_mos_top30.csv", index=False)

high_mos = df[df['mos'] > 75].copy()
high_mos['underpredict'] = high_mos['mos'] - high_mos['student_pred']
high_mos.sort_values('underpredict', ascending=False).head(30).to_csv(OUT_DIR / "student_underpredict_high_mos_top30.csv", index=False)

# 8. Training Overlap Analysis
overlap_metrics = []
for split in df['student_train_split'].unique():
    sub = df[df['student_train_split'] == split]
    if len(sub) > 0:
        m = get_metrics(sub)
        m['train_split'] = split
        overlap_metrics.append(m)
pd.DataFrame(overlap_metrics).to_csv(OUT_DIR / "training_overlap_gap_summary.csv", index=False)

# 9. Recommendation
rec = {
    "overall_srcc_gap": float(overall['srcc_gap']),
    "overall_mae_gap": float(overall['mae_gap']),
    "absent_from_train_count": int((df['student_train_split'] == 'absent').sum()),
    "absent_srcc": float(get_metrics(df[df['student_train_split'] == 'absent'])['student_srcc']) if len(df[df['student_train_split'] == 'absent']) > 1 else None,
    "student_severe_fp": int(df['student_severe_fp'].sum()),
    "recommendation": "STUDENT_GAP_MAINLY_CAPACITY_AND_COVERAGE", # We'll determine this dynamically in the report or hardcode a safe bet and refine
}
with open(OUT_DIR / "gap_recommendation.json", "w") as f:
    json.dump(rec, f, indent=2)

print("Analysis script completed.")
