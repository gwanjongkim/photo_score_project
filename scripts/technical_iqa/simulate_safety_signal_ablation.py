import pandas as pd
import numpy as np
from pathlib import Path
import json
from PIL import Image
from scipy.signal import convolve2d
from scipy.ndimage import median_filter

OUT_DIR = Path("outputs/topiq_flive_safety_signal_ablation_20260608")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Data
preds_path = "outputs/topiq_flive_safety_gate_simulation_20260608/gated_predictions.csv"
df = pd.read_csv(preds_path)

base_col = "topiq_existing_flive_grid_s06_0_100"
gate_a_col = "gate_a_0_100"
prod_col = "production_existing_0_100"
topiq_col = "topiq_existing_0_100"
flive_col = "flive_existing_0_100"
mos_col = "mos_0_100"
mdiqa_col = "mdiqa_selected_0_100"

# Find base severe FPs
base_sfp_mask = (df[mos_col] < 40) & (df[base_col] > 65)
base_sfps = df[base_sfp_mask].copy()

# Find remaining severe FPs after Gate A
gate_a_sfp_mask = (df[mos_col] < 40) & (df[gate_a_col] > 65)
remaining_sfps = df[gate_a_sfp_mask].copy()

# Save scores of remaining FPs
scores_cols = ['image_path', mos_col, prod_col, topiq_col, flive_col, base_col, gate_a_col, mdiqa_col]
if 'student_14105_0_100' in df.columns:
    scores_cols.append('student_14105_0_100')
remaining_sfps[scores_cols].to_csv(OUT_DIR / "remaining_severe_fp_analysis.csv", index=False)

# MDIQA Oracle Gates
def oracle_1(row):
    m = row[mdiqa_col]
    b = row[base_col]
    if m < 50 and b > 65:
        return min(b, m + 10)
    return b

def oracle_2(row):
    m = row[mdiqa_col]
    b = row[base_col]
    if m < 55 and b > 65:
        return min(b, m + 10)
    return b

def oracle_3(row):
    m = row[mdiqa_col]
    b = row[base_col]
    if m < 60 and b > 65:
        return min(b, m + 5)
    return b

df['oracle_1'] = df.apply(oracle_1, axis=1)
df['oracle_2'] = df.apply(oracle_2, axis=1)
df['oracle_3'] = df.apply(oracle_3, axis=1)

from scipy.stats import spearmanr
def get_sfp_and_srcc(pred_col):
    pred = df[pred_col]
    mos = df[mos_col]
    sfp = ((mos < 40) & (pred > 65)).sum()
    srcc = spearmanr(mos, pred)[0]
    return int(sfp), float(srcc)

oracle_metrics = []
for col in ['oracle_1', 'oracle_2', 'oracle_3']:
    sfp, srcc = get_sfp_and_srcc(col)
    oracle_metrics.append({'gate': col, 'severe_fp': sfp, 'srcc': srcc})

pd.DataFrame(oracle_metrics).to_csv(OUT_DIR / "mdiqa_oracle_gate_metrics.csv", index=False)

# Simple Metrics Probe
def compute_simple_metrics(img_path):
    try:
        img = Image.open(img_path).convert('L')
        gray = np.array(img, dtype=np.float32)
        
        # Simple laplacian filter
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        lap = convolve2d(gray, kernel, mode='valid')
        lap_var = np.var(lap)
        
        # Tenengrad
        sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        sx = convolve2d(gray, sobelx, mode='valid')
        sy = convolve2d(gray, sobely, mode='valid')
        tenengrad = np.mean(sx**2 + sy**2)
        
        # Center crop sharpness
        h, w = gray.shape
        cy, cx = h//2, w//2
        crop_size = min(h, w) // 4
        center = gray[cy-crop_size:cy+crop_size, cx-crop_size:cx+crop_size]
        center_lap = convolve2d(center, kernel, mode='valid')
        center_lap_var = np.var(center_lap)
        
        # Noise
        median = median_filter(gray, size=3)
        noise_est = np.std(gray - median)
        
        return {
            'laplacian_var': float(lap_var),
            'tenengrad': float(tenengrad),
            'center_lap_var': float(center_lap_var),
            'noise_est': float(noise_est)
        }
    except Exception:
        return None

sample_df = df.sample(n=100, random_state=42)
# Ensure SFPs are in the sample
sfp_paths = base_sfps['image_path'].tolist()
sample_paths = list(set(sample_df['image_path'].tolist() + sfp_paths))

probe_results = []
for p in sample_paths:
    m = compute_simple_metrics(p)
    if m:
        row = df[df['image_path'] == p].iloc[0]
        m['image_path'] = p
        m['mos'] = row[mos_col]
        m['base_pred'] = row[base_col]
        m['is_sfp'] = p in sfp_paths
        probe_results.append(m)

probe_df = pd.DataFrame(probe_results)
probe_df.to_csv(OUT_DIR / "simple_quality_metric_probe.csv", index=False)

sfp_fixed_by_oracle_1 = get_sfp_and_srcc('oracle_1')[0] == 0

rec = {
    "oracle_fixes_sfp": sfp_fixed_by_oracle_1,
    "best_oracle_sfp": min(m['severe_fp'] for m in oracle_metrics),
    "decision": "TOPIQ_FLIVE_NEEDS_LEARNED_SAFETY_MODEL"
}

with open(OUT_DIR / "ablation_recommendation.json", "w") as f:
    json.dump(rec, f, indent=2)

print("Ablation script complete.")
