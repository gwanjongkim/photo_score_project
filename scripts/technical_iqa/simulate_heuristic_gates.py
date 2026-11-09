import pandas as pd
import numpy as np
from pathlib import Path
import json
from scipy.stats import spearmanr, pearsonr
from scipy.ndimage import laplace, median_filter, sobel
from PIL import Image
import concurrent.futures
import time
import os

OUT_DIR = Path("outputs/topiq_flive_heuristic_gate_full_simulation_20260608")
OUT_DIR.mkdir(parents=True, exist_ok=True)

preds_path = "outputs/official_protocol_technical_signal_benchmark_20260604/official_signal_predictions.csv"
df = pd.read_csv(preds_path)

# Ensure correct base col
base_col = "topiq_existing_flive_grid_s06_0_100"
mos_col = "mos_0_100"

def fast_metrics(img_path):
    try:
        img = Image.open(img_path).convert('L')
        gray = np.array(img, dtype=np.float32)
        
        lap = laplace(gray)
        lap_var = np.var(lap)
        
        sx = sobel(gray, axis=0, mode='constant')
        sy = sobel(gray, axis=1, mode='constant')
        tenengrad = np.mean(sx**2 + sy**2)
        
        med = median_filter(gray, size=3)
        noise_est = np.std(gray - med)
        
        h, w = gray.shape
        cy, cx = h//2, w//2
        crop_size = min(h, w) // 4
        center = gray[cy-crop_size:cy+crop_size, cx-crop_size:cx+crop_size]
        center_lap = laplace(center)
        center_lap_var = np.var(center_lap)
        
        center_med = median_filter(center, size=3)
        center_noise_est = np.std(center - center_med)
        
        return {
            'image_path': img_path,
            'laplacian_var': float(lap_var),
            'tenengrad': float(tenengrad),
            'noise_est': float(noise_est),
            'center_lap_var': float(center_lap_var),
            'center_noise_est': float(center_noise_est)
        }
    except Exception as e:
        print(f"Error {img_path}: {e}")
        return {
            'image_path': img_path,
            'laplacian_var': 0.0,
            'tenengrad': 0.0,
            'noise_est': 0.0,
            'center_lap_var': 0.0,
            'center_noise_est': 0.0
        }

image_paths = df['image_path'].tolist()

print(f"Extracting metrics for {len(image_paths)} images...")
t0 = time.time()
results = []
with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    for res in executor.map(fast_metrics, image_paths, chunksize=100):
        results.append(res)
print(f"Extraction done in {time.time()-t0:.2f}s")

metrics_df = pd.DataFrame(results)
metrics_df.to_csv(OUT_DIR / "image_quality_heuristic_metrics.csv", index=False)

# Merge
df = pd.merge(df, metrics_df, on='image_path', how='left')

# Gate definitions
def gate_a(row):
    if row[base_col] > 65 and row['noise_est'] > 10:
        return min(row[base_col], 60)
    return row[base_col]

def gate_b(row):
    if row[base_col] > 65 and row['noise_est'] > 8:
        return min(row[base_col], 60)
    return row[base_col]

def gate_c(row):
    if row[base_col] > 65 and row['laplacian_var'] > 1500:
        return min(row[base_col], 60)
    return row[base_col]

def gate_d(row):
    if row[base_col] > 65 and row['noise_est'] > 8 and row['laplacian_var'] > 1000:
        return min(row[base_col], 60)
    return row[base_col]

def gate_e(row):
    if row[base_col] > 65 and (row['noise_est'] > 10 or row['laplacian_var'] > 1500):
        return min(row[base_col], 60)
    return row[base_col]

def gate_f(row):
    if row[base_col] > 65 and row['center_noise_est'] > 8:
        return min(row[base_col], 60)
    return row[base_col]

def gate_g(row):
    if row[base_col] > 65 and row['center_lap_var'] < 50:
        return min(row[base_col], 60)
    return row[base_col]

gates = {
    'gate_a_0_100': gate_a,
    'gate_b_0_100': gate_b,
    'gate_c_0_100': gate_c,
    'gate_d_0_100': gate_d,
    'gate_e_0_100': gate_e,
    'gate_f_0_100': gate_f,
    'gate_g_0_100': gate_g
}

for k, f in gates.items():
    df[k] = df.apply(f, axis=1)

df.to_csv(OUT_DIR / "heuristic_gate_predictions.csv", index=False)

def get_metrics(df_sub, pred_col):
    mos = df_sub[mos_col].to_numpy()
    pred = df_sub[pred_col].to_numpy()
    n = len(mos)
    if n < 2: return {}
    mae = np.mean(np.abs(mos - pred))
    rmse = np.sqrt(np.mean((mos - pred)**2))
    srcc = spearmanr(mos, pred)[0]
    plcc = pearsonr(mos, pred)[0]
    fp = ((mos < 40) & (pred > 60)).sum()
    severe_fp = ((mos < 40) & (pred > 65)).sum()
    fn = ((mos > 75) & (pred < 45)).sum()
    return {
        "n": n, "srcc": float(srcc), "plcc": float(plcc), "mae": float(mae), "rmse": float(rmse),
        "pred_min": float(np.min(pred)), "pred_max": float(np.max(pred)), "pred_mean": float(np.mean(pred)), "pred_std": float(np.std(pred)),
        "target_min": float(np.min(mos)), "target_max": float(np.max(mos)), "target_mean": float(np.mean(mos)), "target_std": float(np.std(mos)),
        "fp_count": int(fp), "severe_fp_count": int(severe_fp), "fn_count": int(fn)
    }

eval_cols = [base_col] + list(gates.keys())
overall = []
ds_metrics = []

for c in eval_cols:
    sig_name = c.replace("_0_100", "")
    m = get_metrics(df, c)
    m["signal"] = sig_name
    overall.append(m)
    for ds in df['dataset'].unique():
        sub = df[df['dataset'] == ds]
        m_ds = get_metrics(sub, c)
        m_ds["signal"] = sig_name
        m_ds["dataset"] = ds
        ds_metrics.append(m_ds)

pd.DataFrame(overall).to_csv(OUT_DIR / "heuristic_gate_metrics_overall.csv", index=False)
pd.DataFrame(ds_metrics).to_csv(OUT_DIR / "heuristic_gate_metrics_datasetwise.csv", index=False)

mod_rows = []
for k in gates.keys():
    g_name = k.replace("_0_100", "")
    diff = df[k] - df[base_col]
    mod = diff != 0
    mod_count = mod.sum()
    mod_rows.append({
        "gate": g_name,
        "modified_count": int(mod_count),
        "modified_percent": float(mod_count / len(df) * 100),
        "avg_score_change": float(diff[mod].mean()) if mod_count > 0 else 0.0,
        "max_score_decrease": float(diff.min()) if mod_count > 0 else 0.0
    })

pd.DataFrame(mod_rows).to_csv(OUT_DIR / "heuristic_gate_modified_rows.csv", index=False)

# severe fp audit
base_sfp = (df[mos_col] < 40) & (df[base_col] > 65)
sfp_audit = []
for k in gates.keys():
    g_name = k.replace("_0_100", "")
    still_sfp = (df[mos_col] < 40) & (df[k] > 65)
    fixed = base_sfp & (~still_sfp)
    new_sfp = (~base_sfp) & still_sfp
    
    sfp_audit.append({
        "gate": g_name,
        "original_sfp_fixed": int(fixed.sum()),
        "new_sfp_introduced": int(new_sfp.sum()),
        "total_sfp_remaining": int(still_sfp.sum())
    })
pd.DataFrame(sfp_audit).to_csv(OUT_DIR / "heuristic_gate_severe_fp_audit.csv", index=False)

base_srcc = next(m["srcc"] for m in overall if m["signal"] == base_col.replace("_0_100", ""))

ranking_data = []
for m in overall:
    sig = m["signal"]
    if sig.startswith("gate_"):
        audit_row = next(a for a in sfp_audit if a["gate"] == sig)
        mod_row = next(r for r in mod_rows if r["gate"] == sig)
        
        is_valid = audit_row["total_sfp_remaining"] == 0 and (base_srcc - m["srcc"] <= 0.01)
        srcc_drop = base_srcc - m["srcc"]
        
        ranking_data.append({
            "gate": sig,
            "srcc": m["srcc"],
            "srcc_drop_vs_base": srcc_drop,
            "severe_fp": audit_row["total_sfp_remaining"],
            "modified_rows": mod_row["modified_count"],
            "valid": is_valid
        })

df_rank = pd.DataFrame(ranking_data).sort_values(by=["valid", "srcc_drop_vs_base"], ascending=[False, True])
# Ignore empty df_rank to avoid crashing if empty
if len(df_rank) > 0:
    df_rank.to_csv(OUT_DIR / "gate_candidate_ranking.csv", index=False)

best_gate = df_rank[df_rank["valid"]].iloc[0] if len(df_rank[df_rank["valid"]]) > 0 else None

if best_gate is not None:
    decision = "TOPIQ_FLIVE_HEURISTIC_GATE_READY_FOR_LOG_ONLY_PROFILING"
    rec = {
        "best_gate": best_gate["gate"],
        "decision": decision,
        "base_severe_fp": int(base_sfp.sum()),
        "gate_srcc": float(best_gate["srcc"]),
        "srcc_drop_vs_base": float(best_gate["srcc_drop_vs_base"]),
        "modified_rows": int(best_gate["modified_rows"]),
        "all_base_sfp_fixed": bool(best_gate["severe_fp"] == 0)
    }
else:
    decision = "TOPIQ_FLIVE_HEURISTIC_GATE_NEEDS_TUNING"
    rec = {
        "best_gate": None,
        "decision": decision,
        "base_severe_fp": int(base_sfp.sum()),
        "reason": "No gate achieved 0 severe FP while maintaining acceptable SRCC loss <= 0.01."
    }

with open(OUT_DIR / "heuristic_gate_recommendation.json", "w") as f:
    json.dump(rec, f, indent=2)

print("done")
