import pandas as pd
import numpy as np
from pathlib import Path
import json
from scipy.stats import spearmanr, pearsonr

OUT_DIR = Path("outputs/topiq_flive_safety_gate_simulation_20260608")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Data
preds_path = "outputs/official_protocol_mdiqa_signal_benchmark_20260604/official_mdiqa_vs_candidates_predictions.csv"
df = pd.read_csv(preds_path)

# Extract essential columns
base_col = "topiq_existing_flive_grid_s06_0_100"
prod_col = "production_existing_0_100"
topiq_col = "topiq_existing_0_100"
flive_col = "flive_existing_0_100"
mos_col = "mos_0_100"
mdiqa_col = "mdiqa_selected_0_100"

# Create a clean df for gate simulation
sim_df = df[['dataset', 'image_path', mos_col, prod_col, topiq_col, flive_col, base_col]].copy()
if mdiqa_col in df.columns:
    sim_df[mdiqa_col] = df[mdiqa_col]

# Check base formula
sim_df['recalculated_base'] = 0.6 * sim_df[topiq_col] + 0.4 * sim_df[flive_col]
max_diff = (sim_df['recalculated_base'] - sim_df[base_col]).abs().max()
if max_diff > 0.1:
    print(f"WARNING: Recalculated base diff is large: {max_diff}")
else:
    print("Base TOPIQ+FLIVE formula verified.")

# 2. Gate Definitions
# Gate A: production low + candidate high cap10
def gate_a(row):
    b = row[base_col]
    p = row[prod_col]
    if p < 45 and b > 65:
        return min(b, p + 10)
    return b

# Gate B: production low + candidate high cap12
def gate_b(row):
    b = row[base_col]
    p = row[prod_col]
    if p < 50 and b > 65:
        return min(b, p + 12)
    return b

# Gate C: production very low + candidate optimistic
def gate_c(row):
    b = row[base_col]
    p = row[prod_col]
    if p < 50 and (b - p) > 20:
        return min(b, p + 15)
    return b

# Gate D: FLIVE low + TOPIQ high disagreement
def gate_d(row):
    b = row[base_col]
    f = row[flive_col]
    t = row[topiq_col]
    if f < 50 and t > 65:
        return min(b, max(f + 12, 55))
    return b

# Gate E: production conservative cap
def gate_e(row):
    b = row[base_col]
    p = row[prod_col]
    if p < 55 and b > 65:
        return min(b, p + 15)
    return b

# Gate F: soft blend fallback
def gate_f(row):
    b = row[base_col]
    p = row[prod_col]
    if p < 50 and b > 65:
        return 0.7 * p + 0.3 * b
    return b

# Gate G: two-stage suspicious optimism gate
def gate_g(row):
    b = row[base_col]
    p = row[prod_col]
    if p < 45 and b > 65:
        return min(b, p + 10)
    elif p < 50 and b > 70:
        return min(b, p + 12)
    return b

gates = {
    "gate_a": gate_a,
    "gate_b": gate_b,
    "gate_c": gate_c,
    "gate_d": gate_d,
    "gate_e": gate_e,
    "gate_f": gate_f,
    "gate_g": gate_g,
}

for g_name, g_func in gates.items():
    sim_df[f"{g_name}_0_100"] = sim_df.apply(g_func, axis=1)

# Save predictions
sim_df.to_csv(OUT_DIR / "gated_predictions.csv", index=False)

# 3. Compute Metrics
def get_metrics(df_sub, pred_col):
    mos = df_sub[mos_col].to_numpy()
    pred = df_sub[pred_col].to_numpy()
    
    n = len(mos)
    mae = np.mean(np.abs(mos - pred))
    rmse = np.sqrt(np.mean((mos - pred)**2))
    
    srcc = spearmanr(mos, pred)[0] if n > 1 else np.nan
    plcc = pearsonr(mos, pred)[0] if n > 1 else np.nan
    
    fp = ((mos < 40) & (pred > 60)).sum()
    severe_fp = ((mos < 40) & (pred > 65)).sum()
    fn = ((mos > 75) & (pred < 45)).sum()
    
    return {
        "n": n,
        "srcc": srcc,
        "plcc": plcc,
        "mae": mae,
        "rmse": rmse,
        "pred_min": np.min(pred),
        "pred_max": np.max(pred),
        "pred_mean": np.mean(pred),
        "pred_std": np.std(pred),
        "target_min": np.min(mos),
        "target_max": np.max(mos),
        "target_mean": np.mean(mos),
        "target_std": np.std(mos),
        "fp_count": int(fp),
        "severe_fp_count": int(severe_fp),
        "fn_count": int(fn)
    }

eval_cols = [prod_col, base_col, topiq_col, flive_col]
if mdiqa_col in sim_df.columns:
    eval_cols.append(mdiqa_col)
eval_cols += [f"{g}_0_100" for g in gates.keys()]

overall_metrics = []
ds_metrics = []

for c in eval_cols:
    sig_name = c.replace("_0_100", "")
    # Overall
    m = get_metrics(sim_df, c)
    m["signal"] = sig_name
    overall_metrics.append(m)
    
    # Dataset-wise
    for ds in sim_df["dataset"].unique():
        sub = sim_df[sim_df["dataset"] == ds]
        m_ds = get_metrics(sub, c)
        m_ds["signal"] = sig_name
        m_ds["dataset"] = ds
        ds_metrics.append(m_ds)

pd.DataFrame(overall_metrics).to_csv(OUT_DIR / "gate_metrics_overall.csv", index=False)
pd.DataFrame(ds_metrics).to_csv(OUT_DIR / "gate_metrics_datasetwise.csv", index=False)

# 4. Modified Rows Analysis
mod_rows = []
for g_name in gates.keys():
    col = f"{g_name}_0_100"
    diff = sim_df[col] - sim_df[base_col]
    modified = diff != 0
    
    n_mod = modified.sum()
    pct_mod = (n_mod / len(sim_df)) * 100
    avg_diff = diff[modified].mean() if n_mod > 0 else 0
    max_drop = diff.min()
    
    mod_rows.append({
        "gate": g_name,
        "modified_count": n_mod,
        "modified_percent": pct_mod,
        "avg_score_change_on_modified": avg_diff,
        "max_score_decrease": max_drop
    })

pd.DataFrame(mod_rows).to_csv(OUT_DIR / "gate_modified_rows.csv", index=False)

# 5. Severe FP Fix Audit
base_sfp_mask = (sim_df[mos_col] < 40) & (sim_df[base_col] > 65)
sfp_audit = []
for g_name in gates.keys():
    col = f"{g_name}_0_100"
    still_sfp = (sim_df[mos_col] < 40) & (sim_df[col] > 65)
    fixed = base_sfp_mask & (~still_sfp)
    new_sfp = (~base_sfp_mask) & still_sfp
    
    sfp_audit.append({
        "gate": g_name,
        "original_sfp_fixed": int(fixed.sum()),
        "new_sfp_introduced": int(new_sfp.sum()),
        "total_sfp_remaining": int(still_sfp.sum())
    })

pd.DataFrame(sfp_audit).to_csv(OUT_DIR / "severe_fp_fix_audit.csv", index=False)

# 6. Candidate Ranking & Recommendation
# Filter valid gates: severe FP == 0
prod_srcc = next(m["srcc"] for m in overall_metrics if m["signal"] == prod_col.replace("_0_100", ""))
base_srcc = next(m["srcc"] for m in overall_metrics if m["signal"] == base_col.replace("_0_100", ""))

ranking_data = []
for m in overall_metrics:
    sig = m["signal"]
    if sig.startswith("gate_"):
        audit_row = next(a for a in sfp_audit if a["gate"] == sig)
        mod_row = next(r for r in mod_rows if r["gate"] == sig)
        
        is_valid = audit_row["total_sfp_remaining"] == 0 and m["srcc"] > prod_srcc
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
df_rank.to_csv(OUT_DIR / "gate_candidate_ranking.csv", index=False)

best_gate = df_rank[df_rank["valid"]].iloc[0] if len(df_rank[df_rank["valid"]]) > 0 else None

if best_gate is not None:
    decision = "TOPIQ_FLIVE_GATE_READY_FOR_LOG_ONLY_PROFILING"
    rec = {
        "best_gate": best_gate["gate"],
        "decision": decision,
        "base_severe_fp": int(base_sfp_mask.sum()),
        "gate_srcc": float(best_gate["srcc"]),
        "srcc_drop_vs_base": float(best_gate["srcc_drop_vs_base"]),
        "modified_rows": int(best_gate["modified_rows"]),
        "all_base_sfp_fixed": bool(best_gate["severe_fp"] == 0)
    }
else:
    decision = "TOPIQ_FLIVE_GATE_NEEDS_TUNING"
    rec = {
        "best_gate": None,
        "decision": decision,
        "base_severe_fp": int(base_sfp_mask.sum()),
        "reason": "No gate achieved 0 severe FP while maintaining acceptable SRCC."
    }

with open(OUT_DIR / "gate_recommendation.json", "w") as f:
    json.dump(rec, f, indent=2)

print("Simulation script complete.")
