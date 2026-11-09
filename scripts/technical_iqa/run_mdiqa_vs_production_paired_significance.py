import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
import os

# Paths
OUTPUT_DIR = Path("outputs/paired_production_mdiqa_significance_20260603")
PAIRED_PREDICTIONS = OUTPUT_DIR / "paired_predictions.csv"
SIGNIFICANCE_RESULTS = OUTPUT_DIR / "paired_significance_results.csv"
BOOTSTRAP_SUMMARY = OUTPUT_DIR / "bootstrap_distribution_summary.csv"
RANGE_ALIGNMENT = OUTPUT_DIR / "range_alignment_summary.csv"

def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks

def spearman(x, y):
    return spearmanr(x, y)[0]

def linear_aligned_mae(pred, target):
    if len(pred) < 2 or np.std(pred) == 0:
        return np.nan
    slope, intercept = np.polyfit(pred, target, 1)
    aligned = slope * pred + intercept
    return np.mean(np.abs(aligned - target))

def run_significance(iterations=5000, seed=20260603):
    if not PAIRED_PREDICTIONS.exists():
        print(f"Error: {PAIRED_PREDICTIONS} not found.")
        return

    df = pd.read_csv(PAIRED_PREDICTIONS)
    df = df[df["production_success"]].copy()
    
    if len(df) == 0:
        print("Error: No successful production predictions found.")
        return

    results = []
    bootstrap_rows = []
    range_rows = []
    
    rng = np.random.default_rng(seed)
    
    for ds in df["dataset"].unique():
        vdf = df[df["dataset"] == ds]
        n = len(vdf)
        print(f"Processing {ds} (n={n})...")
        
        mos = vdf["quality_score_0_100"].values
        mdiqa = vdf["mdiqa_pred_0_100"].values
        prod = vdf["production_pred_0_100"].values
        
        # Base metrics
        s_mdiqa = spearman(mdiqa, mos)
        s_prod = spearman(prod, mos)
        delta_s = s_mdiqa - s_prod
        
        p_mdiqa = pearsonr(mdiqa, mos)[0]
        p_prod = pearsonr(prod, mos)[0]
        delta_p = p_mdiqa - p_prod
        
        mae_mdiqa = np.mean(np.abs(mdiqa - mos))
        mae_prod = np.mean(np.abs(prod - mos))
        
        amae_mdiqa = linear_aligned_mae(mdiqa, mos)
        amae_prod = linear_aligned_mae(prod, mos)
        
        # Range check
        range_rows.append({
            "dataset": ds,
            "mdiqa_mean": np.mean(mdiqa),
            "mdiqa_std": np.std(mdiqa),
            "prod_mean": np.mean(prod),
            "prod_std": np.std(prod),
            "mos_mean": np.mean(mos),
            "mos_std": np.std(mos),
        })
        
        # Bootstrap
        deltas = []
        for i in range(iterations):
            indices = rng.choice(n, size=n, replace=True)
            b_mos = mos[indices]
            b_mdiqa = mdiqa[indices]
            b_prod = prod[indices]
            
            bs_mdiqa = spearman(b_mdiqa, b_mos)
            bs_prod = spearman(b_prod, b_mos)
            deltas.append(bs_mdiqa - bs_prod)
        
        deltas = np.array(deltas)
        mean_delta = np.mean(deltas)
        ci_low = np.percentile(deltas, 2.5)
        ci_high = np.percentile(deltas, 97.5)
        p_value = np.mean(deltas <= 0)
        prob_better = np.mean(deltas > 0)
        
        results.append({
            "dataset": ds,
            "n": n,
            "srcc_mdiqa": s_mdiqa,
            "srcc_prod": s_prod,
            "delta_srcc": delta_s,
            "srcc_p_value": p_value,
            "srcc_ci_low": ci_low,
            "srcc_ci_high": ci_high,
            "prob_mdiqa_better": prob_better,
            "plcc_mdiqa": p_mdiqa,
            "plcc_prod": p_prod,
            "mae_mdiqa": mae_mdiqa,
            "mae_prod": mae_prod,
            "amae_mdiqa": amae_mdiqa,
            "amae_prod": amae_prod,
        })
        
        bootstrap_rows.append({
            "dataset": ds,
            "mean_delta": mean_delta,
            "std_delta": np.std(deltas),
            "ci_95": f"[{ci_low:.4f}, {ci_high:.4f}]"
        })

    pd.DataFrame(results).to_csv(SIGNIFICANCE_RESULTS, index=False)
    pd.DataFrame(bootstrap_rows).to_csv(BOOTSTRAP_SUMMARY, index=False)
    pd.DataFrame(range_rows).to_csv(RANGE_ALIGNMENT, index=False)
    print(f"Saved significance results to {SIGNIFICANCE_RESULTS}")

if __name__ == "__main__":
    run_significance()
