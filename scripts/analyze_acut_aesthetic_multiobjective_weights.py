#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Re-evaluate A-cut aesthetic ensemble weights using a combined multi-objective goal.
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from itertools import product
from tqdm import tqdm

def load_benchmarks(ava_path, aadb_path):
    print(f"Loading AVA from {ava_path}...")
    df_ava = pd.read_csv(ava_path)
    print(f"Loading AADB from {aadb_path}...")
    df_aadb = pd.read_csv(aadb_path)
    
    # Scale AADB ground truth to 1-10 if it's 0-1
    if df_aadb['gt_score'].max() <= 1.1:
        df_aadb['gt_score'] = df_aadb['gt_score'] * 10.0
        
    return df_ava, df_aadb

def parse_device_log(log_path):
    if not os.path.exists(log_path):
        return None
    
    print(f"Parsing device log from {log_path}...")
    data = []
    # [AcutAestheticCompare] norms nima=... rgnet=... alamp=... | oldNoIcaa=... newUncal=... calibCandidate=... alampHeavy=... nimaHeavy=... rgnetHeavy=...
    pattern = re.compile(r"\[AcutAestheticCompare\] norms nima=([\d.]+) rgnet=([\d.]+) alamp=([\d.]+) \|")
    
    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                nima, rgnet, alamp = map(float, match.groups())
                data.append({'nima': nima, 'rgnet': rgnet, 'alamp': alamp})
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    df['image_index'] = range(len(df))
    return df

def get_metrics(gt, pred):
    srcc, _ = spearmanr(gt, pred)
    plcc, _ = pearsonr(gt, pred)
    mae = np.mean(np.abs(gt - pred))
    mse = np.mean((gt - pred)**2)
    
    p10 = np.percentile(pred, 10)
    p90 = np.percentile(pred, 90)
    spread = p90 - p10
    
    return {
        'srcc': srcc,
        'plcc': plcc,
        'mae': mae,
        'mse': mse,
        'mean': np.mean(pred),
        'std': np.std(pred),
        'min': np.min(pred),
        'max': np.max(pred),
        'p10': p10,
        'p90': p90,
        'p90_minus_p10': spread
    }

def get_spread_quality(spread):
    # Prefer P90-P10 around 0.25 to 0.55 normalized score.
    # Penalize P90-P10 < 0.10.
    # Penalize P90-P10 > 0.70.
    if spread < 0.10:
        return 0.1 * (spread / 0.10)
    if spread > 0.70:
        return 0.1 * (1.0 - (spread - 0.70) / 0.30)
    
    if 0.25 <= spread <= 0.55:
        return 1.0
    
    if spread < 0.25:
        # linear scale from 0.1 at 0.1 to 1.0 at 0.25
        return 0.1 + 0.9 * (spread - 0.10) / (0.25 - 0.10)
    else:
        # linear scale from 1.0 at 0.55 to 0.1 at 0.70
        return 1.0 - 0.9 * (spread - 0.55) / (0.70 - 0.55)

def normalize_series(s, higher_is_better=True):
    s_min, s_max = s.min(), s.max()
    if s_max == s_min:
        return s * 0.0 + 1.0
    if higher_is_better:
        return (s - s_min) / (s_max - s_min)
    else:
        return (s_max - s) / (s_max - s_min)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ava", default="outputs/aesthetic_cross_predictions_for_weight_search_20260603/aligned_ava_complete_primary.csv")
    parser.add_argument("--aadb", default="outputs/aesthetic_cross_predictions_for_weight_search_20260603/aligned_aadb_complete_primary.csv")
    parser.add_argument("--log", default="data/debug/acut_aesthetic_compare_device_log_20260603.txt")
    parser.add_argument("--user", default="data/debug/acut_user_rank_input_20260603.csv")
    parser.add_argument("--out_dir", default="outputs/acut_aesthetic_weight_multiobjective_20260603")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_ava, df_aadb = load_benchmarks(args.ava, args.aadb)
    df_device = parse_device_log(args.log)
    
    df_user = None
    if os.path.exists(args.user):
        print(f"Loading user rank from {args.user}...")
        df_user = pd.read_csv(args.user)
        if df_device is not None:
            df_user = pd.merge(df_device, df_user, on='image_index')
            if df_user.empty:
                print("Warning: user rank merge resulted in empty dataframe.")
                df_user = None

    # Weights generation
    coarse_weights = []
    for w_nima in np.arange(0, 1.0001, 0.05):
        for w_rgnet in np.arange(0, 1.0001 - w_nima, 0.05):
            w_alamp = 1.0 - w_nima - w_rgnet
            coarse_weights.append((w_nima, w_rgnet, w_alamp))
            
    fixed_candidates = [
        (0.37, 0.41, 0.22),    # newUncal
        (0.125, 0.50, 0.375),  # oldNoIcaaRenorm
        (0.25, 0.30, 0.45),    # alampHeavy
        (0.50, 0.30, 0.20),    # nimaHeavy
        (0.20, 0.55, 0.25),    # rgnetHeavy
    ]
    
    all_candidates = list(set(coarse_weights + fixed_candidates))
    
    def run_eval(candidates):
        results = []
        for w_nima, w_rgnet, w_alamp in tqdm(candidates):
            row = {'w_nima': w_nima, 'w_rgnet': w_rgnet, 'w_alamp': w_alamp}
            
            # Helper for penalty
            penalty = 0.0
            if any(w > 0.65 for w in [w_nima, w_rgnet, w_alamp]):
                penalty += 0.05
            if any(w < 0.05 for w in [w_nima, w_rgnet, w_alamp]):
                penalty += 0.02
            
            row['safety_penalty'] = penalty
            
            # AVA evaluation (normalization needed)
            # Benchmark CSV columns are 1..10
            # We normalize to 0..1 for multiobjective consistency if needed, 
            # but we can use 1..10 directly if all metrics handle it.
            # Let's normalize inputs to 0..1 inside the script for "norms" logic.
            
            ava_nima = (df_ava['nima_pred_scaled_1_10'] - 1.0) / 9.0
            ava_rgnet = (df_ava['rgnet_pred_scaled_1_10'] - 1.0) / 9.0
            ava_alamp = (df_ava['alamp_adaptive_pred_scaled_1_10'] - 1.0) / 9.0
            ava_gt = (df_ava['gt_score'] - 1.0) / 9.0
            
            ava_pred = w_nima * ava_nima + w_rgnet * ava_rgnet + w_alamp * ava_alamp
            ava_m = get_metrics(ava_gt, ava_pred)
            for k, v in ava_m.items():
                row[f'ava_{k}'] = v
                
            aadb_nima = (df_aadb['nima_pred_scaled_1_10'] - 1.0) / 9.0
            aadb_rgnet = (df_aadb['rgnet_pred_scaled_1_10'] - 1.0) / 9.0
            aadb_alamp = (df_aadb['alamp_adaptive_pred_scaled_1_10'] - 1.0) / 9.0
            aadb_gt = (df_aadb['gt_score'] - 1.0) / 9.0
            
            aadb_pred = w_nima * aadb_nima + w_rgnet * aadb_rgnet + w_alamp * aadb_alamp
            aadb_m = get_metrics(aadb_gt, aadb_pred)
            for k, v in aadb_m.items():
                row[f'aadb_{k}'] = v
                
            # Combined benchmark (70/30)
            row['bench_combined_srcc'] = 0.70 * ava_m['srcc'] + 0.30 * aadb_m['srcc']
            row['bench_combined_plcc'] = 0.70 * ava_m['plcc'] + 0.30 * aadb_m['plcc']
            row['bench_combined_mae'] = 0.70 * ava_m['mae'] + 0.30 * aadb_m['mae']
            row['bench_combined_spread'] = 0.70 * ava_m['p90_minus_p10'] + 0.30 * aadb_m['p90_minus_p10']
            
            # Device log evaluation
            if df_device is not None:
                dev_pred = w_nima * df_device['nima'] + w_rgnet * df_device['rgnet'] + w_alamp * df_device['alamp']
                dev_m = get_metrics(dev_pred, dev_pred) # Self-metrics for spread
                for k, v in dev_m.items():
                    if k in ['mean', 'std', 'min', 'max', 'p10', 'p90', 'p90_minus_p10']:
                        row[f'device_{k}'] = v
            
            # User rank evaluation
            if df_user is not None:
                user_pred = w_nima * df_user['nima'] + w_rgnet * df_user['rgnet'] + w_alamp * df_user['alamp']
                # user_rank: lower is better. pred score: higher is better. 
                # so correlation should be negative.
                user_srcc, _ = spearmanr(df_user['user_rank'], user_pred)
                row['user_srcc'] = -user_srcc # make it positive for better
                
                # top3 overlap (simple implementation)
                # user top 3: lowest user_rank
                user_top3_idx = df_user.nsmallest(3, 'user_rank')['image_index'].values
                pred_top3_idx = df_user.nlargest(3, user_pred)['image_index'].values
                row['user_top3_overlap'] = len(set(user_top3_idx).intersection(set(pred_top3_idx))) / 3.0

            results.append(row)
        return pd.DataFrame(results)

    print("Running coarse search...")
    df_results = run_eval(all_candidates)
    
    def compute_final_objective(df):
        # Normalization
        df['bench_srcc_norm'] = normalize_series(df['bench_combined_srcc'])
        df['bench_plcc_norm'] = normalize_series(df['bench_combined_plcc'])
        df['bench_mae_inv_norm'] = normalize_series(df['bench_combined_mae'], higher_is_better=False)
        
        df['bench_spread_quality'] = df['bench_combined_spread'].apply(get_spread_quality)
        df['bench_spread_quality_norm'] = normalize_series(df['bench_spread_quality'])
        
        if df_user is not None:
            df['user_srcc_norm'] = normalize_series(df['user_srcc'])
            df['user_top3_overlap_norm'] = normalize_series(df['user_top3_overlap'])
            
            df['final_objective'] = (
                0.30 * df['bench_srcc_norm']
                + 0.15 * df['bench_plcc_norm']
                + 0.10 * df['bench_mae_inv_norm']
                + 0.15 * df['bench_spread_quality_norm']
                + 0.25 * df['user_srcc_norm']
                + 0.05 * df['user_top3_overlap_norm']
            )
        elif df_device is not None:
            df['device_spread_quality'] = df['device_p90_minus_p10'].apply(get_spread_quality)
            df['device_spread_quality_norm'] = normalize_series(df['device_spread_quality'])
            
            df['final_objective'] = (
                0.35 * df['bench_srcc_norm']
                + 0.15 * df['bench_plcc_norm']
                + 0.10 * df['bench_mae_inv_norm']
                + 0.25 * df['bench_spread_quality_norm']
                + 0.15 * df['device_spread_quality_norm']
            )
        else:
            df['final_objective'] = (
                0.40 * df['bench_srcc_norm']
                + 0.20 * df['bench_plcc_norm']
                + 0.15 * df['bench_mae_inv_norm']
                + 0.25 * df['bench_spread_quality_norm']
            )
        
        df['final_objective'] -= df['safety_penalty']
        return df

    df_results = compute_final_objective(df_results)
    
    # Fine search
    print("Running fine search...")
    top_30 = df_results.nlargest(30, 'final_objective')
    fine_candidates = []
    for _, row in top_30.iterrows():
        w_nima_base, w_rgnet_base, w_alamp_base = row['w_nima'], row['w_rgnet'], row['w_alamp']
        for dn in np.arange(-0.04, 0.0401, 0.01):
            for dr in np.arange(-0.04, 0.0401, 0.01):
                wn = round(w_nima_base + dn, 2)
                wr = round(w_rgnet_base + dr, 2)
                wa = round(1.0 - wn - wr, 2)
                if wn >= 0 and wr >= 0 and wa >= 0:
                    fine_candidates.append((wn, wr, wa))
    
    fine_candidates = list(set(fine_candidates))
    df_fine = run_eval(fine_candidates)
    
    # Merge and re-normalize final objective
    df_all = pd.concat([df_results, df_fine]).drop_duplicates(subset=['w_nima', 'w_rgnet', 'w_alamp'])
    df_all = compute_final_objective(df_all)
    
    df_all.to_csv(os.path.join(args.out_dir, "multiobjective_weight_results.csv"), index=False)
    
    top_candidates = df_all.nlargest(20, 'final_objective')
    top_candidates.to_csv(os.path.join(args.out_dir, "top_candidates.csv"), index=False)
    
    winner = top_candidates.iloc[0]
    v3 = df_all[(df_all['w_nima'] == 0.25) & (df_all['w_rgnet'] == 0.30) & (df_all['w_alamp'] == 0.45)].iloc[0]
    
    print("\nResults Summary:")
    print(f"Benchmark CSV: AVA={len(df_ava)}, AADB={len(df_aadb)}")
    print(f"Device log found: {df_device is not None}")
    print(f"User rank found: {df_user is not None}")
    print(f"Recommended weights: NIMA={winner['w_nima']:.2f}, RGNet={winner['w_rgnet']:.2f}, A-LAMP={winner['w_alamp']:.2f}")
    
    adjust = winner['final_objective'] > v3['final_objective'] + 0.01
    print(f"Should keep V3 or adjust: {'ADJUST' if adjust else 'KEEP V3'}")
    
    # Generate report
    with open(os.path.join(args.out_dir, "report.md"), 'w') as f:
        f.write("# A-cut Aesthetic Ensemble Multi-Objective Re-evaluation\n\n")
        f.write("## 1. Executive Summary\n")
        f.write(f"The analysis optimized ensemble weights across {len(df_all)} candidates using a combined objective of benchmark performance, score spread, and device log consistency.\n\n")
        
        f.write("## 2. Data Availability\n")
        f.write(f"- AVA: {len(df_ava)} images\n")
        f.write(f"- AADB: {len(df_aadb)} images\n")
        f.write(f"- Device logs: {len(df_device) if df_device is not None else 0} entries\n")
        f.write(f"- User rankings: {len(df_user) if df_user is not None else 0} entries\n\n")
        
        f.write("## 3. Metrics Explanation\n")
        f.write("- **Benchmark SRCC**: Ranking alignment on research datasets.\n")
        f.write("- **Spread Quality**: Prefers models that utilize the full 0-1 (or 1-10) range effectively without extreme clustering or over-saturation.\n")
        f.write("- **Device Log Spread**: Stability and range on actual device-processed images.\n\n")
        
        f.write("## 4. Top Candidates\n")
        f.write("| NIMA | RGNet | A-LAMP | Objective | Bench SRCC | Bench Spread | Device Spread |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for _, r in top_candidates.head(10).iterrows():
            f.write(f"| {r['w_nima']:.2f} | {r['w_rgnet']:.2f} | {r['w_alamp']:.2f} | {r['final_objective']:.4f} | {r['bench_combined_srcc']:.4f} | {r['bench_combined_spread']:.4f} | {r.get('device_p90_minus_p10', 0):.4f} |\n")
        
        f.write("\n## 5. Current V3 Comparison\n")
        f.write(f"- **V3 (0.25 / 0.30 / 0.45)**: Objective = {v3['final_objective']:.4f}\n")
        f.write(f"- **Winner ({winner['w_nima']:.2f} / {winner['w_rgnet']:.2f} / {winner['w_alamp']:.2f})**: Objective = {winner['final_objective']:.4f}\n\n")
        
        f.write("## 6. Final Recommendation\n")
        if adjust:
            f.write(f"**ADJUST** to NIMA {winner['w_nima']:.2f} / RGNet {winner['w_rgnet']:.2f} / A-LAMP {winner['w_alamp']:.2f}.\n")
        else:
            f.write("**KEEP V3** as the performance gain from adjustment is marginal or statistically insignificant.\n")
            
        f.write("\n## 7. Caveats\n")
        f.write("- AVA/AADB metrics may not fully capture 'A-cut' product aesthetic which values specific compositional features.\n")
        f.write("- Spread quality is a heuristic to prevent 'always high' or 'always medium' scores.\n")
        f.write("- ICAA is excluded for runtime performance.\n")
        
        if adjust:
            f.write("\n## 8. Next Implementation Prompt\n")
            f.write(f"Update `pozy_app` aesthetic ensemble weights to:\n")
            f.write(f"- NIMA: {winner['w_nima']:.2f}\n")
            f.write(f"- RGNet: {winner['w_rgnet']:.2f}\n")
            f.write(f"- A-LAMP: {winner['w_alamp']:.2f}\n")

    print(f"Report saved to: {os.path.join(args.out_dir, 'report.md')}")

if __name__ == "__main__":
    main()
