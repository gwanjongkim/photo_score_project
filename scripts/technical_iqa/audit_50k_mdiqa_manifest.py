import pandas as pd
import numpy as np
from pathlib import Path
import json

OUT_DIR = Path("outputs/mdiqa_distillation_50k_manifest_audit_20260608")

# Data Sources
BASE_MANIFEST = "outputs/research_protocol_iqa_manifests_20260530/benchmark_all_research_protocol_iqa.csv"
STUDENT_14K_MANIFEST = "outputs/student_distillation_manifest_14105_20260603/student_distillation_manifest.csv"
OFFICIAL_TEST_MANIFEST = "outputs/official_protocol_benchmark_manifest_20260604/official_benchmark_test_manifest.csv"

# Cache Sources
CACHE_1 = "outputs/mdiqa_correct_protocol_full_benchmark_20260601/predictions.csv"
CACHE_2 = "outputs/official_protocol_mdiqa_signal_benchmark_20260604/predictions.csv"

def norm_path(p):
    p = str(p)
    if p.startswith('/home/omen_pc1/photo_score_project/'):
        return p.replace('/home/omen_pc1/photo_score_project/', '')
    return p

# 1. Load Caches
df_c1 = pd.read_csv(CACHE_1)
df_c2 = pd.read_csv(CACHE_2)
cached_paths = set(df_c1['image_path'].apply(norm_path)).union(set(df_c2['image_path'].apply(norm_path)))

# 2. Load Base Manifest
df_base = pd.read_csv(BASE_MANIFEST)
df_base['norm_path'] = df_base['image_path'].apply(norm_path)
df_base['is_cached'] = df_base['norm_path'].isin(cached_paths)

# Drop duplicates since benchmark_all contains multiple runs
df_base = df_base.drop_duplicates(subset=['norm_path']).copy()

# Official Test Paths
df_off_test = pd.read_csv(OFFICIAL_TEST_MANIFEST)
df_off_test['norm_path'] = df_off_test['image_path'].apply(norm_path)
official_test_paths = set(df_off_test['norm_path'])
df_base['is_official_test'] = df_base['norm_path'].isin(official_test_paths)

train_pool = df_base[~df_base['is_official_test']].copy()

pool_koniq = train_pool[(train_pool['dataset'] == 'koniq10k') & (train_pool['content_type'] == 'image')]
pool_spaq = train_pool[(train_pool['dataset'] == 'spaq') & (train_pool['content_type'] == 'image')]
pool_flive_img = train_pool[(train_pool['dataset'] == 'flive') & (train_pool['content_type'] == 'image')]
pool_flive_patch = train_pool[(train_pool['dataset'] == 'flive') & (train_pool['content_type'] == 'patch')]

# Policy A: image-only conservative 50k
policy_a_df = pd.concat([pool_koniq, pool_spaq, pool_flive_img])
policy_a_df = policy_a_df.sample(n=min(50000, len(policy_a_df)), random_state=42)

# Policy B: balanced 50k with sampled FLIVE patches
koniq_sample = pool_koniq.sample(n=min(10000, len(pool_koniq)), random_state=42)
spaq_sample = pool_spaq.sample(n=min(11000, len(pool_spaq)), random_state=42)
flive_img_sample = pool_flive_img.sample(n=min(39000, len(pool_flive_img)), random_state=42)

# Do we need patches to reach 50k?
needed = 50000 - len(koniq_sample) - len(spaq_sample) - len(flive_img_sample)
if needed > 0 and len(pool_flive_patch) > 0:
    flive_patch_sample = pool_flive_patch.sample(n=min(needed, len(pool_flive_patch)), random_state=42)
else:
    flive_patch_sample = pd.DataFrame()

policy_b_df = pd.concat([koniq_sample, spaq_sample, flive_img_sample, flive_patch_sample]).sample(n=min(50000, len(koniq_sample)+len(spaq_sample)+len(flive_img_sample)+len(flive_patch_sample)), random_state=42)

# Policy C: official + hard-absent emphasis
policy_c_df = pd.concat([pool_koniq, pool_spaq]).sample(n=min(20000, len(pool_koniq)+len(pool_spaq)), random_state=42)

# Policy D: teacher-cache-first 50k
cached_pool = train_pool[train_pool['is_cached']]
uncached_pool = train_pool[~train_pool['is_cached']]
policy_d_df = pd.concat([
    cached_pool.sample(n=min(50000, len(cached_pool)), random_state=42),
    uncached_pool.sample(n=max(0, 50000 - len(cached_pool)), random_state=42)
]).head(50000)

policies = {
    "Policy A (Image-only)": policy_a_df,
    "Policy B (Balanced)": policy_b_df,
    "Policy C (Hard-absent)": policy_c_df,
    "Policy D (Cache-first)": policy_d_df
}

policy_summary = []
for name, pdf in policies.items():
    policy_summary.append({
        "policy": name,
        "total_rows": len(pdf),
        "koniq_rows": len(pdf[pdf['dataset'] == 'koniq10k']),
        "spaq_rows": len(pdf[pdf['dataset'] == 'spaq']),
        "flive_img_rows": len(pdf[(pdf['dataset'] == 'flive') & (pdf['content_type'] == 'image')]),
        "flive_patch_rows": len(pdf[(pdf['dataset'] == 'flive') & (pdf['content_type'] == 'patch')]),
        "cached_rows": int(pdf['is_cached'].sum()),
        "missing_rows": int(len(pdf) - pdf['is_cached'].sum()),
    })

pd.DataFrame(policy_summary).to_csv(OUT_DIR / "candidate_50k_manifest_policy_summary.csv", index=False)

# Recommend Policy B
rec_policy_df = policy_b_df
rec_policy_df.to_csv(OUT_DIR / "candidate_50k_manifest_preview.csv", index=False)

missing_in_rec = int(len(rec_policy_df) - rec_policy_df['is_cached'].sum())
speed_it_per_sec = 23102 / 515.0
est_seconds = missing_in_rec / speed_it_per_sec

missing_est = [{
    "missing_rows": missing_in_rec,
    "estimated_speed_it_s": speed_it_per_sec,
    "estimated_runtime_seconds": est_seconds,
    "estimated_runtime_hours": est_seconds / 3600.0
}]
pd.DataFrame(missing_est).to_csv(OUT_DIR / "missing_mdiqa_inference_estimate.csv", index=False)

# Leakage Risk Audit
leakage_audit = [
    {"split": "Official Test", "overlap_with_train": len(set(rec_policy_df['norm_path']).intersection(official_test_paths))}
]
pd.DataFrame(leakage_audit).to_csv(OUT_DIR / "leakage_risk_audit.csv", index=False)

# Recommendation
rec = {
    "recommended_policy": "Policy B (Balanced)",
    "total_target_rows": len(rec_policy_df),
    "missing_inference_rows": missing_in_rec,
    "estimated_runtime_hours": round(est_seconds / 3600.0, 2),
    "decision": "MDIQA_50K_MANIFEST_READY_FOR_SMOKE"
}
with open(OUT_DIR / "recommendation.json", "w") as f:
    json.dump(rec, f, indent=2)

print("Audit complete.")