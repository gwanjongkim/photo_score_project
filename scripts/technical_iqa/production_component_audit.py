import pandas as pd
from scipy.stats import spearmanr
import numpy as np

def audit():
    df = pd.read_csv("outputs/paired_production_mdiqa_significance_expanded_20260603/paired_predictions_expanded.csv")
    vdf = df[df["dataset"] == "koniq10k"].copy()
    mos = vdf["quality_score_0_100"]

    print(f"Dataset: KonIQ-10k (n={len(vdf)})")
    print(f"MDIQA SRCC: {spearmanr(vdf['mdiqa_pred_0_100'], mos)[0]:.4f}")
    print(f"Production Ensemble SRCC: {spearmanr(vdf['production_pred_0_100'], mos)[0]:.4f}")
    print(f"KonIQ-Mobile component SRCC: {spearmanr(vdf['koniq_mobile'], mos)[0]:.4f}")
    print(f"FLIVE-Image component SRCC: {spearmanr(vdf['flive_image_mobile'], mos)[0]:.4f}")

    # Re-weighted ensemble (KonIQ 0.32, FLIVE 0.24 normalized)
    weighted_img = (0.32 * vdf["koniq_mobile"] + 0.24 * vdf["flive_image_mobile"]) / 0.56
    print(f"KonIQ+FLIVE Image-only Ensemble SRCC: {spearmanr(weighted_img, mos)[0]:.4f}")
    
    # 5-component weighted
    # musiq_aadb needs to be on same scale as others (0..100) or we use normalized values.
    # Production uses normalize_unit_score which maps KonIQ/100 and MUSIQ (which is already 0..1)
    
    k_norm = vdf['koniq_mobile'] / 100.0
    f_norm = vdf['flive_image_mobile'] / 100.0
    pm_norm = vdf['flive_patch_mean'] / 100.0
    pmin_norm = vdf['flive_patch_min'] / 100.0
    m_norm = vdf['musiq_aadb'] # already 0..1
    
    # Weights: 0.32, 0.24, 0.22, 0.12, 0.10
    ensemble_5 = 0.32*k_norm + 0.24*f_norm + 0.22*pm_norm + 0.12*pmin_norm + 0.10*m_norm
    print(f"Manual 5-component Ensemble SRCC: {spearmanr(ensemble_5, mos)[0]:.4f}")

if __name__ == "__main__":
    audit()
