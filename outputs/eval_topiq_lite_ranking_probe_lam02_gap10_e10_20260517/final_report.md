# TOPIQ-lite Conservative Ranking Probe Full Evaluation

## 1. Summary
**PARTIAL PASS / FAIL**
- Ranking performance (SRCC) improved significantly over `mixed_112_frozen`.
- However, the "conservative" configuration (`lam02_gap10`) actually caused **more** regression damage (MAE/Bias) than the previous aggressive probe (`lam05_gap05`).
- The increased pair gap (0.10) led to excessive score dispersion (Std Ratio 2.26), which is detrimental for datasets with tight MOS distributions like FLIVE (GT std ~6.0).

## 2. Training Recap
- **Lambda:** 0.2
- **Tau:** 0.1
- **Min Pair Gap:** 0.10
- **Pair Count:** 10,000
- **Quick FLIVE SRCC trend:** 0.44 -> 0.50 (on 1024 samples)
- **Val Reg Loss trend:** 0.0029 -> 0.0048
- **Mode Collapse:** No (Std 0.13 on 0..1 scale)

## 3. Metrics Summary
| Dataset | Model | MAE | RMSE | SRCC | PLCC | Bias | Std Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **flive** | mixed_112_frozen | **5.02** | 6.82 | 0.469 | 0.563 | **-2.37** | 1.22 |
| **flive** | ranking_lam05_e12 | 7.53 | 10.27 | **0.552** | 0.628 | -4.09 | 1.98 |
| **flive** | ranking_lam02_gap10_e10 | 8.59 | 11.83 | 0.547 | 0.625 | -4.46 | 2.26 |
| **koniq** | mixed_112_frozen | 5.80 | 7.73 | 0.863 | 0.884 | 2.45 | 0.85 |
| **koniq** | ranking_lam05_e12 | 7.59 | 9.66 | 0.832 | 0.862 | 3.39 | 1.14 |
| **koniq** | ranking_lam02_gap10_e10 | 8.08 | 10.15 | 0.840 | 0.869 | 2.07 | 1.27 |
| **spaq** | mixed_112_frozen | 7.91 | 10.04 | 0.899 | 0.893 | 1.30 | 0.83 |
| **spaq** | ranking_lam05_e12 | 8.55 | 11.04 | 0.880 | 0.885 | -3.58 | 0.98 |
| **spaq** | ranking_lam02_gap10_e10 | 9.48 | 12.14 | 0.884 | 0.890 | -5.84 | 1.06 |

## 4. Comparison vs mixed_112_frozen
- **Did FLIVE SRCC improve?** Yes (0.469 -> 0.547).
- **Did FLIVE MAE stay closer to mixed_112 than lam05?** No, it got worse (8.59 vs 7.53).
- **Did FLIVE bias stay within ±3?** No (-4.46).
- **Did KonIQ SRCC remain >= 0.84?** Yes (0.840).
- **Did SPAQ SRCC remain >= 0.88?** Yes (0.884).

## 5. Comparison vs ranking_lam05_e12
- **Did lam02 keep most ranking gain?** Yes, SRCC 0.547 vs 0.552.
- **Did lam02 reduce MAE/Bias damage?** No, it increased the damage.
- **Which is better as a deployable candidate?** `ranking_lam05_e12` is technically better than `lam02_gap10`, but neither is strictly "safe" for regression.

## 6. Candidate Decision
**C. ranking_lam02 still damages regression; keep mixed_112_frozen or stick to ranking_lam05 if SRCC is prioritized.**
- The `gap=0.10` setting was too aggressive in pushing score separation.

## 7. Recommended Next Step
- **Another conservative probe:** Try `lambda=0.1` and return to `min_pair_gap=0.05`.
- Or accept `mixed_112_frozen` as the final choice if MAE is the primary constraint.

## 8. Files Created
- `data/processed/topiq_replacement/flive_pairs_gap010_10k.csv`
- `outputs/ranking_topiq_lite_mixed112_flive_pairs_lam02_gap10_e10_20260517/`
- `outputs/eval_topiq_lite_ranking_probe_lam02_gap10_e10_20260517/summary.csv`
- `scripts/eval_topiq_ranking_probe_lam02.py`

## 9. Final NO-GO
- Do not deploy `lam02_gap10` over `lam05_gap05`.
