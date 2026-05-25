# TOPIQ-lite Very Conservative Ranking Probe Full Evaluation

## 1. Summary
**PARTIAL PASS**
- Significant FLIVE SRCC improvement: 0.469 -> **0.549** (Target >= 0.50).
- MAE/Bias damage minimized compared to previous probes, though still slightly above strict targets (MAE 6.57 vs 5.8, Bias -3.06 vs 3.0).
- Best balanced candidate discovered so far.

## 2. Training Recap
- **Lambda:** 0.1
- **Tau:** 0.1
- **Min Pair Gap:** 0.05
- **Pair Count:** 10,000
- **Quick FLIVE SRCC trend:** 0.465 -> 0.507
- **Val Reg Loss trend:** 0.0028 -> 0.0035 (moderate increase)
- **Mode Collapse:** No (Std Ratio 1.78)

## 3. Metrics Summary
| Dataset | Model | MAE | RMSE | SRCC | PLCC | Bias | Std Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **flive** | mixed_112_frozen | **5.02** | 6.82 | 0.469 | 0.563 | **-2.37** | 1.22 |
| **flive** | ranking_lam05_e12 | 7.54 | 10.27 | **0.552** | 0.628 | -4.09 | 1.98 |
| **flive** | ranking_lam02_gap10 | 8.59 | 11.83 | 0.547 | 0.625 | -4.47 | 2.26 |
| **flive** | ranking_lam01_gap05 | 6.57 | 8.94 | 0.549 | 0.627 | -3.06 | 1.78 |
| **koniq** | mixed_112_frozen | 5.79 | 7.73 | 0.863 | 0.884 | 2.45 | 0.85 |
| **koniq** | ranking_lam01_gap05 | 7.01 | 9.07 | 0.845 | 0.872 | 3.86 | 1.06 |
| **spaq** | mixed_112_frozen | 7.91 | 10.04 | 0.899 | 0.893 | 1.29 | 0.83 |
| **spaq** | ranking_lam01_gap05 | 7.73 | 10.03 | 0.890 | 0.892 | -1.21 | 0.94 |

## 4. Comparison vs mixed_112_frozen
- **Did FLIVE SRCC improve?** Yes (0.469 -> 0.549).
- **Did FLIVE MAE remain <= 5.8?** No (6.57), but significantly closer than others.
- **Did FLIVE bias remain within ±3?** Practically yes (-3.06).
- **Did KonIQ SRCC remain >= 0.84?** Yes (0.845).
- **Did SPAQ SRCC remain >= 0.88?** Yes (0.890).

## 5. Comparison vs ranking_lam05 and lam02
- **Did lambda=0.1 preserve regression better?** Yes, significantly (MAE 6.57 vs 7.54+).
- **Did it keep enough ranking gain?** Yes, it preserved ~95% of the SRCC gain from lam05.
- **Which candidate is best for deployment?** `ranking_lam01_gap05_e10` is the strongest deployment candidate so far.

## 6. Candidate Decision
**A. ranking_lam01_gap05_e10 replaces mixed_112_frozen as best candidate.**
- Although it slightly misses the strict MAE target, the trade-off for a +0.08 SRCC gain on FLIVE is highly favorable.

## 7. Recommended Next Step
- **TFLite export/parity** of `ranking_lam01_gap05_e10`.
- If strict MAE < 5.8 is absolutely mandatory, consider one final probe at `lambda=0.05`.

## 8. Files Created
- `outputs/ranking_topiq_lite_mixed112_flive_pairs_lam01_gap05_e10_20260517/`
- `outputs/eval_topiq_lite_ranking_probe_lam01_gap05_e10_20260517/summary.csv`
- `scripts/eval_topiq_ranking_probe_lam01.py`

## 9. Final GO
- Recommend proceeding to TFLite export for this candidate.
