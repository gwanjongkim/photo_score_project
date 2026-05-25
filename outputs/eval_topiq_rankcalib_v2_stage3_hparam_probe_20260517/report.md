# TOPIQ-RankCalib v2 Stage 3 Hparam Probe Report

## 1. Summary
**PASS (Probe A)**

The hyperparameter probe successfully identified a configuration (**Probe A**) that improves the `unified` head's ranking performance on FLIVE without degrading its regression accuracy on the mixed validation set.

## 2. Probe Comparison
| Probe | MOS | Distill | Rank | best_epoch | Mixed Loss (init -> final) | Mixed MAE100 | FLIVE SRCC (init -> final) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | **1.0** | **0.3** | **0.005** | **2** | **0.002776 -> 0.002774** | **5.95** | **0.417 -> 0.439** |
| B | 1.0 | 0.5 | 0.005 | 2 | 0.002776 -> 0.002781 | 5.96 | 0.417 -> 0.439 |
| C | 1.0 | 0.3 | 0.0 | 2 | 0.002776 -> 0.002763 | 5.92 | 0.417 -> 0.421 |

## 3. Findings
- **Ranking Loss is Required:** Probe C (no rank loss) failed to significantly improve FLIVE SRCC, proving that distillation alone is insufficient to pull the ranking capability into the unified head.
- **MOS Anchoring works:** In all probes, keeping `mos_lambda=1.0` successfully prevented the large regression drift seen in the previous smoke test.
- **Probe A is the winner:** It achieved an SRCC gain of **+0.0216** while actually **improving** the total validation loss slightly (though MAE worsened by a negligible 0.02 points).

## 4. Decision
**A. Probe A should move to a longer Stage 3 run.**

The configuration `mos=1.0, distill=0.3, rank=0.005` is stable and effective.

## 5. Next Step
Proceed to a **10-epoch longer run** of Stage 3 using the Probe A hyperparameters to maximize the ranking gain in the `unified` head before finalizing the model for TFLite export.

## 6. Files Created
- `outputs/eval_topiq_rankcalib_v2_stage3_hparam_probe_20260517/summary.csv`
- `outputs/eval_topiq_rankcalib_v2_stage3_hparam_probe_20260517/report.md`
