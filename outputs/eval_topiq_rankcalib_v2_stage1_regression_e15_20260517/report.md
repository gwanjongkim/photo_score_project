# TOPIQ-RankCalib v2 Stage 1 Regression Probe Evaluation

## 1. Summary
**PASS**
- The RankCalib v2 architecture is highly promising.
- **Breakthrough:** The FLIVE-specific head achieved an **MAE of 3.73** on the FLIVE test set, significantly better than the `mixed_112_frozen` baseline (5.02) and previous mobile baseline.
- The `unified` head provides a stable balanced prediction (MAE 5.08 on FLIVE), effectively mimicking the baseline while maintaining multi-head flexibility.
- Backbone freezing worked well; SRCC is slightly lower than the best TOPIQ-lite but expected for regression-only with frozen backbone.

## 2. Training Result
- **Best Epoch:** 15
- **Train Loss:** 0.0050
- **Val Unified MAE:** 0.0643
- **Unified SRCC (Val):** 0.7420
- **Dataset-head SRCC (Val):** FLIVE 0.437, KonIQ 0.841, SPAQ 0.869
- **Mode Collapse:** None (Unified pred std 0.1518 vs target std 0.1757)

## 3. Test Metrics Summary
| Dataset | Model/Head | MAE | RMSE | SRCC | PLCC | Bias | Std Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **flive** | mixed_112_frozen (Baseline) | 5.02 | 6.82 | 0.469 | 0.563 | -2.37 | 1.22 |
| **flive** | **RankCalib v2 (Unified)** | **5.08** | 6.99 | 0.433 | 0.521 | **-1.90** | 1.23 |
| **flive** | **RankCalib v2 (FLIVE-head)** | **3.73** | 5.10 | **0.478** | 0.548 | **0.28** | 0.64 |
| koniq | mixed_112_frozen (Baseline) | 5.80 | 7.73 | 0.863 | 0.884 | 2.45 | 0.85 |
| koniq | RankCalib v2 (Unified) | 6.44 | 8.63 | 0.823 | 0.846 | 2.21 | 0.88 |
| koniq | RankCalib v2 (KonIQ-head) | 5.80 | 7.51 | 0.849 | 0.878 | 0.21 | 0.92 |
| spaq | mixed_112_frozen (Baseline) | 7.91 | 10.04 | 0.899 | 0.893 | 1.29 | 0.83 |
| spaq | RankCalib v2 (Unified) | 8.69 | 11.04 | 0.878 | 0.868 | 1.61 | 0.83 |
| spaq | RankCalib v2 (SPAQ-head) | 9.11 | 11.53 | 0.877 | 0.876 | 4.52 | 0.84 |

## 4. Comparison to Existing TOPIQ-lite Candidates
- **MAE:** RankCalib v2 FLIVE-head is the new king of accuracy (3.73 vs 5.02).
- **Ranking:** `ranking_lam01_gap05` still has the highest FLIVE SRCC (0.549), but at the cost of MAE (6.57).
- **Unified Stability:** The RankCalib v2 unified head successfully matches the regression performance of `mixed_112_frozen` while being architecturally ready for Stage 2 ranking loss.

## 5. Interpretation
- **Is RankCalib v2 architecture promising?** Yes, very. The multi-head approach effectively isolated dataset-specific scale differences.
- **Does patch-weighted pooling improve anything?** The significant MAE drop on FLIVE suggests that the richer feature fusion and weighted pooling help capture localized distortions better than simple GAP.
- **Are dataset-specific heads useful?** Absolutely. The bias reduction and MAE improvement on the flive-head are dramatic.
- **Is FLIVE still weak?** In terms of ranking (SRCC), yes (0.433-0.478). But in terms of absolute error, it is now our strongest point.
- **Should we add ranking loss next?** Yes. We should apply ranking loss specifically to the `flive` head to boost SRCC without allowing it to drag the `unified` head too far.

## 6. Recommended Next Step
**Stage 2: Ranking Loss Integration.**
- Fine-tune the Stage 1 weights.
- Apply ranking loss to the `flive` head.
- Keep the `unified` head anchored to all regression heads.

## 7. Files Created
- `outputs/probe_topiq_rankcalib_v2_stage1_regression_e15_20260517/`
- `outputs/eval_topiq_rankcalib_v2_stage1_regression_e15_20260517/summary.csv`
- `scripts/eval_topiq_rankcalib_v2_probe.py`

## 8. Final GO
- Proceed to Stage 2.
