# TOPIQ+FLIVE Heuristic Safety Gate Full Simulation Report

## 1. Executive Summary
Following the promising initial probe, we ran a full 4,967-image simulation to extract lightweight classical metrics (Laplacian variance, Tenengrad, simple noise estimates) and use them to gate the `topiq_flive_60_40` candidate. The goal was to eliminate 3 severe False Positives (FPs) without complex neural inference.

The full simulation revealed that the distributions of these heuristic metrics overlap heavily between genuine severe FPs (heavy noise, localized blur) and high-quality sharp images (foliage, texture). Any threshold low enough to catch the 3 severe FPs ended up unnecessarily capping 15-20% of the entire dataset, severely degrading the SRCC gains.

**Final Decision: TOPIQ_FLIVE_HEURISTIC_GATE_NOT_WORTH_IT**

## 2. Sources Inspected
- `outputs/official_protocol_technical_signal_benchmark_20260604/official_signal_predictions.csv`
- `outputs/topiq_flive_safety_signal_ablation_20260608/simple_quality_metric_probe.csv`

## 3. Heuristic Metric Extraction
We successfully processed 4,967 images using fast Python implementations (`scipy.ndimage`) to compute:
- Laplacian variance
- Tenengrad sharpness
- Median-subtracted noise estimate
- Center-crop variants

Average noise across the dataset was ~6.25, while the severe FPs ranged from 11.0 to 20.6.

## 4. Gate Definitions
Seven heuristic gates were tested, combining the `topiq_flive_60_40` score with `noise_est` and `laplacian_var` thresholds (e.g., Gate A: cap to 60 if candidate > 65 and noise > 10).

## 5. Overall Gate Metrics
- **Base Candidate SRCC**: 0.8069 (3 Severe FPs)
- **Gate A SRCC**: 0.6810 (0 Severe FPs)
- **Gate C SRCC**: 0.6619 (0 Severe FPs)

The gates that successfully reached 0 severe FPs caused catastrophic drops in overall correlation (-0.12 to -0.14 SRCC loss).

## 6. Dataset-wise Gate Metrics
The damage was concentrated in KonIQ and SPAQ, where high-quality images with natural textures (high Laplacian/noise metrics) were incorrectly penalized and capped at 60.

## 7. Severe FP Fix Audit
Aggressive gates (A, B, C, D, E, F) fixed all 3 original severe FPs, but only by brute-force capping.

## 8. Modified Row and Damage Analysis
- **Gate A (`noise_est > 10`)**: Modified 780 rows (15.7%).
- **Gate C (`laplacian_var > 1500`)**: Modified 886 rows (17.8%).
- **Gate F (`center_noise_est > 8`)**: Modified 1,136 rows (22.8%).

Collateral damage was extreme. Modifying ~15-20% of the dataset negates the precision benefits of the TOPIQ+FLIVE hybrid model.

## 9. Best Heuristic Gate
No heuristic gate provided an acceptable tradeoff. Gate G (`center_lap_var < 50`) modified the fewest rows (63) but failed to fix any of the severe FPs, as the severe FPs had high frequency background details (e.g., center Laplacian > 1200).

## 10. Flutter Feasibility
While computing these metrics on a mobile device is highly feasible (<5ms), deploying them as a safety gate is actively harmful to overall ranking quality.

## 11. Final Decision
**TOPIQ_FLIVE_HEURISTIC_GATE_NOT_WORTH_IT**

## 12. Next Actions
- Abandon simple heuristic metrics for safety gating.
- Accept the `topiq_flive_60_40` candidate (with its 3 known severe FPs) as the primary log-only profiling target, prioritizing its massive overall SRCC improvements.
- Rely on the upcoming Phase 14 student distillation (with expanded data and larger backbone) to learn these nuanced boundaries naturally, rather than hardcoding them.
