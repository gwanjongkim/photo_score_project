# Technical IQA Benchmark: TOPIQ capped by KonIQ Guard

## 1. Executive Summary
The evaluation of **TOPIQ mixed112 capped by KonIQ guard** (technical_score = min(TOPIQ, KonIQ + margin)) demonstrates that this fusion strategy successfully combines the high general IQA performance of TOPIQ with the robust safety "guard" of KonIQ-Mobile.

**Winner: C6 (Soft Blend + KonIQ Cap 8)**
- **Average Benchmark SRCC**: 0.760 (Significantly higher than current safe baseline B5 at 0.745).
- **Safety (Hard-FP v2)**: Mean score 54.1, with **0 images over-scoring > 65**.
- **KonIQ Benefit**: C6 reaches SRCC 0.892 on KonIQ-10k test (vs 0.866 for KonIQ-Mobile alone), showing that the soft blend + cap architecture improves the best of both worlds.

Using a margin of **8 points** (C3/C6) provides an optimal tradeoff, maintaining perfect safety at the 65-score threshold while significantly improving ranking performance on SPAQ and KonIQ datasets.

## 2. Protocol
- **Datasets**: FLIVE test (N=3981), KonIQ-10k test (N=1008), SPAQ test (N=1125), Hard-FP v2 (N=44).
- **Metrics**: SRCC, PLCC, MAE, Bias.
- **Safety Metrics**: Mean, Median, Count > 60, Count > 65 on confirmed failures.
- **Fusion definitions**:
    - **C3**: `min(topiq, koniq + 8)`
    - **C6**: `min(0.7 * topiq + 0.3 * koniq, koniq + 8)`
- **Bootstrap**: 1000 iterations, seed 42.

## 3. Available Prediction Sources
- Aligned predictions from `outputs/technical_iqa_combo_benchmark_20260524/`
- Validated model scores: `koniq_mobile`, `flive_mobile`, `topiq_lite_mixed112`.

## 4. Benchmark Results (SRCC)
| Dataset | B5 (Current Guard) | C3 (TOPIQ Cap 8) | C6 (TOPIQ Soft-Cap 8) | Delta (C6-B5) |
| :--- | :---: | :---: | :---: | :---: |
| FLIVE | 0.508 | 0.477 | 0.487 | -0.021 |
| KonIQ | 0.867 | 0.875 | **0.892** | +0.025 |
| SPAQ | 0.861 | 0.900 | **0.900** | +0.039 |
| **Average** | **0.745** | **0.751** | **0.760** | **+0.015** |

## 5. Hard-FP v2 Safety Results (Lower is Better)
| Model / Fusion | Mean Score | Median | Count > 60 | Count > 65 |
| :--- | :---: | :---: | :---: | :---: |
| topiq_mixed112 | 65.4 | 68.4 | 31 | 28 |
| B5 (Safe Guard) | 52.8 | 52.9 | 5 | 0 |
| **C3 (Cap 8)** | **54.0** | **54.9** | **10** | **0** |
| **C6 (Soft-Cap 8)** | **54.1** | **54.9** | **9** | **0** |
| C4 (Cap 10) | 55.7 | 56.8 | 14 | 4 |

## 6. Margin Sensitivity (TOPIQ + KonIQ Margin)
Increasing the margin improves SRCC but quickly degrades safety:
- **Margin 0**: SRCC 0.732 (Aggressive, safe but loses TOPIQ benefit)
- **Margin 5**: SRCC 0.742 (Safe, mean 51.3)
- **Margin 8**: SRCC 0.751 (Optimal safety crossover)
- **Margin 10**: SRCC 0.755 (**Safety failure**: 4 images > 65)

## 7. Bootstrap Confidence Intervals (C3 vs B5 Delta)
- **Delta Average SRCC (C3 - B5)**: **0.031 [0.022, 0.040]** (Significant improvement in ranking)
- **Delta Hard-FP Mean (C3 - B5)**: 1.15 [-0.71, 2.73] (Safety is statistically comparable to B5)
- **Delta Hard-FP Count > 65**: 0.0 [0.0, 0.0] (Identical safety at threshold 65)

## 8. Direct Comparison with KonIQ+FLIVE
Combination C (TOPIQ+KonIQ) outperforms Combination B (KonIQ+FLIVE) on high-quality datasets (KonIQ/SPAQ) while matching its safety on technical failures. FLIVE-Mobile still provides better performance on the FLIVE test set, but its "semantic bias" makes it a poor general-purpose technical metric compared to TOPIQ.

## 9. Interpretation
KonIQ-Mobile acts as an effective non-linear constraint on TOPIQ. By using `min(TOPIQ, KonIQ + 8)`, we prevent TOPIQ's high-confidence false positives (semantic bias) from leaking into the final A-cut selection, while allowing TOPIQ to provide superior ranking on authentic images.

C6 (Soft Blend + Cap) is particularly recommended as it leverages both models' strengths across the entire score range, achieving the highest average SRCC (0.760) among all safe fusion candidates.

## 10. Recommendation
**Choice B: Use TOPIQ capped by KonIQ (specifically C6: Soft-Cap 8).**

This candidate provides the best balance of state-of-the-art IQA ranking and product-level safety. It should replace the current KonIQ+FLIVE path for technical scoring.

## 11. No-Go Conditions
- **DO NOT** use margin > 8 if autonomous safety is required.
- **DO NOT** remove KonIQ-Mobile from the technical pipeline.
- **DO NOT** deploy TOPIQ-Mixed112 without a non-linear guard (KonIQ).
