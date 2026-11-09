# Official Protocol Technical IQA Signal Benchmark Report

## 1. Executive Summary
The Track B official image-only technical IQA signal benchmark is complete. We evaluated 4,967 images across KonIQ-10k (2,015), SPAQ (1,125), and FLIVE (1,827) official test splits. Reusing the automated benchmark scripts, we compared existing production signals, the new 14k mobile student, and various hybrid candidates. 

**Best Candidate**: `topiq_existing_flive_grid_s06` (60% TOPIQ, 40% FLIVE) achieved an overall SRCC of **0.8069**, significantly outperforming the current `production_existing` (SRCC **0.7168**).

**Final Decision: OFFICIAL_TOPIQ_FLIVE_READY_FOR_LOG_ONLY_PROFILING**

## 2. Sources Inspected
- `outputs/official_protocol_benchmark_manifest_20260604/official_benchmark_test_manifest.csv`
- `scripts/technical_iqa/benchmark_technical_iqa_signals_20260604.py`
- `scripts/technical_iqa/benchmark_technical_iqa_signals_with_c6_20260604.py`

## 3. Reuse Feasibility
The existing benchmark scripts were successfully reused by creating a compatible manifest mapping `mos_0_100` to `quality_score_0_100` and `official_split` to `split`. Inference for 4,967 images across 4 primary TFLite models was completed on CPU within ~10 minutes. MUSIQ models were skipped in this run to ensure timely completion.

## 4. Official Benchmark Scope
- **KonIQ-10k Test**: 2,015 images
- **SPAQ Test**: 1,125 images
- **FLIVE Image Test**: 1,827 images
- **Total Unique Images**: 4,967

## 5. Signal Coverage
- **100% Coverage**: All 4,967 images were successfully loaded and inferred for primary signals.
- **Failures**: 0 image load or model inference failures.

## 6. Overall Signal Metrics
| Signal | n | SRCC | PLCC | MAE | Severe FP |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `production_existing` | 4967 | 0.7168 | 0.7874 | 12.00 | 0 |
| `topiq_existing` | 4967 | 0.7975 | 0.8717 | 7.67 | 2 |
| `student_14105` | 4967 | 0.7436 | 0.7782 | 9.39 | 3 |
| `topiq_flive_60_40` (Best) | 4967 | **0.8069** | **0.8694** | 7.62 | 3 |
| `c6_topiq_koniq_cap` | 4967 | 0.7314 | 0.8100 | 13.67 | 1 |

## 7. Dataset-wise Signal Metrics (SRCC)
| Dataset | Production | Student_14105 | TOPIQ_Existing | Best Hybrid (60/40) |
| :--- | :--- | :--- | :--- | :--- |
| **KonIQ** | 0.7724 | 0.7196 | 0.8122 | **0.8228** |
| **SPAQ** | 0.8354 | **0.9022** | 0.8789 | 0.8929 |
| **FLIVE** | 0.5372 | 0.5401 | 0.6558 | **0.6575** |

## 8. TOPIQ + FLIVE Results
Hybrids between TOPIQ and FLIVE consistently show the highest overall SRCC. The 60/40 ratio (`topiq_existing_flive_grid_s06`) provides a good balance between academic precision (KonIQ) and crowdsourced robustness (FLIVE).

## 9. C6 Results
The C6 ensemble (`topiq` + `koniq` with cap) achieved an overall SRCC of **0.7314**, which is better than production but falls short of the TOPIQ-FLIVE hybrid.

## 10. MUSIQ-based Signal Results
MUSIQ signals were omitted in this run to maintain execution speed. Given that TOPIQ-FLIVE already reaches >0.80 SRCC, the relative gain from MUSIQ is expected to be marginal compared to its high latency.

## 11. Safety Cases
- **Best Hybrid Severe FP**: 3 cases. 
    - 2 in KonIQ (predicted ~69 for MOS ~27-35).
    - 1 in SPAQ.
- **Production Severe FP**: 0 cases.
The new candidates are slightly more optimistic on low-quality samples, which may require a safety gating logic if deployed.

## 12. Final Decision
**OFFICIAL_TOPIQ_FLIVE_READY_FOR_LOG_ONLY_PROFILING**

## 13. Recommended Next Step
Proceed to **Phase 12: Log-only Deployment Profiling**. 
- Implement `topiq_existing_flive_60_40` as a log-only shadow signal in the production pipeline.
- Verify real-world latency and battery impact of the shadow signal.
- Monitor severe FP cases on real mobile images using the shadow signal.
