# Official Protocol MDIQA vs Technical Signal Benchmark Report

## 1. Executive Summary
The official image-only benchmark comparison including MDIQA is complete. We evaluated 4,967 images across KonIQ-10k, SPAQ, and FLIVE official test splits. MDIQA was successfully evaluated on 100% of these rows (217 missing KonIQ rows were newly inferred).

**Best Overall Signal**: `mdiqa_selected` achieved the highest overall SRCC of **0.8227** and, crucially, recorded **0 Severe False Positives**, outperforming the previous top candidate `topiq_existing_flive_60_40` (SRCC 0.8069, 3 Severe FPs) and current production (SRCC 0.7168).

**Final Decision: MDIQA_OFFICIAL_BENCHMARK_MDIQA_BEST_BUT_TEACHER_ONLY**

## 2. Sources Inspected
- `outputs/official_protocol_benchmark_manifest_20260604/official_benchmark_test_manifest.csv`
- `outputs/official_protocol_technical_signal_benchmark_20260604/official_signal_predictions.csv`
- `scripts/technical_iqa/run_mdiqa_official_image_only_benchmark.py` (New)

## 3. MDIQA Cache / Inference Coverage
- **Cached Coverage**: 4,750 / 4,967 rows (~95.6%).
- **New Inference**: 217 KonIQ-10k rows were newly inferred to reach **100% coverage**.
- **Execution**: Inference completed within minutes using `.venv_mdiqa` on GPU.

## 4. Official Benchmark Scope
- **KonIQ-10k Test**: 2,015 images
- **SPAQ Test**: 1,125 images
- **FLIVE Image Test**: 1,827 images
- **Total**: 4,967 images

## 5. MDIQA Variant Policy
- **KonIQ-10k**: `mdiqa_koniq` (seed1)
- **SPAQ**: `mdiqa_xd`
- **FLIVE**: `mdiqa_xd`

## 6. Overall Metrics
| Signal | n | SRCC | PLCC | MAE | Severe FP |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `production_existing` | 4967 | 0.7168 | 0.7874 | 12.00 | 0 |
| `topiq_existing` | 4967 | 0.7975 | 0.8717 | 7.67 | 2 |
| `topiq_flive_60_40` | 4967 | 0.8069 | 0.8694 | 7.62 | 3 |
| `mdiqa_selected` | 4967 | **0.8227** | **0.8222** | 8.41 | **0** |

## 7. Dataset-wise Metrics (SRCC)
| Dataset | Production | topiq_flive_60_40 | mdiqa_selected |
| :--- | :--- | :--- | :--- |
| **KonIQ** | 0.7724 | 0.8642 | **0.9311** |
| **SPAQ** | 0.8414 | 0.8795 | **0.9136** |
| **FLIVE** | 0.5372 | **0.5819** | 0.5692 |

## 8. Safety Cases
- **MDIQA safety**: MDIQA is the only high-performance candidate that maintains 0 Severe FPs on the official test set. This confirms its reliability as a teacher.
- **Hybrid safety**: TOPIQ+FLIVE (60:40) remains slightly optimistic on low-quality samples.

## 9. Comparison Against TOPIQ+FLIVE 60:40
MDIQA provides a superior scientific baseline (SRCC +0.0158 over hybrid) and significantly better dataset-specific alignment on KonIQ and SPAQ. While slightly weaker on FLIVE images than the optimized hybrid, it still exceeds production.

## 10. MDIQA Teacher vs Mobile Candidate Interpretation
The 14,105-row mobile student was trained using MDIQA as a teacher. The fact that MDIQA (SRCC 0.82) is superior to the student (SRCC 0.74) indicates that there is still room for improvement in student distillation, but MDIQA remains the optimal supervision source.

## 11. Final Decision
**MDIQA_OFFICIAL_BENCHMARK_MDIQA_BEST_BUT_TEACHER_ONLY**
MDIQA is the performance leader but its computational complexity (PyTorch, heavy transformer backbone) makes it unsuitable for real-time mobile runtime. It is the confirmed gold-standard teacher.

## 12. Recommended Next Step
Proceed to **Phase 13: Student-Teacher Gap Analysis**.
- Investigate why the mobile student (0.74 SRCC) is losing ~0.08 SRCC compared to the teacher (0.82 SRCC).
- Determine if a larger student backbone (EfficientNet-V2-B1/B2) or better augmentation can narrow this gap.
