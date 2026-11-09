# MDIQA vs Production Paired Significance Report

## 1. Executive Summary
Paired significance testing on n=100 images per dataset confirms that **MDIQA statistically outperforms current Production technical scoring** on both KonIQ-10k and SPAQ. 
- **KonIQ-10k**: MDIQA SRCC **0.9409** vs Production **0.8036** (Delta **+0.1373**, 95% CI **[0.058, 0.249]**, p < 0.001).
- **SPAQ**: MDIQA SRCC **0.9558** vs Production **0.8832** (Delta **+0.0726**, 95% CI **[0.036, 0.120]**, p < 0.001).
MDIQA is ready for consideration as a partial teacher for these datasets.

## 2. Sources Inspected
- `outputs/mdiqa_correct_protocol_full_benchmark_20260601/predictions.csv`
- `src/infer/select_best_shots.py`
- `configs/stage5_reference.json`

## 3. Production Scoring Source
- **Core Logic**: `src/infer/select_best_shots.py` ensemble using `QualityBundle` components.
- **Components**: KonIQ-Mobile (32%), FLIVE-Image (24%), FLIVE-Patch (34%), MUSIQ-AADB (10%).
- **Execution**: Forced CPU fallback in `.venv_gpu` due to CuDNN version mismatch.

## 4. Paired Evaluation Manifest
- **Source**: MDIQA full benchmark predictions.
- **Filtering**: KonIQ-10k and SPAQ only; valid image paths and targets.
- **Sample Size**: n=100 per dataset (total 200) processed for significance.

## 5. Production Prediction Generation
- **Environment**: `.venv_gpu` (CPU execution).
- **Runtime**: ~4.2s per image steady-state.
- **Success Rate**: 100% on evaluated sample.

## 6. Dataset-wise Paired Metrics
| Dataset | n | SRCC (MDIQA) | SRCC (Prod) | Delta SRCC | PLCC (MDIQA) | PLCC (Prod) |
|---|---|---|---|---|---|---|
| KonIQ-10k | 100 | **0.9409** | 0.8036 | **+0.1373** | 0.9654 | 0.7955 |
| SPAQ | 100 | **0.9558** | 0.8832 | **+0.0726** | 0.9659 | 0.8761 |

## 7. Paired Bootstrap Significance Test
- **Iterations**: 5000
- **KonIQ 95% CI**: `[0.0582, 0.2493]`
- **SPAQ 95% CI**: `[0.0363, 0.1199]`
- **Probability MDIQA > Production**: >99.9% for both datasets.

## 8. Range and Scale Alignment
- **KonIQ**: MDIQA Mean 59.8 (Std 4.9) vs Prod Mean 54.4 (Std 7.5).
- **SPAQ**: MDIQA Mean 43.6 (Std 18.8) vs Prod Mean 48.9 (Std 14.6).
- **Observation**: Score distributions are reasonably aligned within the 0..100 range, though MDIQA is tighter on KonIQ.

## 9. Failure Cases
- None in the n=200 sample.

## 10. Hybrid Teacher Decision
MDIQA is a superior signal for KonIQ and SPAQ. The hybrid design (MDIQA for KonIQ/SPAQ, Production for FLIVE) is validated by paired image-level evidence.

## 11. Final Decision
**`HYBRID_TEACHER_READY_FOR_LABEL_DESIGN_REVIEW`**

## 12. Recommended Next Step
Proceed to draft the offline teacher-label generation script using MDIQA (seed1 for KonIQ, xd for SPAQ) and existing Production/MOS for FLIVE and mobile images.
