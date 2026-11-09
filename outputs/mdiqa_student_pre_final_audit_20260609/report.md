# MDIQA Student Pre-Final Audit Report

## 1. Executive Summary
This pre-final audit (Phase 22) confirms the integrity of the MDIQA distillation pipeline. We verified MDIQA teacher inference code, student architecture suitability, and the 50,000-row training manifest. 

**Status**: `READY_FOR_FINAL_KONIQ_X2_LABEL_SCALING_SMOKE`.

The pipeline is technically sound, but the student model performance (0.70-0.74 SRCC) currently lags behind the production hybrid (0.80 SRCC). The identified bottleneck is **teacher label compression** (especially in KonIQ-10k targets) and **dataset-specific scale bias**.

## 2. Sources Inspected
- `outputs/official_protocol_mdiqa_signal_benchmark_20260604/report.md`
- `outputs/mdiqa_distillation_50k_full_targets_20260608/report.md`
- `scripts/technical_iqa/run_mdiqa_official_image_only_benchmark.py`
- `scripts/technical_iqa/train_mobile_student_iqa_50k_smoke.py`
- `outputs/mobile_student_iqa_50k_controlled_b0_20260608/report.md`
- `outputs/mobile_student_iqa_50k_label_scaling_ranking_plan_20260609/report.md`

## 3. Original Goal Alignment
The goal of replacing the KonIQ+FLIVE hybrid (0.80 SRCC) with a single mobile student is **at risk**. While the student beats the current production baseline (0.71 SRCC), it has not yet surpassed the hybrid candidate. One final attempt with label scaling and ranking is recommended before a final decision to terminate the line.

## 4. MDIQA Code Correctness Audit
- **Variants**: Correct (`koniq_seed1` for KonIQ, `mdiqa_xd` for others).
- **Preprocessing**: Correct (384, Bicubic, [0, 1] normalization).
- **Inference**: Verified to match official benchmark reports (0.8227 overall SRCC).

## 5. MDIQA Variant and Preprocessing Audit
See `mdiqa_variant_preprocess_audit.csv`. Variant routing is consistent between official benchmarking and 50k target generation.

## 6. Student Model Architecture Audit
- **Backbone**: EfficientNetV2B0 (Supported and mobile-plausible).
- **Head**: Standard GAP + Dense 256 + Dense 64 + Sigmoid.
- **Export**: TFLite compatibility confirmed via 14k student export.

## 7. Student Preprocessing and Target Scaling Audit
- **Preprocessing**: `include_preprocessing=True` is used, correctly handling [0, 255] float32 inputs.
- **Scaling**: Teacher targets (0-100) are normalized to (0-1) for Sigmoid training. Consistent.

## 8. Training Data Quality Audit
- **Manifest**: Exactly 50,000 valid rows.
- **Purity**: 0 FLIVE patches, 0 leakage into official Track B test set.
- **Coverage**: 33.8k FLIVE, 8.9k SPAQ, 7.1k KonIQ.

## 9. Split, Leakage, and Distribution Audit
- **Overlap**: 0 path/group overlap across train/val/test splits.
- **Consistency**: Internal splits show similar target distributions.

## 10. Failure / Outlier Cases
- **KonIQ Compression**: Teacher scores for KonIQ are compressed to 43.46-69.06. This limits the student's ability to learn fine-grained rankings without scaling.
- **SPAQ/FLIVE Outliers**: Extremely low scores (9.46) identified; verify if these represent genuine quality degradation or teacher failures.

## 11. Root Cause Reassessment
The drop from 0.74 (14k student) to 0.70 (50k student) despite more data is likely due to the inclusion of harder MDIQA targets or changes in dataset weighting. The overall student gap (0.74 vs 0.82 teacher) is primarily due to label compression and the difficulty of a single model learning three disparate dataset scales simultaneously.

## 12. Go/No-Go Recommendation
**GO** for one final experiment: `READY_FOR_FINAL_KONIQ_X2_LABEL_SCALING_SMOKE`.

## 13. Final Decision
Proceed with **Phase 21-S2: Label Scaling Smoke**. If the overall SRCC does not exceed 0.75-0.78, the MDIQA student line should be finalized as a "Research reference only" and not deployed.

## 14. Next Actions
1. Run `scripts/technical_iqa/train_mobile_student_iqa_50k_ranking_smoke.py` with `S2_koniq_x2` scaling.
2. Evaluate on official Track B and compare against hybrid baseline.
