# Technical IQA Teacher Hybrid Design Audit Report

## 1. Executive Summary
Audit of MDIQA, MUSIQ, and Production technical scoring concludes that a **hybrid teacher design** is feasible and numerically promising for **KonIQ-10k** and **SPAQ**, but requires careful **statistical validation** and **dataset-aware risk mitigation**. MDIQA significantly improves KonIQ/SPAQ results but remains slightly below production on FLIVE. The proposed hybrid target uses MDIQA for KonIQ/SPAQ and retains the existing production score for FLIVE. This approach is intended for **offline label generation** only; mobile runtime must remain dataset-agnostic.

## 2. Sources Inspected
- `outputs/mdiqa_correct_protocol_full_benchmark_20260601/report.md`
- `outputs/musiq_kaggle_sanity_fallback_20260601/report.md`
- `outputs/research_protocol_baseline_eval_audit_20260530/report.md`
- `src/infer/select_best_shots.py`
- `configs/stage5_reference.json`

## 3. Current Production Technical Score
The production technical score is a weighted ensemble implemented in `select_best_shots.py`.
- **Core Weights**: KonIQ-Mobile (0.32), FLIVE-Image-Mobile (0.24), FLIVE-Patch-Mean (0.22), FLIVE-Patch-Min (0.12), MUSIQ-AADB (0.10).
- **Relative Ratio**: The relative importance of KonIQ-to-FLIVE (image-only) is exactly **4/7 (~0.57)** to **3/7 (~0.43)**.
- **Status**: C6, TOPIQ, MDIQA, and Official MUSIQ (Kaggle) are currently **log-only/comparison-only** and not used in production.

## 4. MDIQA vs Production Result Summary
MDIQA (correct protocol) shows clear wins on two out of three target datasets.
- **KonIQ-10k**: SRCC 0.9464 vs 0.934 (**+0.0124** on n=20150).
- **SPAQ**: SRCC 0.9136 vs 0.873 (**+0.0406** on n=1125).
- **FLIVE**: SRCC 0.5692 vs 0.574 (**-0.0048** on n=1827).

## 5. Statistical Significance Assessment
- **KonIQ**: The +0.0124 improvement on n=20150 images is likely highly significant due to the large sample size.
- **SPAQ**: The +0.0406 improvement on n=1125 is numerically strong.
- **Limitation**: Final significance cannot be proven without **paired prediction files** for the production baseline on these same sets. Aggregate comparison confirms numerical superiority only.

## 6. MUSIQ Official vs Local Implementation Clarification
It is critical to distinguish between the two MUSIQ implementations in this project:
- **Official MUSIQ (Kaggle)**: Evaluated recently. Strong on SPAQ (0.946) but weak on others. Requires RGB fallback.
- **Local MUSIQ-AADB**: Part of the current production selector bundle. Trained on AADB.
- **Conclusion**: Official MUSIQ is a strong SPAQ-only support signal, but MDIQA provides more balanced improvements.

## 7. Hybrid Teacher Feasibility
A hybrid teacher is valid for **offline label generation**:
- **Target Logic**: `target = MDIQA if dataset in {KonIQ, SPAQ} else Production`.
- **Safety**: This ensures the distilled student never sees a target worse than the current production baseline for any dataset.
- **Runtime**: This logic is strictly for distillation. The resulting student model will be dataset-agnostic at runtime.

## 8. Dataset-Aware Target Risks
- **Overfitting**: A student might learn to distinguish datasets by their visual properties and apply dataset-specific biases.
- **Inconsistency**: Similar quality images from different datasets might receive different targets, confusing the model gradients.
- **Mitigation**: Standardize all evaluations to full-image test splits and ensure score ranges are aligned across datasets.

## 9. Real A-cut Mobile Generalization Risks
Laboratory datasets (KonIQ, SPAQ, FLIVE) often differ from real-world mobile photo distributions.
- **Risk**: A student trained only on these datasets may become unstable on user images.
- **Requirement**: Validation on a **real A-cut mobile holdout set** is mandatory before production acceptance.

## 10. Recommended Validation Gates
1. **Paired Significance**: Run paired bootstrap tests using production-vs-MDIQA per-image scores.
2. **Holdout Parity**: Verify MDIQA predictions on real A-cut mobile images align with production trends.
3. **Range Alignment**: Confirm MDIQA 0..100 scale matches production 0..100 distribution to avoid shifting the final selector output.

## 11. Final Decision
**`HYBRID_TEACHER_NEEDS_PRODUCTION_PREDICTIONS_FIRST`**

## 12. Next Actions
- Generate paired production predictions for the KonIQ-10k and SPAQ test sets used in the MDIQA benchmark.
- Conduct paired statistical significance testing.
- Evaluate MDIQA generalization on real A-cut mobile photos.
