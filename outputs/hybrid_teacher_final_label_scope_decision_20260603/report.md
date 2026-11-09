# Hybrid Teacher Final Label Scope Decision Report

## 1. Executive Summary
After completion of the Phase 5 10k dry-run and the subsequent Unknown/Real-mobile fallback audit, we have determined that the project is ready for full-scale label generation **strictly limited to known datasets** (KonIQ-10k, SPAQ, and FLIVE). Real-mobile fallback data (c6_parity) and product-gold labels (Test_weight) exhibit technical-aesthetic mismatches and environment constraints (CuDNN mismatch) that make them unsuitable for main training labels at this time.

**Final Decision: KNOWN_DATASET_ONLY_LABEL_DRYRUN_READY**

## 2. Phase 5 Known-dataset Dry-run Result
The 10k dry-run using `policy_f_shift_cap3` was successful. MDIQA-source targets for KonIQ and SPAQ showed 0 severe FPs and high SRCC (0.91-0.94) against MOS. FLIVE production scores remained stable (SRCC 0.63). Range calibration improved mean shift by 4 points compared to the baseline.

## 3. Unknown / Real-mobile Fallback Audit Reinterpretation
The audit identified 50 real-mobile rows with existing production scores. While stable, the sample size is too small for full fallback training. Furthermore, bulk production scoring of new images is blocked by a CuDNN mismatch in the `.venv_tf` environment.

## 4. Technical vs Aesthetic/Product Label Separation
A critical finding is the divergence between product labels (Bad/Soso/Good) and technical targets. Product labels incorporate aesthetic preferences and specific device-intent that technical IQA models (NIMA, MUSIQ, MDIQA) are not designed to supervise directly.

## 5. IMG_2785 Case Interpretation
`IMG_2785.JPG` (labeled Bad, scored 68.26) represents a technical-aesthetic mismatch. It contains lens/waterdrop artifacts that the technical model is optimistic about. However, the product "Bad" label also reflects aesthetic rejection. Treating this as a severe technical FP would be misleading.

## 6. Final Label Scope Recommendation
- **KonIQ-10k**: Include (MDIQA seed1 + policy_f_shift_cap3).
- **SPAQ**: Include (MDIQA xd + policy_f_shift_cap3).
- **FLIVE**: Include (Production technical score).
- **Unknown / Fallback**: **Exclude** from Phase 6.

## 7. Excluded / Holdout Data
- `c6_parity` (50 rows): Keep as real-mobile holdout for student validation only.
- `Test_weight` / `product-gold`: Keep for aesthetic/product validation only.

## 8. Remaining Risks
- **Coverage**: Excluding fallback means the student will not see explicit "Unknown" labels during training, potentially reducing robustness on future unseen sources.
- **Environment**: CuDNN mismatch remains a blocker for scoring any new real-mobile data.

## 9. Final Decision
**KNOWN_DATASET_ONLY_LABEL_DRYRUN_READY**

## 10. Recommended Next Step
Prepare the known-dataset-only full-label dry-run script and manifest using the validated `policy_f_shift_cap3` for KonIQ and SPAQ.
