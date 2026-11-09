# Hybrid Teacher Range Calibration 2k Audit Report

## 1. Executive Summary
- Final decision: `CALIBRATION_POLICY_NEEDS_ADJUSTMENT`.
- Selected/recommended policy: `policy_d_group_mean_std`.
- Baseline source mean shift/std ratio/gap mismatch: `22.6235` / `3.8111` / `34.9940`.
- Recommended source mean shift/std ratio/gap mismatch: `17.5569` / `3.1878` / `39.6805`.
- This is a calibration design dry-run only; no final labels were created.

## 2. Sources Inspected
- `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_label_design_review_2k_20260603/hybrid_label_dryrun_2k_manifest.csv`
- `outputs/hybrid_teacher_label_design_review_2k_20260603/report.md`
- `outputs/hybrid_teacher_label_design_review_2k_20260603/range_calibration_audit.csv`
- `outputs/paired_production_mdiqa_significance_expanded_20260603/paired_predictions_expanded.csv`
- `outputs/mdiqa_correct_protocol_full_benchmark_20260601/predictions.csv`
- `outputs/technical_iqa_combo_benchmark_20260524/fusion_predictions_flive.csv`

## 3. Baseline Range Problem
- Policy A MDIQA vs production mean shift: `22.6235`.
- Policy A std ratio: `3.8111`.
- Policy A p95/p05 gap mismatch: `34.9940`.

## 4. Calibration Policies Evaluated
- Policy A: identity strict hybrid baseline.
- Policy B: 80/20 conservative blend where production exists.
- Policy C: source-wise 5-fold cross-fit affine MOS calibration.
- Policy D: source-wise group mean/std alignment to own MOS distribution.
- Policy E: MDIQA-only capped conservative mean/std calibration.

## 5. Calibration Parameters
- Parameters CSV: `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_range_calibration_2k_20260603/calibration_parameters.csv`.
- Parameter rows: `28`.

## 6. Policy Metrics Against MOS
- policy_a_identity / fallback_proxy: n=300, SRCC=0.5678, PLCC=0.7382, MAE=3.1363, severeFP=0, FP=0, FN=0, clipped=0
- policy_a_identity / flive: n=700, SRCC=0.6212, PLCC=0.7551, MAE=3.1214, severeFP=0, FP=0, FN=0, clipped=0
- policy_a_identity / koniq10k: n=500, SRCC=0.9487, PLCC=0.9586, MAE=6.4625, severeFP=0, FP=0, FN=0, clipped=0
- policy_a_identity / spaq: n=500, SRCC=0.9086, PLCC=0.9169, MAE=7.6043, severeFP=0, FP=0, FN=0, clipped=0
- policy_b_blend_80_20 / fallback_proxy: n=300, SRCC=0.5678, PLCC=0.7382, MAE=3.1363, severeFP=0, FP=0, FN=0, clipped=0
- policy_b_blend_80_20 / flive: n=700, SRCC=0.6212, PLCC=0.7551, MAE=3.1214, severeFP=0, FP=0, FN=0, clipped=0
- policy_b_blend_80_20 / koniq10k: n=490, SRCC=0.9393, PLCC=0.9522, MAE=6.4388, severeFP=0, FP=0, FN=0, clipped=0
- policy_b_blend_80_20 / spaq: n=500, SRCC=0.9084, PLCC=0.9176, MAE=7.2888, severeFP=0, FP=0, FN=0, clipped=0
- policy_c_crossfit_affine / fallback_proxy: n=300, SRCC=0.5653, PLCC=0.7317, MAE=3.1122, severeFP=0, FP=0, FN=0, clipped=0
- policy_c_crossfit_affine / flive: n=700, SRCC=0.6179, PLCC=0.7502, MAE=3.0097, severeFP=0, FP=0, FN=0, clipped=0
- policy_c_crossfit_affine / koniq10k: n=500, SRCC=0.9484, PLCC=0.9583, MAE=2.3945, severeFP=0, FP=0, FN=0, clipped=0
- policy_c_crossfit_affine / spaq: n=500, SRCC=0.9078, PLCC=0.9161, MAE=6.3824, severeFP=0, FP=2, FN=0, clipped=0
- policy_d_group_mean_std / fallback_proxy: n=300, SRCC=0.5678, PLCC=0.7382, MAE=3.2917, severeFP=0, FP=0, FN=0, clipped=0
- policy_d_group_mean_std / flive: n=700, SRCC=0.6212, PLCC=0.7551, MAE=3.2126, severeFP=0, FP=0, FN=0, clipped=0
- policy_d_group_mean_std / koniq10k: n=500, SRCC=0.9487, PLCC=0.9586, MAE=2.4346, severeFP=0, FP=0, FN=0, clipped=0
- policy_d_group_mean_std / spaq: n=500, SRCC=0.9086, PLCC=0.9169, MAE=6.4542, severeFP=0, FP=2, FN=0, clipped=0
- policy_e_global_conservative / fallback_proxy: n=300, SRCC=0.5678, PLCC=0.7382, MAE=3.1363, severeFP=0, FP=0, FN=0, clipped=0
- policy_e_global_conservative / flive: n=700, SRCC=0.6212, PLCC=0.7551, MAE=3.1214, severeFP=0, FP=0, FN=0, clipped=0
- policy_e_global_conservative / koniq10k: n=500, SRCC=0.9487, PLCC=0.9586, MAE=3.6670, severeFP=0, FP=0, FN=0, clipped=0
- policy_e_global_conservative / spaq: n=500, SRCC=0.9086, PLCC=0.9169, MAE=6.4542, severeFP=0, FP=2, FN=0, clipped=0

## 7. Safety Cases
- policy_c_crossfit_affine / FP: 2
- policy_d_group_mean_std / FP: 2
- policy_e_global_conservative / FP: 2

## 8. Range Calibration Comparison
- policy_a_identity: mean_shift=22.6235, std_ratio=3.8111, gap_mismatch=34.9940, max_SRCC_drop=0.0000, clipped=0
- policy_b_blend_80_20: mean_shift=21.2622, std_ratio=3.3186, gap_mismatch=29.3181, max_SRCC_drop=0.0095, clipped=0
- policy_c_crossfit_affine: mean_shift=17.5579, std_ratio=3.9588, gap_mismatch=40.5490, max_SRCC_drop=0.0033, clipped=0
- policy_d_group_mean_std: mean_shift=17.5569, std_ratio=3.1878, gap_mismatch=39.6805, max_SRCC_drop=0.0000, clipped=0
- policy_e_global_conservative: mean_shift=18.7778, std_ratio=4.2352, gap_mismatch=41.8502, max_SRCC_drop=0.0000, clipped=0

## 9. Selected / Rejected Policy
{
  "policy": "policy_d_group_mean_std",
  "source_mean_shift": 17.55685156993807,
  "source_mean_shift_improvement": 5.066621276987192,
  "source_std_ratio": 3.187769415917544,
  "source_gap_mismatch": 39.680481232311195,
  "source_gap_mismatch_improvement": -4.6864937644261815,
  "severe_fp": 0.0,
  "clipping_count": 0.0,
  "clipping_pct": 0.0,
  "max_srcc_drop_vs_policy_a": 0.0,
  "fallback_proxy_srcc": 0.5677680863120701,
  "fallback_proxy_srcc_drop_vs_policy_a": 0.0,
  "fallback_proxy_mae": 3.291685179405057,
  "fallback_proxy_mae_delta_vs_policy_a": 0.1554001229496036,
  "flive_srcc": 0.6212216409771099,
  "flive_srcc_drop_vs_policy_a": 0.0,
  "flive_mae": 3.2125849237294384,
  "flive_mae_delta_vs_policy_a": 0.09117318403689367,
  "koniq10k_srcc": 0.948713772081456,
  "koniq10k_srcc_drop_vs_policy_a": 0.0,
  "koniq10k_mae": 2.4346209705500668,
  "koniq10k_mae_delta_vs_policy_a": -4.027834203986478,
  "spaq_srcc": 0.9086102286792862,
  "spaq_srcc_drop_vs_policy_a": 0.0,
  "spaq_mae": 6.454229951486966,
  "spaq_mae_delta_vs_policy_a": -1.1501011352678328,
  "rejection_reason": "helpful_but_failed_selection_gate"
}

## 10. Remaining Risks
- Fallback group remains a FLIVE production proxy, not real mobile/unknown data.
- Policy C/D use MOS-derived calibration; future use must freeze parameters and validate on larger dry-run rows.
- Calibration can improve numeric range continuity while preserving rank, but it changes target semantics and should stay diagnostic until 10k dry-run confirms safety.

## 11. Final Decision
CALIBRATION_POLICY_NEEDS_ADJUSTMENT

## 12. Recommended Next Step
- Adjust calibration policy and rerun this 2k audit before any 10k dry-run.
