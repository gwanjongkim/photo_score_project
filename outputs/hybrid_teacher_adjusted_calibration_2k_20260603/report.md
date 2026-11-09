# Adjusted Hybrid Teacher Calibration 2k Revalidation Report

## 1. Executive Summary
- Run mode: `full_2k`.
- Final decision: `ADJUSTED_CALIBRATION_READY_FOR_10K_DRYRUN`.
- Selected/recommended policy: `policy_f_shift_cap3`.
- Policy A source mean shift/gap mismatch: `22.6235` / `34.9940`.
- Recommended source mean shift/gap mismatch: `19.6235` / `34.9940`.

## 2. Sources Inspected
- `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_label_design_review_2k_20260603/hybrid_label_dryrun_2k_manifest.csv`
- `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_adjusted_calibration_design_20260603/previous_calibration_failure_analysis.md`
- `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_adjusted_calibration_design_20260603/adjusted_calibration_policy_spec.md`

## 3. Previous Calibration Failure Analysis
- Prior strong calibration reduced mean shift but expanded MDIQA-source tails and created two SPAQ FP cases.
- Adjusted policies avoid aggressive variance expansion and add FP guards or diagnostic clamps.

## 4. Adjusted Policies Evaluated
- `policy_a_identity`
- `policy_b_blend_80_20`
- `policy_f_shift_cap3`
- `policy_f_shift_cap5`
- `policy_f_shift_cap8`
- `policy_g_residual_25_cap5`
- `policy_g_residual_50_cap5`
- `policy_h_a_fp_clamp`
- `policy_h_f5_fp_clamp`
- `policy_i_zsoft_25_cap5`
- `policy_i_zsoft_50_cap5`

## 5. Calibration Parameters
- `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_adjusted_calibration_2k_20260603/adjusted_calibration_parameters.csv` rows: `21`.

## 6. Policy Metrics Against MOS
- policy_a_identity / fallback_proxy: n=300, SRCC=0.5678, PLCC=0.7382, MAE=3.1363, severeFP=0, FP=0, FN=0, clipped=0
- policy_a_identity / flive: n=700, SRCC=0.6212, PLCC=0.7551, MAE=3.1214, severeFP=0, FP=0, FN=0, clipped=0
- policy_a_identity / koniq10k: n=500, SRCC=0.9487, PLCC=0.9586, MAE=6.4625, severeFP=0, FP=0, FN=0, clipped=0
- policy_a_identity / spaq: n=500, SRCC=0.9086, PLCC=0.9169, MAE=7.6043, severeFP=0, FP=0, FN=0, clipped=0
- policy_b_blend_80_20 / fallback_proxy: n=300, SRCC=0.5678, PLCC=0.7382, MAE=3.1363, severeFP=0, FP=0, FN=0, clipped=0
- policy_b_blend_80_20 / flive: n=700, SRCC=0.6212, PLCC=0.7551, MAE=3.1214, severeFP=0, FP=0, FN=0, clipped=0
- policy_b_blend_80_20 / koniq10k: n=490, SRCC=0.9393, PLCC=0.9522, MAE=6.4388, severeFP=0, FP=0, FN=0, clipped=0
- policy_b_blend_80_20 / spaq: n=500, SRCC=0.9084, PLCC=0.9176, MAE=7.2888, severeFP=0, FP=0, FN=0, clipped=0
- policy_f_shift_cap3 / fallback_proxy: n=300, SRCC=0.5678, PLCC=0.7382, MAE=3.1363, severeFP=0, FP=0, FN=0, clipped=0
- policy_f_shift_cap3 / flive: n=700, SRCC=0.6212, PLCC=0.7551, MAE=3.1214, severeFP=0, FP=0, FN=0, clipped=0
- policy_f_shift_cap3 / koniq10k: n=500, SRCC=0.9487, PLCC=0.9586, MAE=5.4306, severeFP=0, FP=0, FN=0, clipped=0
- policy_f_shift_cap3 / spaq: n=500, SRCC=0.9086, PLCC=0.9169, MAE=6.5726, severeFP=0, FP=0, FN=0, clipped=0
- policy_f_shift_cap5 / fallback_proxy: n=300, SRCC=0.5678, PLCC=0.7382, MAE=3.1363, severeFP=0, FP=0, FN=0, clipped=0
- policy_f_shift_cap5 / flive: n=700, SRCC=0.6212, PLCC=0.7551, MAE=3.1214, severeFP=0, FP=0, FN=0, clipped=0
- policy_f_shift_cap5 / koniq10k: n=500, SRCC=0.9487, PLCC=0.9586, MAE=5.2619, severeFP=0, FP=0, FN=0, clipped=0
- policy_f_shift_cap5 / spaq: n=500, SRCC=0.9086, PLCC=0.9169, MAE=6.4704, severeFP=0, FP=2, FN=0, clipped=0
- policy_f_shift_cap8 / fallback_proxy: n=300, SRCC=0.5678, PLCC=0.7382, MAE=3.1363, severeFP=0, FP=0, FN=0, clipped=0
- policy_f_shift_cap8 / flive: n=700, SRCC=0.6212, PLCC=0.7551, MAE=3.1214, severeFP=0, FP=0, FN=0, clipped=0
- policy_f_shift_cap8 / koniq10k: n=500, SRCC=0.9487, PLCC=0.9586, MAE=5.9378, severeFP=0, FP=0, FN=0, clipped=0
- policy_f_shift_cap8 / spaq: n=500, SRCC=0.9086, PLCC=0.9169, MAE=7.1317, severeFP=0, FP=2, FN=0, clipped=0
- policy_g_residual_25_cap5 / fallback_proxy: n=300, SRCC=0.5678, PLCC=0.7382, MAE=3.0998, severeFP=0, FP=0, FN=0, clipped=0
- policy_g_residual_25_cap5 / flive: n=700, SRCC=0.6212, PLCC=0.7551, MAE=3.0569, severeFP=0, FP=0, FN=0, clipped=0
- policy_g_residual_25_cap5 / koniq10k: n=500, SRCC=0.9487, PLCC=0.9586, MAE=5.9654, severeFP=0, FP=0, FN=0, clipped=0
- policy_g_residual_25_cap5 / spaq: n=500, SRCC=0.9086, PLCC=0.9169, MAE=7.1474, severeFP=0, FP=0, FN=0, clipped=0
- policy_g_residual_50_cap5 / fallback_proxy: n=300, SRCC=0.5678, PLCC=0.7382, MAE=3.0755, severeFP=0, FP=0, FN=0, clipped=0
- policy_g_residual_50_cap5 / flive: n=700, SRCC=0.6212, PLCC=0.7551, MAE=3.0112, severeFP=0, FP=0, FN=0, clipped=0
- policy_g_residual_50_cap5 / koniq10k: n=500, SRCC=0.9487, PLCC=0.9586, MAE=5.5710, severeFP=0, FP=0, FN=0, clipped=0
- policy_g_residual_50_cap5 / spaq: n=500, SRCC=0.9086, PLCC=0.9169, MAE=6.7722, severeFP=0, FP=0, FN=0, clipped=0
- policy_h_a_fp_clamp / fallback_proxy: n=300, SRCC=0.5678, PLCC=0.7382, MAE=3.1363, severeFP=0, FP=0, FN=0, clipped=0
- policy_h_a_fp_clamp / flive: n=700, SRCC=0.6212, PLCC=0.7551, MAE=3.1214, severeFP=0, FP=0, FN=0, clipped=0
- policy_h_a_fp_clamp / koniq10k: n=500, SRCC=0.9487, PLCC=0.9586, MAE=6.4625, severeFP=0, FP=0, FN=0, clipped=0
- policy_h_a_fp_clamp / spaq: n=500, SRCC=0.9086, PLCC=0.9169, MAE=7.6043, severeFP=0, FP=0, FN=0, clipped=0
- policy_h_f5_fp_clamp / fallback_proxy: n=300, SRCC=0.5678, PLCC=0.7382, MAE=3.1363, severeFP=0, FP=0, FN=0, clipped=0
- policy_h_f5_fp_clamp / flive: n=700, SRCC=0.6212, PLCC=0.7551, MAE=3.1214, severeFP=0, FP=0, FN=0, clipped=0
- policy_h_f5_fp_clamp / koniq10k: n=500, SRCC=0.9487, PLCC=0.9586, MAE=5.2619, severeFP=0, FP=0, FN=0, clipped=0
- policy_h_f5_fp_clamp / spaq: n=500, SRCC=0.9091, PLCC=0.9173, MAE=6.4646, severeFP=0, FP=0, FN=0, clipped=0
- policy_i_zsoft_25_cap5 / fallback_proxy: n=300, SRCC=0.5678, PLCC=0.7382, MAE=3.1303, severeFP=0, FP=0, FN=0, clipped=0
- policy_i_zsoft_25_cap5 / flive: n=700, SRCC=0.6212, PLCC=0.7551, MAE=3.0750, severeFP=0, FP=0, FN=0, clipped=0
- policy_i_zsoft_25_cap5 / koniq10k: n=500, SRCC=0.9487, PLCC=0.9586, MAE=5.0310, severeFP=0, FP=0, FN=0, clipped=0
- policy_i_zsoft_25_cap5 / spaq: n=500, SRCC=0.9086, PLCC=0.9169, MAE=7.1105, severeFP=0, FP=0, FN=0, clipped=0
- policy_i_zsoft_50_cap5 / fallback_proxy: n=300, SRCC=0.5678, PLCC=0.7390, MAE=3.1501, severeFP=0, FP=0, FN=0, clipped=0
- policy_i_zsoft_50_cap5 / flive: n=700, SRCC=0.6212, PLCC=0.7558, MAE=3.0743, severeFP=0, FP=0, FN=0, clipped=0
- policy_i_zsoft_50_cap5 / koniq10k: n=500, SRCC=0.9487, PLCC=0.9578, MAE=3.8069, severeFP=0, FP=0, FN=0, clipped=0
- policy_i_zsoft_50_cap5 / spaq: n=500, SRCC=0.9086, PLCC=0.9169, MAE=6.7191, severeFP=0, FP=0, FN=0, clipped=0

## 7. Safety Cases
- policy_f_shift_cap5 / FP: 2
- policy_f_shift_cap8 / FP: 2

## 8. Range Calibration Comparison
- policy_a_identity: mean_shift=22.6235, gap_mismatch=34.9940, max_SRCC_drop=0.0000, FP=0, clipped=0
- policy_b_blend_80_20: mean_shift=21.2622, gap_mismatch=29.3181, max_SRCC_drop=0.0095, FP=0, clipped=0
- policy_f_shift_cap3: mean_shift=19.6235, gap_mismatch=34.9940, max_SRCC_drop=0.0000, FP=0, clipped=0
- policy_f_shift_cap5: mean_shift=17.6235, gap_mismatch=34.9940, max_SRCC_drop=0.0000, FP=2, clipped=0
- policy_f_shift_cap8: mean_shift=14.6235, gap_mismatch=34.9940, max_SRCC_drop=0.0000, FP=2, clipped=0
- policy_g_residual_25_cap5: mean_shift=21.2317, gap_mismatch=35.0051, max_SRCC_drop=0.0000, FP=0, clipped=0
- policy_g_residual_50_cap5: mean_shift=19.8400, gap_mismatch=35.0389, max_SRCC_drop=0.0000, FP=0, clipped=0
- policy_h_a_fp_clamp: mean_shift=22.6235, gap_mismatch=34.9940, max_SRCC_drop=0.0000, FP=0, clipped=0
- policy_h_f5_fp_clamp: mean_shift=17.6264, gap_mismatch=34.9940, max_SRCC_drop=0.0000, FP=0, clipped=0
- policy_i_zsoft_25_cap5: mean_shift=21.3568, gap_mismatch=36.2498, max_SRCC_drop=0.0000, FP=0, clipped=0
- policy_i_zsoft_50_cap5: mean_shift=20.1098, gap_mismatch=36.8066, max_SRCC_drop=0.0000, FP=0, clipped=0

## 9. Selected / Rejected Policy
{
  "policy": "policy_f_shift_cap3",
  "source_mean_shift": 19.62347284692526,
  "source_mean_shift_improvement": 3.0,
  "source_std_ratio": 3.8110577055434174,
  "source_gap_mismatch": 34.99398746788501,
  "source_gap_mismatch_improvement": 0.0,
  "severe_fp": 0.0,
  "fp": 0.0,
  "clipping_count": 0.0,
  "clipping_pct": 0.0,
  "max_srcc_drop_vs_policy_a": 0.0,
  "fallback_proxy_srcc": 0.5677680863120701,
  "fallback_proxy_srcc_drop_vs_policy_a": 0.0,
  "fallback_proxy_mae": 3.136285056455453,
  "flive_srcc": 0.6212216409771099,
  "flive_srcc_drop_vs_policy_a": 0.0,
  "flive_mae": 3.1214117396925447,
  "koniq10k_srcc": 0.948713772081456,
  "koniq10k_srcc_drop_vs_policy_a": 0.0,
  "koniq10k_mae": 5.430578191525136,
  "spaq_srcc": 0.9086102286792862,
  "spaq_srcc_drop_vs_policy_a": 0.0,
  "spaq_mae": 6.572628422808648,
  "selection_score": 3.0
}

## 10. Remaining Risks
- Fallback proxy remains FLIVE-like, not real mobile/unknown data.
- MOS-guarded policies are diagnostic only and cannot directly label unlabeled future rows.
- Full 2k revalidation is required before any 10k dry-run.

## 11. Final Decision
ADJUSTED_CALIBRATION_READY_FOR_10K_DRYRUN

## 12. Recommended Next Step
- Proceed to a 10k dry-run using the recommended adjusted policy; do not create final labels yet.
