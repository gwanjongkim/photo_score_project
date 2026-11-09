# Adjusted Hybrid Teacher Calibration 2k Revalidation Report

## 1. Executive Summary
- Run mode: `smoke`.
- Final decision: `SMOKE_ONLY_NOT_A_FULL_2K_DECISION`.
- Selected/recommended policy: `policy_f_shift_cap8`.
- Policy A source mean shift/gap mismatch: `26.6655` / `38.7120`.
- Recommended source mean shift/gap mismatch: `18.6655` / `38.7120`.

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
- `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_adjusted_calibration_2k_20260603_smoke/adjusted_calibration_parameters.csv` rows: `21`.

## 6. Policy Metrics Against MOS
- policy_a_identity / fallback_proxy: n=12, SRCC=0.3357, PLCC=0.3317, MAE=2.6599, severeFP=0, FP=0, FN=0, clipped=0
- policy_a_identity / flive: n=12, SRCC=0.2378, PLCC=0.3721, MAE=3.4562, severeFP=0, FP=0, FN=0, clipped=0
- policy_a_identity / koniq10k: n=14, SRCC=0.9648, PLCC=0.9952, MAE=7.3420, severeFP=0, FP=0, FN=0, clipped=0
- policy_a_identity / spaq: n=12, SRCC=0.8392, PLCC=0.9669, MAE=5.0321, severeFP=0, FP=0, FN=0, clipped=0
- policy_b_blend_80_20 / fallback_proxy: n=12, SRCC=0.3357, PLCC=0.3317, MAE=2.6599, severeFP=0, FP=0, FN=0, clipped=0
- policy_b_blend_80_20 / flive: n=12, SRCC=0.2378, PLCC=0.3721, MAE=3.4562, severeFP=0, FP=0, FN=0, clipped=0
- policy_b_blend_80_20 / koniq10k: n=14, SRCC=0.9692, PLCC=0.9870, MAE=7.5500, severeFP=0, FP=0, FN=0, clipped=0
- policy_b_blend_80_20 / spaq: n=12, SRCC=0.7972, PLCC=0.9626, MAE=4.7858, severeFP=0, FP=0, FN=0, clipped=0
- policy_f_shift_cap3 / fallback_proxy: n=12, SRCC=0.3357, PLCC=0.3317, MAE=2.6599, severeFP=0, FP=0, FN=0, clipped=0
- policy_f_shift_cap3 / flive: n=12, SRCC=0.2378, PLCC=0.3721, MAE=3.4562, severeFP=0, FP=0, FN=0, clipped=0
- policy_f_shift_cap3 / koniq10k: n=14, SRCC=0.9648, PLCC=0.9952, MAE=6.4848, severeFP=0, FP=0, FN=0, clipped=0
- policy_f_shift_cap3 / spaq: n=12, SRCC=0.8392, PLCC=0.9669, MAE=3.6688, severeFP=0, FP=0, FN=0, clipped=0
- policy_f_shift_cap5 / fallback_proxy: n=12, SRCC=0.3357, PLCC=0.3317, MAE=2.6599, severeFP=0, FP=0, FN=0, clipped=0
- policy_f_shift_cap5 / flive: n=12, SRCC=0.2378, PLCC=0.3721, MAE=3.4562, severeFP=0, FP=0, FN=0, clipped=0
- policy_f_shift_cap5 / koniq10k: n=14, SRCC=0.9648, PLCC=0.9952, MAE=6.1056, severeFP=0, FP=0, FN=0, clipped=0
- policy_f_shift_cap5 / spaq: n=12, SRCC=0.8392, PLCC=0.9669, MAE=4.0329, severeFP=0, FP=0, FN=0, clipped=0
- policy_f_shift_cap8 / fallback_proxy: n=12, SRCC=0.3357, PLCC=0.3317, MAE=2.6599, severeFP=0, FP=0, FN=0, clipped=0
- policy_f_shift_cap8 / flive: n=12, SRCC=0.2378, PLCC=0.3721, MAE=3.4562, severeFP=0, FP=0, FN=0, clipped=0
- policy_f_shift_cap8 / koniq10k: n=14, SRCC=0.9648, PLCC=0.9952, MAE=6.0592, severeFP=0, FP=0, FN=0, clipped=0
- policy_f_shift_cap8 / spaq: n=12, SRCC=0.8392, PLCC=0.9669, MAE=5.8628, severeFP=0, FP=0, FN=0, clipped=0
- policy_g_residual_25_cap5 / fallback_proxy: n=12, SRCC=0.3357, PLCC=0.3317, MAE=2.6599, severeFP=0, FP=0, FN=0, clipped=0
- policy_g_residual_25_cap5 / flive: n=12, SRCC=0.2378, PLCC=0.3721, MAE=3.4239, severeFP=0, FP=0, FN=0, clipped=0
- policy_g_residual_25_cap5 / koniq10k: n=14, SRCC=0.9648, PLCC=0.9952, MAE=6.8231, severeFP=0, FP=0, FN=0, clipped=0
- policy_g_residual_25_cap5 / spaq: n=12, SRCC=0.8392, PLCC=0.9669, MAE=4.5916, severeFP=0, FP=0, FN=0, clipped=0
- policy_g_residual_50_cap5 / fallback_proxy: n=12, SRCC=0.3357, PLCC=0.3317, MAE=2.6599, severeFP=0, FP=0, FN=0, clipped=0
- policy_g_residual_50_cap5 / flive: n=12, SRCC=0.2378, PLCC=0.3721, MAE=3.4067, severeFP=0, FP=0, FN=0, clipped=0
- policy_g_residual_50_cap5 / koniq10k: n=14, SRCC=0.9648, PLCC=0.9952, MAE=6.3042, severeFP=0, FP=0, FN=0, clipped=0
- policy_g_residual_50_cap5 / spaq: n=12, SRCC=0.8392, PLCC=0.9669, MAE=4.1511, severeFP=0, FP=0, FN=0, clipped=0
- policy_h_a_fp_clamp / fallback_proxy: n=12, SRCC=0.3357, PLCC=0.3317, MAE=2.6599, severeFP=0, FP=0, FN=0, clipped=0
- policy_h_a_fp_clamp / flive: n=12, SRCC=0.2378, PLCC=0.3721, MAE=3.4562, severeFP=0, FP=0, FN=0, clipped=0
- policy_h_a_fp_clamp / koniq10k: n=14, SRCC=0.9648, PLCC=0.9952, MAE=7.3420, severeFP=0, FP=0, FN=0, clipped=0
- policy_h_a_fp_clamp / spaq: n=12, SRCC=0.8392, PLCC=0.9669, MAE=5.0321, severeFP=0, FP=0, FN=0, clipped=0
- policy_h_f5_fp_clamp / fallback_proxy: n=12, SRCC=0.3357, PLCC=0.3317, MAE=2.6599, severeFP=0, FP=0, FN=0, clipped=0
- policy_h_f5_fp_clamp / flive: n=12, SRCC=0.2378, PLCC=0.3721, MAE=3.4562, severeFP=0, FP=0, FN=0, clipped=0
- policy_h_f5_fp_clamp / koniq10k: n=14, SRCC=0.9648, PLCC=0.9952, MAE=6.1056, severeFP=0, FP=0, FN=0, clipped=0
- policy_h_f5_fp_clamp / spaq: n=12, SRCC=0.8392, PLCC=0.9669, MAE=4.0329, severeFP=0, FP=0, FN=0, clipped=0
- policy_i_zsoft_25_cap5 / fallback_proxy: n=12, SRCC=0.3357, PLCC=0.3317, MAE=2.6788, severeFP=0, FP=0, FN=0, clipped=0
- policy_i_zsoft_25_cap5 / flive: n=12, SRCC=0.2378, PLCC=0.3721, MAE=3.5363, severeFP=0, FP=0, FN=0, clipped=0
- policy_i_zsoft_25_cap5 / koniq10k: n=14, SRCC=0.9648, PLCC=0.9952, MAE=5.5519, severeFP=0, FP=0, FN=0, clipped=0
- policy_i_zsoft_25_cap5 / spaq: n=12, SRCC=0.8392, PLCC=0.9669, MAE=4.7202, severeFP=0, FP=0, FN=0, clipped=0
- policy_i_zsoft_50_cap5 / fallback_proxy: n=12, SRCC=0.3357, PLCC=0.3317, MAE=2.6977, severeFP=0, FP=0, FN=0, clipped=0
- policy_i_zsoft_50_cap5 / flive: n=12, SRCC=0.2378, PLCC=0.3721, MAE=3.7090, severeFP=0, FP=0, FN=0, clipped=0
- policy_i_zsoft_50_cap5 / koniq10k: n=14, SRCC=0.9648, PLCC=0.9966, MAE=4.1085, severeFP=0, FP=0, FN=0, clipped=0
- policy_i_zsoft_50_cap5 / spaq: n=12, SRCC=0.8392, PLCC=0.9669, MAE=4.4084, severeFP=0, FP=0, FN=0, clipped=0

## 7. Safety Cases
- None.

## 8. Range Calibration Comparison
- policy_a_identity: mean_shift=26.6655, gap_mismatch=38.7120, max_SRCC_drop=0.0000, FP=0, clipped=0
- policy_b_blend_80_20: mean_shift=24.9514, gap_mismatch=33.6085, max_SRCC_drop=0.0420, FP=0, clipped=0
- policy_f_shift_cap3: mean_shift=23.6655, gap_mismatch=38.7120, max_SRCC_drop=0.0000, FP=0, clipped=0
- policy_f_shift_cap5: mean_shift=21.6655, gap_mismatch=38.7120, max_SRCC_drop=0.0000, FP=0, clipped=0
- policy_f_shift_cap8: mean_shift=18.6655, gap_mismatch=38.7120, max_SRCC_drop=0.0000, FP=0, clipped=0
- policy_g_residual_25_cap5: mean_shift=25.1124, gap_mismatch=39.6471, max_SRCC_drop=0.0000, FP=0, clipped=0
- policy_g_residual_50_cap5: mean_shift=23.5593, gap_mismatch=40.5823, max_SRCC_drop=0.0000, FP=0, clipped=0
- policy_h_a_fp_clamp: mean_shift=26.6655, gap_mismatch=38.7120, max_SRCC_drop=0.0000, FP=0, clipped=0
- policy_h_f5_fp_clamp: mean_shift=21.6655, gap_mismatch=38.7120, max_SRCC_drop=0.0000, FP=0, clipped=0
- policy_i_zsoft_25_cap5: mean_shift=25.3713, gap_mismatch=40.6783, max_SRCC_drop=0.0000, FP=0, clipped=0
- policy_i_zsoft_50_cap5: mean_shift=24.2588, gap_mismatch=40.8658, max_SRCC_drop=0.0000, FP=0, clipped=0

## 9. Selected / Rejected Policy
{
  "policy": "policy_f_shift_cap8",
  "source_mean_shift": 18.665477491342102,
  "source_mean_shift_improvement": 8.0,
  "source_std_ratio": 8.21697762992906,
  "source_gap_mismatch": 38.71201068162918,
  "source_gap_mismatch_improvement": 0.0,
  "severe_fp": 0.0,
  "fp": 0.0,
  "clipping_count": 0.0,
  "clipping_pct": 0.0,
  "max_srcc_drop_vs_policy_a": 0.0,
  "fallback_proxy_srcc": 0.3356643356643357,
  "fallback_proxy_srcc_drop_vs_policy_a": 0.0,
  "fallback_proxy_mae": 2.6598642069231233,
  "flive_srcc": 0.23776223776223776,
  "flive_srcc_drop_vs_policy_a": 0.0,
  "flive_mae": 3.456222748277964,
  "koniq10k_srcc": 0.9648351648351648,
  "koniq10k_srcc_drop_vs_policy_a": 0.0,
  "koniq10k_mae": 6.059207858774911,
  "spaq_srcc": 0.8391608391608392,
  "spaq_srcc_drop_vs_policy_a": 0.0,
  "spaq_mae": 5.862811555465064,
  "selection_score": 8.0,
  "full_2k_decision_if_pattern_holds": "ADJUSTED_CALIBRATION_READY_FOR_10K_DRYRUN",
  "smoke_only": true
}

## 10. Remaining Risks
- Fallback proxy remains FLIVE-like, not real mobile/unknown data.
- MOS-guarded policies are diagnostic only and cannot directly label unlabeled future rows.
- Full 2k revalidation is required before any 10k dry-run.

## 11. Final Decision
SMOKE_ONLY_NOT_A_FULL_2K_DECISION

## 12. Recommended Next Step
- Run the full 2k revalidation command after reviewing smoke output.
