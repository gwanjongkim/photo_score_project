# Context Notes

- Task scope is benchmark/analysis only. Do not modify Flutter code, production scoring, model files, retraining code, or inference outputs.
- Input tables are restricted to `outputs/aesthetic_cross_predictions_for_weight_search_20260603/aligned_ava_complete_primary.csv` and `outputs/aesthetic_cross_predictions_for_weight_search_20260603/aligned_aadb_complete_primary.csv`.
- Output directory is new: `outputs/aesthetic_ensemble_weight_search_complete_20260603/`.
- Column inspection showed AVA `gt_score` is already on a 1..10 scale, while AADB `gt_score` is on a 0..1 scale and must be converted to 1..10 for common-scale metrics.
- Prediction columns with explicit common scale are `nima_pred_scaled_1_10`, `rgnet_pred_scaled_1_10`, `alamp_adaptive_pred_scaled_1_10`, and `icaa_pred_scaled_1_10`.
- Calibration will be fit on the 60% search/train fold and applied to validation/test. Weights are selected on validation only, then reported on held-out test.
- Final research-best and production candidate are both Group A calibrated combined-objective weights: NIMA 0.37, RGNet 0.41, A-LAMP adaptive 0.22, ICAA 0.00.
- Best Group B calibrated combined-objective weights are NIMA 0.33, RGNet 0.36, A-LAMP adaptive 0.22, ICAA 0.09, but this underperforms Group A on held-out combined SRCC by 0.003274 and reduces AADB SRCC by 0.011971.
- Decision: keep ICAA as a separate color subscore for production because it does not improve holistic benchmark performance by the required +0.005 combined SRCC.
