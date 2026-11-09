# Phase 19 Context Notes

- Scope: inverse-frequency corrected W5-v2 B0 smoke only. No B1/B2, no full controlled training, no TFLite export, no Flutter edits, no production `technical_score`, no commit.
- Phase 18 code already applies sample weights only to training dataset through `smoke.make_dataset(..., sample_weight_col="dataset_sample_weight")`.
- Validation/test prediction and official benchmark use unweighted datasets from `smoke.predict_split`.
- W5-v2 weights: FLIVE 0.443, KonIQ-10k 2.433, SPAQ 1.963.
- Need `dataset_weight_effective_contribution.csv` with count/share/weight/effective count/effective share for train/val/test.
- Escalated `.venv_tf` run saw one TensorFlow GPU and completed in `239.38` seconds.
- Train effective contribution hit target: FLIVE `0.300353`, KonIQ-10k `0.349743`, SPAQ `0.349904`.
- Official Track B Overall SRCC was `0.516337`, worse than Phase 18 W5-lite `0.530639` and Phase 16 unweighted B0 `0.703290`.
- No collapse, no NaN/Inf, failure rows `0`, official failure rows `0`.
- Final decision: `WEIGHTED_B0_W5V2_SMOKE_NEEDS_LABEL_SCALING_OR_RANKING`.
