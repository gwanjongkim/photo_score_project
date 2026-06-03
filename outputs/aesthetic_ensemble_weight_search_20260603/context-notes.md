# Context Notes

- Scope is benchmark/analysis only. No Flutter, production scoring, model files, or retraining.
- Existing worktree is dirty with unrelated changes; this task is isolated under `outputs/aesthetic_ensemble_weight_search_20260603/`.
- Primary evidence is constrained to the named 20260603 benchmark/follow-up outputs where possible.
- Initial inventory found primary NIMA AVA, adaptive A-LAMP AVA, RGNet AADB, ICAA ICAA17K, and ICAA SPAQ predictions.
- Complete same-dataset alignments for the requested ensembles are not assumed; missing cross-dataset prediction files must be reported instead of substituted.
- Final primary alignments are incomplete for both requested ensembles: AVA lacks primary RGNet and ICAA predictions, AADB lacks primary NIMA/adaptive A-LAMP/ICAA predictions, and ICAA17K has only ICAA predictions.
- Weight search result CSVs are intentionally header-only because no dataset has all required Group A or Group B model predictions on common rows.
- Non-primary candidates were found for AADB teacher-label caches and an older AVA RGNet benchmark, but they were not mixed into the 20260603 adaptive-primary optimization.
