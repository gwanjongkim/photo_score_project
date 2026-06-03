# Context Notes

- Scope is code-generation/benchmark-prediction only. No Flutter, production scoring, model files, retraining, or weight tuning.
- Requested `assets/models/...` files are not present in this checkout. Equivalent local artifacts matching the requested metadata were found under `forWeights2/models/aesthetic/` and `outputs/flutter_aesthetic_metadata_20260523/`.
- Existing primary predictions to reuse are NIMA AVA, adaptive A-LAMP AVA, RGNet AADB, and ICAA ICAA17K.
- Missing primary same-row predictions are AVA RGNet, AVA ICAA, AADB NIMA, AADB adaptive A-LAMP, and AADB ICAA.
- The script must record failures instead of silently skipping rows and must not use old non-primary predictions for optimization.
- Smoke predictions succeeded for all five missing model/dataset pairs at 20/20 rows.
- Full AADB generated 1000/1000 rows for NIMA, adaptive A-LAMP, and ICAA, with no failures.
- Full AVA generated 25550 prediction rows for RGNet and 25550 prediction rows for ICAA. Both failed on `data/raw/ava/images/3101.jpg` with `not a TIFF file...`; this is recorded in the failure CSVs.
- Complete primary alignments are ready: AVA has 25548 same-row NIMA/RGNet/adaptive A-LAMP/ICAA rows, and AADB has 1000 same-row rows.
- No ensemble weights were tuned in this step; ICAA remains a separate color subscore until the next weight-search step tests its holistic contribution.
