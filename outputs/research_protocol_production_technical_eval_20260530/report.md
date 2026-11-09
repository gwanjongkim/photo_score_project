# Production Technical Score Image-Level Test Evaluation

## 1. Summary
Production technical_score evaluation succeeded as an offline analysis of the confirmed A-cut selector technical component.

## 2. Production Formula Audit
- Formula status: confirmed.
- Exact files inspected: `src/infer/select_best_shots.py`, `src/infer/run_acut_pipeline.py`, `src/infer/predict_mobile_aesthetic.py`, `src/infer/predict_quality_bundle.py`, `configs/stage5_reference.json`, `configs/stage5_student_server.json`.
- Logic found: `DEFAULT_SELECTOR_WEIGHTS` defines technical weights, `score_component(...)` normalizes raw scores to unit scale and renormalizes available weights, and `compute_selector_row(...)` stores the result as `technical_component`.
- Plain language formula: use available technical model scores, normalize to unit scale, renormalize configured weights over available scores, then convert back to `0..100` for this report.
- Mathematical formula for this image-level evaluation: `production_technical_score = (0.32 * KonIQ + 0.24 * FLIVE_image) / 0.56`.
- Score scale and normalization: KonIQ and FLIVE standalone predictions are already `0..100`; selector logic divides by `100` internally, combines on unit scale, and this offline report multiplies back to `0..100`.

## 3. Inputs
- `outputs/research_protocol_image_test_eval_20260530/image_test_manifest.csv`: `4967` rows.
- `outputs/research_protocol_image_test_eval_20260530/image_test_predictions.csv`: `9934` standalone prediction rows.

Datasets/splits:
- flive / official / test / image: `1827` rows.
- koniq10k / run_01 / test / image: `2015` rows.
- spaq / official / test / image: `1125` rows.

## 4. Safety Checks
- Total input rows: `4967`.
- No patch rows: `0` non-image rows.
- No full 271k benchmark was run.
- No training was run.
- No production source was modified.
- No Flutter files were touched.
- C6 skipped.
- production score evaluated only as offline analysis.

## 5. Baselines Compared
- KonIQ standalone.
- FLIVE standalone.
- production_technical_score.

## 6. Metrics
- flive / koniq_standalone: SRCC=0.486333, PLCC=0.602447, MAE=8.94848, RMSE=11.3053, Max AE=52.6934, n=1827.
- flive / flive_standalone: SRCC=0.660292, PLCC=0.773241, MAE=3.80165, RMSE=5.05028, Max AE=31.0912, n=1827.
- flive / production_technical_score: SRCC=0.574431, PLCC=0.702134, MAE=5.61073, RMSE=7.19656, Max AE=33.5583, n=1827.
- koniq10k / koniq_standalone: SRCC=0.936918, PLCC=0.952237, MAE=5.18017, RMSE=6.863, Max AE=23.9526, n=2015.
- koniq10k / flive_standalone: SRCC=0.720587, PLCC=0.763063, MAE=9.64147, RMSE=11.9886, Max AE=37.0794, n=2015.
- koniq10k / production_technical_score: SRCC=0.93446, PLCC=0.94759, MAE=3.25617, RMSE=4.05919, Max AE=15.1193, n=2015.
- spaq / koniq_standalone: SRCC=0.855512, PLCC=0.84514, MAE=9.58066, RMSE=11.8064, Max AE=42.4665, n=1125.
- spaq / flive_standalone: SRCC=0.826652, PLCC=0.681455, MAE=20.8763, RMSE=24.9845, Max AE=53.7123, n=1125.
- spaq / production_technical_score: SRCC=0.873043, PLCC=0.847766, MAE=11.8859, RMSE=14.4976, Max AE=43.898, n=1125.

## 7. Score Range and Calibration
- flive / koniq_standalone: pred min/max/mean/std=13.1912/86.5663/64.3131/10.338; gt min/max/mean/std=13.5203/90.5344/71.8765/7.81573; bias=-7.56343; out_of_range=0.
- flive / flive_standalone: pred min/max/mean/std=16.6976/91.801/72.8298/5.85891; gt min/max/mean/std=13.5203/90.5344/71.8765/7.81573; bias=0.953338; out_of_range=0.
- flive / production_technical_score: pred min/max/mean/std=21.5476/82.6883/67.9631/7.83395; gt min/max/mean/std=13.5203/90.5344/71.8765/7.81573; bias=-3.91338; out_of_range=0.
- koniq10k / koniq_standalone: pred min/max/mean/std=3.81572/89.9204/59.4971/15.39; gt min/max/mean/std=23.8532/84.9057/63.0098/11.0945; bias=-3.51269; out_of_range=0.
- koniq10k / flive_standalone: pred min/max/mean/std=39.5101/88.6997/72.2441/5.81323; gt min/max/mean/std=23.8532/84.9057/63.0098/11.0945; bias=9.23431; out_of_range=0.
- koniq10k / production_technical_score: pred min/max/mean/std=19.6251/86.1679/64.9601/10.8436; gt min/max/mean/std=23.8532/84.9057/63.0098/11.0945; bias=1.95031; out_of_range=0.
- spaq / koniq_standalone: pred min/max/mean/std=6.40328/81.9308/47.6285/19.8796; gt min/max/mean/std=4.8/91.7/47.9479/21.9359; bias=-0.319418; out_of_range=0.
- spaq / flive_standalone: pred min/max/mean/std=17.7014/81.6793/66.3687/9.73597; gt min/max/mean/std=4.8/91.7/47.9479/21.9359; bias=18.4208; out_of_range=0.
- spaq / production_technical_score: pred min/max/mean/std=12.786/78.5352/55.66/14.6786; gt min/max/mean/std=4.8/91.7/47.9479/21.9359; bias=7.71211; out_of_range=0.

## 8. Interpretation
- flive: best SRCC baseline is `flive_standalone` (SRCC=0.660292); production_technical_score SRCC=0.574431.
- koniq10k: best SRCC baseline is `koniq_standalone` (SRCC=0.936918); production_technical_score SRCC=0.93446.
- spaq: best SRCC baseline is `production_technical_score` (SRCC=0.873043); production_technical_score SRCC=0.873043.

This evaluates the existing selector technical component; it is not a proposed replacement and does not change production behavior.

## 9. Decision
A. Production technical_score evaluation passed and results are ready for Gemini analysis.

## 10. Next Action
Analyze `production_technical_metrics.csv`, `production_score_range_checks.csv`, and `production_formula_audit.md` with Gemini.

## 11. Forbidden Actions
- Do not replace production technical_score.
- Do not use C6 as production.
- Do not train.
- Do not run full 271k benchmark.
