# Research-Protocol Technical IQA Image-Level Test Evaluation

## 1. Summary
Image-level test evaluation succeeded for existing standalone KonIQ and FLIVE TFLite baselines.

## 2. Inputs
- `outputs/research_protocol_iqa_manifests_20260530/benchmark_koniq10k_musiq_80_20_10runs.csv`: selected `2015` rows.
- `outputs/research_protocol_iqa_manifests_20260530/benchmark_spaq_official_split.csv`: selected `1125` rows.
- `outputs/research_protocol_iqa_manifests_20260530/benchmark_flive_paq2piq_toolbox_split.csv`: selected `1827` rows.

Expected row counts:
- KonIQ-10k `run_01` test: `2015`.
- SPAQ official test: `1125`.
- FLIVE / PaQ-2-PiQ image test: `1827`.

## 3. Safety Checks
- Total input rows: `4967`.
- Expected total row count: `4967`.
- Max row cap: `4967`.
- No full 271k benchmark was run.
- No training was run.
- No production code was modified.
- No Flutter files were touched.
- C6 skipped.
- production technical_score skipped.
- Patch rows excluded.

## 4. Baselines Evaluated
- KonIQ standalone: `exports/tflite/koniq_mobile.tflite`.
- FLIVE standalone: `exports/tflite/flive_image_mobile.tflite`.

## 5. Metrics
Paper-style metrics are SRCC and PLCC. A-cut diagnostics are MAE, RMSE, and Max AE.

- flive / test / flive_standalone: n=1827, SRCC=0.660292, PLCC=0.773241, MAE=3.80165, RMSE=5.05028, Max AE=31.0912.
- flive / test / koniq_standalone: n=1827, SRCC=0.486333, PLCC=0.602447, MAE=8.94848, RMSE=11.3053, Max AE=52.6934.
- koniq10k / test / flive_standalone: n=2015, SRCC=0.720587, PLCC=0.763063, MAE=9.64147, RMSE=11.9886, Max AE=37.0794.
- koniq10k / test / koniq_standalone: n=2015, SRCC=0.936918, PLCC=0.952237, MAE=5.18017, RMSE=6.863, Max AE=23.9526.
- spaq / test / flive_standalone: n=1125, SRCC=0.826652, PLCC=0.681455, MAE=20.8763, RMSE=24.9845, Max AE=53.7123.
- spaq / test / koniq_standalone: n=1125, SRCC=0.855512, PLCC=0.84514, MAE=9.58066, RMSE=11.8064, Max AE=42.4665.

## 6. Score Range Checks
- flive / flive_standalone: target min/max/mean/std=13.5203/90.5344/71.8765/7.81573; prediction min/max/mean/std=16.6976/91.801/72.8298/5.85891.
- flive / koniq_standalone: target min/max/mean/std=13.5203/90.5344/71.8765/7.81573; prediction min/max/mean/std=13.1912/86.5663/64.3131/10.338.
- koniq10k / flive_standalone: target min/max/mean/std=23.8532/84.9057/63.0098/11.0945; prediction min/max/mean/std=39.5101/88.6997/72.2441/5.81323.
- koniq10k / koniq_standalone: target min/max/mean/std=23.8532/84.9057/63.0098/11.0945; prediction min/max/mean/std=3.81572/89.9204/59.4971/15.39.
- spaq / flive_standalone: target min/max/mean/std=4.8/91.7/47.9479/21.9359; prediction min/max/mean/std=17.7014/81.6793/66.3687/9.73597.
- spaq / koniq_standalone: target min/max/mean/std=4.8/91.7/47.9479/21.9359; prediction min/max/mean/std=6.40328/81.9308/47.6285/19.8796.

## 7. Runtime
- all / all: seconds=74.4601, avg_ms_per_input_image=14.991, avg_ms_per_prediction=7.49548.
- model_invoke_only / flive_standalone: seconds=24.4786, avg_ms_per_input_image=4.92824, avg_ms_per_prediction=4.92824.
- model_invoke_only / koniq_standalone: seconds=25.4167, avg_ms_per_input_image=5.11712, avg_ms_per_prediction=5.11712.

## 8. Issues Found
- No path, preprocessing, model loading, output scale, calibration, or score range issue found.

## 9. Decision
A. Image-level test evaluation passed; ready for analysis with Gemini.

## 10. Next Action
Analyze `image_test_metrics.csv`, `score_range_checks.csv`, and `runtime_summary.csv` with Gemini before any larger benchmark.

## 11. Forbidden Actions
- Do not run full 271k benchmark yet.
- Do not train.
- Do not replace production technical_score.
- Do not use C6 as production.
