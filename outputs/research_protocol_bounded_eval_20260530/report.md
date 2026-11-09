# Research-Protocol Technical IQA Bounded Evaluation

## 1. Summary
Bounded evaluation succeeded for existing standalone KonIQ and FLIVE TFLite baselines.

## 2. Inputs
- `outputs/research_protocol_iqa_manifests_20260530/benchmark_koniq10k_musiq_80_20_10runs.csv`: selected `100` rows.
- `outputs/research_protocol_iqa_manifests_20260530/benchmark_spaq_official_split.csv`: selected `100` rows.
- `outputs/research_protocol_iqa_manifests_20260530/benchmark_flive_paq2piq_toolbox_split.csv`: selected `100` rows.

Selection:
- KonIQ-10k: `run_id=run_01`, `split=test`, `100` sampled rows.
- SPAQ: `split=test`, `100` sampled rows.
- FLIVE / PaQ-2-PiQ: `content_type=image`, `split=test`, `100` sampled rows. Patch rows were excluded.
- Sampling seed: `42`.

## 3. Safety Checks
- Total input rows: `300`.
- Max row cap: `300`.
- No full benchmark was run.
- No training was run.
- No production code was modified.
- No Flutter files were touched.
- C6 skipped.
- production technical_score skipped.

## 4. Baselines Evaluated
- KonIQ standalone: `exports/tflite/koniq_mobile.tflite`.
- FLIVE standalone: `exports/tflite/flive_image_mobile.tflite`.

## 5. Metrics
Paper-style metrics are SRCC and PLCC. A-cut diagnostics are MAE, RMSE, and Max AE.

- flive / test / flive_standalone: n=100, SRCC=0.570321, PLCC=0.712994, MAE=4.42896, RMSE=5.93668, Max AE=20.7749.
- flive / test / koniq_standalone: n=100, SRCC=0.455362, PLCC=0.674402, MAE=8.91018, RMSE=10.8847, Max AE=27.7186.
- koniq10k / test / flive_standalone: n=100, SRCC=0.77813, PLCC=0.814365, MAE=9.37317, RMSE=11.6899, Max AE=27.4649.
- koniq10k / test / koniq_standalone: n=100, SRCC=0.938199, PLCC=0.957073, MAE=5.329, RMSE=6.98862, Max AE=22.3357.
- spaq / test / flive_standalone: n=100, SRCC=0.810263, PLCC=0.663946, MAE=20.7594, RMSE=24.5249, Max AE=53.4136.
- spaq / test / koniq_standalone: n=100, SRCC=0.85426, PLCC=0.831029, MAE=9.28428, RMSE=11.7494, Max AE=37.7016.

## 6. Runtime
- all / all: seconds=4.54721, avg_ms_per_input_image=15.1574, avg_ms_per_prediction=7.57868.
- model_invoke_only / flive_standalone: seconds=1.49976, avg_ms_per_input_image=4.99919, avg_ms_per_prediction=4.99919.
- model_invoke_only / koniq_standalone: seconds=1.55229, avg_ms_per_input_image=5.17431, avg_ms_per_prediction=5.17431.

## 7. Issues Found
- No path, preprocessing, model loading, output scale, or score range issue found.

## 8. Decision
A. Bounded evaluation passed; ready to plan larger split-aware evaluation.

## 9. Next Action
Plan a larger split-aware evaluation with an explicit row budget and the same standalone-baseline adapter pattern.

## 10. Forbidden Actions
- Do not run full benchmark yet.
- Do not train.
- Do not replace production technical_score.
- Do not use C6 as production.
