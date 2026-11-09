# Research-Protocol Technical IQA Smoke Scoring

## 1. Summary
Smoke scoring succeeded for the KonIQ standalone and FLIVE standalone TFLite baselines on the sampled research-protocol manifests.

## 2. Inputs
- `outputs/research_protocol_baseline_eval_audit_20260530/smoke_prep/sampled_manifests/koniq_run01_smoke.csv`: `10` rows.
- `outputs/research_protocol_baseline_eval_audit_20260530/smoke_prep/sampled_manifests/spaq_smoke.csv`: `10` rows.
- `outputs/research_protocol_baseline_eval_audit_20260530/smoke_prep/sampled_manifests/flive_image_smoke.csv`: `15` rows.

## 3. Safety Checks
- Total rows processed: `35` input rows, producing `70` baseline prediction rows.
- Maximum input row cap: `35`.
- No full benchmark was run.
- No training was run.
- No production code was modified.
- No Flutter files were touched.

## 4. Baselines Evaluated
- KonIQ standalone: evaluated with `exports/tflite/koniq_mobile.tflite`.
- FLIVE standalone: evaluated with `exports/tflite/flive_image_mobile.tflite`.
- production technical_score: skipped because this adapter is intentionally limited to standalone TFLite baselines and does not invoke production selector behavior.
- C6 candidate: skipped because C6 is log-only/comparison-only and would require TOPIQ-specific wiring outside the smallest safe adapter.

## 5. Smoke Metrics
Metrics below are sanity checks only. The sampled files contain at most tiny 5-image split samples, so SRCC/PLCC should not be interpreted as benchmark results.

- flive / flive_standalone: n=15, SRCC=0.582143, PLCC=0.712797, MAE=3.79543, RMSE=4.63635, Max AE=8.8106.
- flive / koniq_standalone: n=15, SRCC=0.417857, PLCC=0.422684, MAE=7.73884, RMSE=10.2433, Max AE=24.308.
- koniq10k / flive_standalone: n=10, SRCC=0.806061, PLCC=0.821337, MAE=11.0779, RMSE=13.2657, Max AE=22.9242.
- koniq10k / koniq_standalone: n=10, SRCC=0.915152, PLCC=0.95761, MAE=6.75996, RMSE=8.88813, Max AE=21.4948.
- spaq / flive_standalone: n=10, SRCC=0.575758, PLCC=0.606704, MAE=12.4437, RMSE=15.4989, Max AE=30.0003.
- spaq / koniq_standalone: n=10, SRCC=0.830303, PLCC=0.813403, MAE=9.16269, RMSE=10.2566, Max AE=17.9198.

## 6. Issues Found
- outputs/research_protocol_baseline_eval_audit_20260530/smoke_prep/sampled_manifests/koniq_run01_smoke.csv has no split column; split-level metrics skipped.
- outputs/research_protocol_baseline_eval_audit_20260530/smoke_prep/sampled_manifests/spaq_smoke.csv has no split column; split-level metrics skipped.
- outputs/research_protocol_baseline_eval_audit_20260530/smoke_prep/sampled_manifests/flive_image_smoke.csv has no split column; split-level metrics skipped.

## 7. Decision
A. Smoke scoring passed; ready to plan bounded larger evaluation.

## 8. Next Action
Plan a bounded split-aware evaluation size before running any larger benchmark.

## 9. Forbidden Actions
- Do not run full benchmark yet.
- Do not train.
- Do not replace production technical_score.
- Do not use C6 as production.
