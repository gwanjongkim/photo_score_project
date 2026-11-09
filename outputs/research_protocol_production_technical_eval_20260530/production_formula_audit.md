# Production Formula Audit

## Files Inspected
- `src/infer/select_best_shots.py`
- `src/infer/run_acut_pipeline.py`
- `src/infer/predict_mobile_aesthetic.py`
- `src/infer/predict_quality_bundle.py`
- `configs/stage5_reference.json`
- `configs/stage5_student_server.json`
- `outputs/research_protocol_image_test_eval_20260530/image_test_predictions.csv`

## Confirmed Formula
Formula status: confirmed for the A-cut selector technical component with available image-level scores.

`src/infer/select_best_shots.py:31-49` defines default technical weights:
- `koniq_score`: `0.32`
- `flive_image_score`: `0.24`
- `flive_patch_mean`: `0.22`
- `flive_patch_min`: `0.12`
- `musiq_score`: `0.10`

`src/infer/select_best_shots.py:170-180` normalizes raw scores to unit scale. Values in `[0, 100]` are divided by `100`.

`src/infer/select_best_shots.py:276-306` computes a component score from available model scores only. Missing model scores are skipped, and available weights are renormalized by their available total weight.

`src/infer/select_best_shots.py:336-355` stores this value as `technical_component`; if only the technical component is present, selector `final_score` falls back to the technical component.

`src/infer/run_acut_pipeline.py:373-375` loads selector weights and records, and `src/infer/run_acut_pipeline.py:452-460` calls `rank_records(...)`, which invokes `compute_selector_row(...)`.

For this image-level evaluation, the existing prediction file contains only `koniq_standalone` and `flive_standalone`, so the confirmed selector logic uses the two available technical scores and renormalizes the configured weights:

`technical_component_unit = (0.32 * koniq_unit + 0.24 * flive_image_unit) / (0.32 + 0.24)`

Since both standalone predictions are on `0..100` scale:

`production_technical_score_0_100 = (0.32 * koniq_standalone + 0.24 * flive_standalone) / 0.56`

Equivalent:

`production_technical_score_0_100 = (4/7 * koniq_standalone) + (3/7 * flive_standalone)`

## Uncertainty Notes
- This is an offline evaluation of the selector technical component using the available image-level KonIQ and FLIVE image scores.
- It does not claim to evaluate patch or MUSIQ branches, because the requested evaluation excludes patch rows and reuses the confirmed image-level standalone predictions.
- `src/infer/predict_mobile_aesthetic.py:12-24` contains a separate mobile policy with `0.60/0.40` technical weights. That policy is not used here because the production A-cut pipeline imports and calls the selector path through `run_acut_pipeline.py`.
