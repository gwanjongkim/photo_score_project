# Aesthetic-only comparison: stage5 vs current vs conservative

Dataset: `/home/omen_pc1/photo_score_project/test_vila`

Source score CSV: `/home/omen_pc1/photo_score_project/outputs/test_vila_aesthetic_ensemble_experiment_20260423/scores_with_ensemble.csv`

This is a non-production review package. It reuses existing scores and ranks images using aesthetic scores only.

No technical quality fields are used for ranking here. The prior source CSV contains technical/final columns, but this package ignores them.

## Formulas

- `stage5_student_full_aadb`: `stage5_aesthetic_score`
- `ensemble_current`: `0.30 * rgnet_aadb + 0.30 * alamp_aadb + 0.25 * aadb_composition + 0.15 * nima_ava_unit`
- `ensemble_conservative`: `0.35 * rgnet_aadb + 0.20 * alamp_aadb + 0.30 * aadb_composition + 0.15 * nima_ava_unit`

Component normalization is inherited from the prior experiment:

- AADB, A-Lamp, and RGNet are clipped unit scores.
- NIMA unit score is `(nima_mean_score - 1.0) / 9.0`, clipped to `[0, 1]`.

## Suggested review order

1. Open `review_index.html`.
2. Compare `contact_01_stage5_aesthetic_only_top10.jpg`, `contact_02_current_ensemble_aesthetic_only_top10.jpg`, and `contact_03_conservative_ensemble_aesthetic_only_top10.jpg`.
3. Review `contact_04_major_disagreement_cases.jpg`.
4. Review `contact_05_dark_low_light_focus_cases.jpg`.
5. Use `aesthetic_only_comparison.csv` to mark `human_pick` and `human_note`.

## Dark / low-light focus

The `low_light_group` column is a human-auditable grouping hint for this `test_vila` set only. It is meant to help review night, star-sky, silhouette, and mood-heavy frames without mixing in technical quality.

Key question: does the conservative ensemble keep visually strong dark/silhouette images high while reducing any A-Lamp-driven over-reward of mood-heavy darkness?

## Contact sheet render warnings

Some scored files could not be thumbnailed by Pillow for contact sheets. See `contact_sheet_render_warnings.csv`; scores and CSV rows are still present.

