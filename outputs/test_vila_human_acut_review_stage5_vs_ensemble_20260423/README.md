# Human A-cut Review: Stage5 vs Weighted Ensemble

This package reuses existing scores from:

`/home/omen_pc1/photo_score_project/outputs/test_vila_aesthetic_ensemble_experiment_20260423/scores_with_ensemble.csv`

No model scoring or retraining was run to create this package.

## What To Open First

1. `review_index.html` for a visual overview.
2. `human_acut_comparison_sheet.csv` when you want to record manual choices.

## How To Judge

- Compare `contact_01_stage5_top10.jpg` and `contact_02_ensemble_top10.jpg`.
- Use `contact_03_top10_overlap.jpg` to see what both paths agree on.
- Use `contact_04_top10_disagreements.jpg` to inspect the images that only one top-10 list selected.
- Use `contact_05_ensemble_much_higher_than_stage5.jpg` to check whether stage5 is under-rewarding scenic, silhouette, landmark, or portrait compositions.
- Use `contact_06_stage5_much_higher_than_ensemble.jpg` to check whether the ensemble is over-rewarding mood, darkness, or composition at the expense of usable A-cut choices.
- Then review rejection quality with `contact_07_stage5_bottom10.jpg` and `contact_08_ensemble_bottom10.jpg`.
- Use `contact_09_bottom10_overlap.jpg` to see what both paths reject.
- Use `contact_10_bottom10_disagreements.jpg` to inspect images that only one path placed in the bottom 10.
- Use `contact_11_ensemble_much_lower_than_stage5.jpg` and `contact_12_stage5_much_lower_than_ensemble.jpg` to decide which low-ranked disagreements matter for practical A-cut rejection.

## CSV Fill-In

In `human_acut_comparison_sheet.csv`, fill:

- `human_pick`: suggested values are `stage5`, `ensemble`, `both`, `neither`, or `unsure`.
- `human_note`: short reason, for example `better landmark framing`, `too dark`, `stronger portrait`, or `duplicate`.
- `human_reject_pick`: suggested values are `stage5`, `ensemble`, `both`, `neither`, or `unsure`.
- `human_reject_note`: short reason, for example `properly rejected`, `too harsh`, `too dark`, `weak detail`, or `should not be bottom 10`.

## Rank Difference Convention

`rank_difference = ensemble_final_rank - stage5_final_rank`.

- Negative value: ensemble ranks the image higher.
- Positive value: stage5 ranks the image higher.
- Zero: same final rank.

For bottom-ranked review, the same column also tells you which model rejects an image more strongly:

- Positive value: ensemble ranks the image lower.
- Negative value: stage5 ranks the image lower.

## Summary

- Stage5 top-10 count: 10
- Ensemble top-10 count: 10
- Top-10 overlap count: 4
- Stage5-only top-10: 20250204_170733.jpg, 1720499666964.jpg, 1720499666985.jpg, 1675564674752-15.jpg, 1675564674752-2.jpg, 20250204_135520.jpg
- Ensemble-only top-10: 1675342165226-13.jpg, 1675342165226-3.jpg, 1675342165226-5.jpg, 20250204_163550.jpg, 1704296944756.jpg, 1675564423029-9.jpg
- Stage5 bottom-10 count: 10
- Ensemble bottom-10 count: 10
- Bottom-10 overlap count: 4
- Stage5-only bottom-10: 20250205_072644.jpg, 1720499657653.jpg, 1715855266134.jpg, 20240515_073118.jpg, 20230201_181510.jpg, IMG_2748.JPG
- Ensemble-only bottom-10: 1738744324186.jpg, IMG_2738.JPG, IMG_2739.JPG, 20250129_054504.jpg, 20240517_060242.jpg, 20240517_060245.jpg
