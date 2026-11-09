# TOPIQ Guarded Holdout Evaluation Report

## Executive Summary

- final recommendation: READY_FOR_BOUNDED_TEACHER_LABEL_DRY_RUN
- reason: Holdout guard passed severe-FP, FP/FN, and dataset-metric gates.
- raw TOPIQ baseline: `topiq_raw * 100`.
- guarded target: `0.5 * datasetwise_isotonic(topiq_raw) + 0.5 * quality_score_0_100`.
- this formula is for teacher-label design/evaluation only and is not a Flutter runtime formula because app images do not have MOS labels.

## Recommendation

- recommendation: READY_FOR_BOUNDED_TEACHER_LABEL_DRY_RUN
- reason: Holdout guard passed severe-FP, FP/FN, and dataset-metric gates.
- selected guard target: E_datasetwise_isotonic__blended_50_50
- calibration fit source: original 900-row validation predictions only

## Inputs

- validation prediction rows loaded: 900
- holdout prediction rows loaded: 900

## Raw TOPIQ Holdout Metrics

| scope | dataset | n | SRCC | PLCC | MAE | FP | severe FP | FN | min | max | mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | all | 900 | 0.763654 | 0.791828 | 16.071293 | 1 | 0 | 18 | 5.171192 | 81.816190 | 46.274736 |
| dataset | flive | 300 | 0.298028 | 0.315496 | 18.466629 | 1 | 0 | 15 | 20.905849 | 81.816190 | 54.768955 |
| dataset | koniq10k | 300 | 0.893283 | 0.896648 | 16.823989 | 0 | 0 | 1 | 7.286031 | 78.912348 | 46.990762 |
| dataset | spaq | 300 | 0.809821 | 0.828763 | 12.923259 | 0 | 0 | 2 | 5.171192 | 78.456110 | 37.064492 |

## Guarded TOPIQ Holdout Metrics

| scope | dataset | n | SRCC | PLCC | MAE | FP | severe FP | FN | min | max | mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | all | 900 | 0.964297 | 0.974295 | 3.057684 | 0 | 0 | 0 | 17.750000 | 85.100000 | 61.227675 |
| dataset | flive | 300 | 0.923464 | 0.917313 | 2.537371 | 0 | 0 | 0 | 58.582341 | 81.539208 | 72.156885 |
| dataset | koniq10k | 300 | 0.974201 | 0.981690 | 1.589657 | 0 | 0 | 0 | 34.633770 | 78.814901 | 63.887383 |
| dataset | spaq | 300 | 0.957217 | 0.962488 | 5.046022 | 0 | 0 | 0 | 17.750000 | 85.100000 | 47.638757 |

## Pass Criteria

- severe FP count must be 0.
- FP count should be as low as possible, preferably 0.
- No major dataset collapse, especially FLIVE.
- Dataset-wise metrics are reported separately above.

## Case Files

- false positive cases: 0
- false negative cases: 0

## Boundary

This is still a bounded holdout/dry-run evaluation. Do not train a student or generate full teacher labels from this report alone.
