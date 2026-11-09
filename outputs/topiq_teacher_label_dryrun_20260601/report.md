# TOPIQ Guarded Label Dry-Run Report

## Executive Summary

- final recommendation: READY_FOR_STUDENT_TRAINING_MANIFEST_DESIGN
- reason: Bounded dry-run passed safety, dataset metric, and distribution checks.
- selected target: E_datasetwise_isotonic__blended_50_50
- calibration fit source: validation predictions only
- this target is for teacher-label design/evaluation only, not a Flutter runtime formula.

## Inputs

- validation prediction rows loaded: 900
- dry-run prediction rows loaded: 9000

## Metrics

| scope | dataset | n | SRCC | PLCC | MAE | FP | severe FP | FN | raw mean | calibrated mean | guarded mean | guarded min | guarded max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | all | 9000 | 0.959662 | 0.970630 | 2.929083 | 0 | 0 | 0 | 46.632712 | 61.893488 | 61.997276 | 12.750000 | 87.600000 |
| dataset | flive | 3000 | 0.922787 | 0.922836 | 2.565090 | 0 | 0 | 0 | 53.606138 | 71.475158 | 71.722839 | 44.483567 | 84.736838 |
| dataset | koniq10k | 3000 | 0.978415 | 0.981343 | 1.701633 | 0 | 0 | 0 | 46.817518 | 63.603067 | 63.484771 | 29.478803 | 80.252294 |
| dataset | spaq | 3000 | 0.964724 | 0.962657 | 4.520525 | 0 | 0 | 0 | 39.474481 | 50.602238 | 50.784219 | 12.750000 | 87.600000 |

## False Case Summary

- false positive cases: 0
- false negative cases: 0

## Histogram Output

- histogram CSV: outputs/topiq_teacher_label_dryrun_20260601/score_histograms.csv

## Boundary

Do not train a student from this report alone. The next step is training-manifest design if the recommendation is READY.
