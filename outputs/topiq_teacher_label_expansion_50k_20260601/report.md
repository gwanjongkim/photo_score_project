# TOPIQ Guarded Label 50k Expansion Report

## Executive Summary
- final recommendation: READY_FOR_50K_STUDENT_MANIFEST
- reason: Expanded guarded labels passed basic safety, dataset metric, and distribution checks.
- selected target: E_datasetwise_isotonic__blended_50_50
- calibration fit source: validation predictions only
- this target is for teacher-label design/evaluation only, not a Flutter runtime formula.

## Inputs
- validation prediction rows loaded: 900
- expansion prediction rows loaded: 50000

## Metrics
| scope | dataset | n | source groups | multi-row groups | SRCC | PLCC | MAE | FP | severe FP | FN | raw mean | calibrated mean | guarded mean | guarded std | guarded min | guarded max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | all | 50000 | 41549 | 7258 | 0.945900 | 0.965224 | 2.800063 | 0 | 0 | 0 | 49.680206 | 66.211161 | 66.396579 | 12.619546 | 10.700000 | 87.983333 |
| dataset | flive | 32000 | 23549 | 7258 | 0.924854 | 0.923520 | 2.588026 | 0 | 0 | 0 | 53.499764 | 71.445338 | 71.750489 | 4.555006 | 34.022350 | 87.460718 |
| dataset | koniq10k | 9000 | 9000 | 0 | 0.978121 | 0.980310 | 1.741161 | 0 | 0 | 0 | 46.640452 | 63.493431 | 63.357062 | 10.188629 | 29.478803 | 81.443203 |
| dataset | spaq | 9000 | 9000 | 0 | 0.964015 | 0.961798 | 4.612877 | 0 | 0 | 0 | 39.139309 | 50.318485 | 50.399970 | 18.442116 | 10.700000 | 87.983333 |

## False Case Summary
- false positive cases: 0
- severe false positive cases: 0
- false negative cases: 0
- score histogram CSV: outputs/topiq_teacher_label_expansion_50k_20260601/score_histograms.csv

## Final Recommendation
- READY_FOR_50K_STUDENT_MANIFEST
- Expanded guarded labels passed basic safety, dataset metric, and distribution checks.
