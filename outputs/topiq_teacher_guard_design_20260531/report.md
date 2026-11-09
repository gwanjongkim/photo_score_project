# TOPIQ Teacher Guard Design Report

## Recommendation

- recommendation: USE_GUARDED_TOPIQ_TARGET
- selected candidate: E_datasetwise_isotonic__blended_50_50
- reason: Use guarded target candidate `E_datasetwise_isotonic__blended_50_50` for any distillation screen; raw TOPIQ is not safe as a standalone teacher because FLIVE correlation is weak.

## Use Policy

- TOPIQ should not be used as the sole main teacher yet.
- TOPIQ can be used as one component of a guarded teacher target.
- TOPIQ is credible on KonIQ/SPAQ based on the validation report and raw candidate dataset metrics.
- TOPIQ should not be trusted for FLIVE without a target guard or FLIVE-specific handling.

## Calibration Caveat

All calibration candidates are fit and evaluated in-sample on the 900-row validation set. These numbers are useful for guard design, not final generalization proof.

## Inputs

- prediction rows loaded: 900
- validation result summary: 900/900 succeeded, selected model `topiq_nr`, failed 0.

## Base Candidate Metrics

| candidate | dataset | SRCC | PLCC | MAE | FP | severe FP | FN | min | max | mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_topiq_raw_x100 | all | 0.696265 | 0.768974 | 16.731306 | 0 | 0 | 21 | 4.682437 | 80.431986 | 46.138373 |
| B_global_linear | all | 0.696265 | 0.768974 | 8.231051 | 10 | 4 | 0 | 30.153346 | 87.508829 | 61.542645 |
| C_global_isotonic | all | 0.708367 | 0.823363 | 7.022104 | 14 | 6 | 0 | 19.798058 | 77.760571 | 61.542645 |
| D_datasetwise_linear | all | 0.754366 | 0.851846 | 6.438670 | 8 | 2 | 0 | 21.810482 | 81.081238 | 61.542645 |
| E_datasetwise_isotonic | all | 0.783691 | 0.872632 | 5.874655 | 6 | 1 | 0 | 12.800000 | 81.900000 | 61.542645 |

## Raw TOPIQ Dataset Metrics

| candidate | dataset | SRCC | PLCC | MAE | FP | severe FP | FN | min | max | mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_topiq_raw_x100 | flive | 0.250960 | 0.315023 | 19.720615 | 0 | 0 | 17 | 8.532328 | 80.431986 | 52.470278 |
| A_topiq_raw_x100 | koniq10k | 0.856579 | 0.877003 | 16.446835 | 0 | 0 | 0 | 4.682437 | 73.925787 | 47.939517 |
| A_topiq_raw_x100 | spaq | 0.803045 | 0.814010 | 14.026467 | 0 | 0 | 4 | 5.175031 | 76.506066 | 38.005323 |

## Best Overall Candidates By MAE

| candidate | dataset | SRCC | PLCC | MAE | FP | severe FP | FN | min | max | mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E_datasetwise_isotonic__blended_50_50 | all | 0.963814 | 0.971958 | 2.937328 | 0 | 0 | 0 | 12.800000 | 86.085714 | 61.542645 |
| D_datasetwise_linear__blended_50_50 | all | 0.953070 | 0.968160 | 3.219335 | 0 | 0 | 0 | 17.305241 | 83.868751 | 61.542645 |
| C_global_isotonic__blended_50_50 | all | 0.954427 | 0.963343 | 3.511052 | 0 | 0 | 0 | 16.299029 | 84.230286 | 61.542645 |
| E_datasetwise_isotonic__blended_70_30 | all | 0.921025 | 0.941452 | 4.112259 | 2 | 0 | 0 | 12.800000 | 84.240000 | 61.542645 |
| B_global_linear__blended_50_50 | all | 0.912285 | 0.955449 | 4.115526 | 0 | 0 | 0 | 21.663163 | 85.775707 | 61.542645 |
| E_datasetwise_isotonic__guard_margin_5 | all | 0.862123 | 0.938263 | 4.473112 | 0 | 0 | 0 | 12.800000 | 81.900000 | 60.141102 |
| D_datasetwise_linear__blended_70_30 | all | 0.897717 | 0.932732 | 4.507069 | 3 | 0 | 0 | 19.107338 | 81.485634 | 61.542645 |
| D_datasetwise_linear__guard_margin_5 | all | 0.846790 | 0.928935 | 4.865902 | 0 | 0 | 0 | 13.600000 | 81.081238 | 59.969877 |
| C_global_isotonic__blended_70_30 | all | 0.897363 | 0.921225 | 4.915473 | 0 | 0 | 0 | 17.698641 | 81.642400 | 61.542645 |
| E_datasetwise_isotonic__guard_margin_8 | all | 0.838393 | 0.925134 | 4.952982 | 0 | 0 | 0 | 12.800000 | 81.900000 | 60.620972 |
| C_global_isotonic__guard_margin_5 | all | 0.828925 | 0.926966 | 5.151035 | 0 | 0 | 0 | 13.600000 | 77.760571 | 59.671576 |
| E_datasetwise_isotonic__guard_margin_10 | all | 0.829396 | 0.917273 | 5.171502 | 0 | 0 | 0 | 12.800000 | 81.900000 | 60.839492 |

## Best Guarded Candidates By Safety Then MAE

| candidate | dataset | SRCC | PLCC | MAE | FP | severe FP | FN | min | max | mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E_datasetwise_isotonic__blended_50_50 | all | 0.963814 | 0.971958 | 2.937328 | 0 | 0 | 0 | 12.800000 | 86.085714 | 61.542645 |
| D_datasetwise_linear__blended_50_50 | all | 0.953070 | 0.968160 | 3.219335 | 0 | 0 | 0 | 17.305241 | 83.868751 | 61.542645 |
| C_global_isotonic__blended_50_50 | all | 0.954427 | 0.963343 | 3.511052 | 0 | 0 | 0 | 16.299029 | 84.230286 | 61.542645 |
| E_datasetwise_isotonic__blended_70_30 | all | 0.921025 | 0.941452 | 4.112259 | 2 | 0 | 0 | 12.800000 | 84.240000 | 61.542645 |
| B_global_linear__blended_50_50 | all | 0.912285 | 0.955449 | 4.115526 | 0 | 0 | 0 | 21.663163 | 85.775707 | 61.542645 |
| E_datasetwise_isotonic__guard_margin_5 | all | 0.862123 | 0.938263 | 4.473112 | 0 | 0 | 0 | 12.800000 | 81.900000 | 60.141102 |
| D_datasetwise_linear__blended_70_30 | all | 0.897717 | 0.932732 | 4.507069 | 3 | 0 | 0 | 19.107338 | 81.485634 | 61.542645 |
| D_datasetwise_linear__guard_margin_5 | all | 0.846790 | 0.928935 | 4.865902 | 0 | 0 | 0 | 13.600000 | 81.081238 | 59.969877 |
| C_global_isotonic__blended_70_30 | all | 0.897363 | 0.921225 | 4.915473 | 0 | 0 | 0 | 17.698641 | 81.642400 | 61.542645 |
| E_datasetwise_isotonic__guard_margin_8 | all | 0.838393 | 0.925134 | 4.952982 | 0 | 0 | 0 | 12.800000 | 81.900000 | 60.620972 |
| C_global_isotonic__guard_margin_5 | all | 0.828925 | 0.926966 | 5.151035 | 0 | 0 | 0 | 13.600000 | 77.760571 | 59.671576 |
| E_datasetwise_isotonic__guard_margin_10 | all | 0.829396 | 0.917273 | 5.171502 | 0 | 0 | 0 | 12.800000 | 81.900000 | 60.839492 |

## False Case Files

- false positive cases written: 44
- false negative cases written: 84

## Distillation Decision

Do not train a student yet. Use the selected guarded target only as the next candidate for a separate bounded teacher-label-generation dry run and holdout check.
