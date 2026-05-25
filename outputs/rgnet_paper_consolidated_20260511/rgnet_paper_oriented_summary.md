# RGNet Paper-Oriented Experiment Summary

## 1. Scope
This is a paper-oriented approximation, not an official reproduction.

Practical app RGNet is untouched. Flutter and `forWeights/` are untouched, and `src/models/rgnet.py` was not modified.

## 2. AADB Regression Result
The best AADB paper-oriented result to carry forward is `RGNet-paper-v1 agg_mean_full`.

| Split | SRCC | PLCC | MAE | RMSE |
|---|---:|---:|---:|---:|
| AADB full test | 0.681865 | 0.687753 | 0.119757 | 0.148603 |

Mean aggregation was best for the AADB regression track. That fits the regression objective: the target is a scalar image-level aesthetic score, so stable averaging over region evidence can beat a sharper evidence-pooling rule.

## 3. AVA Classification Result
The best AVA paper-oriented classification result is `RGNet-paper-AVA lse_r4 full-run`.

| Split | Accuracy | F1 | ROC-AUC | AP | BCE |
|---|---:|---:|---:|---:|---:|
| test | 0.769658 | 0.846781 | 0.797906 | 0.901876 | 0.481819 |
| val | 0.771633 | 0.847343 | 0.801449 | 0.897984 | 0.481745 |

Confusion matrix at threshold `0.50`.

| Split | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| test | 3402 | 3921 | 1964 | 16262 |
| val | 3522 | 4026 | 1809 | 16194 |

LSE worked better for AVA classification in this track. The mid-run comparison selected `lse_r4`, and the completed full run improved over the mid-run on F1, ROC-AUC, AP, and BCE. This is plausible because binary classification can benefit from stronger localized region evidence, while mean aggregation can smooth that evidence away.

Threshold calibration adds operating-point detail. Threshold `0.50` is a strong fixed baseline. Best validation F1 selected threshold `0.41` and reached test F1 `0.851918`. Best validation balanced accuracy selected threshold `0.70` and reached test balanced accuracy `0.723243`, reducing test false positives from 3921 to 1856.

## 4. Regression vs Classification Insight
AADB regression favored mean aggregation. AVA classification favored LSE r=4.

This is consistent with scalar score regression and region evidence classification behaving differently. Regression rewards stable image-level score estimation, while binary AVA classification can reward sharper pooling of high-confidence aesthetic evidence.

## 5. Paper Direction
The RGNet paper direction includes AVA classification and AADB regression. This project has now run both tracks as paper-oriented approximations.

The current paired baseline is AADB `RGNet-paper-v1 agg_mean_full` for regression and AVA `RGNet-paper-AVA lse_r4 full-run` for classification.

## 6. Remaining Gap To Official Paper
- Exact DenseASPP not implemented.
- Exact RegionGraph not implemented.
- No official author code or weights.
- Input size and hyperparameters may differ.
- Binary label threshold may be simplified.

## 7. Recommended Next Steps
1. Finish threshold calibration decision using the generated calibration outputs.
2. Preserve RGNet AADB/AVA results as the current baseline.
3. Start A-LAMP-paper-AVA track.
4. Start color aesthetics D0/D1 separately.
