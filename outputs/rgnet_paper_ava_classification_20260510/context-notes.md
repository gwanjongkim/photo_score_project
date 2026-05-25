# Context Notes

## 2026-05-11 Reporting Finalization

- Scope is reporting and threshold calibration only. No model training, full AVA evaluation rerun, Flutter changes, forWeights changes, or src/models/rgnet.py changes.
- Required full-run artifacts were present under outputs/rgnet_paper_ava_classification_20260510/full_train/lse_r4 and outputs/rgnet_paper_ava_classification_20260510/eval/lse_r4.
- Wording must stay at "paper-oriented approximation" and must not claim official RGNet paper reproduction.
- Threshold calibration will use only existing val_predictions.csv and test_predictions.csv.
- Full AVA lse_r4 is now treated as the current RGNet paper-oriented AVA classification baseline.
- Threshold `0.50` remains a strong fixed baseline. Best validation F1 selected `0.41`, improving test F1 to `0.851918` while increasing false positives. Best validation balanced accuracy and Youden J selected `0.70`, improving test balanced accuracy to `0.723243` and reducing test false positives from `3921` to `1856`.
- Consolidated reporting keeps AADB regression (`RGNet-paper-v1 agg_mean_full`) and AVA classification (`RGNet-paper-AVA lse_r4 full-run`) separate because mean aggregation won in regression and LSE r=4 won in classification.
