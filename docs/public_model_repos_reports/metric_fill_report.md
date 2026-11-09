# Metric Fill Report

## Executive Summary
This report documents the extraction of actual benchmark metrics from the A-CUT research workspace and their population into the 11 public model repositories. The primary goal was to replace TBD, TODO, and NEEDS_USER_REVIEW placeholders with confirmed numeric evidence.

## CSV Files and Reports Inspected
- outputs/acut_capstone_final_report_20260526/model_metrics_summary.csv
- outputs/nima_fixed_preproc_eval_20260525/report.md
- outputs/rgnet_v1_aadb_pil_resize_retrain_20260524_tflite/parity_report.md
- outputs/mobile_alamp_v2_full_ava_20260519/report.md
- outputs/technical_iqa_signal_comparison_20260604/report.md
- outputs/technical_iqa_signal_comparison_with_c6_20260604/report.md
- outputs/mdiqa_correct_protocol_full_benchmark_20260601/report.md
- outputs/musiq_kaggle_sanity_fallback_20260601/report.md
- outputs/mobile_student_iqa_14105_checkpoint_eval_20260604/datasetwise_checkpoint_metrics.csv

## Metrics Filled per Repo
1.  **acut-nima-mobile**: SRCC 0.5894, PLCC 0.5929, AUC 0.8018 on AVA.
2.  **acut-rgnet-mobile**: SRCC 0.6819 on AADB, Accuracy 0.7697 on AVA. TFLite FP16 SRCC 0.6647 confirmed.
3.  **acut-mobile-alamp**: Accuracy 0.7633, ROC-AUC 0.7877 on AVA. Mobile v2 Val Accuracy 0.7142.
4.  **acut-icaa-color-aesthetics**: SRCC 0.8791, PLCC 0.8958 on ICAA17K.
5.  **acut-technical-iqa**: KonIQ SRCC 0.866, FLIVE SRCC 0.630, Overall production SRCC 0.6914.
6.  **acut-topiq-mixed112**: SPAQ SRCC 0.899, KonIQ SRCC 0.863, Overall SRCC 0.8148.
7.  **acut-topiq-flive-hybrid**: Overall SRCC 0.8222, MAE 5.0395.
8.  **acut-c6-topiq-koniq**: Overall SRCC 0.7036, MAE 15.3872.
9.  **acut-mdiqa-benchmark**: KonIQ-10k SRCC 0.9464, SPAQ SRCC 0.9136, FLIVE SRCC 0.5692.
10. **acut-musiq-probe**: SPAQ SRCC 0.9460, KonIQ-10k SRCC 0.9275, FLIVE SRCC 0.4815 (Official Kaggle MUSIQ).
11. **acut-student-iqa**: Overall SRCC 0.6771. KonIQ SRCC 0.7230, SPAQ SRCC 0.8984.

## Metrics Still Unresolved
- Specific latency benchmarks for most TFLite models on target Android devices are referenced but not always populated with final "ms" values in the README tables.
- Input/output contracts for some research candidates (MDIQA/MUSIQ) are inferred from scripts.

## Conflicts Found
- None. Selection priority followed the "final" report markers in the workspace.

## Files Modified
- README.md and docs/benchmark_summary.md across all 11 repositories.

## Files Copied into results/
- model_metrics_summary.csv
- metrics_summary.json
- export_metadata.json
- summary.json
- c6_recommendation.json
- dataset_metrics.csv
- datasetwise_checkpoint_metrics.csv

## Final Decision
METRICS_FILLED_READY_FOR_REVIEW
