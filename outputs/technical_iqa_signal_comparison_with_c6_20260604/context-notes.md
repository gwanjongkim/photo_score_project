# C6 signal comparison context notes

- Previous report: `outputs/technical_iqa_signal_comparison_20260604/report.md`.
- Previous decision: `TECHNICAL_SIGNAL_COMPARISON_READY_FOR_LOG_ONLY_PROFILING`.
- Previous selected candidate: `topiq_existing_flive_57_43`.
- Coverage already recovered: overall 1413 / 1413, FLIVE 399 / 399, KonIQ 901 / 901, SPAQ 113 / 113.
- C6 formula: `min(0.7 * TOPIQ_mixed112_score_100 + 0.3 * KonIQ_score_100, KonIQ_score_100 + 8)`.
- C6 remains log-only candidate only. Prior C6 replacement was NO-GO pending parity/generalization.
- `signal_predictions_common.csv` already contains `topiq_existing_0_100` and `koniq_existing_0_100`; no inference needed.
- Scope: offline CSV benchmark only. No Flutter edit, no production scoring edit, no training, no distillation, no TFLite export, no commit.

## Result

- Script: `scripts/technical_iqa/benchmark_technical_iqa_signals_with_c6_20260604.py`.
- Py-compile passed.
- Benchmark run was CSV-only in `.venv_tf`.
- C6 input coverage: overall 1413 / 1413, FLIVE 399 / 399, KonIQ 901 / 901, SPAQ 113 / 113.
- Main C6 formula used: `min(0.7 * topiq_existing + 0.3 * koniq_existing, koniq_existing + 8)`.
- Main C6 overall MAE/SRCC/PLCC: `15.3872/0.7036/0.7886`, FP/severe FP/FN `0/0/14`.
- TOPIQ+FLIVE 57/43 overall MAE/SRCC/PLCC: `5.0395/0.8222/0.8786`, FP/severe FP/FN `0/0/0`.
- C6 dataset SRCC: FLIVE `0.3692`, KonIQ `0.7953`, SPAQ `0.8761`.
- TOPIQ+FLIVE 57/43 dataset SRCC: FLIVE `0.5188`, KonIQ `0.8547`, SPAQ `0.8767`.
- Main C6 cap activation: overall 973 / 1413 (`68.86%`), FLIVE 362 / 399 (`90.73%`), KonIQ 557 / 901 (`61.82%`), SPAQ 54 / 113 (`47.79%`).
- Main C6 mean/max cap clip: `6.1438/23.2421`.
- Final decision: `TOPIQ_FLIVE_REMAINS_BEST_LOG_ONLY_CANDIDATE`.
