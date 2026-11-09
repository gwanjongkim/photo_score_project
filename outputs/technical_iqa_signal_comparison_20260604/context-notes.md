# Technical IQA signal comparison context notes

- Phase 12 decision: `STUDENT_TFLITE_READY_FOR_MOBILE_PROFILING_LOG_ONLY`.
- Phase 13 decision: `HYBRID_STUDENT_FLIVE_NEEDS_RATIO_TUNING`.
- Previous blocker: common rows only 596 / 1413; KonIQ only 84 / 901.
- Production image-level formula already audited as `(4 * KonIQ + 3 * FLIVE_image) / 7` using available standalone image-level scores.
- Existing KonIQ and FLIVE TFLites: `exports/tflite/koniq_mobile.tflite`, `exports/tflite/flive_image_mobile.tflite`.
- TOPIQ mixed112 TFLite exists: `outputs/final_topiq_lite_mixed112_export_20260517/topiq_lite_mixed112_frozen_fp16.tflite`; protocol is resize_with_pad 384, raw 0..255, output * 100.
- MUSIQ local probe TFLites exist and use token inputs: patches, positions, scale_ids, token_mask. Existing 128-row caches lack image_path, so new bounded inference over test rows is safer.
- Student FP16 TFLite parity was already accepted; use it for speed/consistency unless Keras needed.
- Scope: offline WSL benchmark only. No Flutter edit, no production `technical_score` edit, no production scoring code edit, no training, no full distillation, no new TFLite export, no commit.

## Benchmark result

- Script: `scripts/technical_iqa/benchmark_technical_iqa_signals_20260604.py`.
- Py-compile passed.
- Benchmark ran once in `.venv_tf`; inference used CPU/XNNPACK.
- Coverage recovered to full test split: overall `1413/1413`, FLIVE `399/399`, KonIQ-10k `901/901`, SPAQ `113/113`.
- Base overall SRCC: production `0.6914`, KonIQ `0.6579`, FLIVE `0.6388`, TOPIQ `0.8148`, student `0.6771`, MUSIQ probes `0.0207..0.2178`.
- Base dataset winners: FLIVE standalone on MAE/SRCC among base signals for FLIVE; TOPIQ for KonIQ/SPAQ.
- Best hybrid by composite: `topiq_existing_flive_57_43`.
- `topiq_existing_flive_57_43` overall: MAE/SRCC/PLCC `5.0395/0.8222/0.8786`, FP/severe FP/FN `0/0/0`.
- Production overall: MAE/SRCC/PLCC `11.4913/0.6914/0.7756`, FP/severe FP/FN `0/0/0`.
- Dataset SRCC deltas vs production for selected candidate: FLIVE `+0.0540`, KonIQ `+0.0854`, SPAQ `+0.0370`.
- Final decision: `TECHNICAL_SIGNAL_COMPARISON_READY_FOR_LOG_ONLY_PROFILING`.
