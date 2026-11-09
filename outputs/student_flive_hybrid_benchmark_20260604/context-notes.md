# Student + FLIVE hybrid benchmark context notes

- Phase 12 decision: `STUDENT_TFLITE_READY_FOR_MOBILE_PROFILING_LOG_ONLY`.
- Phase 11 decision: `BEST_CHECKPOINT_NEEDS_FLIVE_HYBRID_PLAN`.
- Student checkpoint: `outputs/mobile_student_iqa_14105_full_controlled_20260604/best_by_val_mae_model.keras`.
- Student is strong on KonIQ/SPAQ, weak on FLIVE; not production replacement alone.
- Existing production image-level technical formula was audited as `(0.32 * KonIQ + 0.24 * FLIVE_image) / 0.56`, equivalent to `(4 * KonIQ + 3 * FLIVE_image) / 7`.
- Cached fusion component files provide `koniq_score` and `flive_score` by dataset.
- Primary benchmark uses `outputs/student_distillation_manifest_14105_20260603/test_manifest.csv`.
- Common rows with cached fusion components: FLIVE 399, KonIQ-10k 84, SPAQ 113, total 596.
- KonIQ coverage is limited because student test split does not fully overlap old fusion benchmark split.
- No Flutter edit, no production `technical_score` edit, no production scoring code edit, no TFLite export, no training, no full distillation, no commit.

## Benchmark result

- Script: `scripts/technical_iqa/benchmark_student_flive_hybrid_20260604.py`.
- Py-compile passed.
- Benchmark ran in `.venv_tf`; CUDA build present but `cuInit` failed, so inference ran CPU. This is acceptable for offline WSL benchmark.
- Common rows: 596 total; FLIVE 399, KonIQ-10k 84, SPAQ 113.
- Production_existing overall: MAE/SRCC/PLCC `6.6849/0.6731/0.8302`, FP/severe FP/FN `4/1/0`.
- Student overall: MAE/SRCC/PLCC `8.4303/0.5981/0.8428`, FP/severe FP/FN `1/0/0`.
- FLIVE overall: MAE/SRCC/PLCC `8.0099/0.6907/0.7349`, FP/severe FP/FN `44/24/0`.
- Best composite/fixed ratio: `hybrid_4_6`, student weight `0.4`, FLIVE weight `0.6`.
- `hybrid_4_6` overall: MAE/SRCC/PLCC `6.7456/0.7003/0.8670`, FP/severe FP/FN `8/1/0`.
- FLIVE constraint passed, but KonIQ/SPAQ stability failed: KonIQ SRCC delta vs production `-0.1038`; SPAQ delta `-0.0191`.
- Final decision: `HYBRID_STUDENT_FLIVE_NEEDS_RATIO_TUNING`.
