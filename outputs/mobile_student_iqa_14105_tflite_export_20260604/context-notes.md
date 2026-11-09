# Phase 12 context notes

- Track A Option A remains active: validated known-dataset teacher-cache subset, not MDIQA official full-dataset relabeling.
- Current best checkpoint is `outputs/mobile_student_iqa_14105_full_controlled_20260604/best_by_val_mae_model.keras`.
- Phase 11 decision is `BEST_CHECKPOINT_NEEDS_FLIVE_HYBRID_PLAN`.
- Phase 11 test metrics: overall MAE/SRCC/PLCC 4.7963/0.6945/0.8423, severe FP 0, collapse false, failures 0.
- Dataset-wise Phase 11 test: FLIVE weak at MAE/SRCC/PLCC 8.9626/0.5176/0.5168; KonIQ and SPAQ remain strong.
- Existing `scripts/technical_iqa/test_tflite_parity.py` is hard-coded to old `exports/tflite/koniq_mobile.tflite` and does not meet Phase 12 export/report needs.
- Phase 12 is export/parity/profiling preparation only: no Flutter edit, no production `technical_score` replacement, no production scoring edit, no training/full distillation, no commit.

## Phase 12 result

- Script: `scripts/technical_iqa/export_student_iqa_14105_tflite.py`.
- Py-compile passed.
- FP32 export: `23.683994 MB`.
- FP16 export: `11.963104 MB`.
- TFLite shape: `[1, 256, 256, 3] float32 -> [1, 1] float32`.
- Flex ops: none detected; analyzer reports GPU delegate compatibility for TFLite runtime 2.21.0.
- Parity sample: 300 rows, stratified 100 FLIVE / 100 KonIQ-10k / 100 SPAQ.
- FP32 parity overall MAE/max diff: `0.000005 / 0.000039` on 0..100 scale.
- FP16 parity overall MAE/max diff: `0.018522 / 0.142264` on 0..100 scale.
- Final decision: `STUDENT_TFLITE_READY_FOR_MOBILE_PROFILING_LOG_ONLY`.
- Environment note: TensorFlow built with CUDA, but GPUs visible were `[]`; export/parity ran CPU/XNNPACK. This is acceptable for Phase 12 export audit, but on-device profiling remains required.
