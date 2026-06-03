# Phase 7 student distillation manifest context notes

- Scope is Option A: keep validated cache-subset direction.
- This is not MDIQA official full-dataset relabeling.
- Input manifest: `outputs/known_dataset_full_label_dryrun_20260603/known_dataset_full_label_dryrun_manifest.csv`.
- Coverage audit recommendation: `KNOWN_DATASET_FULL_LABEL_READY_FOR_REVIEW`.
- Valid cache rows: total 14105, KonIQ-10k 8999, SPAQ 1125, FLIVE 3981.
- Label policy: KonIQ MDIQA seed1 plus policy_f_shift_cap3, SPAQ MDIQA xd plus policy_f_shift_cap3, FLIVE production technical score.
- Unknown/fallback excluded; c6_parity real-mobile holdout only; Test_weight/product-gold aesthetic/product validation only.
- Input precheck: no duplicate image_path, all targets in [0,100], all source_group present.
- Script run result: train 11283, val 1409, test 1413.
- Dataset by split: train FLIVE 3184, KonIQ 7199, SPAQ 900; val FLIVE 398, KonIQ 899, SPAQ 112; test FLIVE 399, KonIQ 901, SPAQ 113.
- Leakage audit: duplicate image_path 0, image_path overlap 0, source_group leakage 0.
- Target validity: selected_teacher_target_0_100 out of range 0, target_norm out of range 0.
- Safety counts: FP 0, severe FP 0, FN 0 across all splits.
- Final decision: STUDENT_MANIFEST_14105_READY_FOR_SMOKE_TRAINING.
