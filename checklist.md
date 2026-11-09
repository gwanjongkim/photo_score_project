# Checklist - MDIQA Checkpoint Source Audit

- [x] List local checkpoints in `MDIQA/checkpoints/`
- [x] Review `MDIQA/README.md` for checkpoint details
- [x] Inspect previous report `outputs/mdiqa_variant_sanity_20260601/report.md`
- [x] Inspect previous inventory `outputs/mdiqa_variant_sanity_20260601/checkpoint_inventory.csv`
- [x] Search code for checkpoint loading (e.g., `inference_iqa.py`)
- [x] Analyze official Drive folder content
- [x] Create `checkpoint_drive_listing.csv`
- [x] Create `checkpoint_local_inventory.csv`
- [x] Create `checkpoint_mapping_candidates.csv`
- [x] Create `missing_checkpoint_actions.csv`
- [x] Generate final report `outputs/mdiqa_checkpoint_source_audit_20260601/report.md`

# Checklist - MDIQA Official Variant Sanity Benchmark

- [x] Verify existing MDIQA scripts, reports, and checkpoint inventory
- [x] Download missing official `koniq_xd` and `net_best` checkpoints without overwriting existing files
- [x] Update `scripts/technical_iqa/run_mdiqa_variant_sanity_benchmark.py` (instead of creating new one)
- [x] Run `py_compile` on the benchmark script
- [x] Run the 100-image official variant sanity benchmark once
- [x] Review generated metrics, failures, best variants, and final decision

# Checklist - Phase 4 hybrid teacher label 2k dry-run and range audit

- [x] Inspect Phase 3 hybrid design artifacts, paired predictions, MDIQA benchmark outputs, FLIVE production predictions, research manifest, and production selector weights.
- [x] Create `scripts/technical_iqa/build_hybrid_teacher_label_dryrun_2k.py`.
- [x] Build deterministic 2k-or-best-effort manifest from existing prediction CSVs only.
- [x] Keep FLIVE primary rows image-only and production-scored.
- [x] Define fallback/unknown as a documented production-scored proxy if real mobile rows are unavailable.
- [x] Evaluate Policy A, Policy B, and range-calibration diagnostics.
- [x] Save all requested Phase 4 CSVs and report under `outputs/hybrid_teacher_label_design_review_2k_20260603/`.
- [x] Run `python3 -m py_compile`.
- [x] Run the Phase 4 script once.
- [x] Report row counts, safety counts, range audit, Policy A vs B, fallback definition, paths, and final decision.

# Checklist - Hybrid teacher range calibration 2k audit

- [x] Inspect Phase 4 2k manifest, validation, distributions, policy comparison, range audit, and safety cases.
- [x] Create `scripts/technical_iqa/run_hybrid_teacher_range_calibration_2k.py`.
- [x] Evaluate Policies A-E on the same 2k rows.
- [x] Save calibrated manifest, calibration parameters, policy metrics, range audit, safety cases, policy comparison, failures, recommendation JSON, report, and next actions.
- [x] Run `python3 -m py_compile`.
- [x] Run the calibration script once.
- [x] Inspect outputs and report selected/rejected policy, safety, range before/after, and final decision.

# Checklist - Adjusted hybrid teacher calibration design

- [x] Inspect previous calibration report, recommendation JSON, policy metrics, range audit, Phase 4 manifest, Phase 4 range audit, and prior calibration script.
- [x] Create `outputs/hybrid_teacher_adjusted_calibration_design_20260603/previous_calibration_failure_analysis.md`.
- [x] Create `outputs/hybrid_teacher_adjusted_calibration_design_20260603/adjusted_calibration_policy_spec.md`.
- [x] Create `scripts/technical_iqa/run_hybrid_teacher_adjusted_calibration_2k.py`.
- [x] Implement Policy A/B baselines and adjusted Policy F/G/H/I candidates.
- [x] Support `--smoke` without running full 2k revalidation.
- [x] Run `python3 -m py_compile`.
- [x] Run `python3 scripts/technical_iqa/run_hybrid_teacher_adjusted_calibration_2k.py --smoke`.
- [x] Report docs, implemented policies, validation result, smoke result, and full 2k command.

# Checklist - Phase 5 hybrid teacher label 10k dry-run prep

- [x] Inspect adjusted 2k report, recommendation JSON, input prediction schemas, and 2k dry-run builder.
- [x] Create `scripts/technical_iqa/build_hybrid_teacher_label_dryrun_10k.py`.
- [x] Implement policy_f_shift_cap3 selected target without modifying previous scripts.
- [x] Build requested 10k outputs under `outputs/hybrid_teacher_label_dryrun_10k_20260603/`.
- [x] Support `--smoke` without running the full 10k dry-run.
- [x] Run `python3 -m py_compile`.
- [x] Run `python3 scripts/technical_iqa/build_hybrid_teacher_label_dryrun_10k.py --smoke`.
- [x] Report files, compile result, smoke result, full command, and report path.

# Checklist - Mobile Student IQA 50k MDIQA Smoke Training

- [x] Inspect relevant Mobile Student IQA student trainers and manifest builders.
- [x] Inspect 50k MDIQA target report and final manifest schema.
- [x] Record TensorFlow GPU preflight.
- [x] Create `scripts/technical_iqa/train_mobile_student_iqa_50k_smoke.py`.
- [x] Compile the new smoke trainer.
- [x] Run bounded 50k smoke training only.
- [x] Inspect required output CSVs and report.
- [x] Create handoff document in `/tmp`.
- [x] Report counts, runtime, GPU status, losses, prediction std, failures, collapse status, and final decision.

# Checklist - TensorFlow GPU Visibility Fix for 50k Smoke

- [x] Confirm task scope: no training unless GPU becomes visible; no export, Flutter, production-score edit, commit.
- [x] Inspect trainer `--out_dir` support and prior smoke report.
- [x] Check `.venv_gpu` TensorFlow GPU visibility with clean NVIDIA lib dirs.
- [x] Check `.venv_tf` TensorFlow GPU visibility with clean NVIDIA lib dirs.
- [x] Inspect NVIDIA package library paths and CUDA-related packages.
- [x] Decide target environment for rerun.
- [x] Run bounded 4000/1000/1000 smoke only if GPU is visible.
- [x] Write required GPU preflight artifacts.
- [x] Write handoff document in `/tmp`.
- [x] Report final decision and next action.

# Checklist - Phase 16 Mobile Student IQA 50k Controlled B0

- [x] Confirm scope: one controlled B0 run only; no TFLite, Flutter, production-score edit, model asset edit, commit, B1/B2 experiments.
- [x] Inspect 50k smoke trainer, 14105/v4 trainers, checkpoint evaluators, official benchmark paths, and GPU-fix report.
- [x] Create `scripts/technical_iqa/train_mobile_student_iqa_50k_controlled.py`.
- [x] Compile controlled script.
- [x] Preflight `.venv_tf` GPU with required launch pattern.
- [x] Run one controlled B0 training.
- [x] Verify required outputs and checkpoint files.
- [x] Evaluate official Track B image-only benchmark if feasible.
- [x] Write final report and next actions.
- [x] Write handoff document in `/tmp`.
- [x] Report final decision and key metrics.

# Checklist - Phase 23 Mobile Student IQA 50k Label-Scaled B0 Smoke S2 KonIQ x2

- [x] Confirm scope: one B0 label-scaling smoke only; no B1/B2, full controlled training, TFLite export, Flutter edit, production-score edit, model asset edit, commit, ranking loss, or extra scaling variants.
- [x] Inspect existing 50k smoke, controlled, and ranking runners plus helper seams.
- [x] Create focused `scripts/technical_iqa/train_mobile_student_iqa_50k_label_scaling_smoke.py`.
- [x] Implement S2 KonIQ x2 scaling with audit columns in split manifests.
- [x] Generate `target_scaling_audit.csv` and verify KonIQ std expansion plus FLIVE/SPAQ unchanged.
- [x] Compile the new runner.
- [x] Run exactly one bounded `.venv_tf` B0 smoke with 4000/1000/1000 caps.
- [x] Inspect required outputs, official metrics, failures, collapse checks, and final report.
- [x] Write handoff document in `/tmp`.
- [ ] Report final decision and key metrics.

# Checklist - Adversarial Review of Codex Reproducibility Audit (2026-07-11)

- [ ] Review paper baseline documents and implementation audit files
- [ ] Inspect NIMA, RGNet, A-LAMP, MUSIQ, TOPIQ, ICAA, and MDIQA source code
- [ ] Identify unsupported MATCH decisions, paper interpretation errors, and missing components
- [ ] Identify database-level leakage and preprocessing normalization bugs
- [ ] Identify missing logistic fitting in IQA PLCC calculations
- [ ] Challenge lineage links between Keras checkpoints and mobile TFLite files
- [ ] Re-score all models and create corrected scorecard
- [ ] Review safety of project claims (e.g. production/mobile readiness)
- [ ] Draft 9 cross-check report markdown files under `outputs/capstone_model_audit_20260711/03_independent_crosscheck/`
- [ ] Record executed command lines in `commands_executed.txt` under `outputs/capstone_model_audit_20260711/03_independent_crosscheck/`
- [ ] Verify all created artifacts contain the required structure and Korean headers (first-line Korean comments) where appropriate
