# Plan - MDIQA Checkpoint Source Audit

Audit the official MDIQA checkpoint source and determine whether SPAQ and FLIVE dataset-specific checkpoints are actually available.

## Phase 1: Local Discovery
1. List all files in `MDIQA/checkpoints/`.
2. Review previous sanity/variant reports for inventory clues.
3. Inspect `MDIQA/README.md` for checkpoint documentation.

## Phase 2: Code Evidence
1. Search MDIQA codebase for checkpoint loading logic.
2. Check for hardcoded filenames or dataset-to-checkpoint mapping in scripts.

## Phase 3: External Source Audit
1. Inspect the official Google Drive folder (via web_fetch if possible, or analysis of provided info).
2. Check GitHub repository for release notes or issues mentioning SPAQ/FLIVE checkpoints.

## Phase 4: Mapping and Decision
1. Correlate local files with documentation/code evidence.
2. Determine if generic filenames (e.g., `net_best.pth`) correspond to specific datasets.
3. Draft the final report and CSV artifacts.

# Plan - Mobile Student IQA 50k MDIQA Smoke Training

Goal: verify the 50k MDIQA target manifest can feed the existing Mobile Student IQA student pipeline without dataloader, target-scale, memory, or prediction-collapse failures.

1. Inspect existing student manifest/training modules -> verify: source files and prior report are named in `report.md`.
2. Build a narrow 50k smoke trainer from the 14105 smoke path -> verify: `py_compile` passes.
3. Construct deterministic dataset-balanced train/val/test splits with seed `20260608` -> verify: no duplicate image paths across splits and no official Track B rows in train.
4. Run bounded smoke training only -> verify: `history.csv`, metrics CSVs, prediction summary, failures, and report exist under `outputs/mobile_student_iqa_50k_smoke_20260608/`.
5. Decide next action -> verify: final decision is one of the user-approved 50k smoke decisions.

# Plan - TensorFlow GPU Visibility Fix for 50k Smoke

Goal: diagnose TensorFlow GPU invisibility and, only if fixed, run the intended bounded 50k smoke.

1. Inspect current trainer and prior report -> verify: trainer has `--out_dir`, prior failure was TF GPU visibility.
2. Probe `.venv_gpu` and `.venv_tf` with clean `LD_LIBRARY_PATH` plus NVIDIA wheel library dirs -> verify: command outputs captured.
3. Inspect NVIDIA package library layout and CUDA build flags -> verify: selected venv is evidence-based.
4. If TensorFlow GPU becomes visible, run only 4000/1000/1000 smoke with new output dir -> verify: no overwrite of previous smoke outputs.
5. Write preflight report, commands, outputs, next actions, and handoff -> verify: artifacts exist under `outputs/mobile_student_iqa_50k_gpu_visibility_fix_20260608/`.

# Plan - Phase 16 Mobile Student IQA 50k Controlled B0

Goal: run one controlled EfficientNetV2B0 baseline on the 50k MDIQA targets and evaluate whether it is ready for checkpoint evaluation or B1/B2 comparison.

1. Inspect student trainers, checkpoint evaluators, official benchmark manifests, and GPU-fix artifacts -> verify: sources listed in report.
2. Create `train_mobile_student_iqa_50k_controlled.py` as a sibling to the smoke trainer -> verify: `py_compile` passes.
3. Preflight `.venv_tf` GPU with unsandboxed WSL CUDA launch -> verify: TensorFlow CUDA build, GPU count 1, ImageNet load path.
4. Run one controlled B0 baseline only: 5 head epochs + 10 finetune epochs, batch 16, ImageNet, Huber -> verify: required CSVs and checkpoints exist in `outputs/mobile_student_iqa_50k_controlled_b0_20260608/`.
5. Evaluate best checkpoint on train/val/test plus official Track B image-only benchmark if feasible -> verify: official benchmark metrics CSVs exist or report says why not.
6. Write final report, next actions, and handoff -> verify: final decision is one of the requested Phase 16 decisions.

# Plan - Adversarial Review of Codex Reproducibility Audit (2026-07-11)

Goal: Perform an independent adversarial cross-check of the Codex reproducibility audit to challenge matching decisions, find incorrect interpretations, lineage challenges, dataset leakage, preprocessing bugs, and verify the validity of models and claims.

1. Independently review Codex paper baselines (`01_paper_baseline_codex`) and implementation audits (`02_codex_implementation_audit`).
2. Examine source code files for NIMA, RGNet, A-LAMP, MUSIQ, TOPIQ, ICAA, and MDIQA.
3. Pinpoint:
   - Unsupported MATCH decisions (e.g., NIMA no duplicate rescale, ICAA/DAT line).
   - Preprocessing inconsistencies (e.g., MobileNetV2 ImageNet mismatch, EfficientNetV2B0 double scaling in RGNet).
   - Dataset leakage (FLIVE image/patch split mixing).
   - Incompatible metric units or missing logistic fitting for PLCC.
   - Traceability/lineage gaps between Keras checkpoints and mobile TFLite files.
4. Correct the scorecard and define required resolution tests.
5. Create the 10 required cross-check reports under `outputs/capstone_model_audit_20260711/03_independent_crosscheck/`.
6. Compile the executed commands in `commands_executed.txt`.
