# Context Notes - MDIQA Checkpoint Source Audit

- Koniq sanity passed with high SRCC (~0.94), indicating the Koniq checkpoint is valid.
- SPAQ and FLIVE variants failed in previous runs due to missing checkpoints.
- SPAQ SRCC with Koniq checkpoint was high (0.93), but FLIVE was very low (0.03).
- User wants to know if dataset-specific checkpoints exist or if we should stick to Koniq-only.

# Context Notes - MDIQA Official Variant Sanity Benchmark

- Scope is benchmark sanity only: no training, TFLite export, Flutter edits, production technical_score edits, distillation, or teacher-label creation.
- Official variant mapping is fixed by the checkpoint source audit: `koniq_seed1`, `koniq_xd`, and unified `net_best`; no SPAQ/FLIVE-specific fusion checkpoint names should be searched.
- The same deterministic 100 readable images per FLIVE, KonIQ-10k, and SPAQ dataset must be used across all variants.
- Final sanity output is `outputs/mdiqa_official_variant_sanity_20260601/`; 900 predictions, 0 failures, and decision `MDIQA_PARTIAL_TEACHER_CANDIDATE`.
- Clean command used CPU in this environment (`torch.cuda.is_available: False`) even though the earlier CUDA diagnosis report recorded GPU availability.

# Context Notes - Phase 4 hybrid teacher label 2k dry-run and range audit

## Decisions
- **Scope**: Larger dry-run and audit only. Do not train, export TFLite, modify Flutter, modify production `technical_score`, modify production scoring code, modify model assets, create final full labels, or run 50k/159k generation.
- **Selected label policy**: Policy A strict hybrid remains the selected default: KonIQ-10k uses MDIQA `seed1`, SPAQ uses MDIQA `xd`, FLIVE uses production technical score, and unknown/fallback uses production safety fallback.
- **Inputs**: Use existing prediction CSVs first: paired MDIQA/production for KonIQ/SPAQ, correct-protocol MDIQA full benchmark, FLIVE production fusion predictions, and the research protocol manifest.
- **No inference**: If production scores are unavailable for sampled rows, report target gaps; do not start MDIQA or production inference.
- **Fallback definition**: If real mobile/unknown rows are unavailable, use a documented production-scored proxy group from existing rows rather than inventing real mobile data.
- **Output root**: Write Phase 4 artifacts under `outputs/hybrid_teacher_label_design_review_2k_20260603/`.

## Validation
- Created `scripts/technical_iqa/build_hybrid_teacher_label_dryrun_2k.py`.
- `python3 -m py_compile scripts/technical_iqa/build_hybrid_teacher_label_dryrun_2k.py` passed.
- `python3 scripts/technical_iqa/build_hybrid_teacher_label_dryrun_2k.py` completed and wrote all requested outputs under `outputs/hybrid_teacher_label_design_review_2k_20260603/`.
- Final manifest rows: 2,000 total and 2,000 valid selected Policy A targets. Counts: KonIQ-10k `500`, SPAQ `500`, FLIVE image production `700`, fallback proxy `300`.
- Fallback proxy is explicitly defined as non-duplicated FLIVE image rows with existing production scores, not real mobile/unknown data.
- Safety counts for selected Policy A: severe FP `0`, FP `0`, FN `0`.
- Policy B did not beat Policy A on MOS alignment: KonIQ SRCC A/B `0.9487/0.9393`; SPAQ SRCC A/B `0.9086/0.9084`; FLIVE and fallback are identical because both use production.
- Range audit flagged source range mismatch: MDIQA-source vs production-source mean shift `22.62`, std ratio `3.81`, p95/p05 gap mismatch `34.99`; no clipping or boundary concentration.
- Final decision: `HYBRID_LABEL_2K_NEEDS_RANGE_CALIBRATION`.

# Context Notes - Hybrid teacher range calibration 2k audit

## Decisions
- **Scope**: Calibration design and 2k dry-run audit only. Do not train, start student distillation, export TFLite, modify Flutter, modify production scoring/code, modify model assets, create final full labels, or run 10k/50k/full label generation.
- **Input**: Use `outputs/hybrid_teacher_label_design_review_2k_20260603/hybrid_label_dryrun_2k_manifest.csv` as the fixed 2k row set.
- **Policies**: Evaluate identity Policy A, 80/20 blend Policy B, source-wise cross-fit affine MOS calibration Policy C, source-wise MOS mean/std alignment Policy D, and capped MDIQA-only conservative calibration Policy E.
- **Selection**: Do not select a policy only because range metrics look smoother; require safety, no large clipping, SRCC drop <= 0.02 per dataset, and meaningful range improvement.
- **Output root**: Write artifacts under `outputs/hybrid_teacher_range_calibration_2k_20260603/`.

## Validation
- Created `scripts/technical_iqa/run_hybrid_teacher_range_calibration_2k.py`.
- `python3 -m py_compile scripts/technical_iqa/run_hybrid_teacher_range_calibration_2k.py` passed.
- `python3 scripts/technical_iqa/run_hybrid_teacher_range_calibration_2k.py` completed and wrote all requested outputs under `outputs/hybrid_teacher_range_calibration_2k_20260603/`.
- Policies evaluated: Policy A identity, Policy B 80/20 blend, Policy C 5-fold cross-fit affine, Policy D source mean/std alignment, Policy E capped MDIQA-only conservative calibration.
- Policy A baseline source mean shift/std ratio/gap mismatch: `22.6235` / `3.8111` / `34.9940`.
- Policy B was safest and preserved safety/ranking, but only modestly improved range: mean shift `21.2622`, gap mismatch `29.3181`.
- Policy C/D/E reduced source mean shift more (`17.56` to `18.78`) but worsened p95/p05 gap mismatch (`39.68` to `41.85`) and introduced `2` SPAQ FP cases, though severe FP stayed `0`.
- No policy introduced clipping; all clipping counts were `0`.
- Final decision: `CALIBRATION_POLICY_NEEDS_ADJUSTMENT`.

# Context Notes - Adjusted hybrid teacher calibration design

## Decisions
- **Scope**: Calibration policy design plus runnable 2k revalidation script only. Do not train, start distillation, export TFLite, modify Flutter, modify production scoring/code, modify model assets, create final labels, or run 10k/50k/full generation.
- **Full run policy**: Prepare the full 2k script but only run `--smoke` in this step unless explicitly reopened.
- **Failure diagnosis**: Prior Policy C/D/E failures came from expanding or lifting SPAQ tails enough to create FP cases and from increasing MDIQA-source spread, which worsened p95/p05 gap mismatch even when mean shift improved.
- **Adjusted candidates**: Use weaker monotonic/shift-only or capped local calibrations: Policy F capped source-step smoothing, Policy G residual bias shrinkage with FP guard, Policy H MOS-based diagnostic clamp, and Policy I soft dataset-local z calibration.
- **Output roots**: Design docs under `outputs/hybrid_teacher_adjusted_calibration_design_20260603/`; future full run artifacts under `outputs/hybrid_teacher_adjusted_calibration_2k_20260603/`; smoke artifacts under `outputs/hybrid_teacher_adjusted_calibration_2k_20260603_smoke/`.

## Validation
- Created `outputs/hybrid_teacher_adjusted_calibration_design_20260603/previous_calibration_failure_analysis.md`.
- Created `outputs/hybrid_teacher_adjusted_calibration_design_20260603/adjusted_calibration_policy_spec.md`.
- Created `scripts/technical_iqa/run_hybrid_teacher_adjusted_calibration_2k.py`.
- `python3 -m py_compile scripts/technical_iqa/run_hybrid_teacher_adjusted_calibration_2k.py` passed.
- `python3 scripts/technical_iqa/run_hybrid_teacher_adjusted_calibration_2k.py --smoke` completed and wrote smoke artifacts under `outputs/hybrid_teacher_adjusted_calibration_2k_20260603_smoke/`.
- Smoke final decision is intentionally `SMOKE_ONLY_NOT_A_FULL_2K_DECISION`; the full 2k run is still pending for Gemini or explicit user approval.
- Smoke selected `policy_f_shift_cap8` on 50 rows as a non-final check, with zero severe FP, zero FP, zero clipping, and mean shift reduced from `26.6655` to `18.6655` while gap mismatch stayed `38.7120`.

# Context Notes - Phase 5 hybrid teacher label 10k dry-run prep

## Decisions
- **Scope**: 10k dry-run/audit preparation only. Do not train, start distillation, export TFLite, modify Flutter, modify production scoring/code, modify model assets, create final labels, or run 50k/full generation.
- **Selected target policy**: Use `policy_f_shift_cap3` from the confirmed adjusted 2k revalidation.
- **Policy implementation**: KonIQ/SPAQ use MDIQA targets shifted by at most 3 points toward production-source central tendency; FLIVE and fallback proxy keep production scores unchanged.
- **Fallback definition**: If real mobile/unknown rows are unavailable, use non-overlapping FLIVE image rows with production scores as `fallback_proxy`; document that this is not representative real mobile data.
- **Full run policy**: Prepare the 10k script but only run `--smoke` in this step. Full 10k remains a separate command.
- **Output root**: Write full outputs under `outputs/hybrid_teacher_label_dryrun_10k_20260603/`; smoke outputs under `outputs/hybrid_teacher_label_dryrun_10k_20260603_smoke/`.

## Validation
- Created `scripts/technical_iqa/build_hybrid_teacher_label_dryrun_10k.py`.
- `python3 -m py_compile scripts/technical_iqa/build_hybrid_teacher_label_dryrun_10k.py` passed.
- `python3 scripts/technical_iqa/build_hybrid_teacher_label_dryrun_10k.py --smoke` completed and wrote all requested smoke outputs under `outputs/hybrid_teacher_label_dryrun_10k_20260603_smoke/`.
- Smoke rows: `50` total and `50` valid; counts were KonIQ `20`, SPAQ `10`, FLIVE `15`, fallback proxy `5`.
- Smoke safety: severe FP `0`, FP `0`, FN `0`, clipping `0`, failures `0`.
- Smoke final decision is intentionally `SMOKE_ONLY_NOT_A_FULL_10K_DECISION`; full 10k was not run.
- Cached FLIVE production coverage appears limited to `3,981` image rows in `fusion_predictions_flive.csv`, so the full dry-run may fall below 10k unless another production-scored fallback source is defined.

# Context Notes - Mobile Student IQA 50k MDIQA Smoke Training

## Decisions
- Scope is smoke training only. Do not run full training, export TFLite, modify Flutter, modify production `technical_score`, or commit.
- Use the 14105 smoke trainer as the main module pattern because it is a bounded single-target Huber distillation check with `target_norm` and EfficientNetV2B0.
- Do not reuse v4a/v4b safety-checkpoint training as the base for this smoke because the current objective is 50k MDIQA manifest/dataloader/target-scale validation, not safety-weighted student selection.
- Construct deterministic train/val/test splits inside the 50k smoke script from `final_50k_mdiqa_distillation_manifest.csv`; the provided source `split` column is original dataset provenance, not a student split.
- Treat TensorFlow GPU invisibility as a blocking preflight issue for `READY_FOR_CONTROLLED_TRAINING`; CPU smoke can still document pipeline behavior but cannot earn the ready decision unless accepted explicitly.

## Evidence
- `.venv_gpu` TensorFlow `2.21.0` reports no physical GPU and logs `failed call to cuInit: UNKNOWN ERROR (100)`.
- `.venv_tf` TensorFlow `2.21.0` reports the same no-GPU state.
- `nvidia-smi` sees an RTX 4070-class GPU with low memory use, so the issue is TensorFlow CUDA visibility rather than absent hardware.
- `outputs/mdiqa_distillation_50k_full_targets_20260608/report.md` records 50,000 valid targets, zero Track B overlap, zero failures, zero duplicate image paths, and final decision `MDIQA_50K_FULL_TARGETS_READY_FOR_STUDENT_SMOKE_TRAINING`.

## Validation
- Created `scripts/technical_iqa/train_mobile_student_iqa_50k_smoke.py`.
- `./.venv_gpu/bin/python -m py_compile scripts/technical_iqa/train_mobile_student_iqa_50k_smoke.py` passed.
- Because TensorFlow GPU was invisible, ran a bounded CPU fallback smoke rather than the intended 4k/1k/1k GPU smoke: `./.venv_gpu/bin/python scripts/technical_iqa/train_mobile_student_iqa_50k_smoke.py --max_train_samples 96 --max_val_samples 48 --max_test_samples 48 --batch_size 16 --backbone_weights none`.
- Outputs written under `outputs/mobile_student_iqa_50k_smoke_20260608/`: required manifests, history, metrics, datasetwise metrics, prediction summary, failure cases, report, and next actions.
- Effective counts were train `96`, val `48`, test `48`; image loading failures `0`; NaN/Inf predictions `0`; image-path overlap across splits `0`; official Track B rows in train `0`.
- Final losses were train `0.005933` and val `0.006163`.
- Prediction std was low: train `0.2141`, val `0.2648`, test `0.2608`; collapse check flagged `pred_std_too_low=True`.
- Final decision was `STUDENT_50K_SMOKE_NEEDS_PIPELINE_FIX` because TensorFlow GPU was invisible; this run is not sufficient to advance to controlled training.

# Context Notes - TensorFlow GPU Visibility Fix for 50k Smoke

## Decisions
- Scope is GPU environment diagnosis and bounded rerun only. Do not train unless TensorFlow GPU becomes visible.
- Do not edit the student trainer unless the environment check proves a trainer-side problem; expected fix area is runtime library path.
- If the 4000/1000/1000 smoke runs, use a new output directory to avoid overwriting `outputs/mobile_student_iqa_50k_smoke_20260608/`.
- Use `env -u LD_LIBRARY_PATH` and NVIDIA wheel library dirs as the primary probe for both `.venv_gpu` and `.venv_tf`.

## Validation
- Sandboxed `.venv_gpu` and `.venv_tf` both reported TensorFlow `2.21.0`, CUDA build `True`, but `gpus=[]` with CUDA init error 100.
- Inside the sandbox, `/dev/dxg` was absent and PyTorch in `.venv_gpu` also reported `cuda_available=False`, so the original GPU invisibility was caused by the sandbox/session device interface, not by the trainer.
- Unsandboxed `.venv_gpu` with `/usr/lib/wsl/lib` plus NVIDIA wheel libs made TensorFlow see `GPU:0`, but the 4000/1000/1000 smoke failed because runtime CuDNN was `9.1.0` while TensorFlow was compiled with `9.3.0`.
- Unsandboxed `.venv_tf` with `/usr/lib/wsl/lib` plus `.venv_tf` NVIDIA wheel libs made TensorFlow see `GPU:0` and completed the intended bounded smoke.
- Successful smoke output is `outputs/mobile_student_iqa_50k_smoke_gpu_fixed_venv_tf_20260608/`; counts were train `4000`, val `1000`, test `1000`; runtime `155.24` seconds; final train/val loss `0.002758/0.002619`; prediction std train/val/test `9.4622/8.8156/7.9930`; failures `0`; collapse `False`.
- GPU visibility fix report artifacts are under `outputs/mobile_student_iqa_50k_gpu_visibility_fix_20260608/`.
- Final GPU-fix decision: `TF_GPU_FIXED_AND_50K_SMOKE_PASS`.

# Context Notes - Phase 16 Mobile Student IQA 50k Controlled B0

## Decisions
- Scope is one controlled EfficientNetV2B0 baseline only. Do not run B1/B2, hyperparameter sweeps, TFLite export, Flutter changes, production technical_score edits, model asset edits, or commits.
- Use `.venv_tf` only, launched unsandboxed with `env -u LD_LIBRARY_PATH` and `LD_LIBRARY_PATH=/usr/lib/wsl/lib:<.venv_tf nvidia lib dirs>`.
- Use the 50k smoke trainer as the base module because it already implements deterministic split construction, `target_norm = teacher_target_0_100 / 100`, EfficientNetV2B0, ImageNet loading, Huber loss, and split/output report scaffolding.
- Create a sibling controlled script rather than mutating the smoke script; controlled run needs checkpointing, 5+10 epochs, full 50k split use, and official benchmark evaluation.
- Official Track B image-only evaluation is feasible from `outputs/official_protocol_benchmark_manifest_20260604/official_benchmark_test_manifest.csv`, which has `10448` rows with `mos_0_100`: FLIVE `7308`, KonIQ `2015`, SPAQ `1125`.
- `.venv_tf` preflight passed with TensorFlow `2.21.0`, CUDA build true, one visible RTX 4070 SUPER GPU, and cached ImageNet EfficientNetV2B0 weights loaded.
- Controlled B0 completed in `3242.59` seconds with train/val/test `40000/5004/4996`; best checkpoint is final finetune epoch 10 with val MAE `0.048773` and val loss `0.001486`.
- Official Track B image-only evaluation used `4967` image rows; the original `10448` official manifest includes `5481` FLIVE patch rows that remain excluded for image-only benchmark reporting.
- Final decision is `STUDENT_50K_B0_NEEDS_TUNING`: no collapse and no failures, but official SRCC `0.703290` is below the previous 14k student official SRCC `0.743600`.

# Context Notes - Phase 23 Mobile Student IQA 50k Label-Scaled B0 Smoke S2 KonIQ x2

## Decisions
- Scope is final bounded S2 KonIQ x2 B0 smoke only. Do not run B1/B2, full controlled training, TFLite export, Flutter changes, production `technical_score` edits, Flutter model asset edits, commits, ranking loss, or additional label-scaling variants.
- Use `.venv_tf` only with the validated clean `LD_LIBRARY_PATH` launch pattern.
- Keep existing smoke/controlled/ranking defaults unchanged. Add a focused phase runner that imports the existing smoke module for split/model/metric helpers and the controlled module for official Track B evaluation.
- Apply target scaling before split/cap/validation so audit columns travel into `train_manifest.csv`, `val_manifest.csv`, and `test_manifest.csv`.
- Training and split metrics use scaled targets; official Track B evaluation uses official MOS labels and remains unweighted.
- Architecture skill finding: target transformation is a small Module behind the manifest-preparation seam; avoid broad refactor because this phase has one adapter/policy and the existing helper interface already gives enough leverage.

## Validation
- Created `scripts/technical_iqa/train_mobile_student_iqa_50k_label_scaling_smoke.py`.
- Target audit columns implemented for split manifests: original/scaled teacher target, scaled norm target, policy, factor, delta, clip flag, and KonIQ scaling mean.
- Target scaling audit summarizes per-dataset count, original/scaled min/max/mean/std, std ratio, delta stats, and clipping.
- `./.venv_tf/bin/python -m py_compile scripts/technical_iqa/train_mobile_student_iqa_50k_label_scaling_smoke.py` passed.
- Ran exactly one bounded Phase 23 smoke under `.venv_tf` with 4000/1000/1000 caps, EfficientNetV2B0, ImageNet weights, 1 head epoch, 2 finetune epochs, neutral sample weights, disabled rank loss, and S2 KonIQ x2 scaling.
- Output root is `outputs/mobile_student_iqa_50k_label_scaled_b0_smoke_s2_koniq_x2_20260609/`.
- Train/val/test counts were `4000/1000/1000`; runtime was `244.18` seconds; TensorFlow saw one GPU; ImageNet weights loaded.
- Target scaling audit passed after allowing capped-subset KonIQ mean drift tolerance of `0.01`: KonIQ std ratio `2.0`, clipping `0`, FLIVE/SPAQ max abs delta `0`.
- Final train/val loss was `0.00275685/0.00262994`; prediction std train/val/test was `10.2111/9.6546/8.7258`; NaN/Inf `0`; collapse `False`.
- Failure rows were `0`; official evaluation failures were `0`.
- Official Track B SRCC was `0.540411`, below Phase 16 `0.703290`, Phase 19 `0.516337` only slightly above, below Phase 21? no, above Phase 21 `0.480260`, below 14k student `0.743600`, below MDIQA teacher `0.822700`, and below production existing `0.716815`.
- Official safety was weak: overall FP `74`, severe FP `14`, FN `7`.
- Final decision: `LABEL_SCALED_B0_SMOKE_NO_GO_RESEARCH_REFERENCE_ONLY`.

# Context Notes - Adversarial Review of Codex Reproducibility Audit (2026-07-11)

## Findings
- **MobileNetV2 image scale discrepancy**: MobileNetV2 loaded with `weights='imagenet'` expects inputs in `[-1, 1]` or preprocessed via `mobilenet_v2.preprocess_input`. The dataset loader (`csv_dataset.py`) feeds `[0, 1]` floats directly. This creates a data-to-backbone normalization mismatch that degrades ImageNet feature quality.
- **EfficientNetV2B0 double-scaling bug in RGNet practical**: In `src/models/rgnet.py`, the model constructs `EfficientNetV2B0` without setting `include_preprocessing=False` (unlike `nima_distribution.py`). Since the input pipeline feeds `[0, 1]`, this triggers double-scaling (dividing by 255 again inside the network), compressing inputs to `[0, 0.0039]`.
- **FLIVE database-level leakage**: In `make_flive_csv.py`, the `all_df` dataset is built by concatenating full image and patch rows, and then randomly shuffled/split. Since patches are cropped from the same parent images, patches of a test image can end up in the training split. This introduces visual content leakage between train and test/validation sets.
- **AVA protocol deviation**: The NIMA paper trains on 250,000 images and tests on 5,000 images. The local AVA preprocessor `make_ava_csv.py` uses random 80/10/10 splits on the matched subset. If matched count is lower, the evaluation uses a different test distribution.
- **Missing logistic fitting for PLCC**: In IQA evaluation, objective scores must undergo a non-linear logistic mapping to match subjective MOS scale non-linearities. The local benchmark directly computes PLCC using `pearsonr` on raw predictions, which degrades correlation metrics.
- **Unsupported MATCH for ICAA/DAT**: The local codebase lacks the ported model file `src/models/icaa_dat.py`. The model is NOT_IMPLEMENTED, so the scorecard must be corrected to 0 paper fidelity.
- **AADB attribute head omission**: RGNet and AADB models omit the 11 aesthetic attributes required for joint learning in the AADB paper, predicting only the overall score.
- **Scorecard updates**: Corrected paper fidelity and experimental validity scores downward for RGNet practical (from 48 to 18), KonIQ mobile (from 15 to 5), FLIVE image mobile (from 10 to 5), and ICAA/DAT local line (from 25 to 0) to reflect missing components and double-scaling/leakage bugs. NIMA fixed candidate is adjusted slightly (from 72 to 68) due to backbone-to-dataset normalization mismatch (ImageNet weights vs `[0, 1]` input).

## Decisions
- Write all adversarial reports under `outputs/capstone_model_audit_20260711/03_independent_crosscheck/`.
- Ensure first line of each created markdown file has a one-line Korean comment explaining its role, and ensure no Korean sentences end with colons (complying with project rules).
- Document all disputes, missed issues, lineage challenges, and unsafe claims systematically.
- Provide concrete resolution tests (such as fixing the double-scaling bug and evaluating with/without leakage).
