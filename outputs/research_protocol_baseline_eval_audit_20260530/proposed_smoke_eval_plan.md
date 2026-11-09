# Proposed Smoke Evaluation Plan

## Scope
This plan is for a smoke-only audit of the existing technical scoring baselines on the research-protocol manifests. It does not run full 271k inference and does not change production scoring behavior.

## Exact Smoke Sample
Default sample size: 5 images per dataset/split.

Sampling rule:
- KonIQ-10k: sample 5 from `run_01/train` and 5 from `run_01/test`.
- SPAQ: sample 5 from `train` and 5 from `test`.
- FLIVE: sample 5 image rows from `train`, `val`, and `test`.
- C6 comparison-only: reuse the same sampled image rows from KonIQ and FLIVE.

Why this shape:
- It preserves split awareness.
- It keeps the run bounded.
- It avoids the misleading "first 5 rows of the CSV" shortcut that the existing benchmark scripts would otherwise use.

## Safe Command Path
Recommended first step.

```bash
PYTHONPATH=. ./.venv_gpu/bin/python outputs/research_protocol_baseline_eval_audit_20260530/run_research_protocol_smoke_eval.py --sample_size 5 --output_dir outputs/research_protocol_baseline_eval_audit_20260530/smoke_prep
```

This prepares the smoke CSVs and a command note file only. It does not run full scoring.

## Expected Output Files
The smoke prep script should write:
- `outputs/research_protocol_baseline_eval_audit_20260530/smoke_prep/sampled_manifests/koniq_run01_smoke.csv`
- `outputs/research_protocol_baseline_eval_audit_20260530/smoke_prep/sampled_manifests/spaq_smoke.csv`
- `outputs/research_protocol_baseline_eval_audit_20260530/smoke_prep/sampled_manifests/flive_image_smoke.csv`
- `outputs/research_protocol_baseline_eval_audit_20260530/smoke_prep/smoke_eval_commands.txt`

## Follow-On Evaluation Command Shape
If a later manual smoke run is approved, use the sampled CSVs above and run only the smallest available benchmark harnesses.

Recommended stop-before-full-eval conditions:
- any missing image paths in the sampled CSVs
- any missing or non-numeric score values
- a mismatch between expected and actual split counts
- any sign that a harness is silently using unbounded CSV order instead of the sampled subsets

## Risk Notes
- The existing benchmark script `scripts/eval_final_topiq_candidates_vs_existing_technical.py` supports `--max_samples`, but that is not split-stratified.
- The production pipeline in `src/infer/run_acut_pipeline.py` can be smoke-run on a tiny input set, but it should not be used to replace the existing technical score.
- C6 remains comparison-only and should stay out of production paths.

## Decision
The audit is not yet ready for direct smoke execution without the adapter/prep step because the current benchmark scripts do not expose the exact split-stratified sample shape required here.
