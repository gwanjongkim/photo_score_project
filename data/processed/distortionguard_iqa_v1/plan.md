# DistortionGuard-IQA v1 Synthetic Smoke Plan

## Goal
Create a bounded synthetic distortion dataset for Stage A pretraining smoke validation, using existing clean/general image manifests as sources.

## Scope
- Add only a standalone dataset generator.
- Generate a small smoke dataset from `data/processed/techiqa_guard/smoke_v1.csv`.
- Write distorted-image manifest, ranking-pair CSV, and summary JSON.
- Avoid model, trainer, export, and Flutter changes.

## Verification
1. Compile `src/data/generate_distortionguard_synthetic.py`.
2. Run the requested 20-image smoke command.
3. Inspect the first rows of the manifest and pair CSV.
4. Inspect the summary JSON for processed/generated/failed counts.
