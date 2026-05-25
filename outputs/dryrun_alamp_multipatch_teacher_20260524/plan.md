# Plan - A-LAMP Multi-Patch Teacher Baseline

## Goal
Implement an isolated VGG16-based A-LAMP Multi-Patch teacher baseline that consumes the validated external adaptive patch JSONL manifests.

## Scope
- Add new model, dataset, trainer, and config files only.
- Treat external patch boxes as adaptive selections, not labels.
- Train a Multi-Patch teacher baseline only; do not implement Layout-Aware subnet, TFLite export, Flutter integration, or full A-LAMP reproduction claims.

## Steps
1. Add a JSONL dataset loader for 5 clipped PIL crops per image, resized to 224x224 and VGG16-preprocessed.
2. Add a shared-VGG16 Multi-Patch classifier with GAP, mean+max orderless aggregation, and a binary sigmoid head.
3. Add a trainer/config that defaults to frozen VGG16 backbone and writes the requested artifacts.
4. Verify with `py_compile` and the requested 16/8 one-epoch smoke training only.

## Success Criteria
- `src/models/alamp_multipatch_teacher.py` compiles.
- `src/datasets/alamp_external_patch_dataset.py` compiles.
- `src/train/train_alamp_multipatch_teacher.py` compiles.
- Smoke training writes `best.weights.h5`, `final_model.keras`, `training_history.csv`, and `train_summary.json` under `outputs/dryrun_alamp_multipatch_teacher_20260524`.
