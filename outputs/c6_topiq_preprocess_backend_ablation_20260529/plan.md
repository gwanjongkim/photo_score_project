# Plan

## Goal
Run an isolated TOPIQ preprocessing backend ablation to find which WSL reference backend best matches Android Flutter TOPIQ and C6 scores.

## Assumptions
- Android TOPIQ contract is `384x384` RGB, resize-with-pad, pad value `0`, raw `0..255`, no `/255`, output raw `* 100`.
- KonIQ remains fixed to the latest EXIF-fixed parity behavior.
- Only diagnostic artifacts under this output directory may be created or modified.
- No training, model export, model mutation, Flutter repo edits, production source edits, or scoring formula changes are allowed.

## Verification
1. Run the isolated ablation script with `./.venv_gpu/bin/python`.
2. Confirm all 50 Android CSV images are processed.
3. Confirm both FP16 and FP32 TOPIQ TFLite artifacts are evaluated for all four preprocessing backends.
4. Confirm `backend_summary.csv`, `per_image_comparison_best.csv`, and `report.md` are written.
