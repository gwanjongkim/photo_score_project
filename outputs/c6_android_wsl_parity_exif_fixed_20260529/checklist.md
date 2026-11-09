# Checklist

- [x] Copy original WSL parity script into the EXIF-fixed output directory.
- [x] Import `ImageOps` from PIL in the copied script.
- [x] Apply `ImageOps.exif_transpose(img)` before `convert("RGB")`.
- [x] Preserve HEIF registration and existing model/scoring contracts.
- [x] Run `.venv/bin/python outputs/c6_android_wsl_parity_exif_fixed_20260529/run_wsl_c6_parity_exif_fixed.py`.
- [x] Confirm `wsl_c6_scores.csv`, `android_vs_wsl_parity.csv`, and `report.md` exist.
- [x] Confirm report includes required metrics, before-vs-after comparison, top 10 remaining mismatches, and decision.
