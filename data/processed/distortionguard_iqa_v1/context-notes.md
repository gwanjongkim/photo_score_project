# DistortionGuard-IQA v1 Synthetic Smoke Context Notes

- 2026-05-23: Task starts from `outputs/distortionguard_iqa_v1_design_20260523/report.md`; implementation begins with synthetic dataset generation only.
- 2026-05-23: Source manifest choice is `data/processed/techiqa_guard/smoke_v1.csv` for the requested smoke run; `image_path` and `dataset` provide source image path and source dataset labels.
- 2026-05-23: Distorted samples are written under `data/processed/distortionguard_iqa_v1/synthetic_smoke/`; generated data remains under the new DistortionGuard branch.
- 2026-05-23: Original/reference images are not duplicated for this smoke; pair rows use the original source path as both `ref_image_path` and the better item for original-vs-distorted comparisons.
- 2026-05-23: `expected_quality_order` is represented as a descending within-distortion quality rank where severity 1 maps higher than severity 5.
- 2026-05-23: Smoke verification completed with 20 source images, 1000 distorted images, 3000 pair rows, 0 failed images, and Pillow WebP support enabled.
- 2026-05-23: `.gitignore` has an unanchored `data/` rule, so `src/data/generate_distortionguard_synthetic.py` and `data/processed/...` files are ignored unless force-staged.
