# Context Notes

- Scope is isolated to `src/datasets/alamp_paper_strict_patch_selector.py`, `tools/run_alamp_paper_strict_patch_selection.sh`, and `outputs/alamp_paper_strict_patch_selector_20260512/`.
- Existing V4 selector is not final here because it is component/role based. This task requires the paper objective terms instead.
- Saliency source is a deliberate substitution: the original graph-based saliency implementation from the paper is not locally available for this run, so existing U2-Net saliency maps are used only as the `S` map source.
- MPNet training, layout graph implementation, Flutter, forWeights, practical A-LAMP, and RGNet are out of scope.
- Patch selection should emit exactly five patches per non-skipped image and record objective components for auditability.
- Validation completed on 2026-05-12 with `./.venv_gpu/bin/python -m py_compile src/datasets/alamp_paper_strict_patch_selector.py`, 10-image smoke generation, and full 1024-image train/val/test generation.
- Full generation produced 1024 valid records and 0 skipped records for each split. Train had 25 overlap-threshold relaxations, val had 23, and test had 30.
- V4 comparison is report-only. V4 was not used to select strict patches.
- Manual gate artifact is 50 generated overlays in `outputs/alamp_paper_strict_patch_selector_20260512/overlay_visualizations`.
