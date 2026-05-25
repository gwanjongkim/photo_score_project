# Checklist

- [x] Confirm existing V4 selector is component/role based and must not be reused as final strict selector.
- [x] Confirm available U2-Net saliency metadata for train, val, and test 1024-image subsets.
- [x] Implement isolated strict A-LAMP paper selector objective.
- [x] Add runner script for compile, smoke, full split generation, comparison, and overlays.
- [x] Run `py_compile`.
- [x] Run 10-image smoke generation.
- [x] Generate train/val/test 1024 strict patch JSONLs.
- [x] Compare strict selector against V4.
- [x] Generate 50 overlay visualizations.
- [x] Validate output counts and JSON/report artifacts.
