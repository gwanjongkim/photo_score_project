# RGNet Paper-Oriented v1 AADB Regression Plan

Assumptions:
- v0 must remain reproducible and unchanged.
- v1 is an RGNet-paper-v1 approximation, not official paper code or weights.
- ASPP is the minimum paper-aligned context module for this pass; exact DenseASPP is deferred unless needed.
- Full training is gated on compile, forward, smoke, and mid-run checks.

Steps:
1. Inspect v0 model/train/eval/config and existing v0 full-result artifacts.
2. Add isolated v1 model/train/eval scripts and v1 config.
3. Validate syntax and run a pure forward-pass smoke.
4. Run smoke training and smoke evaluation on tiny AADB subsets.
5. Run a mid-size training gate with ImageNet weights if available.
6. Run full AADB training/evaluation only if prior gates pass and GPU is visible.
7. Write report, metrics summary, and command log with v0/practical comparisons.
