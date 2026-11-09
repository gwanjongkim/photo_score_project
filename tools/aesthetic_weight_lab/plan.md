# Aesthetic Weight Lab Plan

Build a portable, TFLite-only scoring and weight-review tool for team members who clone the repo, add private images under `test_images/`, run one command, and open a local HTML report.

1. Create the new lab structure under `tools/aesthetic_weight_lab/` without modifying training code, Flutter code, or `src/`.
   Verify: new files live in the requested tool/config/model/test-image paths only.
2. Bundle only deployment-usable TFLite models into `models/aesthetic/`.
   Verify: baseline NIMA, RGNet, A-LAMP and the fine-tuned A-LAMP candidate exist there; excluded RGNet retry/fine-tuned artifacts are absent.
3. Add YAML configs that point to swappable model files and weights.
   Verify: baseline config uses baseline A-LAMP; fine-tuned config swaps only A-LAMP.
4. Implement a TFLite runner and report generator.
   Verify: enabled models run from config, scores are written to CSV/JSON, and HTML reranks in-browser from adjustable weights.
5. Document Git LFS, private-image handling, intended use, and limitations.
   Verify: README, model card, SHA256 manifest, and handoff report are present.
6. Validate with compile checks and smoke runs when image samples are available.
   Verify: command log and handoff report record exact results.
