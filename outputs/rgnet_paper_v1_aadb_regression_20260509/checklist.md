# RGNet Paper-Oriented v1 AADB Regression Checklist

- [x] Inspect v0 model, training, evaluation, config, and full-result artifacts.
- [x] Implement isolated `src/models/rgnet_paper_v1.py`.
- [x] Implement isolated v1 training/evaluation scripts.
- [x] Add `configs/paper_benchmarks/rgnet_paper_v1_aadb_regression.yaml`.
- [x] Run `py_compile` on changed/new scripts.
- [x] Run pure forward-pass smoke.
- [x] Run smoke training.
- [x] Validate smoke `train_summary.json`.
- [x] Run smoke evaluation.
- [x] Run mid-size training gate.
- [x] Run full AADB training if gates pass.
- [x] Run full AADB test/val evaluation.
- [x] Write final report, metrics summary, and command log.
- [x] Confirm v0/practical RGNet/Flutter/forWeights were not touched.
