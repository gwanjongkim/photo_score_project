# RGNet Paper-Oriented AADB Regression Context Notes

## 2026-05-09

- The practical RGNet in `src/models/rgnet.py` is intentionally left untouched.
- The new model will be marked as `RGNet-paper-v0 approximation`, not an official reproduction.
- To keep the experiment isolated, planning artifacts live under `outputs/rgnet_paper_aadb_regression_20260509/`.
- The worktree already had many unrelated modified and untracked files before this task; this task will not revert them.
- AADB counts from `data/processed/aadb`: train `7612`, val `846`, test `1000`.
- Prior local paper audit says current practical RGNet uses EfficientNetV2B0 and approximated graph reasoning; this track changes only the new paper-oriented path.
- Prior local comparison numbers available for report: current practical RGNet AADB baseline on AADB val512 SRCC `0.4987`, PLCC `0.5094`, MAE `0.1304`, RMSE `0.1619`; AVA-to-AADB fine-tuned practical RGNet on AADB val512 SRCC `0.6749`, PLCC `0.6835`, MAE `0.1104`, RMSE `0.1381`.
- Implemented `src/models/rgnet_paper.py` as DenseNet121 plus spatial region-node graph reasoning. The model uses random init when `backbone_weights=none`, and attempts ImageNet weights when requested.
- Implemented train/eval scripts with explicit `official_reproduction: false` summary fields and smoke subset controls.
- First smoke train wrote `smoke_train/` and completed, but its save/load check compared against a pre-training forward pass. The train script was corrected, and the verified rerun will use a separate output directory to avoid overwriting.
- Verified smoke train output is `smoke_train_v2/`: 8 train samples, 4 val samples, 1 epoch, random DenseNet weights, val_loss `0.03703468665480614`, val_mae `0.17081350088119507`, save/load max abs diff `0.0`.
- Verified smoke eval output is `smoke_eval_v2/`: test subset 8 samples SRCC `-0.203596464525254`, PLCC `-0.31242404651323874`, MAE `0.1755114048719406`, RMSE `0.20520318169267343`; val subset 8 samples SRCC `0.14371515142959107`, PLCC `0.15893311762891194`, MAE `0.17503324151039124`, RMSE `0.1971614864921894`.
- TensorFlow inside the sandbox did not expose CUDA (`visible_gpus: []`) and logged `failed call to cuInit`; smoke training/eval completed on CPU.
- Final report artifacts written: `rgnet_paper_aadb_report.md`, `metrics_summary.json`, and `command_log.txt`.
