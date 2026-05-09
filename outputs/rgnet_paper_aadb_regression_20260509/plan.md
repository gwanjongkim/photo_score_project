# RGNet Paper-Oriented AADB Regression Plan

Assumptions:
- This track is isolated from the practical RGNet and Flutter app.
- The AADB CSV files use `image_path` and `score`, with scores already normalized to [0, 1].
- Exact RegionGraph details are ambiguous, so this implementation will be labeled `RGNet-paper-v0 approximation`.
- Full training may be expensive; a smoke run is required before any longer run.

Steps:
1. Add a separate DenseNet121-based RGNet-paper-v0 model in `src/models/rgnet_paper.py`.
2. Add an AADB training script that saves `final_model.keras`, `best.weights.h5`, `training_history.csv`, and `train_summary.json`.
3. Add an evaluation script that reports SRCC, PLCC, MAE, and RMSE on test and optional validation data.
4. Add a benchmark config under `configs/paper_benchmarks/`.
5. Run compile checks, smoke training, smoke evaluation, and JSON validation.
6. Write the required report, metrics summary, and command log under this output directory.
