# RGNet-v1 Cascaded DenseASPP ES30 Context Notes

- This is an RGNet-paper-v1 approximation, not an official WACV RGNet reproduction.
- The experiment does not change architecture. It reuses the cascaded DenseASPP + paper-style recipe path.
- Preflight tracked dirty files are unrelated A-LAMP and TechIQA files. They are not modified by this run.
- Previous cascaded DenseASPP paper-recipe best-checkpoint test SRCC: `0.6683026827`.
- Current best RGNet AADB baseline SRCC: `0.6819`.
- Success threshold for this ablation is SRCC greater than `0.6683`; beating the current baseline requires SRCC greater than `0.6819`.
- Trainer early stopping restores best weights when enabled. The primary comparison remains `best.weights.h5`.
- A first outside-sandbox start inherited `disable_early_stopping: true` from the config and was terminated before completing epoch 1. The corrected run explicitly passed `--disable_early_stopping false`.
- Corrected training completed 14 epochs, stopped by EarlyStopping, and restored best weights from epoch 6.
- Best validation result: `val_loss=0.0227430612`, `val_mae=0.1203708574`.
- Best-checkpoint evaluation completed on the full 1000-image AADB test split: SRCC `0.6532747076`, PLCC `0.6641100023`, MAE `0.1213828921`.
- Final-model evaluation was effectively identical to best-checkpoint evaluation because EarlyStopping restored best weights before saving `final_model.keras`: SRCC `0.6532700892`, PLCC `0.6641100357`, MAE `0.1213828772`.
- EarlyStopping improved validation loss versus the previous paper-recipe run but did not improve test SRCC.
