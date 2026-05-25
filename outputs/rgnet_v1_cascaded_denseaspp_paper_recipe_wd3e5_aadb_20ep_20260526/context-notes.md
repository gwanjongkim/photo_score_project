# RGNet-v1 Cascaded DenseASPP WD3e-5 Context Notes

- This is an RGNet-paper-v1 approximation, not an official WACV RGNet reproduction.
- Architecture is unchanged from the cascaded DenseASPP paper-recipe run.
- This ablation changes only optimizer weight decay from `1e-5` to `3e-5`.
- Preflight tracked dirty files are unrelated A-LAMP and TechIQA files. They are intentionally out of scope.
- ES30 best-checkpoint test SRCC: `0.6532747076`.
- Previous DenseASPP paper-recipe best-checkpoint test SRCC: `0.6683026827`.
- Current best RGNet AADB baseline SRCC: `0.6819`.
- Success threshold for continuing DenseASPP route: SRCC greater than `0.6683`.
- Training completed all 20 requested epochs with EarlyStopping disabled.
- Best validation checkpoint: epoch 8, `val_loss=0.0230989102`, `val_mae=0.1200934500`.
- Final epoch showed overfitting relative to best checkpoint: `val_loss=0.0267665926`, `val_mae=0.1292474866`.
- Best-checkpoint evaluation on full AADB test: SRCC `0.6518271602`, PLCC `0.6592002461`, MAE `0.1233198717`.
- Final-model evaluation on full AADB test: SRCC `0.6517927630`, PLCC `0.6602884616`, MAE `0.1251951456`.
- Stronger weight decay did not improve over ES30 or the previous paper-recipe run. It stayed below the DenseASPP route continuation threshold of `0.6683`.
