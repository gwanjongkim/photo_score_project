# RGNet-v1 Cascaded DenseASPP WD3e-5 Plan

1. Preserve the existing cascaded DenseASPP paper-recipe architecture and change only optimizer weight decay.
2. Train the 20-epoch run with `weight_decay=3e-5` and EarlyStopping disabled.
3. Evaluate `best.weights.h5` on the full AADB test split.
4. Evaluate `final_model.keras` if the best-checkpoint evaluation succeeds.
5. Compare against ES30 SRCC `0.6532747076`, prior paper-recipe SRCC `0.6683026827`, and RGNet baseline SRCC `0.6819`.
