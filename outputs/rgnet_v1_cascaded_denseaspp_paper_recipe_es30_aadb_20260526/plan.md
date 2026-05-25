# RGNet-v1 Cascaded DenseASPP ES30 Plan

1. Preserve the cascaded DenseASPP paper-recipe architecture and change only the schedule/early-stopping behavior.
2. Train the 30-epoch early-stopping ablation with restore-best behavior enabled.
3. Evaluate `best.weights.h5` on the full AADB test split, then evaluate `final_model.keras` if the main checkpoint evaluation succeeds.
4. Compare against the previous paper-recipe DenseASPP SRCC `0.6683026827` and the current RGNet baseline SRCC `0.6819`.
