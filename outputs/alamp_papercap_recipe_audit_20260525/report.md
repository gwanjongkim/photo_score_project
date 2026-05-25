# A-LAMP Paper-Capacity Recipe Audit

## 1. Summary
This audit compares our current A-LAMP Multi-Patch "paper-capacity 4096" implementation against the external reference code (`train_complete.py`). We identified significant discrepancies in both architecture and training recipe that explain the poor generalization (Train AUC 0.93 vs Val AUC 0.64) observed in recent midruns. The primary culprit is the **total absence of weight regularization (L2)** in our implementation, combined with a **100x higher learning rate** and lack of data augmentation.

## 2. External Architecture
The external implementation uses a deeper shared projection and unfreezes part of the VGG16 backbone:
- **Shared Backbone:** VGG16 (include_top=False, **Block 5 trainable**) -> Flatten -> **Dense(4096, relu)** -> **Dropout(0.5)** -> **Dense(4096, relu)**.
- **Aggregation:** Orderless Mean + Orderless Max (Result: 8192-dim).
- **Head:** **Dense(4096, relu)** -> **Dropout(0.5)** -> **Dense(4096, relu)** -> Dense(1, sigmoid).
- **Total Depth:** 4 Dense layers (2 shared, 2 in head) + VGG Block 5.

## 3. Current Architecture (Midrun)
Our implementation is slightly shallower in the shared part and keeps the backbone frozen:
- **Shared Backbone:** VGG16 (include_top=False, **Frozen**) -> Flatten -> **Dense(4096, relu)**.
- **Aggregation:** Orderless Mean + Orderless Max (Result: 8192-dim).
- **Head:** **Dense(4096, relu)** -> **Dropout(0.5)** -> **Dense(4096, relu)** -> **Dropout(0.5)** -> Dense(1, sigmoid).
- **Total Depth:** 3 Dense layers (1 shared, 2 in head).

## 4. Architecture Differences
- **Missing Layer:** We are missing one 4096-dim Dense layer in the shared patch subnet before aggregation.
- **Backbone Trainability:** External unfreezes VGG16 from layer 15 (`block5_conv1`), whereas we keep it frozen.
- **Dropout Placement:** Minor difference in dropout at the very top (we have an extra dropout after the second head layer).

## 5. External Training Recipe
- **Optimizer:** SGD with momentum 0.9.
- **Learning Rate:** **1.0e-6** (extremely conservative).
- **Regularization:** **L2 1e-5** manually added to ALL Conv2D and Dense layers (kernels and biases).
- **Augmentation:** Horizontal Flip.
- **Batch Size:** 4.
- **Epochs:** 8.

## 6. Current Training Recipe
- **Optimizer:** Adam.
- **Learning Rate:** **1.0e-4** (100x higher than external).
- **Regularization:** **None**. No L2 weight decay is applied.
- **Augmentation:** None.
- **Batch Size:** 1 (in paper-capacity run).

## 7. High-Impact Differences
1. **L2 Regularization (Critical):** A 153M parameter model with no weight decay is guaranteed to overfit AVA-sized datasets (or subsets). External uses 1e-5; we use 0.
2. **Learning Rate (High):** Our LR is 100x higher than the reference. Combined with Adam (which can be more aggressive), this leads to rapid convergence to a local minimum/memorization.
3. **Shared Depth (Medium):** The external shared projection is twice as deep (2 layers vs 1).
4. **Data Augmentation (Medium):** Lack of horizontal flips reduces effective dataset size.

## 8. Interpretation of Smallrun and Midrun Results
- **Smallrun (512 samples):** Val AUC ~0.5. With only 512 samples and 153M params, the model likely memorized the training set instantly without learning any generalizable aesthetic features.
- **Midrun (8192 samples):** Train AUC 0.93, Val AUC 0.64. The increased data helped pull Val AUC above chance, but the lack of regularization allowed the model to overfit heavily. The "signal" is present but the model is too "loose."

## 9. Recommended Next Experiment
**DO NOT run larger training (full AVA) as-is.** It will likely result in the same overfitting pattern and wasted compute.

**Recommended Plan (Recipe Adjustment):**
1. **Implement L2 Regularization:** Add `kernel_regularizer=l2(1e-5)` and `bias_regularizer=l2(1e-5)` to all Dense layers and the projection layer.
2. **Lower Learning Rate:** Reduce to `2e-5` (if using Adam) or `1e-6` (if using SGD).
3. **Enable Horizontal Flip:** Add simple random flip to the data pipeline.
4. **Match Shared Depth:** Add the second Dense(4096) layer to the shared patch subnet.
5. **Unfreeze Block 5:** Consider unfreezing VGG16 Block 5 ONLY if L2 is active.

## 10. Final Judgment
The current "paper-capacity" implementation is a **structural match** in spirit but a **recipe mismatch** in practice. The large trainable head (153M params) requires the strict regularization used in the original paper. Without it, the "paper-capacity" actually becomes a liability (overfitting) rather than an asset (accuracy). 

**Next Step Recommendation:** Adjust the model definition to include L2 and the extra shared layer, then rerun the 8192 midrun to verify Val AUC improvement (targeting >0.70) before committing to a full run.
