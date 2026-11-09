# Recommended Next Action

## 1. Reliability Verdict
**CSV_RELIABLE_FOR_WEIGHT_SEARCH**
The audit confirms that the WSL benchmark CSVs are technically sound, correctly aligned, and follow a deterministic split protocol with no major leakage.

## 2. Root Cause of Intuition Mismatch
The weight search selected **RGNet (0.41)** over **A-LAMP (0.22)** because:
- **RGNet** is the "AADB hero" (SRCC 0.66), dominating the compositional/ranking signals in that dataset.
- **A-LAMP** is the "AVA hero" (SRCC 0.58), dominating the holistic aesthetic signals.
- The previous search objective (Combined 70/30 SRCC) balanced these two, but because A-LAMP's AADB performance was relatively poor (SRCC 0.30), the optimizer reduced its weight to prevent AADB degradation.
- **Real-device intuition** likely values the "A-LAMP style" (vibrant, high-contrast aesthetics) more than the "RGNet style" (compositional structure), especially for A-cut selection.

## 3. Recommended Next Steps
1. **Multi-Objective Search**: Do not just optimize for SRCC. Add a constraint or objective for "A-LAMP prominence" or optimize for a combined rank that penalizes low A-LAMP weights if A-LAMP is known to match product requirements better.
2. **Collect Product-Specific Triplets**: Collect 50-100 image triplets ranked by the product team. Use this as a "Gold Dataset" to validate any candidate weights.
3. **WSL/Flutter Parity Check**: Sample 20 images from the aligned CSV, run them through the Flutter implementation, and compare the log scores with the `_pred_scaled_1_10` values in the CSV.
   - If difference > 0.05, investigate resize/normalization mismatches.
   - If difference < 0.05, the WSL benchmark is a perfectly valid proxy for the mobile device.
4. **Calibrated vs Raw**: The search used "Calibrated" weights (linear fit). Ensure the Flutter implementation uses the same slope/intercept if it wants to match the ensemble logic exactly, OR use "Raw Scaled" weights if the implementation only does simple averaging.
