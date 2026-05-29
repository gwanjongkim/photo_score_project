# Decision Summary: C6 Android Parity

**Final Decision:** C6 should remain **log-only** and must **not** replace the production `technical_score`.

While C6 shows strong rank correlation (SRCC 0.97), the absolute error (Max AE ~4.6) is too high for a production-grade replacement. Tensor-level investigation reveals that the preprocessing pipelines between Android and WSL remain distinct, and simple dimension matching does not close the gap.

### Strongest Evidence
- **Max AE Failure:** Even with the best identified backend (PIL Lanczos), the Max AE of 4.6256 exceeds the target threshold of 3.0.
- **Tensor Mismatch:** Mean relative checksum difference (0.013) and patch mean differences (0.36) confirm the input tensors are not identical.
- **Diagnostic Failure:** Source-dimension matching worsened the parity metrics, suggesting deeper differences in decoding or interpolation kernels.
- **Branch Sensitivity:** 3 images show significant errors specifically due to flipping the `min()` cap branch in the C6 formula.

### Next Action
Document the identified discrepancies and maintain C6 as a debug-only metric. No further tensor-level tuning is recommended at this stage.
