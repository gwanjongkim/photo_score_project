# TOPIQ-lite ranking_lam01_gap05 Final Candidate TFLite Export Report

## 1. Summary
**PASS**
- Successful export to FP32 and FP16 TFLite.
- Builtin-only conversion confirmed (No Flex/Select TF Ops).
- No Erfc ops present.
- Excellent parity with Keras model (max diff < 0.05 on 100 scale).

## 2. Export Results
| Model | Size | Builtin-only | Flex | Erfc | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| FP32 | 24MB | Yes | No | No | PASS |
| FP16 | 12MB | Yes | No | No | PASS |

## 3. Tensor Signatures
- **Input:** `serving_default_topiq_input:0`, shape=[1, 384, 384, 3], dtype=float32
- **Output:** `StatefulPartitionedCall_1:0`, shape=[1, 1], dtype=float32

## 4. Op Analysis
Unique TFLite ops:
ADD, CONCATENATION, CONV_2D, DELEGATE, DEPTHWISE_CONV_2D, FULLY_CONNECTED, LOGISTIC, MEAN, MUL, PACK, RESHAPE, RESIZE_BILINEAR, SHAPE, STRIDED_SLICE, SUB.
(FP16 adds DEQUANTIZE).

## 5. Parity Results (8 samples per dataset)
| Dataset | FP32 max diff | FP16 max diff | FP16 mean diff | PASS? |
| :--- | :---: | :---: | :---: | :---: |
| SPAQ | 0.01 | 0.03 | 0.02 | YES |
| KonIQ | 0.01 | 0.04 | 0.01 | YES |
| FLIVE | 0.02 | 0.04 | 0.01 | YES |
*Diffs recorded on 0-100 scale.*

## 6. Replacement Implication
- This is the current best ranking-enhanced single-model candidate.
- Export success confirms technical feasibility for mobile deployment.
- **Performance caveat:** While ranking SRCC is much higher than the baseline, FLIVE MAE (6.57) is still higher than the old mobile_flive (~4.4 in some tests). Final approval depends on product acceptance of this trade-off.

## 7. Recommended Next Step
**Android smoke test with ranking_lam01 FP16.**
- Verify on-device latency and output range.

## 8. Files Created
- `outputs/final_topiq_lite_ranking_lam01_export_20260517/topiq_lite_ranking_lam01_gap05_fp32.tflite`
- `outputs/final_topiq_lite_ranking_lam01_export_20260517/topiq_lite_ranking_lam01_gap05_fp16.tflite`
- `outputs/final_topiq_lite_ranking_lam01_export_20260517/summary.csv`
- `outputs/final_topiq_lite_ranking_lam01_export_20260517/tflite_ops.txt`
- `outputs/final_topiq_lite_ranking_lam01_export_20260517/parity_*.csv`

## 9. Final GO
- Ready for mobile integration testing.
