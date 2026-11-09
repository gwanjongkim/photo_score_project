# ICAA17K DAT GridSample-Free v2 Implementation Result

## Scope

Implemented `experiments/icaa_export_safe_v2/` as an isolated GridSample-free copy of the existing export-safe ICAA17K DAT model. Official repository files, A-cut source files, Flutter files, and `experiments/icaa_export_safe/` v1 files were not modified.

No training was run. No TFLite conversion was started.

## Files Created

- `experiments/icaa_export_safe_v2/export_safe_models/dat.py`
- `experiments/icaa_export_safe_v2/export_safe_models/dat_blocks.py`
- `experiments/icaa_export_safe_v2/export_safe_models/SoftHistogram.py`
- `experiments/icaa_export_safe_v2/export_safe_models/__init__.py`
- `experiments/icaa_export_safe_v2/scripts/parity_v1_vs_v2_pytorch.py`
- `experiments/icaa_export_safe_v2/scripts/export_and_parity_onnx_v2.py`
- `experiments/icaa_export_safe_v2/reports/grid_sample_free_implementation_result.md`

## Files Modified

- `experiments/icaa_export_safe_v2/export_safe_models/dat_blocks.py`

The v2 files were copied from `experiments/icaa_export_safe/export_safe_models/` before modifying the v2 copy.

## F.grid_sample Call Sites Replaced

In v1 source `experiments/icaa_export_safe/export_safe_models/dat_blocks.py`, the two `F.grid_sample` usages were:

- Spatial feature sampling in `DAttentionBaseline.forward`, around v1 line `283`.
- Relative position bias sampling in `DAttentionBaseline.forward`, around v1 line `319`.

In v2 `experiments/icaa_export_safe_v2/export_safe_models/dat_blocks.py`, both are replaced with `manual_bilinear_grid_sample`:

- Spatial feature sampling now calls `manual_bilinear_grid_sample(...)` around v2 line `353`.
- Relative position bias sampling now calls `manual_bilinear_grid_sample(...)` around v2 line `389`.

`rg` confirms no `F.grid_sample`, NumPy, `np.*`, or `torch.from_numpy` usage remains in the v2 model files.

## Manual Bilinear Sampler Summary

Implemented `manual_bilinear_grid_sample(input, grid)` in v2 `dat_blocks.py`.

Behavior replicated:

- Input tensor format: NCHW.
- Grid format: `[N, H_out, W_out, 2]`, normalized x/y coordinates in `[-1, 1]`.
- `align_corners=True`: normalized `-1` maps to index `0`, normalized `+1` maps to index `size - 1`.
- `padding_mode='zeros'`: invalid corner samples are masked to zero contribution.
- Bilinear sampling uses `x0/x1/y0/y1`, four corner weights, clamped gather indices, and flattened NHWC gather via `torch.index_select`.
- No Python loops over pixels and no custom ops.

## PyTorch v1 vs v2 Parity

Command:

```bash
python experiments/icaa_export_safe_v2/scripts/parity_v1_vs_v2_pytorch.py \
  --limit 32 \
  --batch_size 1 \
  --device cpu
```

Artifacts:

- `outputs/icaa_export_safe_v2_pytorch_parity_20260514_215433/parity_report.json`
- `outputs/icaa_export_safe_v2_pytorch_parity_20260514_215433/parity_predictions.csv`
- `outputs/icaa_export_safe_v2_pytorch_parity_20260514_215433/parity_log.txt`

Result:

- Status: passed.
- Pass level: acceptable.
- Evaluable images: `32`.
- Missing images skipped: `0`.
- Full tensor max abs diff: `0.0000248551`.
- Full tensor mean abs diff: `0.0000004787`.
- MOS max abs diff: `0.0000248551`.
- MOS mean abs diff: `0.0000008270`.
- Color max abs diff: `0.0000032187`.
- Color mean abs diff: `0.0000001304`.

Threshold interpretation:

- Preferred `max_abs_diff <= 1e-5`: not met.
- Acceptable `max_abs_diff <= 1e-4`: met.

## ONNX Export Result

Command:

```bash
python experiments/icaa_export_safe_v2/scripts/export_and_parity_onnx_v2.py \
  --limit 32 \
  --batch_size 1 \
  --device cpu
```

Artifacts:

- `outputs/icaa_export_safe_v2_onnx_20260514_215455/icaa_dat_export_safe_v2.onnx`
- `outputs/icaa_export_safe_v2_onnx_20260514_215455/onnx_op_histogram.json`
- `outputs/icaa_export_safe_v2_onnx_20260514_215455/onnx_parity_report.json`
- `outputs/icaa_export_safe_v2_onnx_20260514_215455/onnx_parity_predictions.csv`
- `outputs/icaa_export_safe_v2_onnx_20260514_215455/onnx_export_log.txt`

Result:

- ONNX export opset: `17`.
- `onnx.checker.check_model`: passed.
- ONNX Runtime provider: `CPUExecutionProvider`.
- ONNX model size: about `336M`.
- GridSample count: `0`.

Export warnings:

- `torch.as_tensor` constants registered during tracing.
- Tensor-to-Python-float trace warnings.

These warnings are tied to fixed static shape export and did not block ONNX Runtime parity for `[1, 3, 224, 224]`.

## ONNX Operator Histogram

From `outputs/icaa_export_safe_v2_onnx_20260514_215455/onnx_op_histogram.json`:

```json
{
  "Abs": 1,
  "Add": 528,
  "And": 200,
  "Cast": 382,
  "Clip": 80,
  "Concat": 541,
  "Constant": 4255,
  "ConstantOfShape": 160,
  "Conv": 86,
  "Div": 189,
  "Einsum": 48,
  "Equal": 20,
  "Erf": 34,
  "Expand": 40,
  "Flatten": 1,
  "Floor": 40,
  "Gather": 845,
  "Gemm": 3,
  "GlobalAveragePool": 1,
  "GreaterOrEqual": 80,
  "LayerNormalization": 63,
  "LessOrEqual": 110,
  "MatMul": 76,
  "Mul": 974,
  "Range": 40,
  "Reciprocal": 20,
  "Relu": 1,
  "Reshape": 588,
  "Shape": 925,
  "Sigmoid": 12,
  "Slice": 178,
  "Softmax": 24,
  "Squeeze": 1,
  "Sub": 170,
  "Tanh": 20,
  "Tile": 20,
  "Transpose": 302,
  "Unsqueeze": 1446,
  "Where": 130
}
```

`GridSample`: `0`.

## ONNX Runtime Parity

Result from `outputs/icaa_export_safe_v2_onnx_20260514_215455/onnx_parity_report.json`:

- Status: passed.
- Evaluable images: `32`.
- Missing images skipped: `0`.
- Full tensor max abs diff: `0.0000203252`.
- Full tensor mean abs diff: `0.0000014622`.
- MOS max abs diff: `0.0000203252`.
- MOS mean abs diff: `0.0000018096`.
- Color max abs diff: `0.0000133514`.
- Color mean abs diff: `0.0000011148`.

Pass thresholds:

- `max_abs_diff <= 1e-4`: met.
- `mean_abs_diff <= 1e-5`: met.

## TFLite Retry Readiness

It is safe to retry TFLite conversion from the specific GridSample-removal standpoint:

- The v2 ONNX graph has `GridSample` count `0`.
- ONNX checker passed.
- ONNX Runtime parity against v2 PyTorch passed under the requested thresholds.
- Static single-image shape `[1, 3, 224, 224]` was used.

This does not claim TFLite conversion success. TFLite conversion and LiteRT runtime validation still need to be run as a separate step.
