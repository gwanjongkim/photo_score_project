# ICAA DAT export-safe v3 final pooling result

## Summary

v3 was created under `experiments/icaa_export_safe_v3/` by copying the v2 export-safe model and changing only the final pooling path in `export_safe_models/dat.py`.

The v3 PyTorch and ONNX gates passed, and the ONNX graph no longer contains `GlobalAveragePool`.

The v3 FP32 TFLite conversion completed, but the generated TFLite model is still not input-sensitive. It returns the same output for zero, one, random, and real-image inputs. Do not proceed to TFLite real-image parity yet.

## Files Created

- `experiments/icaa_export_safe_v3/export_safe_models/dat.py`
- `experiments/icaa_export_safe_v3/export_safe_models/dat_blocks.py`
- `experiments/icaa_export_safe_v3/export_safe_models/SoftHistogram.py`
- `experiments/icaa_export_safe_v3/export_safe_models/__init__.py`
- `experiments/icaa_export_safe_v3/scripts/parity_v2_vs_v3_pytorch.py`
- `experiments/icaa_export_safe_v3/scripts/export_and_parity_onnx_v3.py`
- `experiments/icaa_export_safe_v3/scripts/convert_and_probe_tflite_v3.py`
- `experiments/icaa_export_safe_v3/reports/final_pooling_safe_result.md`

## Files Modified

- `experiments/icaa_export_safe_v3/export_safe_models/dat.py`

No official repo files, A-cut source files, Flutter files, v1 files, or v2 files were modified.

## Final Pooling Change

v2 used:

```python
x = self.cls_norm(x)
x = F.adaptive_avg_pool2d(x, 1)
x = torch.flatten(x, 1)
```

v3 uses explicit NCHW spatial flattening and mean:

```python
x = self.cls_norm(x)
B, C, H, W = x.shape
x = x.reshape(B, C, H * W)
x = x.mean(dim=2)
```

Reason: the v2 TFLite debug showed the input signal survived through `cls_norm`, then collapsed immediately after the converted final `GlobalAveragePool` / `MEAN` path. The v3 change removes ONNX `GlobalAveragePool` and makes the intended NCHW spatial reduction explicit.

## PyTorch v2 vs v3 Parity

Requested CUDA command:

```bash
python experiments/icaa_export_safe_v3/scripts/parity_v2_vs_v3_pytorch.py --limit 256 --batch_size 1 --device cuda
```

Result: failed before inference because `torch.cuda.is_available() is False` in this Python environment.

CPU fallback command:

```bash
python experiments/icaa_export_safe_v3/scripts/parity_v2_vs_v3_pytorch.py --limit 256 --batch_size 1 --device cpu
```

Artifacts:

- `outputs/icaa_export_safe_v3_pytorch_parity_20260515_170845/parity_report.json`
- `outputs/icaa_export_safe_v3_pytorch_parity_20260515_170845/parity_predictions.csv`
- `outputs/icaa_export_safe_v3_pytorch_parity_20260515_170845/parity_log.txt`

Result: passed, preferred.

- evaluable images: `256`
- missing image rows skipped: `0`
- full tensor max abs diff: `0.0`
- full tensor mean abs diff: `0.0`
- MOS max / mean diff: `0.0` / `0.0`
- color max / mean diff: `0.0` / `0.0`

## ONNX Export and Runtime Parity

Command:

```bash
python experiments/icaa_export_safe_v3/scripts/export_and_parity_onnx_v3.py --limit 32 --device cpu
```

Artifacts:

- `outputs/icaa_export_safe_v3_onnx_20260515_171049/icaa_dat_export_safe_v3.onnx`
- `outputs/icaa_export_safe_v3_onnx_20260515_171049/onnx_op_histogram.json`
- `outputs/icaa_export_safe_v3_onnx_20260515_171049/onnx_parity_report.json`
- `outputs/icaa_export_safe_v3_onnx_20260515_171049/onnx_parity_predictions.csv`
- `outputs/icaa_export_safe_v3_onnx_20260515_171049/onnx_export_log.txt`

Result: passed.

- ONNX checker: passed
- GridSample count: `0`
- GlobalAveragePool count: `0`
- full tensor max abs diff: `2.0325183868408203e-05`
- full tensor mean abs diff: `1.4621764421463013e-06`
- MOS max / mean diff: `2.0325183868408203e-05` / `1.8076971173286438e-06`
- color max / mean diff: `1.33514404296875e-05` / `1.1166557669639587e-06`

ONNX export warnings:

- `TracerWarning: torch.as_tensor results are registered as constants in the trace...` count `140`
- `TracerWarning: Converting a tensor to a Python float might cause the trace to be incorrect...` count `80`

## ONNX Operator Histogram

```json
{
  "Abs": 1,
  "Add": 528,
  "And": 200,
  "Cast": 382,
  "Clip": 80,
  "Concat": 542,
  "Constant": 4262,
  "ConstantOfShape": 160,
  "Conv": 86,
  "Div": 189,
  "Einsum": 48,
  "Equal": 20,
  "Erf": 34,
  "Expand": 40,
  "Floor": 40,
  "Gather": 849,
  "Gemm": 3,
  "GreaterOrEqual": 80,
  "LayerNormalization": 63,
  "LessOrEqual": 110,
  "MatMul": 76,
  "Mul": 975,
  "Range": 40,
  "Reciprocal": 20,
  "ReduceMean": 1,
  "Relu": 1,
  "Reshape": 589,
  "Shape": 929,
  "Sigmoid": 12,
  "Slice": 178,
  "Softmax": 24,
  "Squeeze": 1,
  "Sub": 170,
  "Tanh": 20,
  "Tile": 20,
  "Transpose": 302,
  "Unsqueeze": 1449,
  "Where": 130
}
```

## TFLite Conversion and Probe

Command:

```bash
python experiments/icaa_export_safe_v3/scripts/convert_and_probe_tflite_v3.py
```

Internal onnx2tf command:

```bash
/home/omen_pc1/photo_score_project/.venv_gpu/bin/onnx2tf -i /home/omen_pc1/photo_score_project/outputs/icaa_export_safe_v3_onnx_20260515_171049/icaa_dat_export_safe_v3.onnx -o /home/omen_pc1/photo_score_project/outputs/icaa_export_safe_v3_tflite_probe_20260515_171113/onnx2tf_no_flex -tb flatbuffer_direct -n
```

Artifacts:

- `outputs/icaa_export_safe_v3_tflite_probe_20260515_171113/onnx2tf_no_flex/icaa_dat_export_safe_v3_float32.tflite`
- `outputs/icaa_export_safe_v3_tflite_probe_20260515_171113/tflite_probe_report.json`
- `outputs/icaa_export_safe_v3_tflite_probe_20260515_171113/tflite_probe_log.txt`
- `outputs/icaa_export_safe_v3_tflite_probe_20260515_171113/onnx2tf_stdout.txt`
- `outputs/icaa_export_safe_v3_tflite_probe_20260515_171113/onnx2tf_stderr.txt`

Conversion result:

- onnx2tf return code: `0`
- FP32 TFLite load: succeeded
- allocate_tensors: succeeded
- input shape: `[1, 224, 224, 3]`
- input dtype: `float32`
- output shape: `[1, 2]`
- output dtype: `float32`

Note: this local `onnx2tf` flatbuffer-direct path automatically emitted a float16 sidecar. The v3 probe did not load, evaluate, or use that sidecar.

## TFLite Sensitivity Results

Real-image probe source:

- `data/raw/icaa17k/complementary/orange_blue/juiedesai19391601268.jpg`

FP32 TFLite outputs:

- zero input: `[[0.5152620077133179, 0.5379891395568848]]`
- one input: `[[0.5152620077133179, 0.5379891395568848]]`
- random input: `[[0.5152620077133179, 0.5379891395568848]]`
- real image input: `[[0.5152620077133179, 0.5379891395568848]]`

Diffs vs zero:

- one max / mean diff: `0.0` / `0.0`
- random max / mean diff: `0.0` / `0.0`
- real image max / mean diff: `0.0` / `0.0`
- max non-zero-input diff vs zero: `0.0`

Result: failed. The generated FP32 TFLite model is still constant-output.

## Decision

v3 is PyTorch-equivalent to v2 and exports to ONNX without `GridSample` or `GlobalAveragePool`, but the no-Flex FP32 TFLite output is still not input-sensitive.

It is not safe to proceed to TFLite real-image parity yet.

No deployment success is claimed.
