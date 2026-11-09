# ICAA17K ONNX Parity Debug Report

## Identified Risky Operations and Fixes

### 1. `DAttentionBaseline.forward`: `offset_range` creation
- **Issue:** `torch.tensor([1.0 / Hk, 1.0 / Wk], ...)` was used in the forward pass. `Hk` and `Wk` were Python ints derived from `offset.size()`. This registered the result as a constant in the trace and produced TracerWarnings.
- **Fix:** Replaced with `torch.stack([1.0 / torch.as_tensor(Hk, ...), 1.0 / torch.as_tensor(Wk, ...)])`. This ensures the operation is tracked by the tracer and avoids data flow breaks.

### 2. `DAttentionBaseline._get_ref_points`: `torch.linspace` and Coordinate Normalization
- **Issue:** Used `torch.linspace` with float arguments and in-place operations (`div_`, `mul_`, `sub_`) on slices of the reference grid. These can be problematic for ONNX export and caused "converting a tensor to float" warnings.
- **Fix:** 
  - Replaced `torch.linspace` with `torch.arange` followed by explicit division by `torch.as_tensor(H_key, ...)`.
  - Replaced in-place slice operations with out-of-place tensor operations.
  - Specified `indexing='ij'` for `torch.meshgrid` to avoid warnings and ensure consistent behavior across PyTorch versions.

### 3. `DAttentionBaseline.forward`: Grid Construction for `grid_sample`
- **Issue:** Used `pos[..., (1, 0)]` for coordinate swapping (y, x -> x, y).
- **Fix:** Replaced with explicit `torch.chunk` and `torch.cat` to be more robust for various ONNX opsets.

### 4. `DAttentionBaseline._get_ref_points`: Dynamic Batch Size
- **Issue:** `int(B)` was used in `expand`, which forced a constant batch size in the trace.
- **Fix:** Removed `int()` cast to allow dynamic batch size in TorchScript (and potentially ONNX if dynamic axes are enabled).

## Verification Results

### PyTorch Parity (Original vs Export-Safe)
- **Command:** `python experiments/icaa_export_safe/scripts/parity_original_vs_export_safe.py --limit 256 --batch_size 1 --device cpu`
- **Result:** Max Abs Diff = **0.0000000000**
- **Status:** PASS

### TorchScript Parity (Export-Safe PyTorch vs TorchScript)
- **Command:** `python experiments/icaa_export_safe/scripts/export_and_parity_torchscript_export_safe.py --limit 32 --batch_size 8 --device cpu`
- **Result:** Max Abs Diff = **0.0000000000**
- **Status:** PASS (Dynamic batch size confirmed)

### ONNX Parity (Export-Safe PyTorch vs ONNX Runtime)
- **Command:** `python experiments/icaa_export_safe/scripts/export_and_parity_onnx_export_safe.py --limit 32 --device cpu`
- **Result (Opset 17):**
  - Full tensor max_abs_diff: **0.0000201464**
  - Full tensor mean_abs_diff: **0.0000012964**
  - MOS max_abs_diff: **0.0000201464**
  - Color max_abs_diff: **0.0000115633**
- **Status:** **PASS** (Thresholds: max 1e-4, mean 1e-5)

## Conclusion
The ONNX parity issue was successfully resolved by eliminating all data-flow-breaking Python operations in the `forward` pass of `DAttentionBaseline`. The model now exports to ONNX with high numerical fidelity.

**Recommended Next Step:**
- The model is now ready for **TFLite conversion** via the ONNX path.
