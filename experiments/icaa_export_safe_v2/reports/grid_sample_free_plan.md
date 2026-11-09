# ICAA17K DAT GridSample-Free Implementation Plan

## 1. Context and Problem
The current export-safe implementation (`v1`) successfully exports to ONNX but fails during TFLite conversion and runtime testing. The primary cause is the `F.grid_sample` operator, which translates to an unsupported `ONNX_GRIDSAMPLE` custom op in TFLite and triggers `FILL` dimension errors in the `ai_edge_litert` runtime. To achieve a pure TFLite export without relying on the Flex delegate (`SELECT_TF_OPS`), `F.grid_sample` must be replaced with a TFLite-friendly manual implementation.

## 2. Where `F.grid_sample` is Used
In `experiments/icaa_export_safe/export_safe_models/dat_blocks.py`, `F.grid_sample` is called twice within `DAttentionBaseline.forward`:

### Call 1: Spatial Feature Sampling
```python
x_sampled = F.grid_sample(
    input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
    grid=grid,
    mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
```
- **Input Tensor Shape:** `[N, C, H_in, W_in]` where `N = B * self.n_groups`, `H_in = H`, `W_in = W`.
- **Grid Tensor Shape:** `[N, H_out, W_out, 2]` where `H_out = Hk`, `W_out = Wk`.

### Call 2: Relative Position Bias Sampling
```python
attn_bias = F.grid_sample(
    input=rpe_bias.reshape(B * self.n_groups, self.n_group_heads, 2 * H - 1, 2 * W - 1),
    grid=disp_grid,
    mode='bilinear', align_corners=True
)  # B * g, h_g, HW, Ns
```
- **Input Tensor Shape:** `[N, C, H_in, W_in]` where `N = B * self.n_groups`, `H_in = 2*H - 1`, `W_in = 2*W - 1`.
- **Grid Tensor Shape:** `[N, H_out, W_out, 2]` where `H_out = H*W`, `W_out = N_sample`.

## 3. Exact Behavior to Replicate
Both calls share identical configurations:
- **`mode='bilinear'`:** Requires standard 2D bilinear interpolation using 4 corner pixels.
- **`align_corners=True`:** The mapping from normalized coordinates `[-1, 1]` to pixel indices `[0, size - 1]` follows the formula: `index = (coord + 1) / 2 * (size - 1)`.
- **`padding_mode='zeros'`:** (PyTorch default) Any coordinate mapping outside the valid pixel range must yield exactly `0.0`.

## 4. Proposed Manual Bilinear Sampling Formula
To ensure maximum TFLite compatibility, the replacement uses 1D indexing (`torch.index_select`) on a flattened input, avoiding complex N-dimensional gathers.

```python
import torch

def manual_grid_sample(input, grid):
    """
    Pure PyTorch replacement for F.grid_sample(..., mode='bilinear', align_corners=True, padding_mode='zeros').
    """
    N, C, H_in, W_in = input.shape
    _, H_out, W_out, _ = grid.shape
    
    x = grid[..., 0]
    y = grid[..., 1]
    
    W_in_f = torch.as_tensor(W_in, dtype=x.dtype, device=x.device)
    H_in_f = torch.as_tensor(H_in, dtype=y.dtype, device=y.device)

    # align_corners=True mapping
    ix = (x + 1.0) / 2.0 * (W_in_f - 1.0)
    iy = (y + 1.0) / 2.0 * (H_in_f - 1.0)

    # Corner coordinates
    ix_nw = torch.floor(ix)
    iy_nw = torch.floor(iy)
    ix_ne = ix_nw + 1.0
    iy_ne = iy_nw
    ix_sw = ix_nw
    iy_sw = iy_nw + 1.0
    ix_se = ix_nw + 1.0
    iy_se = iy_nw + 1.0

    # Weights
    w_e = ix - ix_nw
    w_w = 1.0 - w_e
    w_s = iy - iy_nw
    w_n = 1.0 - w_s

    w_nw = w_n * w_w
    w_ne = w_n * w_e
    w_sw = w_s * w_w
    w_se = w_s * w_e

    # Out-of-bounds masks (padding_mode='zeros')
    valid_nw = (ix_nw >= 0.0) & (ix_nw < W_in_f) & (iy_nw >= 0.0) & (iy_nw < H_in_f)
    valid_ne = (ix_ne >= 0.0) & (ix_ne < W_in_f) & (iy_ne >= 0.0) & (iy_ne < H_in_f)
    valid_sw = (ix_sw >= 0.0) & (ix_sw < W_in_f) & (iy_sw >= 0.0) & (iy_sw < H_in_f)
    valid_se = (ix_se >= 0.0) & (ix_se < W_in_f) & (iy_se >= 0.0) & (iy_se < H_in_f)

    # Apply masks to weights
    w_nw = torch.where(valid_nw, w_nw, torch.zeros_like(w_nw)).unsqueeze(-1)
    w_ne = torch.where(valid_ne, w_ne, torch.zeros_like(w_ne)).unsqueeze(-1)
    w_sw = torch.where(valid_sw, w_sw, torch.zeros_like(w_sw)).unsqueeze(-1)
    w_se = torch.where(valid_se, w_se, torch.zeros_like(w_se)).unsqueeze(-1)

    # Clamp indices to avoid out-of-bounds gather (weights will zero out invalid values anyway)
    ix_nw_c = torch.clamp(ix_nw, 0.0, W_in_f - 1.0).to(torch.int64)
    iy_nw_c = torch.clamp(iy_nw, 0.0, H_in_f - 1.0).to(torch.int64)
    ix_ne_c = torch.clamp(ix_ne, 0.0, W_in_f - 1.0).to(torch.int64)
    iy_ne_c = torch.clamp(iy_ne, 0.0, H_in_f - 1.0).to(torch.int64)
    ix_sw_c = torch.clamp(ix_sw, 0.0, W_in_f - 1.0).to(torch.int64)
    iy_sw_c = torch.clamp(iy_sw, 0.0, H_in_f - 1.0).to(torch.int64)
    ix_se_c = torch.clamp(ix_se, 0.0, W_in_f - 1.0).to(torch.int64)
    iy_se_c = torch.clamp(iy_se, 0.0, H_in_f - 1.0).to(torch.int64)

    # Flatten logic
    N_t = torch.as_tensor(N, dtype=torch.int64, device=input.device)
    W_in_t = torch.as_tensor(W_in, dtype=torch.int64, device=input.device)
    H_in_t = torch.as_tensor(H_in, dtype=torch.int64, device=input.device)
    
    b_idx = torch.arange(N_t, device=input.device).view(N, 1, 1)
    b_offset = b_idx * (H_in_t * W_in_t)

    idx_nw = (b_offset + iy_nw_c * W_in_t + ix_nw_c).view(-1)
    idx_ne = (b_offset + iy_ne_c * W_in_t + ix_ne_c).view(-1)
    idx_sw = (b_offset + iy_sw_c * W_in_t + ix_sw_c).view(-1)
    idx_se = (b_offset + iy_se_c * W_in_t + ix_se_c).view(-1)

    input_perm = input.permute(0, 2, 3, 1).reshape(-1, C)

    val_nw = torch.index_select(input_perm, 0, idx_nw).view(N, H_out, W_out, C)
    val_ne = torch.index_select(input_perm, 0, idx_ne).view(N, H_out, W_out, C)
    val_sw = torch.index_select(input_perm, 0, idx_sw).view(N, H_out, W_out, C)
    val_se = torch.index_select(input_perm, 0, idx_se).view(N, H_out, W_out, C)

    output = val_nw * w_nw + val_ne * w_ne + val_sw * w_sw + val_se * w_se
    return output.permute(0, 3, 1, 2)
```

## 5. Expected ONNX Ops and TFLite Compatibility
**ONNX Ops:**
- `Floor`, `Clamp`, `Cast`
- `Add`, `Sub`, `Mul`, `Div`
- `Where`, `GreaterOrEqual`, `Less`
- `Gather` (from `index_select`)
- `Reshape`, `Transpose`

**TFLite Compatibility:**
- **EXCELLENT.** Every single PyTorch/ONNX operation used above maps directly to standard TFLite nodes (`GATHER`, `MUL`, `ADD`, `LESS`, `SELECT`, `RESHAPE`, `TRANSPOSE`).
- The `FILL` runtime error and `ONNX_GRIDSAMPLE` custom ops will be completely eliminated.

## 6. Known Risks
- **Index Arithmetic Overflow:** The flattened index calculation `(b_idx * H_in * W_in) + ...` could theoretically overflow `int32`. However, for standard sizes like `B=16`, `H=224`, `W=224`, the maximum index is well within `int32` bounds (approx. 2 billion). `int64` is used explicitly to guarantee safety during tracing.
- **Trace Complexity:** The model graph node count will increase since `grid_sample` is replaced by many basic arithmetic nodes. This is standard for deployment without custom delegates and is perfectly handled by `onnx2tf`.

## 7. Recommended Implementation Steps and Parity Test Plan
1. **Setup:** Create `experiments/icaa_export_safe_v2/models/` and copy the `export_safe_models` (`v1`) into it. Update local imports.
2. **Implement:** Inject the `manual_grid_sample` function into `dat_blocks.py` and replace both `F.grid_sample` calls.
3. **PyTorch Parity:** Run a 1-to-1 PyTorch evaluation script to compare the official checkpoint against the `v2` implementation at `batch_size=1`. Max absolute difference must be `0.0` or extremely close to it.
4. **ONNX Export:** Export to ONNX (`opset 17`) and verify ORT parity.
5. **TFLite Validation:** Use `onnx2tf` to generate a TFLite file, then verify with `tf.lite.Interpreter` that it loads without Flex ops and completes a dummy inference successfully.
