# ICAA17K DAT Native TensorFlow Reimplementation Plan

## 1. Confirmed Facts
- The official PyTorch DAT model reproduces the paper's metrics (Accuracy@0.5: ~0.9635, SRCC: ~0.8806, PLCC: ~0.8981) on the 1test.csv dataset.
- The model configuration uses a mix of Local ('L'), Shifted Window ('S'), and Deformable/Delegate ('D') attention blocks: `stage_spec=[["L","S"], ["L","S"], ["L","D"] * 9, ["L","D"]]`.
- Direct TFLite conversion from ONNX failed because the original and export-safe PyTorch models rely on `F.grid_sample` (or manual equivalents that hit shape constraints) which translate to unsupported or problematic custom operations in the `ai_edge_litert` runtime.
- The goal is an exact native TensorFlow/Keras reproduction, mapped directly from the official PyTorch checkpoint, to bypass the intermediate ONNX serialization issues.

## 2. Open Questions
- Can the complex `DAttentionBaseline` and its manual grid sampling be efficiently implemented in standard TensorFlow operations without hitting dynamic shape constraints during TFLite conversion?
- Will `tf.gather` / `tf.gather_nd` behave equivalently to PyTorch's `index_select` under TFLite XNNPACK?
- How should grouped 1D convolutions in `SoftHistogram` be mapped to Keras `Conv1D` effectively while preserving exact parameter semantics?

## 3. Module Inventory
| Module Name | Purpose |
|---|---|
| **Patch Projection** | Initial overlapping convolution to tokenize the image. |
| **TransformerStage** | Groups attention and MLP blocks, handles downsampling. |
| **LocalAttention (L)** | Standard window-based self-attention. |
| **ShiftWindowAttention (S)** | Window-based attention with spatial shifting (`torch.roll`). |
| **DAttentionBaseline (D)** | Computes offsets, constructs grids, and samples keys/values. |
| **TransformerMLP** | Standard two-layer feed-forward network. |
| **TransformerMLPWithConv** | MLP containing an intermediate 3x3 Depthwise Convolution. |
| **LayerNormProxy** | Layer normalization applied over the channel dimension. |
| **SoftHistogram** | Computes color aesthetic histograms using 1D convolutions. |
| **DAT** | Top-level model orchestrating stages and the final classification head. |

## 4. PyTorch to TensorFlow Weight Mapping Table
When migrating weights from PyTorch to TensorFlow, tensor transpositions are required due to different layout and convolution kernel standards.

| PyTorch Parameter | Keras Equivalent | Transposition Rule (`np.transpose`) |
|---|---|---|
| `nn.Conv2d.weight` | `Conv2D.kernel` | `(out_c, in_c, H, W) -> (H, W, in_c, out_c)` (rule: `(2, 3, 1, 0)`) |
| `nn.Conv2d.bias` | `Conv2D.bias` | Direct Copy |
| `nn.Linear.weight` | `Dense.kernel` | `(out_c, in_c) -> (in_c, out_c)` (rule: `(1, 0)`) |
| `nn.Linear.bias` | `Dense.bias` | Direct Copy |
| `nn.LayerNorm.weight` | `LayerNormalization.gamma` | Direct Copy |
| `nn.LayerNorm.bias` | `LayerNormalization.beta` | Direct Copy |
| `SoftHistogram` Conv1D weight | `Conv1D.kernel` | `(out_c, in_c, K) -> (K, in_c, out_c)` (rule: `(2, 1, 0)`) |

## 5. Layout Decision (NHWC vs NCHW)
**Decision**: Implement the TensorFlow model natively in **NHWC (channels_last)**.
**Evidence**: 
- TFLite is highly optimized for NHWC layouts. 
- If the model is built in NCHW internally within TF, the TFLite converter inserts `TRANSPOSE` operations before and after nearly every convolution and normalization layer, heavily degrading mobile performance and breaking operator fusions. 
- Therefore, all PyTorch `einops` patterns and convolution configurations must be translated to assume NHWC tensors.

## 6. High-Risk Modules
1. **DAttentionBaseline**: Reimplementing the manual bilinear sampler in pure TF requires careful mapping of `torch.index_select` to `tf.gather` and avoiding dimension broadcasting ambiguities.
2. **SoftHistogram**: The `grouped=self.num_bins * self.in_channels` property of PyTorch `Conv1d` needs precise mapping to Keras `Conv1D` or `DepthwiseConv1D`.
3. **ShiftWindowAttention**: PyTorch's `torch.roll` and corresponding attention masking logic must be strictly verified against TF's `tf.roll`.

## 7. Staged Implementation Plan
* **Stage A**: Build testing skeleton and PyTorch checkpoint inspector (`inspect_pytorch_state_dict.py`, `build_tf_dat_skeleton.py`).
* **Stage B**: Implement base layers and verification scripts (`LayerNormProxy`, `TransformerMLP`, `SoftHistogram`) (`parity_module_*.py`).
* **Stage C**: Implement and test `Patch Projection` and downsampling blocks.
* **Stage D**: Implement `LocalAttention` and `ShiftWindowAttention` with NHWC logic.
* **Stage E**: Implement `DAttentionBaseline` including TF manual bilinear sampling.
* **Stage F**: Implement `TransformerStage` and full `DAT` model assembly.
* **Stage G**: Execute full TF vs PyTorch parity test (`parity_full_tf_vs_pytorch.py`).
* **Stage H**: Export to TensorFlow `SavedModel` (`export_tf_savedmodel.py`).
* **Stage I**: Convert to TFLite FP32 (`convert_tf_to_tflite.py`).
* **Stage J**: Verify TFLite Runtime parity (`parity_tflite_tf.py`).

## 8. Exact Parity Thresholds
- **Module-level Parity**: `max_abs_diff <= 1e-5`
- **Block-level Parity**: `max_abs_diff <= 1e-5`
- **Full Model (TF vs PyTorch)**: `max_abs_diff <= 1e-4`, `mean_abs_diff <= 1e-5`
- **TFLite vs TF**: `max_abs_diff <= 1e-4`, `mean_abs_diff <= 1e-5`

## 9. Estimated Effort and Risks
- **Effort**: Medium-High. Manual translation of `einops` expressions to NHWC format requires meticulous attention to detail.
- **Risks**: The manual bilinear sampler might still produce a TFLite graph that the runtime rejects if shapes are not perfectly inferred by the converter. If TFLite rejects the `tf.gather` operations, alternative simplifications (e.g., standard spatial attention) might be necessary.

## 10. Exact Next Implementation Step
The immediate next step is **Stage A**:
Create `inspect_pytorch_state_dict.py` to extract and document the exact shape of every parameter in the official PyTorch checkpoint, which will serve as the source of truth for the `map_pytorch_weights_to_tf.py` script.
