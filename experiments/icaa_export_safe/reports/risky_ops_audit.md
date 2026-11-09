# ICAA17K Export-Safe Audit Report

## Identified Risky Operations

### 1. NumPy Conversions in `dat_blocks.py`
In `DAttentionBaseline.forward`, several tensors are converted to NumPy arrays, processed, and converted back to PyTorch.
- **Location:** `external/icaa_official_repo/ICAA17K_code/models/dat_blocks.py` (Lines 255-268)
- **Code:**
  ```python
  offset_x = torch.chunk(offset, 2, dim=3)[0].detach().cpu().numpy()
  offset_y = torch.chunk(offset, 2, dim=3)[1].detach().cpu().numpy()
  reference_x = torch.chunk(reference, 2, dim=3)[0].detach().cpu().numpy()
  reference_y = torch.chunk(reference, 2, dim=3)[1].detach().cpu().numpy()

  temp_x = offset_x * reference_x
  temp_y = offset_y * reference_y

  temp_x = np.where(temp_x <= 0.00, 0, 1)
  temp_y = np.where(temp_y <= 0.00, 0, 1)
  offset_temp = temp_x * temp_y
  manu_offset = np.where(offset_temp <= 0.00, 0.25, 1)

  manu_offset_ = torch.cat((torch.from_numpy(manu_offset), torch.from_numpy(manu_offset)), 3).float().to(device)
  ```
- **Impact:**
  - Breaks TorchScript tracing (registers values as constants).
  - Breaks gradient flow (though not critical for inference-only export).
  - Likely cause of real-image parity failure.

### 2. Batch Averaging in `dat_blocks.py`
The code averages reference points and gates across the batch dimension.
- **Location:** `external/icaa_official_repo/ICAA17K_code/models/dat_blocks.py` (Lines 245-251)
- **Code:**
  ```python
  referencek = self.ref_point14(x)
  referencek = torch.mean(referencek, dim=0).unsqueeze(dim=0)
  # ...
  gate = self.ref_gate(x)
  gate = torch.mean(gate, dim=0).unsqueeze(dim=0)
  ```
- **Impact:**
  - Makes individual predictions dependent on other images in the same batch.
  - Causes shape mismatches when `batch_size > 1` if the tracing was done with `batch_size=1`.
  - Explains why `batch_size > 1` evaluation failed with `RuntimeError`.

### 3. Missing Device/DType in Tensor Creation
- **Location:** `external/icaa_official_repo/ICAA17K_code/models/SoftHistogram.py` (Line 41)
- **Code:** `+ torch.tensor(10e-6)`
- **Location:** `external/icaa_official_repo/ICAA17K_code/models/dat_blocks.py` (Line 112)
- **Code:** `.masked_fill(..., float(-100.0))` (using Python float)

## Plan for Export-Safe Model
1. Copy model files to `experiments/icaa_export_safe/models/`.
2. Replace NumPy `np.where` with `torch.where`.
3. Replace `.cpu().numpy()` and `torch.from_numpy` with direct PyTorch operations.
4. Investigate if `torch.mean(..., dim=0)` can be removed or if it's essential for the paper's logic (if essential, we must document that `batch_size=1` is required or handle it per-sample).
5. Ensure all tensor creations use the appropriate `device` and `dtype`.
