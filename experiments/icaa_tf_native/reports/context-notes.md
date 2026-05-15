# Stage A Context Notes

- Scope is inspection only for native TensorFlow/Keras DAT preparation.
- Official repo files, A-cut source files, Flutter files, training, TensorFlow implementation, and TFLite conversion are out of scope.
- The export-safe v3 DAT package is used as a read-only PyTorch model reference for strict checkpoint loading and dummy output shape.
- Unknown or high-risk mappings must be marked explicitly instead of inferred beyond available evidence.
- The official checkpoint loaded as an `OrderedDict` state_dict with 487 tensors and 87,138,907 total state_dict tensor elements.
- The export-safe v3 DAT model strict-loaded the checkpoint with no missing or unexpected keys; the primary dummy output shape for `[1, 3, 224, 224]` is `[1, 2]`.
- Depthwise Conv2D, SoftHistogram grouped Conv1D, and positional/buffer tensors are intentionally marked as uncertain or manual-inspection categories for Stage B.

## Stage B Context Notes

- Stage B validates only basic layer mapping rules using checkpoint tensors. It does not implement DAT, train, or convert to TFLite.
- TensorFlow/Keras parity is checked in NHWC where appropriate, then converted back to PyTorch layout for comparison.
- LayerNorm parity must respect `LayerNormProxy`: PyTorch receives NCHW, transposes to NHWC, applies channel-only LayerNorm, and transposes back.
- Final Stage B run output is `outputs/icaa_tf_native_stage_b_20260515_174854/`.
- Conv2D, Dense, channel LayerNorm, GELU, and isolated MLP parity all passed the preferred `max_abs_diff <= 1e-5` threshold.
- GELU should use TensorFlow exact mode for DAT parity because PyTorch DAT uses default `nn.GELU()` semantics; TensorFlow tanh approximation was not close enough to PyTorch default.
- It is safe to proceed to Stage C patch projection parity only; this does not establish full TensorFlow DAT feasibility.

## Stage C Context Notes

- Stage C implements only the patch projection block: Conv2D plus channel LayerNormalization in NHWC.
- The PyTorch reference is strictly `export_safe_v3 DAT.model.patch_proj`; no later DAT stages are invoked.
- The expected random-input output shape is PyTorch NCHW `[1, 128, 56, 56]` and TensorFlow NHWC `[1, 56, 56, 128]` before transposing for comparison.
- Final Stage C run output is `outputs/icaa_tf_native_stage_c_20260515_175817/`.
- Random input parity passed the preferred threshold with max_abs_diff `3.81469727e-06` and mean_abs_diff `8.1713118e-08`.
- Real ICAA17K image parity on three local test images passed the preferred threshold with max_abs_diff `1.90734863e-06` and mean_abs_diff `4.95252301e-08`.
- It is safe to proceed to Stage D SoftHistogram parity only; this does not establish full TensorFlow DAT feasibility.

## Stage D Context Notes

- PyTorch `SoftHistogram` is instantiated in DAT as `SoftHistogram(n_features=36, n_examples=6, num_bins=6, quantiles=False)`.
- The forward path expects input shaped `[batch, 36]`, transposes to `[36, batch]`, unsqueezes to Conv1d layout `[1, 36, batch]`, and returns `[1, 216, batch]` for `quantiles=False`.
- The module uses grouped Conv1d twice: feature-to-bin expansion with groups `36`, then per-bin width scaling with groups `216`.
- `hist_feature.centers` is a registered PyTorch parameter but is not referenced by `forward`; `hist_feature.widths` aliases `bin_widths_conv.weight`.
- Final Stage D run output is `outputs/icaa_tf_native_stage_d_20260515_180922/`.
- DAT hook inspection confirmed SoftHistogram input shape `[1, 36]` and output shape `[1, 216, 1]` for a dummy image and one real ICAA17K image.
- Random, zeros, ones, and captured real pre-SoftHistogram parity all passed the preferred `max_abs_diff <= 1e-5` threshold.
- The largest Stage D max_abs_diff was `5.96046448e-08`.
- It is safe to proceed to Stage E local attention / block parity only; this does not establish full TensorFlow DAT feasibility.

## Stage E Context Notes

- The Stage E PyTorch target is `TransformerStage` stage `0`, depth index `0`, not the full stage and not full DAT.
- Stage `0` block `0` uses `LayerNormProxy`, `LocalAttention`, residual add, `LayerNormProxy`, `TransformerMLP`, residual add.
- Stage `0` block `0` has attention spec `"L"`; no deformable/delegate attention is included in Stage E.
- The expected block input/output shape is PyTorch NCHW `[batch, 128, 56, 56]`; the TensorFlow implementation uses NHWC `[batch, 56, 56, 128]`.
- The first drop path rate is `0.0`, so `stages.0.drop_path.0` is `Identity`; eval mode has no dropout randomness.
- Final Stage E run output is `outputs/icaa_tf_native_stage_e_20260515_181850/`.
- Stage E random block input parity passed the preferred threshold with max_abs_diff `9.53674316e-06` and mean_abs_diff `6.46088381e-07`.
- Stage E deterministic random-image patch-projection parity passed the preferred threshold with max_abs_diff `6.19888306e-06` and mean_abs_diff `5.73447096e-07`.
- Stage E three-real-image patch-projection parity passed the preferred threshold with max_abs_diff `6.19888306e-06` and mean_abs_diff `4.23493987e-07`.
- Subcomponent debug diffs for norm0, attention, residuals, norm1, MLP, and output all stayed within the preferred threshold.
- It is safe to proceed to Stage F deformable/delegate attention parity only; this does not establish full TensorFlow DAT feasibility.

## Stage F Context Notes

- The first deformable/delegate attention module is `stages.2.attns.1`, class `DAttentionBaseline`.
- It is in DAT stage index `2`, attention/block index `1`, corresponding to the first `"D"` entry in `stage_spec[2]`.
- The module input/output shape is expected to be PyTorch NCHW `[batch, 512, 14, 14]`; TensorFlow uses NHWC `[batch, 14, 14, 512]`.
- The offset branch is depthwise Conv2D `[128,1,5,5]`, channel LayerNorm, exact GELU, and 1x1 Conv2D to two offset channels.
- The module uses `use_pe=True`, `dwc_pe=False`, `fixed_pe=False`, and an `rpe_table` with shape `[16, 27, 27]`.
- TensorFlow `DAttentionBaseline` uses explicit `PytorchLikeConv1x1` matrix-multiply projections for the 1x1 PyTorch Conv2d layers to reduce backend accumulation drift, especially around `ref_gate`.
- The TensorFlow manual bilinear sampler matches the export-safe PyTorch sampler with `align_corners=True`, `padding_mode=zeros`, zeroed invalid-corner weights, and clamped gather indices.
- Stage F deterministic random module input uses `np.random.randn(...)*0.5` to keep the synthetic tensor near the observed captured DAttention tensor scale while remaining deterministic.
- Stage F deterministic random-image input uses uniform RGB `[0,1)` followed by the same ImageNet mean/std preprocessing used for real ICAA17K image checks.
- Final Stage F run output is `outputs/icaa_tf_native_stage_f_20260515_183945/`.
- Stage F random module input parity passed the preferred threshold with max_abs_diff `8.16583633e-06` and mean_abs_diff `8.04580907e-07`.
- Stage F captured deterministic random-image input parity passed the preferred threshold with max_abs_diff `4.66406345e-06` and mean_abs_diff `7.68889038e-07`.
- Stage F captured three-real-image input parity passed the preferred threshold with max_abs_diff `8.10623169e-06` and mean_abs_diff `4.83207145e-07`.
- It is safe to proceed to Stage G full `TransformerStage` parity only; this does not establish full TensorFlow DAT feasibility.

## Stage G-0 Context Notes

- Scope is only DAT `TransformerStage` stage `0`; full DAT, later stages, training, and TFLite conversion remain out of scope.
- Stage `0` uses `TransformerStage` with depth `2` and `stage_spec=["L", "S"]`.
- Stage `0` has `proj=Identity` because `dim_in == dim_embed == 128`; DAT down projection is applied after `model.stages[0](x)` in `DAT.forward`, so it is not part of Stage G-0.
- The expected stage input/output shape is PyTorch NCHW `[batch, 128, 56, 56]`; TensorFlow should use NHWC `[batch, 56, 56, 128]`.
- Stage G-0 must reuse the verified Stage E local attention and MLP mapping, then add shifted-window attention semantics for block `1`.
- Drop path must be called in the same residual order, but eval mode makes both stage-0 `DropPath` modules deterministic identity.
- Final Stage G-0 run output is `outputs/icaa_tf_native_stage_g0_20260516_014248/`.
- Stage G-0 deterministic random stage input parity passed the acceptable threshold with max_abs_diff `1.04904175e-05` and mean_abs_diff `7.89989087e-07`.
- Stage G-0 deterministic random-image patch-projection parity passed the preferred threshold with max_abs_diff `6.19888306e-06` and mean_abs_diff `7.25404448e-07`.
- Stage G-0 three-real-image patch-projection parity passed the acceptable threshold with max_abs_diff `1.14887953e-05` and mean_abs_diff `6.86473015e-07`.
- The worst recorded subcomponent diffs were `block1_output` for random and real-image cases and `block1_after_attn_residual` for the random-image patch-projection case; all remained below the acceptable threshold.
- It is safe to proceed to Stage G-1 `TransformerStage` stage `1` parity only; this does not establish full TensorFlow DAT feasibility.

## Stage G-1 Context Notes

- Scope is only DAT `TransformerStage` stage `1`; full DAT, stages `2`/`3`, training, and TFLite conversion remain out of scope.
- Stage `1` uses `TransformerStage` with depth `2` and `stage_spec=["L", "S"]`.
- Stage `1` has `proj=Identity` because `dim_in == dim_embed == 256`; DAT down projection is applied after `model.stages[1](x)` in `DAT.forward`, so it is not part of Stage G-1.
- The expected stage input/output shape is PyTorch NCHW `[batch, 256, 28, 28]`; TensorFlow should use NHWC `[batch, 28, 28, 256]`.
- Stage `1` block `0` uses `LocalAttention` and block `1` uses `ShiftWindowAttention` with `attn_mask` shape `[16, 49, 49]`.
- Both stage-1 drop path modules are `DropPath` with nonzero configured probabilities, but eval mode makes them deterministic identity.
- Final Stage G-1 run output is `outputs/icaa_tf_native_stage_g1_20260516_022555/`.
- Stage G-1 deterministic random stage input parity passed the preferred threshold with max_abs_diff `7.62939453e-06` and mean_abs_diff `1.18373578e-06`.
- Stage G-1 captured deterministic random-image input parity passed the preferred threshold with max_abs_diff `3.81469727e-06` and mean_abs_diff `5.34230878e-07`.
- Stage G-1 captured three-real-image input parity passed the preferred threshold with max_abs_diff `5.24520874e-06` and mean_abs_diff `6.35120614e-07`.
- The worst recorded subcomponent diff for all Stage G-1 cases was `block1_output`; all remained below the preferred threshold.
- It is safe to proceed to Stage G-2 `TransformerStage` stage `2` parity only; this does not establish full TensorFlow DAT feasibility.
