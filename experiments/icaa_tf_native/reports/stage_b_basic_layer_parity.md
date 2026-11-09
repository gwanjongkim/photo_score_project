# Stage B Basic Layer Parity

## Scope
- Exact command used: `python experiments/icaa_tf_native/scripts/stage_b_basic_layer_parity.py`
- Checkpoint path: `/home/omen_pc1/photo_score_project/weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth`
- Output directory: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_stage_b_20260515_174854`
- Status: `ok`
- Seeds: torch=123, numpy=123, tensorflow=123

## Checkpoint Tensors Used
- Conv2D: `patch_proj.0.weight`, `patch_proj.0.bias`
- Dense: `hst_head.weight`, `hst_head.bias`
- LayerNorm: `stages.0.layer_norms.0.norm.weight`, `stages.0.layer_norms.0.norm.bias`
- GELU: parameter-free function check
- MLP: `stages.0.mlps.0.chunk.linear1.weight`, `stages.0.mlps.0.chunk.linear1.bias`, `stages.0.mlps.0.chunk.linear2.weight`, `stages.0.mlps.0.chunk.linear2.bias`

## Mapping Rules Verified
- Conv2D: PyTorch `[out_c, in_c, kH, kW]` -> TensorFlow `[kH, kW, in_c, out_c]`; input NCHW -> NHWC and output back to NCHW.
- Dense: PyTorch `[out_features, in_features]` -> TensorFlow `[in_features, out_features]`; bias direct copy.
- LayerNorm: `LayerNormProxy` channel-only normalization maps to TensorFlow `LayerNormalization(axis=-1, epsilon=1e-5)` on NHWC tensors.
- GELU: selected `exact` mode by direct PyTorch-vs-TensorFlow comparison.
- MLP: isolated `linear1 -> GELU -> linear2` mapping with dropout omitted because parity is evaluated in inference form.

## Results
| Test | Status | Max abs diff | Mean abs diff | Output shape |
| --- | --- | --- | --- | --- |
| Conv2D patch projection | pass_preferred | 4.76837158e-07 | 5.03491506e-08 | [2, 128, 4, 5] |
| Dense hst_head | pass_preferred | 5.96046448e-07 | 1.30401332e-07 | [4, 36] |
| LayerNormProxy channel LayerNorm | pass_preferred | 7.15255737e-07 | 2.19274678e-08 | [2, 128, 5, 7] |
| GELU | pass_preferred | 2.38418579e-07 | 3.8804604e-08 | [3, 7, 11] |
| Isolated TransformerMLP linear1-GELU-linear2 | pass_preferred | 9.53674316e-07 | 2.07778797e-07 | [2, 5, 128] |

## GELU Approximate Mode Check
- Exact max_abs_diff: 2.38418579e-07
- Tanh approximate max_abs_diff: 0.00039434433
- Chosen mode: `exact`

## Unresolved Issues
- none

## Stage C Patch Projection Parity Decision
- Safe to proceed to Stage C patch projection parity: True
- Scope of this decision: basic Conv2D, Dense, LayerNorm, GELU, and isolated MLP mapping only. This does not establish full TensorFlow DAT feasibility.

## Warnings
- stage_b_basic_layer_parity.py:48: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.

## Runtime Log Notices
- none

## Errors
- none
