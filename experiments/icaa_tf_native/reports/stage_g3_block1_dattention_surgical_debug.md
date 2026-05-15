# Stage G-3 Block1 DAttention Surgical Debug

## Scope
- Exact command used: `python experiments/icaa_tf_native/scripts/debug_stage_g3_block1_dattention.py`
- Checkpoint path: `/home/omen_pc1/photo_score_project/weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth`
- PyTorch reference model path: `/home/omen_pc1/photo_score_project/experiments/icaa_export_safe_v3/export_safe_models`
- Output directory: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_stage_g3_dattention_debug_20260516_034800`
- Status: `ok`
- Exact block tested: Stage `3` block `1`
- PyTorch module: `stages.3.attns.1`
- TensorFlow module: `TFTransformerStage3.attns[1]` / `TFDAttentionBaseline`
- Seeds: torch=123, numpy=123, tensorflow=123

## Checkpoint Keys Used
- `stages.3.attns.1.rpe_table`
- `stages.3.attns.1.conv_offset.0.weight`
- `stages.3.attns.1.conv_offset.0.bias`
- `stages.3.attns.1.conv_offset.1.norm.weight`
- `stages.3.attns.1.conv_offset.1.norm.bias`
- `stages.3.attns.1.conv_offset.3.weight`
- `stages.3.attns.1.proj_q.weight`
- `stages.3.attns.1.proj_q.bias`
- `stages.3.attns.1.proj_k.weight`
- `stages.3.attns.1.proj_k.bias`
- `stages.3.attns.1.proj_v.weight`
- `stages.3.attns.1.proj_v.bias`
- `stages.3.attns.1.proj_out.weight`
- `stages.3.attns.1.proj_out.bias`
- `stages.3.attns.1.ref_point14.weight`
- `stages.3.attns.1.ref_point14.bias`
- `stages.3.attns.1.ref_gate.weight`
- `stages.3.attns.1.ref_gate.bias`

## Direct Same-Input DAttention Decision
| Case | Direct same-input DAttention | Max abs diff | Mean abs diff | First diverging subcomponent | Likely component |
| --- | --- | --- | --- | --- | --- |
| deterministic random stage-3 input | True | 8.58306885e-05 | 1.26867951e-06 | None | None |
| captured stage-3 input from deterministic random image | True | 1.33514404e-05 | 5.75906931e-08 | None | None |
| captured stage-3 input from 3 real ICAA17K images | True | 2.28881836e-05 | 9.75057972e-08 | None | None |

## First Diverging Subcomponent
- deterministic random stage-3 input: none
- captured stage-3 input from deterministic random image: none
- captured stage-3 input from 3 real ICAA17K images: none

## Interpretation
- Direct same-input DAttention parity passed for all cases: True
- Supports numeric drift: True
- Supports implementation bug: False
- Code modification recommended: False
- Full DAT assembly should remain blocked: True
- Decision note: Direct same-input DAttention passed; this supports accumulated numeric drift before or around Stage G-3 rather than a structural Stage 3 block1 DAttention implementation bug.

## Captured Subcomponents
- block1 DAttention input
- q
- offset
- pos
- reference
- sampled feature
- k
- v
- attention logits before relative position bias
- relative position bias / sampled RPE bias
- attention logits after bias
- softmax attention
- weighted attention output before projection
- output projection
- final DAttention output

## Debug Tensor Summary
- Debug summary artifact: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_stage_g3_dattention_debug_20260516_034800/optional_debug_tensors_summary.json`
- PyTorch manual reconstruction of DAttention is compared against the PyTorch module output for each case.

## Real Image Inputs
- Performed: True
- Image paths: ['/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/juiedesai19391601268.jpg', '/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/162814817@N0627431714937.jpg', '/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/148560147@N0337039646742.jpg']

## Warnings
- debug_stage_g3_block1_dattention.py:112: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
- __init__.py:49: Importing from timm.models.layers is deprecated, please import via timm.layers
- functional.py:534: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3595.)
- layer.py:424: `build()` was called on layer 'tf_transformer_stage3', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.

## Runtime Log Notices
- none

## Errors
- none
