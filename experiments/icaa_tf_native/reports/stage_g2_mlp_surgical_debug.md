# Stage G-2 MLP Surgical Debug

## Scope
- Exact command used: `python experiments/icaa_tf_native/scripts/debug_stage_g2_block_mlp.py`
- Checkpoint path: `/home/omen_pc1/photo_score_project/weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth`
- PyTorch reference model path: `/home/omen_pc1/photo_score_project/experiments/icaa_export_safe_v3/export_safe_models`
- Output directory: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_stage_g2_mlp_debug_20260516_031415`
- Status: `ok`
- Target blocks: [3, 4]
- Seeds: torch=123, numpy=123, tensorflow=123

## Checkpoint Keys Used
- `stages.2.mlps.3.chunk.linear1.weight`
- `stages.2.mlps.3.chunk.linear1.bias`
- `stages.2.mlps.3.chunk.linear2.weight`
- `stages.2.mlps.3.chunk.linear2.bias`
- `stages.2.mlps.4.chunk.linear1.weight`
- `stages.2.mlps.4.chunk.linear1.bias`
- `stages.2.mlps.4.chunk.linear2.weight`
- `stages.2.mlps.4.chunk.linear2.bias`

## Direct Same-Input MLP Decision
| Case | Block | Direct same-input MLP | First full-flow failure | First direct failure |
| --- | --- | --- | --- | --- |
| deterministic random stage-2 input | 3 | pass | block3_mlp_output | None |
| deterministic random stage-2 input | 4 | pass | block4_block_input | None |
| captured stage-2 input from deterministic random image | 3 | pass | None | None |
| captured stage-2 input from deterministic random image | 4 | pass | block4_mlp_output | None |
| captured stage-2 input from 3 real ICAA17K images | 3 | pass | block3_mlp_output | None |
| captured stage-2 input from 3 real ICAA17K images | 4 | pass | block4_block_input | None |

## Interpretation
- Block 3 direct same-input MLP parity passed: True
- Block 4 direct same-input MLP parity passed: True
- First diverging subcomponent in full-flow comparisons: block3_mlp_output
- First diverging subcomponent in direct same-input comparisons: None
- Supports numeric drift: True
- Supports implementation bug: False
- Code modification recommended: False
- Stage G-2 rerun recommendation: Do not change MLP implementation; investigate accumulated numeric drift or explicitly approve a relaxed full-stage gate before rerun.

## Debug Tensor Summary
- Debug summary artifact: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_stage_g2_mlp_debug_20260516_031415/optional_debug_tensors_summary.json`
- Captured components per target block: block input, norm1 output, attention output, after-attention residual, norm2 output, MLP linear1 output, GELU output, MLP linear2 output, and after-MLP residual.
- Direct same-input MLP tests use PyTorch-captured `norm2_output` as the input to both MLP implementations.

## Real Image Inputs
- Performed: True
- Image paths: ['/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/juiedesai19391601268.jpg', '/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/162814817@N0627431714937.jpg', '/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/148560147@N0337039646742.jpg']

## Warnings
- debug_stage_g2_block_mlp.py:108: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
- __init__.py:49: Importing from timm.models.layers is deprecated, please import via timm.layers
- functional.py:534: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3595.)
- layer.py:424: `build()` was called on layer 'tf_transformer_stage2', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.

## Runtime Log Notices
- none

## Errors
- none
