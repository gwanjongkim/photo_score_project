# Stage A State Dict Inspection

## Scope
- Checkpoint path: `/home/omen_pc1/photo_score_project/weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth`
- Exact command used: `python experiments/icaa_tf_native/scripts/inspect_pytorch_state_dict.py`
- Inspection status: `ok`
- Checkpoint object type: `OrderedDict`
- State dict confirmation: `checkpoint object is an OrderedDict state_dict`

## Totals
- Total parameter tensors: 487
- Total parameter count: 87138907
- CSV inventory: `/home/omen_pc1/photo_score_project/experiments/icaa_tf_native/weight_maps/pytorch_state_dict_inventory.csv`
- JSON summary: `/home/omen_pc1/photo_score_project/experiments/icaa_tf_native/weight_maps/pytorch_state_dict_summary.json`

## Top-Level Module Breakdown
| top_module | tensor_count | numel |
| --- | --- | --- |
| patch_proj | 4 | 6528 |
| stages | 458 | 84334580 |
| down_projs | 9 | 2756096 |
| cls_norm | 2 | 2048 |
| hst_head | 2 | 36900 |
| SoftHistogram-related layers | 6 | 1296 |
| class_head | 2 | 217 |
| class_head2 | 2 | 217 |
| others | 2 | 1025 |

## Discovered Layer Categories
| likely_tf_layer_type | tensor_count |
| --- | --- |
| Bias | 151 |
| Conv2D | 74 |
| Dense | 80 |
| DepthwiseConv2D | 10 |
| LayerNormalization_beta | 63 |
| LayerNormalization_gamma | 63 |
| PositionalEncoding_or_Buffer | 40 |
| SoftHistogram | 6 |

## Uncertain/Unknown Parameter Groups
- `stages.0.attns.0.relative_position_bias_table` shape=[169, 4] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.0.attns.0.relative_position_index` shape=[49, 49] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.0.attns.1.relative_position_bias_table` shape=[169, 4] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.0.attns.1.relative_position_index` shape=[49, 49] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.0.attns.1.attn_mask` shape=[64, 49, 49] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.1.attns.0.relative_position_bias_table` shape=[169, 8] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.1.attns.0.relative_position_index` shape=[49, 49] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.1.attns.1.relative_position_bias_table` shape=[169, 8] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.1.attns.1.relative_position_index` shape=[49, 49] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.1.attns.1.attn_mask` shape=[16, 49, 49] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.2.attns.0.relative_position_bias_table` shape=[169, 16] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.2.attns.0.relative_position_index` shape=[49, 49] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.2.attns.1.rpe_table` shape=[16, 27, 27] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.2.attns.1.conv_offset.0.weight` shape=[128, 1, 5, 5] type=DepthwiseConv2D rule=Depthwise Conv if present -> inspect carefully, do not guess
- `stages.2.attns.2.relative_position_bias_table` shape=[169, 16] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.2.attns.2.relative_position_index` shape=[49, 49] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.2.attns.3.rpe_table` shape=[16, 27, 27] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.2.attns.3.conv_offset.0.weight` shape=[128, 1, 5, 5] type=DepthwiseConv2D rule=Depthwise Conv if present -> inspect carefully, do not guess
- `stages.2.attns.4.relative_position_bias_table` shape=[169, 16] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.2.attns.4.relative_position_index` shape=[49, 49] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.2.attns.5.rpe_table` shape=[16, 27, 27] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.2.attns.5.conv_offset.0.weight` shape=[128, 1, 5, 5] type=DepthwiseConv2D rule=Depthwise Conv if present -> inspect carefully, do not guess
- `stages.2.attns.6.relative_position_bias_table` shape=[169, 16] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.2.attns.6.relative_position_index` shape=[49, 49] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.2.attns.7.rpe_table` shape=[16, 27, 27] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.2.attns.7.conv_offset.0.weight` shape=[128, 1, 5, 5] type=DepthwiseConv2D rule=Depthwise Conv if present -> inspect carefully, do not guess
- `stages.2.attns.8.relative_position_bias_table` shape=[169, 16] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.2.attns.8.relative_position_index` shape=[49, 49] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.2.attns.9.rpe_table` shape=[16, 27, 27] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.2.attns.9.conv_offset.0.weight` shape=[128, 1, 5, 5] type=DepthwiseConv2D rule=Depthwise Conv if present -> inspect carefully, do not guess
- `stages.2.attns.10.relative_position_bias_table` shape=[169, 16] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.2.attns.10.relative_position_index` shape=[49, 49] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.2.attns.11.rpe_table` shape=[16, 27, 27] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.2.attns.11.conv_offset.0.weight` shape=[128, 1, 5, 5] type=DepthwiseConv2D rule=Depthwise Conv if present -> inspect carefully, do not guess
- `stages.2.attns.12.relative_position_bias_table` shape=[169, 16] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.2.attns.12.relative_position_index` shape=[49, 49] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.2.attns.13.rpe_table` shape=[16, 27, 27] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.2.attns.13.conv_offset.0.weight` shape=[128, 1, 5, 5] type=DepthwiseConv2D rule=Depthwise Conv if present -> inspect carefully, do not guess
- `stages.2.attns.14.relative_position_bias_table` shape=[169, 16] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- `stages.2.attns.14.relative_position_index` shape=[49, 49] type=PositionalEncoding_or_Buffer rule=Unknown -> mark unknown
- ... 16 additional uncertain rows are listed in the CSV.

## Export-Safe v3 Model Check
- Model class: `icaa_export_safe_v3_export_safe_models_stage_a.dat.DAT`
- Model parameter count: 86912997
- Strict checkpoint load success: True
- Missing keys after strict checkpoint load: []
- Unexpected keys after strict checkpoint load: []
- Strict load error: None
- Dummy input shape: [1, 3, 224, 224]
- Dummy output shape summary: `{"type": "tuple", "length": 3, "primary_item": {"type": "Tensor", "shape": [1, 2]}, "auxiliary_item_count": 2, "auxiliary_items": [{"type": "list", "length": 4, "first_item": {"type": "list", "length": 2, "first_item": {"type": "NoneType", "shape": null}}}, {"type": "list", "length": 4, "first_item": {"type": "list", "length": 2, "first_item": {"type": "NoneType", "shape": null}}}]}`

## Proposed First TensorFlow Implementation Target for Stage B
Start with isolated base-layer parity before assembling DAT: `LayerNormProxy`, plain `Conv2D`/`Dense` transposition loaders, and the small `SoftHistogram` grouped-Conv1D path. Do not begin deformable attention until these lower-risk mappings have exact PyTorch-vs-TensorFlow parity evidence.

## Warnings
- inspect_pytorch_state_dict.py:114: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
- __init__.py:49: Importing from timm.models.layers is deprecated, please import via timm.layers
- functional.py:534: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3595.)

## Errors
- none
