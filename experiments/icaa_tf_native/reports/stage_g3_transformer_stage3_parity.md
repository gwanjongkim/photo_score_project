# Stage G-3 TransformerStage 3 Parity

## Scope
- Exact command used: `python experiments/icaa_tf_native/scripts/stage_g3_transformer_stage3_parity.py`
- Checkpoint path: `/home/omen_pc1/photo_score_project/weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth`
- PyTorch reference model path: `/home/omen_pc1/photo_score_project/experiments/icaa_export_safe_v3/export_safe_models`
- Output directory: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_stage_g3_20260516_033136`
- Status: `parity_failed`
- Seeds: torch=123, numpy=123, tensorflow=123
- Stage G-2 full strict parity status: unresolved; this Stage G-3 report does not claim Stage G-2 success.

## PyTorch Classes Inspected
```json
{
  "class_name": "TransformerStage",
  "stage_index": 3,
  "stage_input_shape_nchw": [
    null,
    1024,
    7,
    7
  ],
  "stage_output_shape_nchw": [
    null,
    1024,
    7,
    7
  ],
  "depth": 2,
  "stage_spec": [
    "L",
    "D"
  ],
  "proj_class": "Identity",
  "attention_blocks": [
    {
      "index": 0,
      "class_name": "LocalAttention",
      "attention_type": "L",
      "window_size": [
        7,
        7
      ],
      "heads": 32,
      "scale": 0.1767766952966369
    },
    {
      "index": 1,
      "class_name": "DAttentionBaseline",
      "attention_type": "D",
      "window_size": null,
      "heads": 32,
      "scale": 0.1767766952966369
    }
  ],
  "deformable_attention_details": {
    "q_size": [
      7,
      7
    ],
    "kv_size": [
      7,
      7
    ],
    "n_heads": 32,
    "n_head_channels": 32,
    "n_groups": 8,
    "offset_range_factor": 2,
    "use_pe": true,
    "dwc_pe": false,
    "fixed_pe": false,
    "no_off": false,
    "rpe_table_shape": [
      32,
      13,
      13
    ]
  },
  "mlp_blocks": [
    {
      "index": 0,
      "class_name": "TransformerMLP",
      "hidden_dim": 4096
    },
    {
      "index": 1,
      "class_name": "TransformerMLP",
      "hidden_dim": 4096
    }
  ],
  "layer_norms": [
    {
      "index": 0,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        1024
      ],
      "eps": 1e-05
    },
    {
      "index": 1,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        1024
      ],
      "eps": 1e-05
    },
    {
      "index": 2,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        1024
      ],
      "eps": 1e-05
    },
    {
      "index": 3,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        1024
      ],
      "eps": 1e-05
    }
  ],
  "drop_path": [
    {
      "index": 0,
      "class_name": "DropPath",
      "drop_prob": 0.47826087474823
    },
    {
      "index": 1,
      "class_name": "DropPath",
      "drop_prob": 0.5
    }
  ],
  "drop_path_eval_behavior": "Both stage-3 drop_path modules are DropPath with nonzero configured probabilities, but eval mode disables them and makes them identity.",
  "residual_order": [
    "x = proj(x)",
    "x0 = x",
    "attn_out = attns[d](layer_norms[2*d](x))",
    "x = drop_path[d](attn_out) + x0",
    "x0 = x",
    "mlp_out = mlps[d](layer_norms[2*d+1](x))",
    "x = drop_path[d](mlp_out) + x0"
  ],
  "down_projection_relationship": {
    "is_part_of_stage3": false,
    "dat_forward_order": "DAT.forward calls model.stages[3](x) and does not apply a down projection afterward because down_projs are only used when i < 3.",
    "down_proj_after_stage3": null
  }
}
```

## Checkpoint Tensors Used
- `stages.3.layer_norms.0.norm.weight`
- `stages.3.layer_norms.0.norm.bias`
- `stages.3.layer_norms.1.norm.weight`
- `stages.3.layer_norms.1.norm.bias`
- `stages.3.layer_norms.2.norm.weight`
- `stages.3.layer_norms.2.norm.bias`
- `stages.3.layer_norms.3.norm.weight`
- `stages.3.layer_norms.3.norm.bias`
- `stages.3.mlps.0.chunk.linear1.weight`
- `stages.3.mlps.0.chunk.linear1.bias`
- `stages.3.mlps.0.chunk.linear2.weight`
- `stages.3.mlps.0.chunk.linear2.bias`
- `stages.3.mlps.1.chunk.linear1.weight`
- `stages.3.mlps.1.chunk.linear1.bias`
- `stages.3.mlps.1.chunk.linear2.weight`
- `stages.3.mlps.1.chunk.linear2.bias`
- `stages.3.attns.0.relative_position_bias_table`
- `stages.3.attns.0.relative_position_index`
- `stages.3.attns.0.proj_qkv.weight`
- `stages.3.attns.0.proj_qkv.bias`
- `stages.3.attns.0.proj_out.weight`
- `stages.3.attns.0.proj_out.bias`
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

## Stage 3 Shapes
- PyTorch input shape: [1, 1024, 7, 7]
- PyTorch output shape: [1, 1024, 7, 7]
- TensorFlow input shape: [1, 7, 7, 1024]
- TensorFlow output shape: [1, 7, 7, 1024]

## TensorFlow Implementation Summary
```json
{
  "class_name": "TFTransformerStage3",
  "target_pytorch_module": "stages.3",
  "input_layout": "NHWC/channels_last",
  "proj": "Identity; no projection weights are used for stage 3",
  "depth": 2,
  "stage_spec": [
    "L",
    "D"
  ],
  "blocks": [
    {
      "index": 0,
      "attention_type": "L",
      "attention": "TFLocalAttention mapped from PyTorch LocalAttention",
      "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
      "layer_norms": [
        "layer_norm_0",
        "layer_norm_1"
      ],
      "drop_path_eval": "identity in eval mode"
    },
    {
      "index": 1,
      "attention_type": "D",
      "attention": "TFDAttentionBaseline mapped from PyTorch DAttentionBaseline",
      "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
      "layer_norms": [
        "layer_norm_2",
        "layer_norm_3"
      ],
      "drop_path_eval": "identity in eval mode"
    }
  ],
  "deformable_attention_config": {
    "q_size": [
      7,
      7
    ],
    "kv_size": [
      7,
      7
    ],
    "heads": 32,
    "head_channels": 32,
    "groups": 8,
    "offset_range_factor": 2,
    "use_pe": true,
    "dwc_pe": false,
    "fixed_pe": false,
    "no_off": false
  },
  "residual_order": [
    "x0 = x",
    "attn_out = attns[d](layer_norms[2*d](x))",
    "x = drop_path[d](attn_out) + x0",
    "x0 = x",
    "mlp_out = mlps[d](layer_norms[2*d+1](x))",
    "x = drop_path[d](mlp_out) + x0"
  ]
}
```

## Weight Mapping Rules Used
- Stage projection: PyTorch `Identity`, so no projection tensor is mapped.
- LocalAttention linear kernels: PyTorch `[out, in]` -> TensorFlow Dense `[in, out]`.
- LocalAttention relative position bias table and index: direct copy / integer gather indices.
- DAttention 1x1 Conv2D projections: PyTorch `[out_c, in_c, 1, 1]` -> explicit NHWC matrix multiply in `PytorchLikeConv1x1`.
- DAttention depthwise offset Conv2D: PyTorch `[channel, 1, kH, kW]` -> TensorFlow `[kH, kW, channel, 1]`.
- DAttention `rpe_table`: direct copy as `[heads, 2H-1, 2W-1]`.
- LayerNorm gamma/beta and all biases: direct copy.
- GELU: TensorFlow exact GELU to match PyTorch default `nn.GELU()`.
- Layout: PyTorch NCHW input/output, TensorFlow NHWC input/output; TensorFlow outputs are transposed back to NCHW before comparison.

## Parity Results
| Test | Status | Max abs diff | Mean abs diff | PyTorch input | PyTorch output | TensorFlow input | TensorFlow output before transpose | First divergence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deterministic random stage-3 input | fail | 0.000225067139 | 2.39951373e-06 | [1, 1024, 7, 7] | [1, 1024, 7, 7] | [1, 7, 7, 1024] | [1, 7, 7, 1024] | block1_attn_out |
| captured stage-3 input from deterministic random image | fail | 0.000122070312 | 1.33776734e-06 | [1, 1024, 7, 7] | [1, 1024, 7, 7] | [1, 7, 7, 1024] | [1, 7, 7, 1024] | block1_output |
| captured stage-3 input from 3 real ICAA17K images | fail | 0.000122070312 | 1.3142278e-06 | [3, 1024, 7, 7] | [3, 1024, 7, 7] | [3, 7, 7, 1024] | [3, 7, 7, 1024] | block1_after_attn_residual |

## First Diverging Block
- deterministic random stage-3 input: block1_attn_out
- captured stage-3 input from deterministic random image: block1_output
- captured stage-3 input from 3 real ICAA17K images: block1_after_attn_residual

## Debug Tensor Summary
- Debug summary artifact: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_stage_g3_20260516_033136/optional_debug_tensors_summary.json`
- Per-block diffs are recorded for attention output, after-attention residual, MLP output, after-MLP residual, and final stage output.
- DAttention `pos` and `reference` diffs are recorded for the deformable block.

## Real Image Inputs
- Performed: True
- Image paths: ['/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/juiedesai19391601268.jpg', '/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/162814817@N0627431714937.jpg', '/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/148560147@N0337039646742.jpg']

## Unresolved Issues
- deterministic random stage-3 input failed; first diverging block: block1_attn_out; likely subcomponent: block 1 attention type D
- captured stage-3 input from deterministic random image failed; first diverging block: block1_output; likely subcomponent: block1 MLP residual, layout, or drop path
- captured stage-3 input from 3 real ICAA17K images failed; first diverging block: block1_after_attn_residual; likely subcomponent: block1 attention residual, layout, or drop path

## Full TensorFlow DAT Assembly Decision
- Safe to proceed to full TensorFlow DAT assembly parity: False
- Scope of this decision: Stage G-3 `TransformerStage` stage `3` only; Stage G-2 full strict parity remains unresolved.

## Warnings
- stage_g3_transformer_stage3_parity.py:107: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
- __init__.py:49: Importing from timm.models.layers is deprecated, please import via timm.layers
- functional.py:534: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3595.)
- layer.py:424: `build()` was called on layer 'tf_transformer_stage3', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.

## Runtime Log Notices
- none

## Errors
- none
