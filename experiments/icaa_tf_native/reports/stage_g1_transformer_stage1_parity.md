# Stage G-1 TransformerStage 1 Parity

## Scope
- Exact command used: `python experiments/icaa_tf_native/scripts/stage_g1_transformer_stage1_parity.py`
- Checkpoint path: `/home/omen_pc1/photo_score_project/weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth`
- PyTorch reference model path: `/home/omen_pc1/photo_score_project/experiments/icaa_export_safe_v3/export_safe_models`
- Output directory: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_stage_g1_20260516_022555`
- Status: `ok`
- Seeds: torch=123, numpy=123, tensorflow=123

## PyTorch Classes Inspected
```json
{
  "class_name": "TransformerStage",
  "stage_index": 1,
  "stage_input_shape_nchw": [
    null,
    256,
    28,
    28
  ],
  "stage_output_shape_nchw": [
    null,
    256,
    28,
    28
  ],
  "depth": 2,
  "stage_spec": [
    "L",
    "S"
  ],
  "proj_class": "Identity",
  "attention_blocks": [
    {
      "index": 0,
      "class_name": "LocalAttention",
      "window_size": [
        7,
        7
      ],
      "heads": 8,
      "scale": 0.1767766952966369
    },
    {
      "index": 1,
      "class_name": "ShiftWindowAttention",
      "window_size": [
        7,
        7
      ],
      "heads": 8,
      "scale": 0.1767766952966369
    }
  ],
  "shift_window_details": {
    "shift_size": 4,
    "fmap_size": [
      28,
      28
    ],
    "attn_mask_shape": [
      16,
      49,
      49
    ],
    "attn_mask_min": -100.0,
    "attn_mask_max": 0.0
  },
  "mlp_blocks": [
    {
      "index": 0,
      "class_name": "TransformerMLP",
      "hidden_dim": 1024
    },
    {
      "index": 1,
      "class_name": "TransformerMLP",
      "hidden_dim": 1024
    }
  ],
  "layer_norms": [
    {
      "index": 0,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        256
      ],
      "eps": 1e-05
    },
    {
      "index": 1,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        256
      ],
      "eps": 1e-05
    },
    {
      "index": 2,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        256
      ],
      "eps": 1e-05
    },
    {
      "index": 3,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        256
      ],
      "eps": 1e-05
    }
  ],
  "drop_path": [
    {
      "index": 0,
      "class_name": "DropPath",
      "drop_prob": 0.043478261679410934
    },
    {
      "index": 1,
      "class_name": "DropPath",
      "drop_prob": 0.06521739065647125
    }
  ],
  "drop_path_eval_behavior": "Both stage-1 drop_path modules are DropPath with nonzero configured probabilities, but eval mode disables them and makes them identity.",
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
    "is_part_of_stage1": false,
    "dat_forward_order": "DAT.forward calls model.stages[1](x), then applies model.down_projs[1](x) outside the stage when i < 3.",
    "down_proj1_class": "Sequential"
  }
}
```

## Checkpoint Tensors Used
- `stages.1.layer_norms.0.norm.weight`
- `stages.1.layer_norms.0.norm.bias`
- `stages.1.layer_norms.1.norm.weight`
- `stages.1.layer_norms.1.norm.bias`
- `stages.1.layer_norms.2.norm.weight`
- `stages.1.layer_norms.2.norm.bias`
- `stages.1.layer_norms.3.norm.weight`
- `stages.1.layer_norms.3.norm.bias`
- `stages.1.mlps.0.chunk.linear1.weight`
- `stages.1.mlps.0.chunk.linear1.bias`
- `stages.1.mlps.0.chunk.linear2.weight`
- `stages.1.mlps.0.chunk.linear2.bias`
- `stages.1.mlps.1.chunk.linear1.weight`
- `stages.1.mlps.1.chunk.linear1.bias`
- `stages.1.mlps.1.chunk.linear2.weight`
- `stages.1.mlps.1.chunk.linear2.bias`
- `stages.1.attns.0.relative_position_bias_table`
- `stages.1.attns.0.relative_position_index`
- `stages.1.attns.0.proj_qkv.weight`
- `stages.1.attns.0.proj_qkv.bias`
- `stages.1.attns.0.proj_out.weight`
- `stages.1.attns.0.proj_out.bias`
- `stages.1.attns.1.relative_position_bias_table`
- `stages.1.attns.1.relative_position_index`
- `stages.1.attns.1.attn_mask`
- `stages.1.attns.1.proj_qkv.weight`
- `stages.1.attns.1.proj_qkv.bias`
- `stages.1.attns.1.proj_out.weight`
- `stages.1.attns.1.proj_out.bias`

## Stage 1 Shapes
- PyTorch input shape: [1, 256, 28, 28]
- PyTorch output shape: [1, 256, 28, 28]
- TensorFlow input shape: [1, 28, 28, 256]
- TensorFlow output shape: [1, 28, 28, 256]

## TensorFlow Implementation Summary
```json
{
  "class_name": "TFTransformerStage1",
  "target_pytorch_module": "stages.1",
  "input_layout": "NHWC/channels_last",
  "proj": "Identity; no projection weights are used for stage 1",
  "depth": 2,
  "stage_spec": [
    "L",
    "S"
  ],
  "blocks": [
    {
      "index": 0,
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
      "attention": "TFShiftWindowAttention mapped from PyTorch ShiftWindowAttention",
      "shift_size": 4,
      "attn_mask": "direct copy from stages.1.attns.1.attn_mask",
      "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
      "layer_norms": [
        "layer_norm_2",
        "layer_norm_3"
      ],
      "drop_path_eval": "identity in eval mode"
    }
  ],
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
- Linear kernels: PyTorch `[out, in]` -> TensorFlow Dense `[in, out]`.
- Linear biases: direct copy.
- LayerNorm gamma/beta: direct copy.
- Relative position bias table: direct copy.
- Relative position index: copied as integer gather indices.
- Shift-window attention mask: direct copy from `stages.1.attns.1.attn_mask`.
- GELU: TensorFlow exact GELU to match PyTorch default `nn.GELU()`.
- Layout: PyTorch NCHW input/output, TensorFlow NHWC input/output; TensorFlow outputs are transposed back to NCHW before comparison.

## Parity Results
| Test | Status | Max abs diff | Mean abs diff | PyTorch input | PyTorch output | TensorFlow input | TensorFlow output before transpose |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deterministic random stage-1 input | pass_preferred | 7.62939453e-06 | 1.18373578e-06 | [1, 256, 28, 28] | [1, 256, 28, 28] | [1, 28, 28, 256] | [1, 28, 28, 256] |
| captured stage-1 input from deterministic random image | pass_preferred | 3.81469727e-06 | 5.34230878e-07 | [1, 256, 28, 28] | [1, 256, 28, 28] | [1, 28, 28, 256] | [1, 28, 28, 256] |
| captured stage-1 input from 3 real ICAA17K images | pass_preferred | 5.24520874e-06 | 6.35120614e-07 | [3, 256, 28, 28] | [3, 256, 28, 28] | [3, 28, 28, 256] | [3, 28, 28, 256] |

## Debug Tensor Summary
- Debug summary artifact: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_stage_g1_20260516_022555/optional_debug_tensors_summary.json`
- Subcomponent diffs are recorded for both attention blocks, both MLP blocks, residual adds, LayerNorm outputs, and final stage output.

## Real Image Inputs
- Performed: True
- Image paths: ['/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/juiedesai19391601268.jpg', '/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/162814817@N0627431714937.jpg', '/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/148560147@N0337039646742.jpg']

## Unresolved Issues
- none

## Stage G-2 TransformerStage 2 Decision
- Safe to proceed to Stage G-2 TransformerStage 2 parity: True
- Scope of this decision: `TransformerStage` stage `1` only. This does not establish full TensorFlow DAT feasibility.

## Warnings
- stage_g1_transformer_stage1_parity.py:107: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
- __init__.py:49: Importing from timm.models.layers is deprecated, please import via timm.layers
- functional.py:534: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3595.)

## Runtime Log Notices
- none

## Errors
- none
