# Stage E Local Attention Block Parity

## Scope
- Exact command used: `python experiments/icaa_tf_native/scripts/stage_e_local_attention_block_parity.py`
- Checkpoint path: `/home/omen_pc1/photo_score_project/weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth`
- PyTorch reference model path: `/home/omen_pc1/photo_score_project/experiments/icaa_export_safe_v3/export_safe_models`
- Output directory: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_stage_e_20260515_181850`
- Status: `ok`
- Seeds: torch=123, numpy=123, tensorflow=123

## PyTorch Classes Inspected
```json
{
  "stage_class": "TransformerStage",
  "stage_proj_class": "Identity",
  "stage_depths": 2,
  "target_depth_index": 0,
  "attention_class": "LocalAttention",
  "mlp_class": "TransformerMLP",
  "layer_norm_proxy_class": "LayerNormProxy",
  "inner_layer_norm_class": "LayerNorm",
  "drop_path_class": "Identity",
  "drop_path_eval_behavior": "Identity for stage 0 block 0 because the first drop_path_rate is 0.0",
  "residual_order": [
    "x0 = x",
    "attn_out = attns[0](layer_norms[0](x))",
    "x = drop_path[0](attn_out) + x0",
    "x0 = x",
    "mlp_out = mlps[0](layer_norms[1](x))",
    "x = drop_path[0](mlp_out) + x0"
  ],
  "attention_window_size": [
    7,
    7
  ],
  "attention_heads": 4,
  "attention_scale": 0.1767766952966369
}
```

## Checkpoint Tensors Used
- `stages.0.layer_norms.0.norm.weight`
- `stages.0.layer_norms.0.norm.bias`
- `stages.0.attns.0.relative_position_bias_table`
- `stages.0.attns.0.relative_position_index`
- `stages.0.attns.0.proj_qkv.weight`
- `stages.0.attns.0.proj_qkv.bias`
- `stages.0.attns.0.proj_out.weight`
- `stages.0.attns.0.proj_out.bias`
- `stages.0.layer_norms.1.norm.weight`
- `stages.0.layer_norms.1.norm.bias`
- `stages.0.mlps.0.chunk.linear1.weight`
- `stages.0.mlps.0.chunk.linear1.bias`
- `stages.0.mlps.0.chunk.linear2.weight`
- `stages.0.mlps.0.chunk.linear2.bias`

## Block Shapes
- PyTorch input shape: [1, 128, 56, 56]
- PyTorch output shape: [1, 128, 56, 56]
- TensorFlow input shape: [1, 56, 56, 128]
- TensorFlow output shape: [1, 56, 56, 128]

## TensorFlow Implementation Summary
```json
{
  "class_name": "TFFirstLocalAttentionBlock",
  "input_layout": "NHWC/channels_last",
  "target_pytorch_block": "stages.0 depth index 0",
  "layers": [
    {
      "name": "layer_norm_0",
      "pytorch_class": "LayerNormProxy",
      "tf_layer": "tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)"
    },
    {
      "name": "local_attention",
      "pytorch_class": "LocalAttention",
      "heads": 4,
      "window_size": [
        7,
        7
      ],
      "qkv_projection": "Dense mapped from proj_qkv Linear",
      "relative_position_bias": "gather table with relative_position_index"
    },
    {
      "name": "residual_0",
      "operation": "attn_out + block_input",
      "drop_path": "identity for stage 0 block 0"
    },
    {
      "name": "layer_norm_1",
      "pytorch_class": "LayerNormProxy",
      "tf_layer": "tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)"
    },
    {
      "name": "mlp",
      "pytorch_class": "TransformerMLP",
      "operation": "Dense -> exact GELU -> Dense, dropout omitted in eval"
    },
    {
      "name": "residual_1",
      "operation": "mlp_out + after_attn_residual",
      "drop_path": "identity for stage 0 block 0"
    }
  ]
}
```

## Weight Mapping Rules Used
- Linear kernels: PyTorch `[out, in]` -> TensorFlow Dense `[in, out]`.
- Linear biases: direct copy.
- LayerNorm gamma/beta: direct copy.
- Relative position bias table: direct copy.
- Relative position index: copied as integer gather indices.
- GELU: TensorFlow exact GELU to match PyTorch default `nn.GELU()`.
- Layout: PyTorch NCHW input/output, TensorFlow NHWC input/output; TensorFlow outputs are transposed back to NCHW before comparison.

## Parity Results
| Test | Status | Max abs diff | Mean abs diff | PyTorch shape | TensorFlow shape before transpose |
| --- | --- | --- | --- | --- | --- |
| random block input | pass_preferred | 9.53674316e-06 | 6.46088381e-07 | [1, 128, 56, 56] | [1, 56, 56, 128] |
| patch projection output from deterministic random image | pass_preferred | 6.19888306e-06 | 5.73447096e-07 | [1, 128, 56, 56] | [1, 56, 56, 128] |
| patch projection output from 3 real ICAA17K images | pass_preferred | 6.19888306e-06 | 4.23493987e-07 | [3, 128, 56, 56] | [3, 56, 56, 128] |

## Debug Tensor Summary
- Debug summary artifact: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_stage_e_20260515_181850/optional_debug_tensors_summary.json`
- Subcomponent diffs are recorded for `norm0`, `attn_out`, `after_attn_residual`, `norm1`, `mlp_out`, and `output`.

## Real Image Inputs
- Performed: True
- Image paths: ['/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/juiedesai19391601268.jpg', '/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/162814817@N0627431714937.jpg', '/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/148560147@N0337039646742.jpg']

## Unresolved Issues
- none

## Stage F Deformable / Delegate Attention Decision
- Safe to proceed to Stage F deformable/delegate attention parity: True
- Scope of this decision: stage-0 depth-0 local-attention block only. This does not establish full TensorFlow DAT feasibility.

## Warnings
- stage_e_local_attention_block_parity.py:124: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
- __init__.py:49: Importing from timm.models.layers is deprecated, please import via timm.layers
- functional.py:534: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3595.)

## Runtime Log Notices
- none

## Errors
- none
