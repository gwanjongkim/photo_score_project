# Stage F Deformable Attention Parity

## Scope
- Exact command used: `python experiments/icaa_tf_native/scripts/stage_f_deformable_attention_parity.py`
- Checkpoint path: `/home/omen_pc1/photo_score_project/weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth`
- PyTorch reference model path: `/home/omen_pc1/photo_score_project/experiments/icaa_export_safe_v3/export_safe_models`
- Output directory: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_stage_f_20260516_113458`
- Status: `ok`
- Seeds: torch=123, numpy=123, tensorflow=123

## PyTorch Module Inspection
```json
{
  "class_name": "DAttentionBaseline",
  "selected_stage_index": 2,
  "selected_attention_index": 1,
  "first_d_in_stage_spec": "stage_spec[2][1]",
  "q_size": [
    14,
    14
  ],
  "kv_size": [
    14,
    14
  ],
  "n_heads": 16,
  "n_head_channels": 32,
  "n_groups": 4,
  "n_group_channels": 128,
  "n_group_heads": 4,
  "scale": 0.1767766952966369,
  "offset_range_factor": 2,
  "use_pe": true,
  "dwc_pe": false,
  "fixed_pe": false,
  "no_off": false,
  "offset_branch": [
    "Conv2d",
    "LayerNormProxy",
    "GELU",
    "Conv2d"
  ],
  "qkv_projection_classes": {
    "proj_q": "Conv2d",
    "proj_k": "Conv2d",
    "proj_v": "Conv2d",
    "proj_out": "Conv2d"
  },
  "rpe_table_shape": [
    16,
    27,
    27
  ],
  "dropout_eval_behavior": "proj_drop and attn_drop are Dropout(0.0) in this config; eval mode disables dropout randomness."
}
```

## Checkpoint Tensors Used
- `stages.2.attns.1.rpe_table`
- `stages.2.attns.1.conv_offset.0.weight`
- `stages.2.attns.1.conv_offset.0.bias`
- `stages.2.attns.1.conv_offset.1.norm.weight`
- `stages.2.attns.1.conv_offset.1.norm.bias`
- `stages.2.attns.1.conv_offset.3.weight`
- `stages.2.attns.1.proj_q.weight`
- `stages.2.attns.1.proj_q.bias`
- `stages.2.attns.1.proj_k.weight`
- `stages.2.attns.1.proj_k.bias`
- `stages.2.attns.1.proj_v.weight`
- `stages.2.attns.1.proj_v.bias`
- `stages.2.attns.1.proj_out.weight`
- `stages.2.attns.1.proj_out.bias`
- `stages.2.attns.1.ref_point14.weight`
- `stages.2.attns.1.ref_point14.bias`
- `stages.2.attns.1.ref_gate.weight`
- `stages.2.attns.1.ref_gate.bias`

## DAttentionBaseline Shapes
- PyTorch input shape: [1, 512, 14, 14]
- PyTorch output shape: [1, 512, 14, 14]
- TensorFlow input shape: [1, 14, 14, 512]
- TensorFlow output shape: [1, 14, 14, 512]

## Test Input Generation
```json
{
  "random_module_input": {
    "distribution": "numpy randn scaled by RANDOM_MODULE_INPUT_SCALE",
    "scale": 0.5,
    "reason": "keeps the synthetic block input near observed captured DAttention tensor scale while still using deterministic random values",
    "seed": 123
  },
  "captured_random_image_input": {
    "distribution": "numpy uniform [0, 1) RGB image with ImageNet mean/std preprocessing",
    "seed": 123
  },
  "real_image_input": {
    "source": "external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv",
    "preprocessing": "PIL bilinear resize to 224x224, RGB [0,1], ImageNet mean/std normalization"
  }
}
```

## TensorFlow Implementation Summary
```json
{
  "class_name": "TFDAttentionBaseline",
  "target_pytorch_module": "stages.2.attns.1",
  "input_layout": "NHWC/channels_last",
  "q_size": [
    14,
    14
  ],
  "heads": 16,
  "head_channels": 32,
  "groups": 4,
  "group_channels": 128,
  "offset_branch": [
    "DepthwiseConv2D kernel 5 stride 1 padding same, mapped from grouped Conv2d",
    "LayerNormalization(axis=-1, epsilon=1e-5)",
    "exact GELU",
    "PytorchLikeConv1x1 to 2 channels without bias"
  ],
  "standard_1x1_projections": [
    "PytorchLikeConv1x1 for proj_q/proj_k/proj_v/proj_out"
  ],
  "sampling": [
    "sigmoid(ref_point14) for reference points",
    "sigmoid(ref_gate) for gate values",
    "offset * manu_offset + reference for grid coordinates",
    "manual_bilinear_grid_sample_nhwc for sampling features and RPE bias"
  ],
  "attention": "scaled dot-product multi-head attention with groups and relative position bias",
  "rpe_table_shape": [
    16,
    27,
    27
  ]
}
```

## Manual Bilinear Sampler Summary
- Normalized grid coordinates are mapped with `align_corners=True`: `(coord + 1) * 0.5 * (size - 1)`.
- Out-of-range corner weights are zeroed to match `padding_mode='zeros'`.
- Corner indices are clamped only for safe `tf.gather` indexing.
- Implemented with TensorFlow tensor ops, no NumPy or pixel loops in the sampler.

## Weight Mapping Rules Used
- 1x1 Conv2D projections: PyTorch `[out_c, in_c, 1, 1]` -> `PytorchLikeConv1x1` kernel `[out_c, in_c]`, applied as NHWC flat matrix multiplication with `kernel^T`.
- Depthwise offset Conv2D: PyTorch `[channel, 1, kH, kW]` -> TensorFlow `[kH, kW, channel, 1]`.
- Conv biases and LayerNorm gamma/beta: direct copy.
- `rpe_table`: direct copy as `[heads, 2H-1, 2W-1]`.
- Layout: PyTorch NCHW input/output, TensorFlow NHWC input/output; TensorFlow output is transposed back to NCHW before comparison.

## Parity Results
| Test | Status | Max abs diff | Mean abs diff | PyTorch shape | TensorFlow shape before transpose |
| --- | --- | --- | --- | --- | --- |
| random DAttentionBaseline input | pass_preferred | 8.16583633e-06 | 8.04580907e-07 | [1, 512, 14, 14] | [1, 14, 14, 512] |
| captured random-image DAttentionBaseline input | pass_preferred | 4.66406345e-06 | 7.68889038e-07 | [1, 512, 14, 14] | [1, 14, 14, 512] |
| captured 3-real-image DAttentionBaseline input | pass_preferred | 8.10623169e-06 | 4.83207145e-07 | [3, 512, 14, 14] | [3, 14, 14, 512] |

## Debug Tensor Summary
- Debug summary artifact: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_stage_f_20260516_113458/optional_debug_tensors_summary.json`
- Debug diffs include returned `pos` and `reference` tensors for each test case.

## Real Image Inputs
- Performed: True
- Image paths: ['/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/juiedesai19391601268.jpg', '/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/162814817@N0627431714937.jpg', '/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/148560147@N0337039646742.jpg']

## Unresolved Issues
- none

## Stage G Full TransformerStage Decision
- Safe to proceed to Stage G full TransformerStage parity: True
- Scope of this decision: `DAttentionBaseline` module parity only. This does not establish full TensorFlow DAT feasibility.

## Warnings
- stage_f_deformable_attention_parity.py:130: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
- __init__.py:49: Importing from timm.models.layers is deprecated, please import via timm.layers
- functional.py:534: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3595.)

## Runtime Log Notices
- none

## Errors
- none
