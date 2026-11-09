# Stage C Patch Projection Parity

## Scope
- Exact command used: `python experiments/icaa_tf_native/scripts/stage_c_patch_projection_parity.py`
- Checkpoint path: `/home/omen_pc1/photo_score_project/weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth`
- PyTorch reference model path: `/home/omen_pc1/photo_score_project/experiments/icaa_export_safe_v3/export_safe_models`
- Output directory: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_stage_c_20260515_175817`
- Status: `ok`
- Seeds: torch=123, numpy=123, tensorflow=123

## Checkpoint Tensors Used
- `patch_proj.0.weight`
- `patch_proj.0.bias`
- `patch_proj.1.norm.weight`
- `patch_proj.1.norm.bias`

## TensorFlow Patch Projection Structure
```json
{
  "input_layout": "NHWC/channels_last",
  "layers": [
    {
      "name": "patch_proj_conv",
      "type": "tf.keras.layers.Conv2D",
      "filters": 128,
      "kernel_size": [
        4,
        4
      ],
      "strides": [
        4,
        4
      ],
      "padding": "valid",
      "use_bias": true
    },
    {
      "name": "patch_proj_norm",
      "type": "tf.keras.layers.LayerNormalization",
      "axis": -1,
      "epsilon": 1e-05
    }
  ]
}
```

## Weight Mapping Rules Used
- Conv2D kernel: PyTorch `[out_c, in_c, kH, kW]` -> TensorFlow `[kH, kW, in_c, out_c]`.
- Conv2D bias: direct copy.
- LayerNorm gamma/beta: direct copy.
- Layout: PyTorch NCHW input/output, TensorFlow NHWC input/output; TensorFlow output is transposed back to NCHW before comparison.

## Random Input Result
- PyTorch output shape: [1, 128, 56, 56]
- TensorFlow output shape before transpose: [1, 56, 56, 128]
- TensorFlow output shape after transpose: [1, 128, 56, 56]
- max_abs_diff: 3.81469727e-06
- mean_abs_diff: 8.1713118e-08
- pass/fail: pass_preferred

## Real Image Result
- Status: pass_preferred
- PyTorch output shape: [3, 128, 56, 56]
- TensorFlow output shape before transpose: [3, 56, 56, 128]
- TensorFlow output shape after transpose: [3, 128, 56, 56]
- max_abs_diff: 1.90734863e-06
- mean_abs_diff: 4.95252301e-08
- Image paths: ['/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/juiedesai19391601268.jpg', '/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/162814817@N0627431714937.jpg', '/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/148560147@N0337039646742.jpg']

## Unresolved Issues
- none

## Stage D SoftHistogram Parity Decision
- Safe to proceed to Stage D SoftHistogram parity: True
- Scope of this decision: patch projection parity only. This does not establish full TensorFlow DAT feasibility.

## Warnings
- stage_c_patch_projection_parity.py:107: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
- __init__.py:49: Importing from timm.models.layers is deprecated, please import via timm.layers
- functional.py:534: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3595.)

## Runtime Log Notices
- none

## Errors
- none
