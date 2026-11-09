# Stage D SoftHistogram Parity

## Scope
- Exact command used: `python experiments/icaa_tf_native/scripts/stage_d_soft_histogram_parity.py`
- Checkpoint path: `/home/omen_pc1/photo_score_project/weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth`
- PyTorch reference model path: `/home/omen_pc1/photo_score_project/experiments/icaa_export_safe_v3/export_safe_models`
- Output directory: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_stage_d_20260515_180922`
- Status: `ok`
- Seeds: torch=123, numpy=123, tensorflow=123

## PyTorch SoftHistogram Inspection
- Class: `SoftHistogram`
- Constructor arguments: `{"n_features": 36, "n_examples": 6, "num_bins": 6, "quantiles": false}`
- Has trainable parameters: True
- Buffers: []
- `centers` aliases `bin_centers_conv.bias`: False
- `widths` aliases `bin_widths_conv.weight`: True
- Conv1D/groups: `{"bin_centers_conv": {"in_channels": 36, "out_channels": 216, "kernel_size": [1], "groups": 36, "bias": true}, "bin_widths_conv": {"in_channels": 216, "out_channels": 216, "kernel_size": [1], "groups": 216, "bias": true}}`

## Expected Shapes From DAT Path
- Dummy SoftHistogram input shape: [1, 36]
- Dummy SoftHistogram output shape: [1, 216, 1]
- Real captured SoftHistogram input shape: [1, 36]
- Real captured SoftHistogram output shape: [1, 216, 1]

## TensorFlow Implementation Summary
```json
{
  "class_name": "TFSoftHistogram",
  "constructor_args": {
    "n_features": 36,
    "n_examples": 6,
    "num_bins": 6,
    "quantiles": false
  },
  "input_shape": "[batch, n_features]",
  "output_shape_when_quantiles_false": "[1, n_features * num_bins, batch]",
  "math": [
    "transpose input [batch, features] -> [features, batch]",
    "repeat each feature row num_bins times to emulate grouped Conv1d expansion",
    "expanded * bin_centers_conv.weight + bin_centers_conv.bias",
    "abs",
    "value * bin_widths_conv.weight + bin_widths_conv.bias",
    "relu"
  ],
  "variables": [
    {
      "name": "centers",
      "trainable": true,
      "used_in_forward": false
    },
    {
      "name": "bin_centers_conv_weight",
      "trainable": false,
      "used_in_forward": true
    },
    {
      "name": "bin_centers_conv_bias",
      "trainable": true,
      "used_in_forward": true
    },
    {
      "name": "bin_widths_conv_weight",
      "trainable": true,
      "used_in_forward": true
    },
    {
      "name": "bin_widths_conv_bias",
      "trainable": false,
      "used_in_forward": true
    }
  ]
}
```

## Weight and State Mapping
- Copied `hist_feature.bin_centers_conv.weight` to the TensorFlow feature-to-bin weight vector.
- Copied `hist_feature.bin_centers_conv.bias` to the TensorFlow feature-to-bin bias vector.
- Copied `hist_feature.bin_widths_conv.weight` to the TensorFlow per-bin width weight vector.
- Copied `hist_feature.bin_widths_conv.bias` to the TensorFlow per-bin width bias vector.
- Copied `hist_feature.centers` for state completeness, but PyTorch `forward` does not reference it.
- `hist_feature.widths` aliases `bin_widths_conv.weight` in PyTorch; the TensorFlow layer aliases the same variable.

## Parity Results
| Test | Status | Max abs diff | Mean abs diff | PyTorch shape | TensorFlow shape |
| --- | --- | --- | --- | --- | --- |
| random expected-shape input | pass_preferred | 5.96046448e-08 | 7.58855423e-10 | [1, 216, 1] | [1, 216, 1] |
| zeros expected-shape input | pass_preferred | 2.98023224e-08 | 1.41423062e-09 | [1, 216, 1] | [1, 216, 1] |
| ones expected-shape input | pass_preferred | 0 | 0 | [1, 216, 1] | [1, 216, 1] |
| captured real pre-SoftHistogram tensor | pass_preferred | 5.96046448e-08 | 1.23314003e-09 | [1, 216, 1] | [1, 216, 1] |

## Captured Real Tensor
- Performed: True
- Image path: `/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/juiedesai19391601268.jpg`

## Unresolved Issues
- none

## Stage E Local Attention / Block Parity Decision
- Safe to proceed to Stage E local attention / block parity: True
- Scope of this decision: SoftHistogram parity only. This does not establish full TensorFlow DAT feasibility.

## Warnings
- stage_d_soft_histogram_parity.py:116: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
- __init__.py:49: Importing from timm.models.layers is deprecated, please import via timm.layers
- functional.py:534: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3595.)

## Runtime Log Notices
- none

## Errors
- none
