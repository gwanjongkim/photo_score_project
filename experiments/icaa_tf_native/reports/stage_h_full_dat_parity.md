# Stage H Full DAT Parity

## Scope
- Exact command used: `python experiments/icaa_tf_native/scripts/stage_h_full_dat_parity.py`
- Checkpoint path: `/home/omen_pc1/photo_score_project/weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth`
- PyTorch reference model path: `/home/omen_pc1/photo_score_project/experiments/icaa_export_safe_v3/export_safe_models`
- Output directory: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_stage_h_full_dat_20260516_040212`
- Status: `ok`
- Seeds: torch=123, numpy=123, tensorflow=123
- SavedModel export: not performed
- TFLite conversion: not performed
- Flutter changes: none

## PyTorch Forward Order Confirmed
1. `patch_proj`
2. `stages[0]`
3. `down_projs[0]`
4. `stages[1]`
5. `down_projs[1]`
6. `stages[2]`
7. `down_projs[2]`
8. `stages[3]`
9. `cls_norm`
10. `reshape NCHW to [B, C, H*W] and mean over spatial axis`
11. `hst_head`
12. `hist_feature`
13. `rearrange 'b p w -> w p b'`
14. `squeeze dim 2`
15. `class_head + sigmoid`
16. `class_head2 + sigmoid`
17. `concat [MOS, color] to [B, 2]`

## TensorFlow Full DAT Structure Summary
```json
{
  "class_name": "TFICAA17KDAT",
  "input_layout": "NHWC/channels_last [B, 224, 224, 3]",
  "inference_only": true,
  "forward_order": [
    "patch_proj",
    "stage0",
    "down_proj0",
    "stage1",
    "down_proj1",
    "stage2",
    "down_proj2",
    "stage3",
    "cls_norm",
    "spatial mean over NHWC H/W",
    "hst_head",
    "TFSoftHistogram",
    "transpose/squeeze histogram to [B, 216]",
    "sigmoid(class_head)",
    "sigmoid(class_head2)",
    "concat [mos, color] -> [B, 2]"
  ],
  "components": {
    "patch_proj": {
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
    },
    "stage0": {
      "class_name": "TFTransformerStage0",
      "target_pytorch_module": "stages.0",
      "input_layout": "NHWC/channels_last",
      "proj": "Identity; no projection weights are used for stage 0",
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
          "drop_path_eval": "identity"
        },
        {
          "index": 1,
          "attention": "TFShiftWindowAttention mapped from PyTorch ShiftWindowAttention",
          "shift_size": 4,
          "attn_mask": "direct copy from stages.0.attns.1.attn_mask",
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
    },
    "down_proj0": {
      "class_name": "TFDownProjection",
      "conv": {
        "type": "tf.keras.layers.Conv2D",
        "filters": 256,
        "kernel_size": [
          2,
          2
        ],
        "strides": [
          2,
          2
        ],
        "padding": "valid",
        "use_bias": false
      },
      "norm": {
        "type": "tf.keras.layers.LayerNormalization",
        "axis": -1,
        "epsilon": 1e-05
      }
    },
    "stage1": {
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
    },
    "down_proj1": {
      "class_name": "TFDownProjection",
      "conv": {
        "type": "tf.keras.layers.Conv2D",
        "filters": 512,
        "kernel_size": [
          2,
          2
        ],
        "strides": [
          2,
          2
        ],
        "padding": "valid",
        "use_bias": false
      },
      "norm": {
        "type": "tf.keras.layers.LayerNormalization",
        "axis": -1,
        "epsilon": 1e-05
      }
    },
    "stage2": {
      "class_name": "TFTransformerStage2",
      "target_pytorch_module": "stages.2",
      "input_layout": "NHWC/channels_last",
      "proj": "Identity; no projection weights are used for stage 2",
      "depth": 18,
      "stage_spec": [
        "L",
        "D",
        "L",
        "D",
        "L",
        "D",
        "L",
        "D",
        "L",
        "D",
        "L",
        "D",
        "L",
        "D",
        "L",
        "D",
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
        },
        {
          "index": 2,
          "attention_type": "L",
          "attention": "TFLocalAttention mapped from PyTorch LocalAttention",
          "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
          "layer_norms": [
            "layer_norm_4",
            "layer_norm_5"
          ],
          "drop_path_eval": "identity in eval mode"
        },
        {
          "index": 3,
          "attention_type": "D",
          "attention": "TFDAttentionBaseline mapped from PyTorch DAttentionBaseline",
          "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
          "layer_norms": [
            "layer_norm_6",
            "layer_norm_7"
          ],
          "drop_path_eval": "identity in eval mode"
        },
        {
          "index": 4,
          "attention_type": "L",
          "attention": "TFLocalAttention mapped from PyTorch LocalAttention",
          "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
          "layer_norms": [
            "layer_norm_8",
            "layer_norm_9"
          ],
          "drop_path_eval": "identity in eval mode"
        },
        {
          "index": 5,
          "attention_type": "D",
          "attention": "TFDAttentionBaseline mapped from PyTorch DAttentionBaseline",
          "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
          "layer_norms": [
            "layer_norm_10",
            "layer_norm_11"
          ],
          "drop_path_eval": "identity in eval mode"
        },
        {
          "index": 6,
          "attention_type": "L",
          "attention": "TFLocalAttention mapped from PyTorch LocalAttention",
          "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
          "layer_norms": [
            "layer_norm_12",
            "layer_norm_13"
          ],
          "drop_path_eval": "identity in eval mode"
        },
        {
          "index": 7,
          "attention_type": "D",
          "attention": "TFDAttentionBaseline mapped from PyTorch DAttentionBaseline",
          "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
          "layer_norms": [
            "layer_norm_14",
            "layer_norm_15"
          ],
          "drop_path_eval": "identity in eval mode"
        },
        {
          "index": 8,
          "attention_type": "L",
          "attention": "TFLocalAttention mapped from PyTorch LocalAttention",
          "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
          "layer_norms": [
            "layer_norm_16",
            "layer_norm_17"
          ],
          "drop_path_eval": "identity in eval mode"
        },
        {
          "index": 9,
          "attention_type": "D",
          "attention": "TFDAttentionBaseline mapped from PyTorch DAttentionBaseline",
          "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
          "layer_norms": [
            "layer_norm_18",
            "layer_norm_19"
          ],
          "drop_path_eval": "identity in eval mode"
        },
        {
          "index": 10,
          "attention_type": "L",
          "attention": "TFLocalAttention mapped from PyTorch LocalAttention",
          "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
          "layer_norms": [
            "layer_norm_20",
            "layer_norm_21"
          ],
          "drop_path_eval": "identity in eval mode"
        },
        {
          "index": 11,
          "attention_type": "D",
          "attention": "TFDAttentionBaseline mapped from PyTorch DAttentionBaseline",
          "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
          "layer_norms": [
            "layer_norm_22",
            "layer_norm_23"
          ],
          "drop_path_eval": "identity in eval mode"
        },
        {
          "index": 12,
          "attention_type": "L",
          "attention": "TFLocalAttention mapped from PyTorch LocalAttention",
          "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
          "layer_norms": [
            "layer_norm_24",
            "layer_norm_25"
          ],
          "drop_path_eval": "identity in eval mode"
        },
        {
          "index": 13,
          "attention_type": "D",
          "attention": "TFDAttentionBaseline mapped from PyTorch DAttentionBaseline",
          "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
          "layer_norms": [
            "layer_norm_26",
            "layer_norm_27"
          ],
          "drop_path_eval": "identity in eval mode"
        },
        {
          "index": 14,
          "attention_type": "L",
          "attention": "TFLocalAttention mapped from PyTorch LocalAttention",
          "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
          "layer_norms": [
            "layer_norm_28",
            "layer_norm_29"
          ],
          "drop_path_eval": "identity in eval mode"
        },
        {
          "index": 15,
          "attention_type": "D",
          "attention": "TFDAttentionBaseline mapped from PyTorch DAttentionBaseline",
          "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
          "layer_norms": [
            "layer_norm_30",
            "layer_norm_31"
          ],
          "drop_path_eval": "identity in eval mode"
        },
        {
          "index": 16,
          "attention_type": "L",
          "attention": "TFLocalAttention mapped from PyTorch LocalAttention",
          "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
          "layer_norms": [
            "layer_norm_32",
            "layer_norm_33"
          ],
          "drop_path_eval": "identity in eval mode"
        },
        {
          "index": 17,
          "attention_type": "D",
          "attention": "TFDAttentionBaseline mapped from PyTorch DAttentionBaseline",
          "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
          "layer_norms": [
            "layer_norm_34",
            "layer_norm_35"
          ],
          "drop_path_eval": "identity in eval mode"
        }
      ],
      "deformable_attention_config": {
        "q_size": [
          14,
          14
        ],
        "kv_size": [
          14,
          14
        ],
        "heads": 16,
        "head_channels": 32,
        "groups": 4,
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
    },
    "down_proj2": {
      "class_name": "TFDownProjection",
      "conv": {
        "type": "tf.keras.layers.Conv2D",
        "filters": 1024,
        "kernel_size": [
          2,
          2
        ],
        "strides": [
          2,
          2
        ],
        "padding": "valid",
        "use_bias": false
      },
      "norm": {
        "type": "tf.keras.layers.LayerNormalization",
        "axis": -1,
        "epsilon": 1e-05
      }
    },
    "stage3": {
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
    },
    "soft_histogram": {
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
  },
  "heads": {
    "cls_norm": "LayerNormalization(axis=-1, epsilon=1e-5)",
    "hst_head": "Dense(36), mapped from PyTorch hst_head",
    "class_head": "Dense(1) + sigmoid, mapped from PyTorch class_head",
    "class_head2": "Dense(1) + sigmoid, mapped from PyTorch class_head2",
    "cls_head": "Present in PyTorch state_dict but unused by DAT.forward and intentionally not used in TF forward."
  }
}
```

## Checkpoint Groups Mapped
- `patch_proj`: 4 tensors
- `stages.0`: 29 tensors
- `down_projs.0`: 3 tensors
- `stages.1`: 29 tensors
- `down_projs.1`: 3 tensors
- `stages.2`: 360 tensors
- `down_projs.2`: 3 tensors
- `stages.3`: 40 tensors
- `cls_norm`: 2 tensors
- `hst_head`: 2 tensors
- `hist_feature`: 6 tensors
- `class_head`: 2 tensors
- `class_head2`: 2 tensors

## Final Output Parity
| Case | Status | B | Full max | Full mean | MOS max | MOS mean | Color max | Color mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deterministic random normalized image input | pass_preferred | 1 | 5.96046448e-08 | 2.98023224e-08 | 0 | 0 | 5.96046448e-08 | 5.96046448e-08 |
| 16 real ICAA17K test images | pass_preferred | 16 | 8.94069672e-07 | 1.44354999e-07 | 8.94069672e-07 | 1.50874257e-07 | 5.36441803e-07 | 1.37835741e-07 |
| 64 real ICAA17K test images | pass_preferred | 64 | 0.000402867794 | 4.35148831e-06 | 0.000402867794 | 7.25523569e-06 | 6.08563423e-05 | 1.44774094e-06 |

## Intermediate Drift Summary
| Case | Largest intermediate tensor diff | Max abs diff | Mean abs diff |
| --- | --- | --- | --- |
| deterministic random normalized image input | stage2 | 0.0294189453 | 5.02577786e-05 |
| 16 real ICAA17K test images | stage2 | 0.0688476562 | 6.82425525e-05 |
| 64 real ICAA17K test images | stage3 | 19.1984253 | 0.000217693072 |

## Real Image Inputs
- Requested count: 16
- Performed count: 16
- Optional 64-image run performed: True

## Stage G-2/G-3 Drift Impact
- Significant final-score impact detected: False
- Interpretation: Accumulated Stage G-2/G-3 feature drift did not materially affect final scores under preferred thresholds.

## Decisions
- Safe to proceed to Stage I TensorFlow SavedModel export: True
- Safe to proceed to TFLite conversion later: True

## Unresolved Issues
- none

## Artifacts
- JSON report: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_stage_h_full_dat_20260516_040212/stage_h_full_dat_report.json`
- Predictions CSV: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_stage_h_full_dat_20260516_040212/stage_h_full_dat_predictions.csv`
- Log: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_stage_h_full_dat_20260516_040212/stage_h_full_dat_log.txt`
- Intermediate diff summary: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_stage_h_full_dat_20260516_040212/optional_intermediate_diff_summary.json`

## Warnings
- stage_h_full_dat_parity.py:132: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
- __init__.py:49: Importing from timm.models.layers is deprecated, please import via timm.layers
- functional.py:534: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3595.)
- layer.py:424: `build()` was called on layer 'stage2', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
- layer.py:424: `build()` was called on layer 'stage3', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.

## Runtime Log Notices
- none

## Errors
- none
