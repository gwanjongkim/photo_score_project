# Stage G-2 TransformerStage 2 Parity

## Scope
- Exact command used: `python experiments/icaa_tf_native/scripts/stage_g2_transformer_stage2_parity.py`
- Checkpoint path: `/home/omen_pc1/photo_score_project/weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth`
- PyTorch reference model path: `/home/omen_pc1/photo_score_project/experiments/icaa_export_safe_v3/export_safe_models`
- Output directory: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_stage_g2_20260516_024326`
- Status: `parity_failed`
- Seeds: torch=123, numpy=123, tensorflow=123

## PyTorch Classes Inspected
```json
{
  "class_name": "TransformerStage",
  "stage_index": 2,
  "stage_input_shape_nchw": [
    null,
    512,
    14,
    14
  ],
  "stage_output_shape_nchw": [
    null,
    512,
    14,
    14
  ],
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
      "heads": 16,
      "scale": 0.1767766952966369
    },
    {
      "index": 1,
      "class_name": "DAttentionBaseline",
      "attention_type": "D",
      "window_size": null,
      "heads": 16,
      "scale": 0.1767766952966369
    },
    {
      "index": 2,
      "class_name": "LocalAttention",
      "attention_type": "L",
      "window_size": [
        7,
        7
      ],
      "heads": 16,
      "scale": 0.1767766952966369
    },
    {
      "index": 3,
      "class_name": "DAttentionBaseline",
      "attention_type": "D",
      "window_size": null,
      "heads": 16,
      "scale": 0.1767766952966369
    },
    {
      "index": 4,
      "class_name": "LocalAttention",
      "attention_type": "L",
      "window_size": [
        7,
        7
      ],
      "heads": 16,
      "scale": 0.1767766952966369
    },
    {
      "index": 5,
      "class_name": "DAttentionBaseline",
      "attention_type": "D",
      "window_size": null,
      "heads": 16,
      "scale": 0.1767766952966369
    },
    {
      "index": 6,
      "class_name": "LocalAttention",
      "attention_type": "L",
      "window_size": [
        7,
        7
      ],
      "heads": 16,
      "scale": 0.1767766952966369
    },
    {
      "index": 7,
      "class_name": "DAttentionBaseline",
      "attention_type": "D",
      "window_size": null,
      "heads": 16,
      "scale": 0.1767766952966369
    },
    {
      "index": 8,
      "class_name": "LocalAttention",
      "attention_type": "L",
      "window_size": [
        7,
        7
      ],
      "heads": 16,
      "scale": 0.1767766952966369
    },
    {
      "index": 9,
      "class_name": "DAttentionBaseline",
      "attention_type": "D",
      "window_size": null,
      "heads": 16,
      "scale": 0.1767766952966369
    },
    {
      "index": 10,
      "class_name": "LocalAttention",
      "attention_type": "L",
      "window_size": [
        7,
        7
      ],
      "heads": 16,
      "scale": 0.1767766952966369
    },
    {
      "index": 11,
      "class_name": "DAttentionBaseline",
      "attention_type": "D",
      "window_size": null,
      "heads": 16,
      "scale": 0.1767766952966369
    },
    {
      "index": 12,
      "class_name": "LocalAttention",
      "attention_type": "L",
      "window_size": [
        7,
        7
      ],
      "heads": 16,
      "scale": 0.1767766952966369
    },
    {
      "index": 13,
      "class_name": "DAttentionBaseline",
      "attention_type": "D",
      "window_size": null,
      "heads": 16,
      "scale": 0.1767766952966369
    },
    {
      "index": 14,
      "class_name": "LocalAttention",
      "attention_type": "L",
      "window_size": [
        7,
        7
      ],
      "heads": 16,
      "scale": 0.1767766952966369
    },
    {
      "index": 15,
      "class_name": "DAttentionBaseline",
      "attention_type": "D",
      "window_size": null,
      "heads": 16,
      "scale": 0.1767766952966369
    },
    {
      "index": 16,
      "class_name": "LocalAttention",
      "attention_type": "L",
      "window_size": [
        7,
        7
      ],
      "heads": 16,
      "scale": 0.1767766952966369
    },
    {
      "index": 17,
      "class_name": "DAttentionBaseline",
      "attention_type": "D",
      "window_size": null,
      "heads": 16,
      "scale": 0.1767766952966369
    }
  ],
  "deformable_attention_details": {
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
    "offset_range_factor": 2,
    "use_pe": true,
    "dwc_pe": false,
    "fixed_pe": false,
    "no_off": false,
    "rpe_table_shape": [
      16,
      27,
      27
    ]
  },
  "mlp_blocks": [
    {
      "index": 0,
      "class_name": "TransformerMLP",
      "hidden_dim": 2048
    },
    {
      "index": 1,
      "class_name": "TransformerMLP",
      "hidden_dim": 2048
    },
    {
      "index": 2,
      "class_name": "TransformerMLP",
      "hidden_dim": 2048
    },
    {
      "index": 3,
      "class_name": "TransformerMLP",
      "hidden_dim": 2048
    },
    {
      "index": 4,
      "class_name": "TransformerMLP",
      "hidden_dim": 2048
    },
    {
      "index": 5,
      "class_name": "TransformerMLP",
      "hidden_dim": 2048
    },
    {
      "index": 6,
      "class_name": "TransformerMLP",
      "hidden_dim": 2048
    },
    {
      "index": 7,
      "class_name": "TransformerMLP",
      "hidden_dim": 2048
    },
    {
      "index": 8,
      "class_name": "TransformerMLP",
      "hidden_dim": 2048
    },
    {
      "index": 9,
      "class_name": "TransformerMLP",
      "hidden_dim": 2048
    },
    {
      "index": 10,
      "class_name": "TransformerMLP",
      "hidden_dim": 2048
    },
    {
      "index": 11,
      "class_name": "TransformerMLP",
      "hidden_dim": 2048
    },
    {
      "index": 12,
      "class_name": "TransformerMLP",
      "hidden_dim": 2048
    },
    {
      "index": 13,
      "class_name": "TransformerMLP",
      "hidden_dim": 2048
    },
    {
      "index": 14,
      "class_name": "TransformerMLP",
      "hidden_dim": 2048
    },
    {
      "index": 15,
      "class_name": "TransformerMLP",
      "hidden_dim": 2048
    },
    {
      "index": 16,
      "class_name": "TransformerMLP",
      "hidden_dim": 2048
    },
    {
      "index": 17,
      "class_name": "TransformerMLP",
      "hidden_dim": 2048
    }
  ],
  "layer_norms": [
    {
      "index": 0,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 1,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 2,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 3,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 4,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 5,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 6,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 7,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 8,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 9,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 10,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 11,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 12,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 13,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 14,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 15,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 16,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 17,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 18,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 19,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 20,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 21,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 22,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 23,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 24,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 25,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 26,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 27,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 28,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 29,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 30,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 31,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 32,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 33,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 34,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    },
    {
      "index": 35,
      "class_name": "LayerNormProxy",
      "inner_class_name": "LayerNorm",
      "normalized_shape": [
        512
      ],
      "eps": 1e-05
    }
  ],
  "drop_path": [
    {
      "index": 0,
      "class_name": "DropPath",
      "drop_prob": 0.08695652335882187
    },
    {
      "index": 1,
      "class_name": "DropPath",
      "drop_prob": 0.10869565606117249
    },
    {
      "index": 2,
      "class_name": "DropPath",
      "drop_prob": 0.1304347813129425
    },
    {
      "index": 3,
      "class_name": "DropPath",
      "drop_prob": 0.15217392146587372
    },
    {
      "index": 4,
      "class_name": "DropPath",
      "drop_prob": 0.17391304671764374
    },
    {
      "index": 5,
      "class_name": "DropPath",
      "drop_prob": 0.19565217196941376
    },
    {
      "index": 6,
      "class_name": "DropPath",
      "drop_prob": 0.21739131212234497
    },
    {
      "index": 7,
      "class_name": "DropPath",
      "drop_prob": 0.239130437374115
    },
    {
      "index": 8,
      "class_name": "DropPath",
      "drop_prob": 0.260869562625885
    },
    {
      "index": 9,
      "class_name": "DropPath",
      "drop_prob": 0.28260868787765503
    },
    {
      "index": 10,
      "class_name": "DropPath",
      "drop_prob": 0.30434781312942505
    },
    {
      "index": 11,
      "class_name": "DropPath",
      "drop_prob": 0.32608693838119507
    },
    {
      "index": 12,
      "class_name": "DropPath",
      "drop_prob": 0.3478260934352875
    },
    {
      "index": 13,
      "class_name": "DropPath",
      "drop_prob": 0.3695652186870575
    },
    {
      "index": 14,
      "class_name": "DropPath",
      "drop_prob": 0.3913043439388275
    },
    {
      "index": 15,
      "class_name": "DropPath",
      "drop_prob": 0.41304346919059753
    },
    {
      "index": 16,
      "class_name": "DropPath",
      "drop_prob": 0.43478259444236755
    },
    {
      "index": 17,
      "class_name": "DropPath",
      "drop_prob": 0.45652174949645996
    }
  ],
  "drop_path_eval_behavior": "All stage-2 drop_path modules are DropPath with nonzero configured probabilities, but eval mode disables them and makes them identity.",
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
    "is_part_of_stage2": false,
    "dat_forward_order": "DAT.forward calls model.stages[2](x), then applies model.down_projs[2](x) outside the stage when i < 3.",
    "down_proj2_class": "Sequential"
  }
}
```

## Checkpoint Tensors Used
- `stages.2.layer_norms.0.norm.weight`
- `stages.2.layer_norms.0.norm.bias`
- `stages.2.layer_norms.1.norm.weight`
- `stages.2.layer_norms.1.norm.bias`
- `stages.2.layer_norms.2.norm.weight`
- `stages.2.layer_norms.2.norm.bias`
- `stages.2.layer_norms.3.norm.weight`
- `stages.2.layer_norms.3.norm.bias`
- `stages.2.layer_norms.4.norm.weight`
- `stages.2.layer_norms.4.norm.bias`
- `stages.2.layer_norms.5.norm.weight`
- `stages.2.layer_norms.5.norm.bias`
- `stages.2.layer_norms.6.norm.weight`
- `stages.2.layer_norms.6.norm.bias`
- `stages.2.layer_norms.7.norm.weight`
- `stages.2.layer_norms.7.norm.bias`
- `stages.2.layer_norms.8.norm.weight`
- `stages.2.layer_norms.8.norm.bias`
- `stages.2.layer_norms.9.norm.weight`
- `stages.2.layer_norms.9.norm.bias`
- `stages.2.layer_norms.10.norm.weight`
- `stages.2.layer_norms.10.norm.bias`
- `stages.2.layer_norms.11.norm.weight`
- `stages.2.layer_norms.11.norm.bias`
- `stages.2.layer_norms.12.norm.weight`
- `stages.2.layer_norms.12.norm.bias`
- `stages.2.layer_norms.13.norm.weight`
- `stages.2.layer_norms.13.norm.bias`
- `stages.2.layer_norms.14.norm.weight`
- `stages.2.layer_norms.14.norm.bias`
- `stages.2.layer_norms.15.norm.weight`
- `stages.2.layer_norms.15.norm.bias`
- `stages.2.layer_norms.16.norm.weight`
- `stages.2.layer_norms.16.norm.bias`
- `stages.2.layer_norms.17.norm.weight`
- `stages.2.layer_norms.17.norm.bias`
- `stages.2.layer_norms.18.norm.weight`
- `stages.2.layer_norms.18.norm.bias`
- `stages.2.layer_norms.19.norm.weight`
- `stages.2.layer_norms.19.norm.bias`
- `stages.2.layer_norms.20.norm.weight`
- `stages.2.layer_norms.20.norm.bias`
- `stages.2.layer_norms.21.norm.weight`
- `stages.2.layer_norms.21.norm.bias`
- `stages.2.layer_norms.22.norm.weight`
- `stages.2.layer_norms.22.norm.bias`
- `stages.2.layer_norms.23.norm.weight`
- `stages.2.layer_norms.23.norm.bias`
- `stages.2.layer_norms.24.norm.weight`
- `stages.2.layer_norms.24.norm.bias`
- `stages.2.layer_norms.25.norm.weight`
- `stages.2.layer_norms.25.norm.bias`
- `stages.2.layer_norms.26.norm.weight`
- `stages.2.layer_norms.26.norm.bias`
- `stages.2.layer_norms.27.norm.weight`
- `stages.2.layer_norms.27.norm.bias`
- `stages.2.layer_norms.28.norm.weight`
- `stages.2.layer_norms.28.norm.bias`
- `stages.2.layer_norms.29.norm.weight`
- `stages.2.layer_norms.29.norm.bias`
- `stages.2.layer_norms.30.norm.weight`
- `stages.2.layer_norms.30.norm.bias`
- `stages.2.layer_norms.31.norm.weight`
- `stages.2.layer_norms.31.norm.bias`
- `stages.2.layer_norms.32.norm.weight`
- `stages.2.layer_norms.32.norm.bias`
- `stages.2.layer_norms.33.norm.weight`
- `stages.2.layer_norms.33.norm.bias`
- `stages.2.layer_norms.34.norm.weight`
- `stages.2.layer_norms.34.norm.bias`
- `stages.2.layer_norms.35.norm.weight`
- `stages.2.layer_norms.35.norm.bias`
- `stages.2.mlps.0.chunk.linear1.weight`
- `stages.2.mlps.0.chunk.linear1.bias`
- `stages.2.mlps.0.chunk.linear2.weight`
- `stages.2.mlps.0.chunk.linear2.bias`
- `stages.2.mlps.1.chunk.linear1.weight`
- `stages.2.mlps.1.chunk.linear1.bias`
- `stages.2.mlps.1.chunk.linear2.weight`
- `stages.2.mlps.1.chunk.linear2.bias`
- `stages.2.mlps.2.chunk.linear1.weight`
- `stages.2.mlps.2.chunk.linear1.bias`
- `stages.2.mlps.2.chunk.linear2.weight`
- `stages.2.mlps.2.chunk.linear2.bias`
- `stages.2.mlps.3.chunk.linear1.weight`
- `stages.2.mlps.3.chunk.linear1.bias`
- `stages.2.mlps.3.chunk.linear2.weight`
- `stages.2.mlps.3.chunk.linear2.bias`
- `stages.2.mlps.4.chunk.linear1.weight`
- `stages.2.mlps.4.chunk.linear1.bias`
- `stages.2.mlps.4.chunk.linear2.weight`
- `stages.2.mlps.4.chunk.linear2.bias`
- `stages.2.mlps.5.chunk.linear1.weight`
- `stages.2.mlps.5.chunk.linear1.bias`
- `stages.2.mlps.5.chunk.linear2.weight`
- `stages.2.mlps.5.chunk.linear2.bias`
- `stages.2.mlps.6.chunk.linear1.weight`
- `stages.2.mlps.6.chunk.linear1.bias`
- `stages.2.mlps.6.chunk.linear2.weight`
- `stages.2.mlps.6.chunk.linear2.bias`
- `stages.2.mlps.7.chunk.linear1.weight`
- `stages.2.mlps.7.chunk.linear1.bias`
- `stages.2.mlps.7.chunk.linear2.weight`
- `stages.2.mlps.7.chunk.linear2.bias`
- `stages.2.mlps.8.chunk.linear1.weight`
- `stages.2.mlps.8.chunk.linear1.bias`
- `stages.2.mlps.8.chunk.linear2.weight`
- `stages.2.mlps.8.chunk.linear2.bias`
- `stages.2.mlps.9.chunk.linear1.weight`
- `stages.2.mlps.9.chunk.linear1.bias`
- `stages.2.mlps.9.chunk.linear2.weight`
- `stages.2.mlps.9.chunk.linear2.bias`
- `stages.2.mlps.10.chunk.linear1.weight`
- `stages.2.mlps.10.chunk.linear1.bias`
- `stages.2.mlps.10.chunk.linear2.weight`
- `stages.2.mlps.10.chunk.linear2.bias`
- `stages.2.mlps.11.chunk.linear1.weight`
- `stages.2.mlps.11.chunk.linear1.bias`
- `stages.2.mlps.11.chunk.linear2.weight`
- `stages.2.mlps.11.chunk.linear2.bias`
- `stages.2.mlps.12.chunk.linear1.weight`
- `stages.2.mlps.12.chunk.linear1.bias`
- `stages.2.mlps.12.chunk.linear2.weight`
- `stages.2.mlps.12.chunk.linear2.bias`
- `stages.2.mlps.13.chunk.linear1.weight`
- `stages.2.mlps.13.chunk.linear1.bias`
- `stages.2.mlps.13.chunk.linear2.weight`
- `stages.2.mlps.13.chunk.linear2.bias`
- `stages.2.mlps.14.chunk.linear1.weight`
- `stages.2.mlps.14.chunk.linear1.bias`
- `stages.2.mlps.14.chunk.linear2.weight`
- `stages.2.mlps.14.chunk.linear2.bias`
- `stages.2.mlps.15.chunk.linear1.weight`
- `stages.2.mlps.15.chunk.linear1.bias`
- `stages.2.mlps.15.chunk.linear2.weight`
- `stages.2.mlps.15.chunk.linear2.bias`
- `stages.2.mlps.16.chunk.linear1.weight`
- `stages.2.mlps.16.chunk.linear1.bias`
- `stages.2.mlps.16.chunk.linear2.weight`
- `stages.2.mlps.16.chunk.linear2.bias`
- `stages.2.mlps.17.chunk.linear1.weight`
- `stages.2.mlps.17.chunk.linear1.bias`
- `stages.2.mlps.17.chunk.linear2.weight`
- `stages.2.mlps.17.chunk.linear2.bias`
- `stages.2.attns.0.relative_position_bias_table`
- `stages.2.attns.0.relative_position_index`
- `stages.2.attns.0.proj_qkv.weight`
- `stages.2.attns.0.proj_qkv.bias`
- `stages.2.attns.0.proj_out.weight`
- `stages.2.attns.0.proj_out.bias`
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
- `stages.2.attns.2.relative_position_bias_table`
- `stages.2.attns.2.relative_position_index`
- `stages.2.attns.2.proj_qkv.weight`
- `stages.2.attns.2.proj_qkv.bias`
- `stages.2.attns.2.proj_out.weight`
- `stages.2.attns.2.proj_out.bias`
- `stages.2.attns.3.rpe_table`
- `stages.2.attns.3.conv_offset.0.weight`
- `stages.2.attns.3.conv_offset.0.bias`
- `stages.2.attns.3.conv_offset.1.norm.weight`
- `stages.2.attns.3.conv_offset.1.norm.bias`
- `stages.2.attns.3.conv_offset.3.weight`
- `stages.2.attns.3.proj_q.weight`
- `stages.2.attns.3.proj_q.bias`
- `stages.2.attns.3.proj_k.weight`
- `stages.2.attns.3.proj_k.bias`
- `stages.2.attns.3.proj_v.weight`
- `stages.2.attns.3.proj_v.bias`
- `stages.2.attns.3.proj_out.weight`
- `stages.2.attns.3.proj_out.bias`
- `stages.2.attns.3.ref_point14.weight`
- `stages.2.attns.3.ref_point14.bias`
- `stages.2.attns.3.ref_gate.weight`
- `stages.2.attns.3.ref_gate.bias`
- `stages.2.attns.4.relative_position_bias_table`
- `stages.2.attns.4.relative_position_index`
- `stages.2.attns.4.proj_qkv.weight`
- `stages.2.attns.4.proj_qkv.bias`
- `stages.2.attns.4.proj_out.weight`
- `stages.2.attns.4.proj_out.bias`
- `stages.2.attns.5.rpe_table`
- `stages.2.attns.5.conv_offset.0.weight`
- `stages.2.attns.5.conv_offset.0.bias`
- `stages.2.attns.5.conv_offset.1.norm.weight`
- `stages.2.attns.5.conv_offset.1.norm.bias`
- `stages.2.attns.5.conv_offset.3.weight`
- `stages.2.attns.5.proj_q.weight`
- `stages.2.attns.5.proj_q.bias`
- `stages.2.attns.5.proj_k.weight`
- `stages.2.attns.5.proj_k.bias`
- `stages.2.attns.5.proj_v.weight`
- `stages.2.attns.5.proj_v.bias`
- `stages.2.attns.5.proj_out.weight`
- `stages.2.attns.5.proj_out.bias`
- `stages.2.attns.5.ref_point14.weight`
- `stages.2.attns.5.ref_point14.bias`
- `stages.2.attns.5.ref_gate.weight`
- `stages.2.attns.5.ref_gate.bias`
- `stages.2.attns.6.relative_position_bias_table`
- `stages.2.attns.6.relative_position_index`
- `stages.2.attns.6.proj_qkv.weight`
- `stages.2.attns.6.proj_qkv.bias`
- `stages.2.attns.6.proj_out.weight`
- `stages.2.attns.6.proj_out.bias`
- `stages.2.attns.7.rpe_table`
- `stages.2.attns.7.conv_offset.0.weight`
- `stages.2.attns.7.conv_offset.0.bias`
- `stages.2.attns.7.conv_offset.1.norm.weight`
- `stages.2.attns.7.conv_offset.1.norm.bias`
- `stages.2.attns.7.conv_offset.3.weight`
- `stages.2.attns.7.proj_q.weight`
- `stages.2.attns.7.proj_q.bias`
- `stages.2.attns.7.proj_k.weight`
- `stages.2.attns.7.proj_k.bias`
- `stages.2.attns.7.proj_v.weight`
- `stages.2.attns.7.proj_v.bias`
- `stages.2.attns.7.proj_out.weight`
- `stages.2.attns.7.proj_out.bias`
- `stages.2.attns.7.ref_point14.weight`
- `stages.2.attns.7.ref_point14.bias`
- `stages.2.attns.7.ref_gate.weight`
- `stages.2.attns.7.ref_gate.bias`
- `stages.2.attns.8.relative_position_bias_table`
- `stages.2.attns.8.relative_position_index`
- `stages.2.attns.8.proj_qkv.weight`
- `stages.2.attns.8.proj_qkv.bias`
- `stages.2.attns.8.proj_out.weight`
- `stages.2.attns.8.proj_out.bias`
- `stages.2.attns.9.rpe_table`
- `stages.2.attns.9.conv_offset.0.weight`
- `stages.2.attns.9.conv_offset.0.bias`
- `stages.2.attns.9.conv_offset.1.norm.weight`
- `stages.2.attns.9.conv_offset.1.norm.bias`
- `stages.2.attns.9.conv_offset.3.weight`
- `stages.2.attns.9.proj_q.weight`
- `stages.2.attns.9.proj_q.bias`
- `stages.2.attns.9.proj_k.weight`
- `stages.2.attns.9.proj_k.bias`
- `stages.2.attns.9.proj_v.weight`
- `stages.2.attns.9.proj_v.bias`
- `stages.2.attns.9.proj_out.weight`
- `stages.2.attns.9.proj_out.bias`
- `stages.2.attns.9.ref_point14.weight`
- `stages.2.attns.9.ref_point14.bias`
- `stages.2.attns.9.ref_gate.weight`
- `stages.2.attns.9.ref_gate.bias`
- `stages.2.attns.10.relative_position_bias_table`
- `stages.2.attns.10.relative_position_index`
- `stages.2.attns.10.proj_qkv.weight`
- `stages.2.attns.10.proj_qkv.bias`
- `stages.2.attns.10.proj_out.weight`
- `stages.2.attns.10.proj_out.bias`
- `stages.2.attns.11.rpe_table`
- `stages.2.attns.11.conv_offset.0.weight`
- `stages.2.attns.11.conv_offset.0.bias`
- `stages.2.attns.11.conv_offset.1.norm.weight`
- `stages.2.attns.11.conv_offset.1.norm.bias`
- `stages.2.attns.11.conv_offset.3.weight`
- `stages.2.attns.11.proj_q.weight`
- `stages.2.attns.11.proj_q.bias`
- `stages.2.attns.11.proj_k.weight`
- `stages.2.attns.11.proj_k.bias`
- `stages.2.attns.11.proj_v.weight`
- `stages.2.attns.11.proj_v.bias`
- `stages.2.attns.11.proj_out.weight`
- `stages.2.attns.11.proj_out.bias`
- `stages.2.attns.11.ref_point14.weight`
- `stages.2.attns.11.ref_point14.bias`
- `stages.2.attns.11.ref_gate.weight`
- `stages.2.attns.11.ref_gate.bias`
- `stages.2.attns.12.relative_position_bias_table`
- `stages.2.attns.12.relative_position_index`
- `stages.2.attns.12.proj_qkv.weight`
- `stages.2.attns.12.proj_qkv.bias`
- `stages.2.attns.12.proj_out.weight`
- `stages.2.attns.12.proj_out.bias`
- `stages.2.attns.13.rpe_table`
- `stages.2.attns.13.conv_offset.0.weight`
- `stages.2.attns.13.conv_offset.0.bias`
- `stages.2.attns.13.conv_offset.1.norm.weight`
- `stages.2.attns.13.conv_offset.1.norm.bias`
- `stages.2.attns.13.conv_offset.3.weight`
- `stages.2.attns.13.proj_q.weight`
- `stages.2.attns.13.proj_q.bias`
- `stages.2.attns.13.proj_k.weight`
- `stages.2.attns.13.proj_k.bias`
- `stages.2.attns.13.proj_v.weight`
- `stages.2.attns.13.proj_v.bias`
- `stages.2.attns.13.proj_out.weight`
- `stages.2.attns.13.proj_out.bias`
- `stages.2.attns.13.ref_point14.weight`
- `stages.2.attns.13.ref_point14.bias`
- `stages.2.attns.13.ref_gate.weight`
- `stages.2.attns.13.ref_gate.bias`
- `stages.2.attns.14.relative_position_bias_table`
- `stages.2.attns.14.relative_position_index`
- `stages.2.attns.14.proj_qkv.weight`
- `stages.2.attns.14.proj_qkv.bias`
- `stages.2.attns.14.proj_out.weight`
- `stages.2.attns.14.proj_out.bias`
- `stages.2.attns.15.rpe_table`
- `stages.2.attns.15.conv_offset.0.weight`
- `stages.2.attns.15.conv_offset.0.bias`
- `stages.2.attns.15.conv_offset.1.norm.weight`
- `stages.2.attns.15.conv_offset.1.norm.bias`
- `stages.2.attns.15.conv_offset.3.weight`
- `stages.2.attns.15.proj_q.weight`
- `stages.2.attns.15.proj_q.bias`
- `stages.2.attns.15.proj_k.weight`
- `stages.2.attns.15.proj_k.bias`
- `stages.2.attns.15.proj_v.weight`
- `stages.2.attns.15.proj_v.bias`
- `stages.2.attns.15.proj_out.weight`
- `stages.2.attns.15.proj_out.bias`
- `stages.2.attns.15.ref_point14.weight`
- `stages.2.attns.15.ref_point14.bias`
- `stages.2.attns.15.ref_gate.weight`
- `stages.2.attns.15.ref_gate.bias`
- `stages.2.attns.16.relative_position_bias_table`
- `stages.2.attns.16.relative_position_index`
- `stages.2.attns.16.proj_qkv.weight`
- `stages.2.attns.16.proj_qkv.bias`
- `stages.2.attns.16.proj_out.weight`
- `stages.2.attns.16.proj_out.bias`
- `stages.2.attns.17.rpe_table`
- `stages.2.attns.17.conv_offset.0.weight`
- `stages.2.attns.17.conv_offset.0.bias`
- `stages.2.attns.17.conv_offset.1.norm.weight`
- `stages.2.attns.17.conv_offset.1.norm.bias`
- `stages.2.attns.17.conv_offset.3.weight`
- `stages.2.attns.17.proj_q.weight`
- `stages.2.attns.17.proj_q.bias`
- `stages.2.attns.17.proj_k.weight`
- `stages.2.attns.17.proj_k.bias`
- `stages.2.attns.17.proj_v.weight`
- `stages.2.attns.17.proj_v.bias`
- `stages.2.attns.17.proj_out.weight`
- `stages.2.attns.17.proj_out.bias`
- `stages.2.attns.17.ref_point14.weight`
- `stages.2.attns.17.ref_point14.bias`
- `stages.2.attns.17.ref_gate.weight`
- `stages.2.attns.17.ref_gate.bias`

## Stage 2 Shapes
- PyTorch input shape: [1, 512, 14, 14]
- PyTorch output shape: [1, 512, 14, 14]
- TensorFlow input shape: [1, 14, 14, 512]
- TensorFlow output shape: [1, 14, 14, 512]

## TensorFlow Implementation Summary
```json
{
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
| deterministic random stage-2 input | fail | 0.946899414 | 0.00426508626 | [1, 512, 14, 14] | [1, 512, 14, 14] | [1, 14, 14, 512] | [1, 14, 14, 512] | block3_mlp_out |
| captured stage-2 input from deterministic random image | fail | 0.0103759766 | 1.89226757e-05 | [1, 512, 14, 14] | [1, 512, 14, 14] | [1, 14, 14, 512] | [1, 14, 14, 512] | block4_mlp_out |
| captured stage-2 input from 3 real ICAA17K images | fail | 0.015625 | 3.22220258e-05 | [3, 512, 14, 14] | [3, 512, 14, 14] | [3, 14, 14, 512] | [3, 14, 14, 512] | block3_mlp_out |

## First Diverging Block
- deterministic random stage-2 input: block3_mlp_out
- captured stage-2 input from deterministic random image: block4_mlp_out
- captured stage-2 input from 3 real ICAA17K images: block3_mlp_out

## Debug Tensor Summary
- Debug summary artifact: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_stage_g2_20260516_024326/optional_debug_tensors_summary.json`
- Per-block diffs are recorded for attention output, after-attention residual, MLP output, after-MLP residual, and final stage output.
- DAttention `pos` and `reference` diffs are recorded for each deformable block.

## Real Image Inputs
- Performed: True
- Image paths: ['/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/juiedesai19391601268.jpg', '/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/162814817@N0627431714937.jpg', '/home/omen_pc1/photo_score_project/data/raw/icaa17k/complementary/orange_blue/148560147@N0337039646742.jpg']

## Unresolved Issues
- deterministic random stage-2 input failed; first diverging block: block3_mlp_out; likely subcomponent: block3 MLP
- captured stage-2 input from deterministic random image failed; first diverging block: block4_mlp_out; likely subcomponent: block4 MLP
- captured stage-2 input from 3 real ICAA17K images failed; first diverging block: block3_mlp_out; likely subcomponent: block3 MLP

## Stage G-3 TransformerStage 3 Decision
- Safe to proceed to Stage G-3 TransformerStage 3 parity: False
- Scope of this decision: `TransformerStage` stage `2` only. This does not establish full TensorFlow DAT feasibility.

## Warnings
- stage_g2_transformer_stage2_parity.py:107: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
- __init__.py:49: Importing from timm.models.layers is deprecated, please import via timm.layers
- functional.py:534: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3595.)
- layer.py:424: `build()` was called on layer 'tf_transformer_stage2', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.

## Runtime Log Notices
- none

## Errors
- none
