# Mobile A-LAMP v2 TFLite Parity Report

This export uses the existing Mobile A-LAMP v2 best checkpoint. No retraining was run.

## Command

```bash
PYTHONPATH=. /home/omen_pc1/photo_score_project/.venv_gpu/bin/python tools/export_mobile_alamp_v2_tflite.py
```

## Inputs

- Keras model: `outputs/mobile_alamp_v2_pretrained_4096/best_val_auc_model.keras`
- Validation JSONL: `outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/val_patch_boxes_4096_v4.jsonl`
- Sample count: `32`
- Preprocessing: `mobilenetv3_include_preprocessing_float_pixels_0_255`
- Pixel convention: `full_image` and `patches` are RGB `float32` pixels in `[0,255]`; MobileNetV3Small preprocessing stays inside the model.

## Outputs

- FP32 TFLite: `outputs/mobile_alamp_v2_pretrained_4096_tflite/mobile_alamp_v2_fp32.tflite`
- FP16 TFLite: `outputs/mobile_alamp_v2_pretrained_4096_tflite/mobile_alamp_v2_fp16.tflite`
- Export metadata: `outputs/mobile_alamp_v2_pretrained_4096_tflite/export_metadata.json`
- JSON report: `outputs/mobile_alamp_v2_pretrained_4096_tflite/parity_report.json`
- Markdown report: `outputs/mobile_alamp_v2_pretrained_4096_tflite/parity_report.md`

## Metrics

| Model | Max abs diff | Mean abs diff | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: |
| FP32 TFLite | 3.933906555e-06 | 8.684583008e-07 | 1 | 1 |
| FP16 TFLite | 0.00742906332 | 0.001990574878 | 0.9999468908 | 0.9996334311 |

## Shapes

```json
{
  "loaded_inputs": {
    "full_image": [
      32,
      384,
      384,
      3
    ],
    "patches": [
      32,
      5,
      224,
      224,
      3
    ]
  },
  "keras_inputs": [
    {
      "name": "full_image",
      "shape": [
        null,
        384,
        384,
        3
      ],
      "dtype": "float32"
    },
    {
      "name": "patches",
      "shape": [
        null,
        5,
        224,
        224,
        3
      ],
      "dtype": "float32"
    }
  ],
  "keras_outputs": [
    {
      "name": "keras_tensor_772",
      "shape": [
        null,
        1
      ],
      "dtype": "float32"
    }
  ],
  "fp32_tflite_inputs": [
    {
      "name": "serving_default_patches:0",
      "index": 0,
      "shape": [
        1,
        5,
        224,
        224,
        3
      ],
      "shape_signature": [
        -1,
        5,
        224,
        224,
        3
      ],
      "dtype": "float32",
      "quantization": [
        0.0,
        0
      ]
    },
    {
      "name": "serving_default_full_image:0",
      "index": 1,
      "shape": [
        1,
        384,
        384,
        3
      ],
      "shape_signature": [
        -1,
        384,
        384,
        3
      ],
      "dtype": "float32",
      "quantization": [
        0.0,
        0
      ]
    }
  ],
  "fp32_tflite_outputs": [
    {
      "name": "StatefulPartitionedCall_1:0",
      "index": 372,
      "shape": [
        1,
        1
      ],
      "shape_signature": [
        -1,
        1
      ],
      "dtype": "float32",
      "quantization": [
        0.0,
        0
      ]
    }
  ],
  "fp16_tflite_inputs": [
    {
      "name": "serving_default_patches:0",
      "index": 0,
      "shape": [
        1,
        5,
        224,
        224,
        3
      ],
      "shape_signature": [
        -1,
        5,
        224,
        224,
        3
      ],
      "dtype": "float32",
      "quantization": [
        0.0,
        0
      ]
    },
    {
      "name": "serving_default_full_image:0",
      "index": 1,
      "shape": [
        1,
        384,
        384,
        3
      ],
      "shape_signature": [
        -1,
        384,
        384,
        3
      ],
      "dtype": "float32",
      "quantization": [
        0.0,
        0
      ]
    }
  ],
  "fp16_tflite_outputs": [
    {
      "name": "StatefulPartitionedCall_1:0",
      "index": 493,
      "shape": [
        1,
        1
      ],
      "shape_signature": [
        -1,
        1
      ],
      "dtype": "float32",
      "quantization": [
        0.0,
        0
      ]
    }
  ]
}
```

## Per-Sample Predictions

| # | row_index | image_path | Keras | FP32 | FP16 | abs FP32 | abs FP16 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/85837.jpg` | 0.7567785978 | 0.7567796707 | 0.7573247552 | 1.072883606e-06 | 0.0005461573601 |
| 1 | 1 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/861325.jpg` | 0.9503389597 | 0.9503390193 | 0.9499717951 | 5.960464478e-08 | 0.0003671646118 |
| 2 | 2 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/209262.jpg` | 0.1881596446 | 0.1881592274 | 0.1881838739 | 4.172325134e-07 | 2.42292881e-05 |
| 3 | 3 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/41026.jpg` | 0.5968427658 | 0.5968444943 | 0.6018509865 | 1.728534698e-06 | 0.005008220673 |
| 4 | 4 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/834998.jpg` | 0.8823598623 | 0.8823601007 | 0.8831678629 | 2.384185791e-07 | 0.0008080005646 |
| 5 | 5 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/68438.jpg` | 0.6762887836 | 0.6762848496 | 0.6746233702 | 3.933906555e-06 | 0.00166541338 |
| 6 | 6 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/742886.jpg` | 0.9178994298 | 0.9178994298 | 0.9165924191 | 0 | 0.001307010651 |
| 7 | 7 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/117139.jpg` | 0.8683306575 | 0.8683291674 | 0.8664979935 | 1.490116119e-06 | 0.001832664013 |
| 8 | 8 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/178577.jpg` | 0.1846657544 | 0.184665665 | 0.1894973963 | 8.940696716e-08 | 0.004831641912 |
| 9 | 9 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/11179.jpg` | 0.607286036 | 0.607285738 | 0.6086809635 | 2.980232239e-07 | 0.001394927502 |
| 10 | 10 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/748396.jpg` | 0.5816312432 | 0.581628859 | 0.5773974061 | 2.384185791e-06 | 0.004233837128 |
| 11 | 11 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/131273.jpg` | 0.7816132307 | 0.7816128731 | 0.7789771557 | 3.576278687e-07 | 0.00263607502 |
| 12 | 12 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/910203.jpg` | 0.6116899252 | 0.6116884351 | 0.6099714041 | 1.490116119e-06 | 0.001718521118 |
| 13 | 13 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/118468.jpg` | 0.3850563169 | 0.3850551844 | 0.379591316 | 1.132488251e-06 | 0.005465000868 |
| 14 | 14 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/475173.jpg` | 0.8880787492 | 0.8880786896 | 0.8889110088 | 5.960464478e-08 | 0.000832259655 |
| 15 | 15 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/838102.jpg` | 0.3773426116 | 0.3773392737 | 0.3774617016 | 3.337860107e-06 | 0.0001190900803 |
| 16 | 16 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/424289.jpg` | 0.1789943427 | 0.1789944321 | 0.177993089 | 8.940696716e-08 | 0.001001253724 |
| 17 | 17 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/764508.jpg` | 0.9743950963 | 0.9743949771 | 0.9743278623 | 1.192092896e-07 | 6.723403931e-05 |
| 18 | 18 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/119911.jpg` | 0.573327601 | 0.5733240247 | 0.5658985376 | 3.576278687e-06 | 0.00742906332 |
| 19 | 19 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/457758.jpg` | 0.5000315309 | 0.5000311732 | 0.5014545918 | 3.576278687e-07 | 0.001423060894 |
| 20 | 20 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/10088.jpg` | 0.2606380582 | 0.2606386542 | 0.2606081665 | 5.960464478e-07 | 2.989172935e-05 |
| 21 | 21 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/669150.jpg` | 0.934992671 | 0.9349932075 | 0.9332941175 | 5.36441803e-07 | 0.001698553562 |
| 22 | 22 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/16483.jpg` | 0.4523201585 | 0.452321142 | 0.4526618719 | 9.834766388e-07 | 0.0003417134285 |
| 23 | 23 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/934395.jpg` | 0.9464141726 | 0.946414113 | 0.9455731511 | 5.960464478e-08 | 0.0008410215378 |
| 24 | 24 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/228809.jpg` | 0.3606521189 | 0.3606526256 | 0.3636932075 | 5.066394806e-07 | 0.003041088581 |
| 25 | 25 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/166702.jpg` | 0.6048331261 | 0.6048332453 | 0.6027947664 | 1.192092896e-07 | 0.002038359642 |
| 26 | 26 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/235911.jpg` | 0.7749649882 | 0.7749643326 | 0.7775743008 | 6.556510925e-07 | 0.002609312534 |
| 27 | 27 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/331424.jpg` | 0.5546528101 | 0.5546521544 | 0.5509287119 | 6.556510925e-07 | 0.003724098206 |
| 28 | 28 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/547797.jpg` | 0.8598139882 | 0.8598138094 | 0.8590658903 | 1.788139343e-07 | 0.0007480978966 |
| 29 | 29 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/319962.jpg` | 0.7277500629 | 0.7277505398 | 0.7261390686 | 4.768371582e-07 | 0.001610994339 |
| 30 | 30 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/4886.jpg` | 0.2236607075 | 0.2236613184 | 0.2250087559 | 6.109476089e-07 | 0.001348048449 |
| 31 | 31 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/312548.jpg` | 0.7729229331 | 0.7729227543 | 0.7699665427 | 1.788139343e-07 | 0.002956390381 |
