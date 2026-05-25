# Mobile A-LAMP v2 TFLite Parity Report

This export uses the existing Mobile A-LAMP v2 best checkpoint. No retraining was run.

## Command

```bash
PYTHONPATH=. /home/omen_pc1/photo_score_project/.venv_gpu/bin/python tools/export_mobile_alamp_v2_tflite.py --model_path outputs/mobile_alamp_v2_full_ava_20260519/best_val_auc_model.keras --val_patch_jsonl outputs/alamp_v4_full_ava_20260517/subsets/val_patch_boxes_full_v4.jsonl --output_dir outputs/mobile_alamp_v2_full_ava_20260519_tflite --max_samples 64
```

## Inputs

- Keras model: `outputs/mobile_alamp_v2_full_ava_20260519/best_val_auc_model.keras`
- Validation JSONL: `outputs/alamp_v4_full_ava_20260517/subsets/val_patch_boxes_full_v4.jsonl`
- Sample count: `64`
- Preprocessing: `mobilenetv3_include_preprocessing_float_pixels_0_255`
- Pixel convention: `full_image` and `patches` are RGB `float32` pixels in `[0,255]`; MobileNetV3Small preprocessing stays inside the model.

## Outputs

- FP32 TFLite: `outputs/mobile_alamp_v2_full_ava_20260519_tflite/mobile_alamp_v2_fp32.tflite`
- FP16 TFLite: `outputs/mobile_alamp_v2_full_ava_20260519_tflite/mobile_alamp_v2_fp16.tflite`
- Export metadata: `outputs/mobile_alamp_v2_full_ava_20260519_tflite/export_metadata.json`
- JSON report: `outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_report.json`
- Markdown report: `outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_report.md`

## Metrics

| Model | Max abs diff | Mean abs diff | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: |
| FP32 TFLite | 0.006583631039 | 0.001611811225 | 0.9999739538 | 0.9999542125 |
| FP16 TFLite | 0.01139107347 | 0.002795987297 | 0.9999169379 | 0.9998168498 |

## Shapes

```json
{
  "loaded_inputs": {
    "full_image": [
      64,
      384,
      384,
      3
    ],
    "patches": [
      64,
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
| 0 | 0 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/85837.jpg` | 0.5116729736 | 0.5123627186 | 0.5098074675 | 0.0006897449493 | 0.001865506172 |
| 1 | 1 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/861325.jpg` | 0.9661675692 | 0.9666752219 | 0.9677164555 | 0.0005076527596 | 0.001548886299 |
| 2 | 2 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/209262.jpg` | 0.1611162871 | 0.1621768624 | 0.1629777253 | 0.001060575247 | 0.001861438155 |
| 3 | 3 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/41026.jpg` | 0.1656836569 | 0.1701576263 | 0.1703637242 | 0.00447396934 | 0.004680067301 |
| 4 | 4 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/834998.jpg` | 0.9988629818 | 0.9989024401 | 0.9989173412 | 3.945827484e-05 | 5.435943604e-05 |
| 5 | 5 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/68438.jpg` | 0.4011227489 | 0.4044595659 | 0.4074000418 | 0.003336817026 | 0.006277292967 |
| 6 | 6 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/742886.jpg` | 0.9584951997 | 0.9587107301 | 0.9586027861 | 0.0002155303955 | 0.0001075863838 |
| 7 | 7 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/117139.jpg` | 0.6015060544 | 0.6010029316 | 0.5999498963 | 0.0005031228065 | 0.001556158066 |
| 8 | 8 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/178577.jpg` | 0.08944541216 | 0.09106549621 | 0.09242770821 | 0.001620084047 | 0.00298229605 |
| 9 | 9 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/11179.jpg` | 0.6894568801 | 0.6895828247 | 0.6905382872 | 0.0001259446144 | 0.00108140707 |
| 10 | 10 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/748396.jpg` | 0.5053908229 | 0.5073798895 | 0.5082030892 | 0.001989066601 | 0.00281226635 |
| 11 | 11 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/131273.jpg` | 0.4192280769 | 0.4217737913 | 0.4225952923 | 0.002545714378 | 0.003367215395 |
| 12 | 12 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/910203.jpg` | 0.7076379657 | 0.7086936235 | 0.7109261751 | 0.001055657864 | 0.003288209438 |
| 13 | 13 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/118468.jpg` | 0.2861688733 | 0.2859979272 | 0.2849751115 | 0.0001709461212 | 0.001193761826 |
| 14 | 14 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/475173.jpg` | 0.9267235398 | 0.9274496436 | 0.9293993115 | 0.0007261037827 | 0.002675771713 |
| 15 | 15 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/838102.jpg` | 0.2437618226 | 0.2422062308 | 0.2399892807 | 0.001555591822 | 0.003772541881 |
| 16 | 16 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/424289.jpg` | 0.2136275917 | 0.2138076425 | 0.2116805166 | 0.0001800507307 | 0.001947075129 |
| 17 | 17 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/764508.jpg` | 0.8759515882 | 0.8774068356 | 0.8766704798 | 0.001455247402 | 0.0007188916206 |
| 18 | 18 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/119911.jpg` | 0.4245668948 | 0.4243736565 | 0.4195502698 | 0.0001932382584 | 0.005016624928 |
| 19 | 19 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/457758.jpg` | 0.2537130713 | 0.2564483583 | 0.2560281754 | 0.002735286951 | 0.002315104008 |
| 20 | 20 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/10088.jpg` | 0.1201200783 | 0.12450625 | 0.1264392138 | 0.004386171699 | 0.006319135427 |
| 21 | 21 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/669150.jpg` | 0.8495436311 | 0.8494647741 | 0.8499411345 | 7.885694504e-05 | 0.000397503376 |
| 22 | 22 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/16483.jpg` | 0.1591653675 | 0.1606532633 | 0.1602950096 | 0.001487895846 | 0.001129642129 |
| 23 | 23 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/934395.jpg` | 0.9702412486 | 0.9702506065 | 0.9703310132 | 9.35792923e-06 | 8.976459503e-05 |
| 24 | 24 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/228809.jpg` | 0.2012582421 | 0.20502536 | 0.2074632198 | 0.003767117858 | 0.006204977632 |
| 25 | 25 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/166702.jpg` | 0.3315307796 | 0.3320828676 | 0.3304653466 | 0.0005520880222 | 0.001065433025 |
| 26 | 26 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/235911.jpg` | 0.44295156 | 0.4439818859 | 0.4483004212 | 0.00103032589 | 0.005348861217 |
| 27 | 27 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/331424.jpg` | 0.5325478911 | 0.531645298 | 0.5287123322 | 0.0009025931358 | 0.003835558891 |
| 28 | 28 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/547797.jpg` | 0.8767092824 | 0.8796803951 | 0.8803448081 | 0.002971112728 | 0.003635525703 |
| 29 | 29 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/319962.jpg` | 0.5758452415 | 0.5768605471 | 0.5784801245 | 0.001015305519 | 0.002634882927 |
| 30 | 30 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/4886.jpg` | 0.06049736217 | 0.06022316962 | 0.05952230096 | 0.0002741925418 | 0.000975061208 |
| 31 | 31 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/312548.jpg` | 0.7938192487 | 0.7943885922 | 0.7924305797 | 0.0005693435669 | 0.001388669014 |
| 32 | 32 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/284733.jpg` | 0.3182456195 | 0.3218635023 | 0.3250317276 | 0.003617882729 | 0.006786108017 |
| 33 | 33 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/156796.jpg` | 0.317469269 | 0.3186689913 | 0.3172884583 | 0.00119972229 | 0.0001808106899 |
| 34 | 34 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/538500.jpg` | 0.394784838 | 0.3959199488 | 0.3932141066 | 0.001135110855 | 0.001570731401 |
| 35 | 35 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/566052.jpg` | 0.6284579039 | 0.6343529224 | 0.6351190805 | 0.005895018578 | 0.006661176682 |
| 36 | 36 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/3791.jpg` | 0.006428394932 | 0.006627156399 | 0.006774876267 | 0.0001987614669 | 0.0003464813344 |
| 37 | 37 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/342264.jpg` | 0.425009191 | 0.4301809371 | 0.4247051477 | 0.005171746016 | 0.000304043293 |
| 38 | 38 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/703710.jpg` | 0.2593868971 | 0.2581140995 | 0.2626248002 | 0.001272797585 | 0.003237903118 |
| 39 | 39 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/489498.jpg` | 0.3675874174 | 0.363368839 | 0.3561963439 | 0.004218578339 | 0.01139107347 |
| 40 | 40 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/632088.jpg` | 0.965862453 | 0.9662417173 | 0.9666903615 | 0.0003792643547 | 0.0008279085159 |
| 41 | 41 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/570519.jpg` | 0.5058569312 | 0.5070890188 | 0.5028322935 | 0.001232087612 | 0.003024637699 |
| 42 | 42 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/799448.jpg` | 0.6124833822 | 0.6143081188 | 0.6179839969 | 0.001824736595 | 0.005500614643 |
| 43 | 43 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/501957.jpg` | 0.4400283992 | 0.4404613674 | 0.4394846261 | 0.0004329681396 | 0.0005437731743 |
| 44 | 44 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/706194.jpg` | 0.5903604627 | 0.5867700577 | 0.5841612816 | 0.003590404987 | 0.00619918108 |
| 45 | 45 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/926048.jpg` | 0.1255530268 | 0.1250718683 | 0.1237359941 | 0.0004811584949 | 0.001817032695 |
| 46 | 46 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/542862.jpg` | 0.453507483 | 0.4549798071 | 0.4537830651 | 0.001472324133 | 0.0002755820751 |
| 47 | 47 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/20783.jpg` | 0.7289994359 | 0.7318941355 | 0.733284831 | 0.002894699574 | 0.004285395145 |
| 48 | 48 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/82222.jpg` | 0.6104463935 | 0.6097170115 | 0.6133309007 | 0.0007293820381 | 0.002884507179 |
| 49 | 49 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/458518.jpg` | 0.2264924198 | 0.2281107008 | 0.223790586 | 0.001618281007 | 0.002701833844 |
| 50 | 50 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/797291.jpg` | 0.4969172478 | 0.4935581386 | 0.4907374978 | 0.003359109163 | 0.006179749966 |
| 51 | 51 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/426443.jpg` | 0.4376567006 | 0.4373780787 | 0.4325141609 | 0.000278621912 | 0.00514253974 |
| 52 | 52 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/222085.jpg` | 0.1547603458 | 0.1544993073 | 0.1575700045 | 0.0002610385418 | 0.002809658647 |
| 53 | 53 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/937285.jpg` | 0.1060889289 | 0.10612946 | 0.1039864272 | 4.053115845e-05 | 0.00210250169 |
| 54 | 54 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/167447.jpg` | 0.5612436533 | 0.5600494146 | 0.5619286299 | 0.001194238663 | 0.0006849765778 |
| 55 | 55 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/354294.jpg` | 0.1504421532 | 0.1508616358 | 0.1507139206 | 0.0004194825888 | 0.0002717673779 |
| 56 | 56 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/79160.jpg` | 0.4664230347 | 0.4663464725 | 0.4631931186 | 7.656216621e-05 | 0.003229916096 |
| 57 | 57 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/422318.jpg` | 0.9115281701 | 0.9125880003 | 0.9136064053 | 0.001059830189 | 0.002078235149 |
| 58 | 58 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/626025.jpg` | 0.4559260905 | 0.4625097215 | 0.4641354084 | 0.006583631039 | 0.008209317923 |
| 59 | 59 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/856654.jpg` | 0.2705456316 | 0.2741599381 | 0.2726894617 | 0.00361430645 | 0.002143830061 |
| 60 | 60 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/122211.jpg` | 0.8122159243 | 0.8133411407 | 0.8145927191 | 0.001125216484 | 0.002376794815 |
| 61 | 61 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/578618.jpg` | 0.08357302099 | 0.08475936204 | 0.08466073871 | 0.001186341047 | 0.001087717712 |
| 62 | 62 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/467965.jpg` | 0.3108431399 | 0.3137273788 | 0.3099386692 | 0.002884238958 | 0.0009044706821 |
| 63 | 63 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/458514.jpg` | 0.8474367857 | 0.8489204645 | 0.8484722376 | 0.001483678818 | 0.001035451889 |
