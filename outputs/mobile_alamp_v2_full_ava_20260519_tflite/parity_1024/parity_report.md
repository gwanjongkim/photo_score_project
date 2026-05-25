# Mobile A-LAMP v2 TFLite Parity Report

This export uses the existing Mobile A-LAMP v2 best checkpoint. No retraining was run.

## Command

```bash
PYTHONPATH=. /home/omen_pc1/photo_score_project/.venv_gpu/bin/python tools/export_mobile_alamp_v2_tflite.py --model_path outputs/mobile_alamp_v2_full_ava_20260519/best_val_auc_model.keras --val_patch_jsonl outputs/alamp_v4_full_ava_20260517/subsets/val_patch_boxes_full_v4.jsonl --output_dir outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_1024/ --max_samples 1024
```

## Inputs

- Keras model: `outputs/mobile_alamp_v2_full_ava_20260519/best_val_auc_model.keras`
- Validation JSONL: `outputs/alamp_v4_full_ava_20260517/subsets/val_patch_boxes_full_v4.jsonl`
- Sample count: `1024`
- Preprocessing: `mobilenetv3_include_preprocessing_float_pixels_0_255`
- Pixel convention: `full_image` and `patches` are RGB `float32` pixels in `[0,255]`; MobileNetV3Small preprocessing stays inside the model.

## Outputs

- FP32 TFLite: `outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_1024/mobile_alamp_v2_fp32.tflite`
- FP16 TFLite: `outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_1024/mobile_alamp_v2_fp16.tflite`
- Export metadata: `outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_1024/export_metadata.json`
- JSON report: `outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_1024/parity_report.json`
- Markdown report: `outputs/mobile_alamp_v2_full_ava_20260519_tflite/parity_1024/parity_report.md`

## Metrics

| Model | Max abs diff | Mean abs diff | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: |
| FP32 TFLite | 0.01110419631 | 0.001367614837 | 0.9999829244 | 0.9999731555 |
| FP16 TFLite | 0.02769544721 | 0.002673436888 | 0.9998949519 | 0.9998649506 |

## Shapes

```json
{
  "loaded_inputs": {
    "full_image": [
      1024,
      384,
      384,
      3
    ],
    "patches": [
      1024,
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
| 0 | 0 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/85837.jpg` | 0.5115974545 | 0.5123627186 | 0.5098074675 | 0.0007652640343 | 0.001789987087 |
| 1 | 1 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/861325.jpg` | 0.9661947489 | 0.9666752219 | 0.9677164555 | 0.0004804730415 | 0.001521706581 |
| 2 | 2 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/209262.jpg` | 0.1608172953 | 0.1621768624 | 0.1629777253 | 0.001359567046 | 0.002160429955 |
| 3 | 3 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/41026.jpg` | 0.1656983644 | 0.1701576263 | 0.1703637242 | 0.004459261894 | 0.004665359855 |
| 4 | 4 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/834998.jpg` | 0.9988611937 | 0.9989024401 | 0.9989173412 | 4.124641418e-05 | 5.614757538e-05 |
| 5 | 5 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/68438.jpg` | 0.4013153315 | 0.4044595659 | 0.4074000418 | 0.003144234419 | 0.00608471036 |
| 6 | 6 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/742886.jpg` | 0.9584927559 | 0.9587107301 | 0.9586027861 | 0.0002179741859 | 0.0001100301743 |
| 7 | 7 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/117139.jpg` | 0.6011940837 | 0.6010029316 | 0.5999498963 | 0.0001911520958 | 0.001244187355 |
| 8 | 8 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/178577.jpg` | 0.08944325894 | 0.09106549621 | 0.09242770821 | 0.001622237265 | 0.002984449267 |
| 9 | 9 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/11179.jpg` | 0.6895168424 | 0.6895828247 | 0.6905382872 | 6.598234177e-05 | 0.001021444798 |
| 10 | 10 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/748396.jpg` | 0.505267024 | 0.5073798895 | 0.5082030892 | 0.002112865448 | 0.002936065197 |
| 11 | 11 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/131273.jpg` | 0.4191395938 | 0.4217737913 | 0.4225952923 | 0.002634197474 | 0.00345569849 |
| 12 | 12 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/910203.jpg` | 0.7078572512 | 0.7086936235 | 0.7109261751 | 0.0008363723755 | 0.00306892395 |
| 13 | 13 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/118468.jpg` | 0.2859935164 | 0.2859979272 | 0.2849751115 | 4.410743713e-06 | 0.001018404961 |
| 14 | 14 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/475173.jpg` | 0.9267604947 | 0.9274496436 | 0.9293993115 | 0.0006891489029 | 0.002638816833 |
| 15 | 15 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/838102.jpg` | 0.2437535226 | 0.2422062308 | 0.2399892807 | 0.001547291875 | 0.003764241934 |
| 16 | 16 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/424289.jpg` | 0.2136751562 | 0.2138076425 | 0.2116805166 | 0.0001324862242 | 0.001994639635 |
| 17 | 17 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/764508.jpg` | 0.8760768771 | 0.8774068356 | 0.8766704798 | 0.001329958439 | 0.0005936026573 |
| 18 | 18 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/119911.jpg` | 0.4247249365 | 0.4243736565 | 0.4195502698 | 0.000351279974 | 0.005174666643 |
| 19 | 19 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/457758.jpg` | 0.2537507713 | 0.2564483583 | 0.2560281754 | 0.002697587013 | 0.00227740407 |
| 20 | 20 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/10088.jpg` | 0.1201610491 | 0.12450625 | 0.1264392138 | 0.004345200956 | 0.006278164685 |
| 21 | 21 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/669150.jpg` | 0.8497026563 | 0.8494647741 | 0.8499411345 | 0.0002378821373 | 0.0002384781837 |
| 22 | 22 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/16483.jpg` | 0.1590944082 | 0.1606532633 | 0.1602950096 | 0.001558855176 | 0.001200601459 |
| 23 | 23 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/934395.jpg` | 0.9702128768 | 0.9702506065 | 0.9703310132 | 3.772974014e-05 | 0.0001181364059 |
| 24 | 24 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/228809.jpg` | 0.2008908242 | 0.20502536 | 0.2074632198 | 0.004134535789 | 0.006572395563 |
| 25 | 25 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/166702.jpg` | 0.3315674067 | 0.3320828676 | 0.3304653466 | 0.000515460968 | 0.00110206008 |
| 26 | 26 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/235911.jpg` | 0.4428506494 | 0.4439818859 | 0.4483004212 | 0.001131236553 | 0.005449771881 |
| 27 | 27 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/331424.jpg` | 0.5319122076 | 0.531645298 | 0.5287123322 | 0.0002669095993 | 0.003199875355 |
| 28 | 28 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/547797.jpg` | 0.8766732216 | 0.8796803951 | 0.8803448081 | 0.003007173538 | 0.003671586514 |
| 29 | 29 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/319962.jpg` | 0.5756519437 | 0.5768605471 | 0.5784801245 | 0.001208603382 | 0.00282818079 |
| 30 | 30 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/4886.jpg` | 0.06059945375 | 0.06022316962 | 0.05952230096 | 0.0003762841225 | 0.001077152789 |
| 31 | 31 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/312548.jpg` | 0.7934714556 | 0.7943885922 | 0.7924305797 | 0.0009171366692 | 0.001040875912 |
| 32 | 32 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/284733.jpg` | 0.3183404207 | 0.3218635023 | 0.3250317276 | 0.003523081541 | 0.006691306829 |
| 33 | 33 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/156796.jpg` | 0.3173727691 | 0.3186689913 | 0.3172884583 | 0.00129622221 | 8.431077003e-05 |
| 34 | 34 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/538500.jpg` | 0.394697845 | 0.3959199488 | 0.3932141066 | 0.001222103834 | 0.001483738422 |
| 35 | 35 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/566052.jpg` | 0.630145967 | 0.6343529224 | 0.6351190805 | 0.004206955433 | 0.004973113537 |
| 36 | 36 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/3791.jpg` | 0.006421814207 | 0.006627156399 | 0.006774876267 | 0.0002053421922 | 0.0003530620597 |
| 37 | 37 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/342264.jpg` | 0.425091356 | 0.4301809371 | 0.4247051477 | 0.005089581013 | 0.0003862082958 |
| 38 | 38 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/703710.jpg` | 0.2593244016 | 0.2581140995 | 0.2626248002 | 0.001210302114 | 0.003300398588 |
| 39 | 39 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/489498.jpg` | 0.3677473962 | 0.363368839 | 0.3561963439 | 0.004378557205 | 0.01155105233 |
| 40 | 40 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/632088.jpg` | 0.9658648968 | 0.9662417173 | 0.9666903615 | 0.0003768205643 | 0.0008254647255 |
| 41 | 41 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/570519.jpg` | 0.5054554939 | 0.5070890188 | 0.5028322935 | 0.001633524895 | 0.002623200417 |
| 42 | 42 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/799448.jpg` | 0.6121604443 | 0.6143081188 | 0.6179839969 | 0.002147674561 | 0.005823552608 |
| 43 | 43 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/501957.jpg` | 0.4401705265 | 0.4404613674 | 0.4394846261 | 0.0002908408642 | 0.0006859004498 |
| 44 | 44 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/706194.jpg` | 0.5895638466 | 0.5867700577 | 0.5841612816 | 0.00279378891 | 0.005402565002 |
| 45 | 45 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/926048.jpg` | 0.1255709678 | 0.1250718683 | 0.1237359941 | 0.000499099493 | 0.001834973693 |
| 46 | 46 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/542862.jpg` | 0.4531350434 | 0.4549798071 | 0.4537830651 | 0.001844763756 | 0.000648021698 |
| 47 | 47 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/20783.jpg` | 0.7287077308 | 0.7318941355 | 0.733284831 | 0.003186404705 | 0.004577100277 |
| 48 | 48 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/82222.jpg` | 0.6108790636 | 0.6097170115 | 0.6133309007 | 0.001162052155 | 0.002451837063 |
| 49 | 49 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/458518.jpg` | 0.2265646011 | 0.2281107008 | 0.223790586 | 0.001546099782 | 0.002774015069 |
| 50 | 50 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/797291.jpg` | 0.4961901903 | 0.4935581386 | 0.4907374978 | 0.002632051706 | 0.005452692509 |
| 51 | 51 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/426443.jpg` | 0.4375860095 | 0.4373780787 | 0.4325141609 | 0.0002079308033 | 0.005071848631 |
| 52 | 52 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/222085.jpg` | 0.1547513157 | 0.1544993073 | 0.1575700045 | 0.0002520084381 | 0.00281868875 |
| 53 | 53 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/937285.jpg` | 0.1059098244 | 0.10612946 | 0.1039864272 | 0.0002196356654 | 0.001923397183 |
| 54 | 54 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/167447.jpg` | 0.561254859 | 0.5600494146 | 0.5619286299 | 0.001205444336 | 0.0006737709045 |
| 55 | 55 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/354294.jpg` | 0.1504455805 | 0.1508616358 | 0.1507139206 | 0.0004160553217 | 0.0002683401108 |
| 56 | 56 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/79160.jpg` | 0.4664000571 | 0.4663464725 | 0.4631931186 | 5.358457565e-05 | 0.003206938505 |
| 57 | 57 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/422318.jpg` | 0.9116066694 | 0.9125880003 | 0.9136064053 | 0.0009813308716 | 0.001999735832 |
| 58 | 58 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/626025.jpg` | 0.4557250738 | 0.4625097215 | 0.4641354084 | 0.006784647703 | 0.008410334587 |
| 59 | 59 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/856654.jpg` | 0.2703461647 | 0.2741599381 | 0.2726894617 | 0.003813773394 | 0.002343297005 |
| 60 | 60 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/122211.jpg` | 0.812058866 | 0.8133411407 | 0.8145927191 | 0.001282274723 | 0.002533853054 |
| 61 | 61 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/578618.jpg` | 0.08364529163 | 0.08475936204 | 0.08466073871 | 0.001114070415 | 0.00101544708 |
| 62 | 62 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/467965.jpg` | 0.3108434677 | 0.3137273788 | 0.3099386692 | 0.002883911133 | 0.0009047985077 |
| 63 | 63 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/458514.jpg` | 0.8472101688 | 0.8489204645 | 0.8484722376 | 0.001710295677 | 0.001262068748 |
| 64 | 64 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/274709.jpg` | 0.9687771201 | 0.9688370824 | 0.9693626165 | 5.996227264e-05 | 0.0005854964256 |
| 65 | 65 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/795487.jpg` | 0.9448446631 | 0.9457758665 | 0.9461314082 | 0.0009312033653 | 0.001286745071 |
| 66 | 66 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/784601.jpg` | 0.1561829746 | 0.1578662395 | 0.1569870114 | 0.001683264971 | 0.0008040368557 |
| 67 | 67 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/812483.jpg` | 0.1711387932 | 0.1721405983 | 0.1735680699 | 0.001001805067 | 0.002429276705 |
| 68 | 68 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/234406.jpg` | 0.1936582029 | 0.190845713 | 0.1793284416 | 0.002812489867 | 0.01432976127 |
| 69 | 69 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/294792.jpg` | 0.2659916282 | 0.2673812211 | 0.2581366599 | 0.001389592886 | 0.007854968309 |
| 70 | 70 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/237325.jpg` | 0.7142362595 | 0.7167088389 | 0.7143546343 | 0.002472579479 | 0.0001183748245 |
| 71 | 71 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/326481.jpg` | 0.2577435076 | 0.2583772242 | 0.2552542686 | 0.0006337165833 | 0.002489238977 |
| 72 | 72 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/479244.jpg` | 0.9003556371 | 0.9019351006 | 0.9019252658 | 0.001579463482 | 0.001569628716 |
| 73 | 73 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/916904.jpg` | 0.6701315045 | 0.6707720757 | 0.6648968458 | 0.0006405711174 | 0.005234658718 |
| 74 | 74 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/372504.jpg` | 0.263009429 | 0.2630554438 | 0.2583810389 | 4.601478577e-05 | 0.004628390074 |
| 75 | 75 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/830214.jpg` | 0.8049436212 | 0.8049539328 | 0.8021948934 | 1.031160355e-05 | 0.002748727798 |
| 76 | 76 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/20882.jpg` | 0.7533264756 | 0.7536644936 | 0.7541764975 | 0.0003380179405 | 0.0008500218391 |
| 77 | 77 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/663205.jpg` | 0.5117583275 | 0.5115636587 | 0.5109820366 | 0.0001946687698 | 0.0007762908936 |
| 78 | 78 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/264711.jpg` | 0.2506213486 | 0.2531227469 | 0.2550280392 | 0.002501398325 | 0.004406690598 |
| 79 | 79 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/760375.jpg` | 0.7082825899 | 0.7087717056 | 0.7058268785 | 0.000489115715 | 0.002455711365 |
| 80 | 80 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/83564.jpg` | 0.8672375679 | 0.8668842912 | 0.8644626737 | 0.0003532767296 | 0.002774894238 |
| 81 | 81 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/126891.jpg` | 0.08143931627 | 0.08247221261 | 0.08224150538 | 0.00103289634 | 0.0008021891117 |
| 82 | 82 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/867436.jpg` | 0.6716914177 | 0.6709426641 | 0.6724746227 | 0.0007487535477 | 0.0007832050323 |
| 83 | 83 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/54761.jpg` | 0.9081126451 | 0.9091403484 | 0.9098958969 | 0.001027703285 | 0.001783251762 |
| 84 | 84 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/942249.jpg` | 0.6084969044 | 0.6105345488 | 0.6083050966 | 0.002037644386 | 0.0001918077469 |
| 85 | 85 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/371535.jpg` | 0.3728671968 | 0.3736485839 | 0.3782212734 | 0.0007813870907 | 0.005354076624 |
| 86 | 86 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/6671.jpg` | 0.6582787037 | 0.6588336229 | 0.6608664989 | 0.0005549192429 | 0.002587795258 |
| 87 | 87 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/4148.jpg` | 0.2797715068 | 0.2818959653 | 0.2842322886 | 0.002124458551 | 0.004460781813 |
| 88 | 88 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/556287.jpg` | 0.2973024845 | 0.3046086431 | 0.2974832654 | 0.007306158543 | 0.0001807808876 |
| 89 | 89 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/72239.jpg` | 0.3084923029 | 0.3089039922 | 0.3068868518 | 0.0004116892815 | 0.001605451107 |
| 90 | 90 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/93930.jpg` | 0.1695222259 | 0.171294257 | 0.173718974 | 0.001772031188 | 0.004196748137 |
| 91 | 91 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/123435.jpg` | 0.220955193 | 0.2207598686 | 0.2179895639 | 0.0001953244209 | 0.002965629101 |
| 92 | 92 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/533693.jpg` | 0.8461161852 | 0.8475677967 | 0.8458843231 | 0.001451611519 | 0.0002318620682 |
| 93 | 93 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/20078.jpg` | 0.3460421264 | 0.3473187983 | 0.3450838029 | 0.001276671886 | 0.0009583234787 |
| 94 | 94 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/64694.jpg` | 0.5095096827 | 0.5114299059 | 0.5094736814 | 0.001920223236 | 3.600120544e-05 |
| 95 | 95 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/758224.jpg` | 0.7598462105 | 0.7626422644 | 0.7606621981 | 0.002796053886 | 0.000815987587 |
| 96 | 96 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/499090.jpg` | 0.6181058884 | 0.6199799776 | 0.603138268 | 0.001874089241 | 0.01496762037 |
| 97 | 97 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/169781.jpg` | 0.571946919 | 0.5752086043 | 0.5809875727 | 0.003261685371 | 0.009040653706 |
| 98 | 98 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/178499.jpg` | 0.5549866557 | 0.558408916 | 0.5561757088 | 0.003422260284 | 0.001189053059 |
| 99 | 99 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/755523.jpg` | 0.8058481216 | 0.8047787547 | 0.8075763583 | 0.001069366932 | 0.001728236675 |
| 100 | 100 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/611801.jpg` | 0.9290809631 | 0.9297611713 | 0.931420505 | 0.0006802082062 | 0.002339541912 |
| 101 | 101 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/412306.jpg` | 0.2686516643 | 0.2673692405 | 0.262917161 | 0.001282423735 | 0.005734503269 |
| 102 | 102 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/534941.jpg` | 0.7211415172 | 0.7206299305 | 0.718215704 | 0.0005115866661 | 0.002925813198 |
| 103 | 103 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/402231.jpg` | 0.9126140475 | 0.9126862288 | 0.9118810892 | 7.218122482e-05 | 0.0007329583168 |
| 104 | 104 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/483993.jpg` | 0.7771514058 | 0.7769089937 | 0.7747641206 | 0.0002424120903 | 0.002387285233 |
| 105 | 105 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/704363.jpg` | 0.2775096595 | 0.2837248743 | 0.2878128588 | 0.006215214729 | 0.01030319929 |
| 106 | 106 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/539623.jpg` | 0.5401393771 | 0.5412507057 | 0.5413181782 | 0.001111328602 | 0.00117880106 |
| 107 | 107 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/13104.jpg` | 0.7843883038 | 0.7847816944 | 0.7874813676 | 0.0003933906555 | 0.003093063831 |
| 108 | 108 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/302992.jpg` | 0.4676832557 | 0.4721206129 | 0.4748999774 | 0.004437357187 | 0.007216721773 |
| 109 | 109 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/93859.jpg` | 0.5788501501 | 0.5791899562 | 0.5758858919 | 0.0003398060799 | 0.002964258194 |
| 110 | 110 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/599709.jpg` | 0.4072529674 | 0.4163455665 | 0.4193388522 | 0.009092599154 | 0.01208588481 |
| 111 | 111 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/713512.jpg` | 0.8957546353 | 0.896027267 | 0.9013342261 | 0.0002726316452 | 0.005579590797 |
| 112 | 112 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/664967.jpg` | 0.2079081833 | 0.2086643577 | 0.2047791183 | 0.0007561743259 | 0.003129065037 |
| 113 | 113 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/847477.jpg` | 0.1584376991 | 0.1591834873 | 0.1547325104 | 0.0007457882166 | 0.003705188632 |
| 114 | 114 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/842483.jpg` | 0.6040612459 | 0.6049479246 | 0.6072585583 | 0.0008866786957 | 0.003197312355 |
| 115 | 115 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/494624.jpg` | 0.7302041054 | 0.7320020199 | 0.7344244719 | 0.001797914505 | 0.004220366478 |
| 116 | 116 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/126601.jpg` | 0.05514801666 | 0.05711357296 | 0.05851099268 | 0.001965556294 | 0.003362976015 |
| 117 | 117 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/113683.jpg` | 0.2325261533 | 0.2336048782 | 0.2345458269 | 0.001078724861 | 0.002019673586 |
| 118 | 118 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/449788.jpg` | 0.65700984 | 0.6574727297 | 0.658649683 | 0.0004628896713 | 0.001639842987 |
| 119 | 119 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/738128.jpg` | 0.9281005263 | 0.9291419387 | 0.9282360077 | 0.001041412354 | 0.0001354813576 |
| 120 | 120 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/305721.jpg` | 0.6996210814 | 0.6982353926 | 0.7004132867 | 0.001385688782 | 0.0007922053337 |
| 121 | 121 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/942314.jpg` | 0.8676319718 | 0.8676502705 | 0.8666379452 | 1.829862595e-05 | 0.0009940266609 |
| 122 | 122 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/487826.jpg` | 0.4570627511 | 0.4579596817 | 0.4602863789 | 0.0008969306946 | 0.003223627806 |
| 123 | 123 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/874239.jpg` | 0.940390408 | 0.9413326979 | 0.9420371652 | 0.0009422898293 | 0.001646757126 |
| 124 | 124 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/822459.jpg` | 0.2695179284 | 0.2705897391 | 0.2703744173 | 0.001071810722 | 0.0008564889431 |
| 125 | 125 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/239152.jpg` | 0.2996459305 | 0.2993570566 | 0.2998406589 | 0.0002888739109 | 0.0001947283745 |
| 126 | 126 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/337116.jpg` | 0.7567503452 | 0.7591731548 | 0.7613463402 | 0.002422809601 | 0.004595994949 |
| 127 | 127 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/135563.jpg` | 0.1658331752 | 0.1659058928 | 0.1646347195 | 7.271766663e-05 | 0.001198455691 |
| 128 | 128 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/770779.jpg` | 0.7142903805 | 0.7149268389 | 0.7142504454 | 0.0006364583969 | 3.9935112e-05 |
| 129 | 129 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/278953.jpg` | 0.20655711 | 0.2091626376 | 0.2145413756 | 0.002605527639 | 0.007984265685 |
| 130 | 130 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/445661.jpg` | 0.6017075181 | 0.6030412912 | 0.6028746367 | 0.001333773136 | 0.001167118549 |
| 131 | 131 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/9687.jpg` | 0.1713537276 | 0.1714600325 | 0.1728444993 | 0.000106304884 | 0.00149077177 |
| 132 | 132 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/564409.jpg` | 0.8862483501 | 0.8871249557 | 0.8871212006 | 0.0008766055107 | 0.0008728504181 |
| 133 | 133 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/498143.jpg` | 0.1762371957 | 0.1773554385 | 0.1752351075 | 0.001118242741 | 0.001002088189 |
| 134 | 134 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/403.jpg` | 0.07638905942 | 0.07658349723 | 0.07512800395 | 0.0001944378018 | 0.00126105547 |
| 135 | 135 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/202379.jpg` | 0.8599011898 | 0.8624953032 | 0.8644375205 | 0.00259411335 | 0.0045363307 |
| 136 | 136 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/403725.jpg` | 0.7611926794 | 0.76195997 | 0.7618010044 | 0.0007672905922 | 0.0006083250046 |
| 137 | 137 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/930404.jpg` | 0.3045408726 | 0.3056342006 | 0.302341193 | 0.001093327999 | 0.002199679613 |
| 138 | 138 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/196768.jpg` | 0.1209060401 | 0.1218775585 | 0.1213786229 | 0.0009715184569 | 0.0004725828767 |
| 139 | 139 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/38193.jpg` | 0.7003076077 | 0.7015075088 | 0.6973649859 | 0.001199901104 | 0.002942621708 |
| 140 | 140 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/740221.jpg` | 0.5721331239 | 0.5733263493 | 0.5776783824 | 0.001193225384 | 0.005545258522 |
| 141 | 141 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/427529.jpg` | 0.93878299 | 0.9402138591 | 0.9409199357 | 0.001430869102 | 0.002136945724 |
| 142 | 142 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/482535.jpg` | 0.2215484977 | 0.2225571573 | 0.2220828831 | 0.001008659601 | 0.0005343854427 |
| 143 | 143 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/138763.jpg` | 0.1103296876 | 0.1137227044 | 0.1149152443 | 0.003393016756 | 0.004585556686 |
| 144 | 144 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/832311.jpg` | 0.5054264069 | 0.5051953197 | 0.5000895858 | 0.0002310872078 | 0.005336821079 |
| 145 | 145 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/381134.jpg` | 0.1216647848 | 0.1230066493 | 0.1216508448 | 0.001341864467 | 1.39400363e-05 |
| 146 | 146 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/795612.jpg` | 0.1764356345 | 0.1759332716 | 0.1727032065 | 0.0005023628473 | 0.003732427955 |
| 147 | 147 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/430303.jpg` | 0.3895634413 | 0.3907858431 | 0.3883565962 | 0.001222401857 | 0.001206845045 |
| 148 | 148 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/903977.jpg` | 0.6989629865 | 0.7000320554 | 0.6991125345 | 0.001069068909 | 0.0001495480537 |
| 149 | 149 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/296287.jpg` | 0.6487235427 | 0.6499359608 | 0.6465633512 | 0.001212418079 | 0.002160191536 |
| 150 | 150 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/846762.jpg` | 0.1062916219 | 0.1079923585 | 0.1081810817 | 0.001700736582 | 0.001889459789 |
| 151 | 151 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/479025.jpg` | 0.887206614 | 0.8881191611 | 0.8893934488 | 0.0009125471115 | 0.002186834812 |
| 152 | 152 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/550487.jpg` | 0.8789613247 | 0.880417347 | 0.880305171 | 0.001456022263 | 0.001343846321 |
| 153 | 153 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/217762.jpg` | 0.2533438504 | 0.2558706999 | 0.2540231943 | 0.002526849508 | 0.0006793439388 |
| 154 | 154 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/246011.jpg` | 0.5192924142 | 0.5147635341 | 0.5096380711 | 0.004528880119 | 0.009654343128 |
| 155 | 155 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/268280.jpg` | 0.04998548329 | 0.05010356754 | 0.05054173246 | 0.0001180842519 | 0.0005562491715 |
| 156 | 156 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/108946.jpg` | 0.741047442 | 0.7392286062 | 0.7406501174 | 0.001818835735 | 0.0003973245621 |
| 157 | 157 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/125801.jpg` | 0.687944591 | 0.6890257597 | 0.6911743879 | 0.001081168652 | 0.003229796886 |
| 158 | 158 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/159527.jpg` | 0.2712751031 | 0.267737329 | 0.2627846003 | 0.003537774086 | 0.008490502834 |
| 159 | 159 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/43001.jpg` | 0.4947536886 | 0.4998959303 | 0.4962198436 | 0.005142241716 | 0.001466155052 |
| 160 | 160 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/117196.jpg` | 0.3198147118 | 0.3211060166 | 0.3272319734 | 0.001291304827 | 0.0074172616 |
| 161 | 161 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/949146.jpg` | 0.9916574359 | 0.991648972 | 0.9916480184 | 8.463859558e-06 | 9.417533875e-06 |
| 162 | 162 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/592761.jpg` | 0.5543006063 | 0.5556373 | 0.5541668534 | 0.001336693764 | 0.0001337528229 |
| 163 | 163 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/394147.jpg` | 0.2689679861 | 0.2698003352 | 0.2687127888 | 0.000832349062 | 0.0002551972866 |
| 164 | 164 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/19660.jpg` | 0.01903843135 | 0.01906633377 | 0.01859188639 | 2.790242434e-05 | 0.0004465449601 |
| 165 | 165 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/270397.jpg` | 0.1768513024 | 0.1773094684 | 0.1777440608 | 0.0004581660032 | 0.0008927583694 |
| 166 | 166 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/10944.jpg` | 0.3011105359 | 0.3006153107 | 0.302739948 | 0.0004952251911 | 0.001629412174 |
| 167 | 167 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/834312.jpg` | 0.8334289193 | 0.8339110017 | 0.8344651461 | 0.0004820823669 | 0.001036226749 |
| 168 | 168 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/696446.jpg` | 0.1611016691 | 0.1610548049 | 0.1609359831 | 4.686415195e-05 | 0.0001656860113 |
| 169 | 169 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/816198.jpg` | 0.3821139932 | 0.3820100725 | 0.3757106364 | 0.0001039206982 | 0.006403356791 |
| 170 | 170 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/181544.jpg` | 0.2688412964 | 0.2699197829 | 0.2686810493 | 0.001078486443 | 0.0001602470875 |
| 171 | 171 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/299194.jpg` | 0.4198234677 | 0.421392262 | 0.4165387154 | 0.00156879425 | 0.003284752369 |
| 172 | 172 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/127943.jpg` | 0.5282941461 | 0.5281729698 | 0.5321726799 | 0.0001211762428 | 0.00387853384 |
| 173 | 173 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/456297.jpg` | 0.9593398571 | 0.9604820013 | 0.960449636 | 0.001142144203 | 0.001109778881 |
| 174 | 174 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/491884.jpg` | 0.8843153119 | 0.8850514889 | 0.8854343891 | 0.0007361769676 | 0.001119077206 |
| 175 | 175 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/857948.jpg` | 0.184330672 | 0.1850775927 | 0.1868865937 | 0.0007469207048 | 0.002555921674 |
| 176 | 176 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/903.jpg` | 0.2964258194 | 0.2932803035 | 0.2897020876 | 0.003145515919 | 0.006723731756 |
| 177 | 177 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/575335.jpg` | 0.6500950456 | 0.6515967846 | 0.6502506733 | 0.001501739025 | 0.0001556277275 |
| 178 | 178 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/52328.jpg` | 0.9042325616 | 0.9046024084 | 0.9046806097 | 0.0003698468208 | 0.0004480481148 |
| 179 | 179 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/329404.jpg` | 0.8668799996 | 0.8693797588 | 0.8688495159 | 0.002499759197 | 0.001969516277 |
| 180 | 180 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/894624.jpg` | 0.9930484295 | 0.9931609631 | 0.9930925369 | 0.0001125335693 | 4.410743713e-05 |
| 181 | 181 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/362478.jpg` | 0.7077351809 | 0.7071855068 | 0.7063227892 | 0.0005496740341 | 0.001412391663 |
| 182 | 182 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/371080.jpg` | 0.6187392473 | 0.6218915582 | 0.6231323481 | 0.003152310848 | 0.004393100739 |
| 183 | 183 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/756697.jpg` | 0.9908339381 | 0.9908374548 | 0.9909519553 | 3.516674042e-06 | 0.0001180171967 |
| 184 | 184 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/376698.jpg` | 0.9384233952 | 0.9385780692 | 0.9380367994 | 0.0001546740532 | 0.000386595726 |
| 185 | 185 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/607467.jpg` | 0.3421879113 | 0.3442223966 | 0.3446678817 | 0.00203448534 | 0.002479970455 |
| 186 | 186 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/651714.jpg` | 0.4019372165 | 0.4018711746 | 0.3987562954 | 6.604194641e-05 | 0.003180921078 |
| 187 | 187 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/754955.jpg` | 0.8619902134 | 0.8618935943 | 0.8612272739 | 9.661912918e-05 | 0.0007629394531 |
| 188 | 188 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/315286.jpg` | 0.8214443922 | 0.8214660287 | 0.8204045892 | 2.163648605e-05 | 0.001039803028 |
| 189 | 189 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/787841.jpg` | 0.6881000996 | 0.6888666153 | 0.6914697289 | 0.0007665157318 | 0.003369629383 |
| 190 | 190 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/532390.jpg` | 0.4161015749 | 0.4167163074 | 0.4120388925 | 0.0006147325039 | 0.00406268239 |
| 191 | 191 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/818267.jpg` | 0.3678689897 | 0.3684025109 | 0.3649511039 | 0.0005335211754 | 0.00291788578 |
| 192 | 192 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/932185.jpg` | 0.7664570808 | 0.7672539949 | 0.7638332248 | 0.0007969141006 | 0.002623856068 |
| 193 | 193 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/742875.jpg` | 0.8747449517 | 0.8764746189 | 0.877606988 | 0.001729667187 | 0.002862036228 |
| 194 | 194 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/656976.jpg` | 0.08855123818 | 0.09005822986 | 0.0892861113 | 0.001506991684 | 0.000734873116 |
| 195 | 195 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/344479.jpg` | 0.3830100596 | 0.3844274879 | 0.3842488229 | 0.001417428255 | 0.001238763332 |
| 196 | 196 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/703965.jpg` | 0.6029874086 | 0.6058768034 | 0.6058896184 | 0.00288939476 | 0.002902209759 |
| 197 | 197 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/513636.jpg` | 0.4718457758 | 0.4724506438 | 0.4721392989 | 0.0006048679352 | 0.0002935230732 |
| 198 | 198 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/557036.jpg` | 0.5755035281 | 0.5756685138 | 0.5761334896 | 0.0001649856567 | 0.0006299614906 |
| 199 | 199 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/706766.jpg` | 0.1623788625 | 0.1633661091 | 0.1634960771 | 0.0009872466326 | 0.001117214561 |
| 200 | 200 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/358226.jpg` | 0.04834924266 | 0.04839361086 | 0.04736699909 | 4.436820745e-05 | 0.0009822435677 |
| 201 | 201 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/8593.jpg` | 0.4016551971 | 0.4020066559 | 0.3919541836 | 0.0003514587879 | 0.009701013565 |
| 202 | 202 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/510961.jpg` | 0.7916832566 | 0.7924262285 | 0.7877371907 | 0.0007429718971 | 0.003946065903 |
| 203 | 203 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/76934.jpg` | 0.2918132246 | 0.2926039696 | 0.2881095111 | 0.0007907450199 | 0.003703713417 |
| 204 | 204 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/106654.jpg` | 0.05421742052 | 0.05466050655 | 0.05349813774 | 0.0004430860281 | 0.0007192827761 |
| 205 | 205 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/909521.jpg` | 0.6468259692 | 0.6479024291 | 0.6554839611 | 0.001076459885 | 0.008657991886 |
| 206 | 206 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/78637.jpg` | 0.2402276993 | 0.2404283136 | 0.2406929582 | 0.0002006143332 | 0.000465258956 |
| 207 | 207 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/711882.jpg` | 0.4025153518 | 0.4049702585 | 0.4000237584 | 0.002454906702 | 0.002491593361 |
| 208 | 208 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/430450.jpg` | 0.6525956988 | 0.6528437138 | 0.6586064696 | 0.0002480149269 | 0.006010770798 |
| 209 | 209 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/576316.jpg` | 0.5502163768 | 0.5503104925 | 0.5487928391 | 9.41157341e-05 | 0.001423537731 |
| 210 | 210 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/93137.jpg` | 0.2283591181 | 0.2299372703 | 0.228429243 | 0.00157815218 | 7.012486458e-05 |
| 211 | 211 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/426624.jpg` | 0.5171808004 | 0.5196897984 | 0.5200257301 | 0.002508997917 | 0.002844929695 |
| 212 | 212 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/133469.jpg` | 0.7067782283 | 0.7076129913 | 0.7045975924 | 0.0008347630501 | 0.002180635929 |
| 213 | 213 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/552496.jpg` | 0.5533373952 | 0.5563108921 | 0.5577192903 | 0.002973496914 | 0.004381895065 |
| 214 | 214 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/503646.jpg` | 0.9370878339 | 0.9376865625 | 0.9371289611 | 0.0005987286568 | 4.11272049e-05 |
| 215 | 215 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/42617.jpg` | 0.1892690957 | 0.1891577095 | 0.1869679093 | 0.0001113861799 | 0.002301186323 |
| 216 | 216 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/494720.jpg` | 0.4828119278 | 0.4848525226 | 0.4805606008 | 0.002040594816 | 0.002251327038 |
| 217 | 217 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/802009.jpg` | 0.794598043 | 0.7970026731 | 0.796882391 | 0.002404630184 | 0.002284348011 |
| 218 | 218 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/280263.jpg` | 0.4388027489 | 0.4397059977 | 0.4337788224 | 0.0009032487869 | 0.005023926497 |
| 219 | 219 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/798000.jpg` | 0.7540524602 | 0.7553239465 | 0.7528495789 | 0.001271486282 | 0.001202881336 |
| 220 | 220 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/516677.jpg` | 0.4905670881 | 0.4903392196 | 0.4910931587 | 0.000227868557 | 0.0005260705948 |
| 221 | 221 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/292772.jpg` | 0.481454432 | 0.4809094369 | 0.4820864797 | 0.0005449950695 | 0.0006320476532 |
| 222 | 222 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/543948.jpg` | 0.4456566274 | 0.4482398331 | 0.4492734373 | 0.0025832057 | 0.003616809845 |
| 223 | 223 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/922725.jpg` | 0.9135274887 | 0.9155932665 | 0.9161797166 | 0.002065777779 | 0.002652227879 |
| 224 | 224 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/485640.jpg` | 0.8056482673 | 0.8086904287 | 0.8100304008 | 0.003042161465 | 0.004382133484 |
| 225 | 225 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/15966.jpg` | 0.02870463952 | 0.02925461903 | 0.02955307811 | 0.0005499795079 | 0.0008484385908 |
| 226 | 226 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/9804.jpg` | 0.2492582202 | 0.2501628101 | 0.24820292 | 0.0009045898914 | 0.001055300236 |
| 227 | 227 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/648315.jpg` | 0.322368294 | 0.3237463832 | 0.3203105032 | 0.00137808919 | 0.002057790756 |
| 228 | 228 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/15980.jpg` | 0.3192430437 | 0.3214701712 | 0.3245497346 | 0.002227127552 | 0.005306690931 |
| 229 | 229 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/477198.jpg` | 0.919801116 | 0.9201399088 | 0.9202447534 | 0.0003387928009 | 0.0004436373711 |
| 230 | 230 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/194101.jpg` | 0.5310073495 | 0.5307190418 | 0.5301758051 | 0.0002883076668 | 0.0008315443993 |
| 231 | 231 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/749244.jpg` | 0.3241870105 | 0.3242015541 | 0.3166706264 | 1.454353333e-05 | 0.007516384125 |
| 232 | 232 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/593274.jpg` | 0.7058413625 | 0.708121717 | 0.7080340385 | 0.0022803545 | 0.002192676067 |
| 233 | 233 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/533484.jpg` | 0.5822715759 | 0.5833771229 | 0.5833079219 | 0.001105546951 | 0.001036345959 |
| 234 | 234 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/570268.jpg` | 0.9350509644 | 0.9353299737 | 0.9328450561 | 0.0002790093422 | 0.002205908298 |
| 235 | 235 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/380947.jpg` | 0.4114108384 | 0.4128903449 | 0.4162512124 | 0.001479506493 | 0.004840373993 |
| 236 | 236 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/320581.jpg` | 0.1995417774 | 0.1990510225 | 0.1983331293 | 0.0004907548428 | 0.001208648086 |
| 237 | 237 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/439043.jpg` | 0.5713868737 | 0.5742789507 | 0.5790379643 | 0.002892076969 | 0.007651090622 |
| 238 | 238 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/710114.jpg` | 0.4054378867 | 0.4070920944 | 0.4076951146 | 0.001654207706 | 0.002257227898 |
| 239 | 239 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/900069.jpg` | 0.9382733107 | 0.938105166 | 0.9408195019 | 0.0001681447029 | 0.002546191216 |
| 240 | 240 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/74222.jpg` | 0.06715734303 | 0.06777834147 | 0.06731328368 | 0.0006209984422 | 0.0001559406519 |
| 241 | 241 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/117318.jpg` | 0.2477221787 | 0.2462431639 | 0.244468689 | 0.001479014754 | 0.003253489733 |
| 242 | 242 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/291847.jpg` | 0.264459461 | 0.2681512535 | 0.2703294456 | 0.003691792488 | 0.005869984627 |
| 243 | 243 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/298812.jpg` | 0.802633822 | 0.8042111397 | 0.801671207 | 0.001577317715 | 0.0009626150131 |
| 244 | 244 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/881641.jpg` | 0.4134443104 | 0.4150497019 | 0.4180004597 | 0.001605391502 | 0.004556149244 |
| 245 | 245 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/170305.jpg` | 0.3446725607 | 0.3467047811 | 0.3446211219 | 0.002032220364 | 5.143880844e-05 |
| 246 | 246 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/41283.jpg` | 0.5300442576 | 0.5313971043 | 0.5277163982 | 0.001352846622 | 0.002327859402 |
| 247 | 247 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/18904.jpg` | 0.2919206023 | 0.2915533781 | 0.2920327187 | 0.0003672242165 | 0.0001121163368 |
| 248 | 248 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/204602.jpg` | 0.203543514 | 0.2037029266 | 0.2038260847 | 0.0001594126225 | 0.0002825707197 |
| 249 | 249 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/328905.jpg` | 0.5809691548 | 0.5842244029 | 0.5811219215 | 0.00325524807 | 0.0001527667046 |
| 250 | 250 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/139886.jpg` | 0.8061119914 | 0.8072835207 | 0.802183032 | 0.001171529293 | 0.00392895937 |
| 251 | 251 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/670929.jpg` | 0.9114509225 | 0.912504673 | 0.9106677771 | 0.001053750515 | 0.0007831454277 |
| 252 | 252 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/537874.jpg` | 0.6539210081 | 0.6535822153 | 0.6498414278 | 0.0003387928009 | 0.004079580307 |
| 253 | 253 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/319776.jpg` | 0.2346196771 | 0.2346461564 | 0.2335268855 | 2.647936344e-05 | 0.001092791557 |
| 254 | 254 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/144826.jpg` | 0.388029784 | 0.3913105428 | 0.3815572262 | 0.003280758858 | 0.006472557783 |
| 255 | 255 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/611368.jpg` | 0.9354051948 | 0.9358420372 | 0.9345903993 | 0.0004368424416 | 0.0008147954941 |
| 256 | 256 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/514341.jpg` | 0.5107236505 | 0.5100402832 | 0.5077496171 | 0.0006833672523 | 0.002974033356 |
| 257 | 257 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/759530.jpg` | 0.6597107649 | 0.6598929167 | 0.6591228247 | 0.0001821517944 | 0.0005879402161 |
| 258 | 258 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/280.jpg` | 0.3016117513 | 0.3034981787 | 0.3017290235 | 0.001886427402 | 0.0001172721386 |
| 259 | 259 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/303444.jpg` | 0.5155798197 | 0.5217789412 | 0.5225325823 | 0.006199121475 | 0.006952762604 |
| 260 | 260 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/592759.jpg` | 0.7370941043 | 0.7399829626 | 0.7388379574 | 0.002888858318 | 0.001743853092 |
| 261 | 261 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/392964.jpg` | 0.7199565172 | 0.7203428745 | 0.7199952602 | 0.0003863573074 | 3.87430191e-05 |
| 262 | 262 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/855657.jpg` | 0.3349530101 | 0.3331330121 | 0.3240749836 | 0.001819998026 | 0.01087802649 |
| 263 | 263 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/328471.jpg` | 0.4905048609 | 0.490565598 | 0.4899831414 | 6.073713303e-05 | 0.0005217194557 |
| 264 | 264 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/723095.jpg` | 0.9776250124 | 0.9773769975 | 0.977747798 | 0.0002480149269 | 0.0001227855682 |
| 265 | 265 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/15187.jpg` | 0.368850559 | 0.3710129857 | 0.3708220124 | 0.00216242671 | 0.001971453428 |
| 266 | 266 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/738046.jpg` | 0.4943672121 | 0.4941871762 | 0.4939379692 | 0.0001800358295 | 0.0004292428493 |
| 267 | 267 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/660190.jpg` | 0.709025979 | 0.7093710899 | 0.7085670829 | 0.0003451108932 | 0.0004588961601 |
| 268 | 268 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/924314.jpg` | 0.9927338362 | 0.9928104877 | 0.9928361773 | 7.665157318e-05 | 0.0001023411751 |
| 269 | 269 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/6048.jpg` | 0.2633034587 | 0.2639565766 | 0.2620782852 | 0.0006531178951 | 0.001225173473 |
| 270 | 270 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/265419.jpg` | 0.08296369016 | 0.08261451125 | 0.08083884418 | 0.0003491789103 | 0.002124845982 |
| 271 | 271 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/30646.jpg` | 0.6387440562 | 0.6409711242 | 0.6362498999 | 0.002227067947 | 0.002494156361 |
| 272 | 272 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/94703.jpg` | 0.07537934184 | 0.07614789903 | 0.07726147771 | 0.0007685571909 | 0.001882135868 |
| 273 | 273 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/262695.jpg` | 0.8861991763 | 0.8864640594 | 0.8914008737 | 0.0002648830414 | 0.00520169735 |
| 274 | 274 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/180818.jpg` | 0.1771036983 | 0.1776785105 | 0.1755921841 | 0.0005748122931 | 0.001511514187 |
| 275 | 275 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/372237.jpg` | 0.7052237988 | 0.7079447508 | 0.7056148052 | 0.002720952034 | 0.0003910064697 |
| 276 | 276 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/19177.jpg` | 0.370772332 | 0.3754318357 | 0.3765406609 | 0.004659503698 | 0.005768328905 |
| 277 | 277 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/6399.jpg` | 0.4045106769 | 0.4030199349 | 0.4049187005 | 0.001490741968 | 0.0004080235958 |
| 278 | 278 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/57753.jpg` | 0.0468971841 | 0.04695857316 | 0.04657734558 | 6.138905883e-05 | 0.0003198385239 |
| 279 | 279 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/863564.jpg` | 0.6737921238 | 0.6743358374 | 0.6724264622 | 0.0005437135696 | 0.001365661621 |
| 280 | 280 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/460141.jpg` | 0.3869268894 | 0.3875788152 | 0.3819936216 | 0.0006519258022 | 0.004933267832 |
| 281 | 281 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/621944.jpg` | 0.8810299039 | 0.8811041713 | 0.8806900382 | 7.426738739e-05 | 0.0003398656845 |
| 282 | 282 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/674651.jpg` | 0.9124946594 | 0.9132998586 | 0.9092404842 | 0.0008051991463 | 0.003254175186 |
| 283 | 283 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/138733.jpg` | 0.2093027085 | 0.210994035 | 0.2057422251 | 0.001691326499 | 0.003560483456 |
| 284 | 284 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/934873.jpg` | 0.8609257936 | 0.8618115783 | 0.8594725132 | 0.000885784626 | 0.001453280449 |
| 285 | 285 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/120316.jpg` | 0.2498146445 | 0.2507629693 | 0.2499123514 | 0.0009483247995 | 9.770691395e-05 |
| 286 | 286 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/917354.jpg` | 0.9322443604 | 0.9325353503 | 0.9322205782 | 0.0002909898758 | 2.378225327e-05 |
| 287 | 287 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/809721.jpg` | 0.2523032725 | 0.2548357248 | 0.2508130968 | 0.002532452345 | 0.001490175724 |
| 288 | 288 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/372034.jpg` | 0.8993442059 | 0.9003673196 | 0.9002566338 | 0.001023113728 | 0.0009124279022 |
| 289 | 289 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/274393.jpg` | 0.9105451107 | 0.9103245139 | 0.9108656049 | 0.0002205967903 | 0.000320494175 |
| 290 | 290 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/921177.jpg` | 0.9248949885 | 0.9257999063 | 0.9271968007 | 0.000904917717 | 0.002301812172 |
| 291 | 291 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/23969.jpg` | 0.3783720732 | 0.3809595406 | 0.3781733215 | 0.002587467432 | 0.000198751688 |
| 292 | 292 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/528382.jpg` | 0.1902218759 | 0.1915940642 | 0.1877568513 | 0.00137218833 | 0.00246502459 |
| 293 | 293 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/432104.jpg` | 0.7509046197 | 0.7572695613 | 0.7600200176 | 0.006364941597 | 0.00911539793 |
| 294 | 294 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/42566.jpg` | 0.03963048011 | 0.04016199708 | 0.04071430489 | 0.0005315169692 | 0.001083824784 |
| 295 | 295 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/775694.jpg` | 0.2616988719 | 0.2624602616 | 0.2719140053 | 0.0007613897324 | 0.01021513343 |
| 296 | 296 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/804283.jpg` | 0.6027565002 | 0.6030443907 | 0.6036731005 | 0.0002878904343 | 0.0009166002274 |
| 297 | 297 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/280281.jpg` | 0.469776541 | 0.4730257094 | 0.4694435596 | 0.003249168396 | 0.000332981348 |
| 298 | 298 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/379965.jpg` | 0.2855494618 | 0.2865256071 | 0.2916970551 | 0.0009761452675 | 0.00614759326 |
| 299 | 299 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/951104.jpg` | 0.04878823459 | 0.04845261946 | 0.04920731485 | 0.0003356151283 | 0.0004190802574 |
| 300 | 300 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/761278.jpg` | 0.2350064069 | 0.2346327901 | 0.2346190661 | 0.0003736168146 | 0.0003873407841 |
| 301 | 301 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/934023.jpg` | 0.5267646313 | 0.5273628235 | 0.5204862356 | 0.000598192215 | 0.006278395653 |
| 302 | 302 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/234452.jpg` | 0.7387027144 | 0.7414164543 | 0.7328215837 | 0.002713739872 | 0.005881130695 |
| 303 | 303 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/112345.jpg` | 0.01665139385 | 0.01659280434 | 0.01588393934 | 5.858950317e-05 | 0.000767454505 |
| 304 | 304 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/349274.jpg` | 0.2162264287 | 0.216558978 | 0.2173005491 | 0.0003325492144 | 0.001074120402 |
| 305 | 305 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/156447.jpg` | 0.6707646251 | 0.6709418297 | 0.6731668711 | 0.0001772046089 | 0.002402245998 |
| 306 | 306 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/227013.jpg` | 0.08103171736 | 0.08170987666 | 0.08102565259 | 0.0006781592965 | 6.064772606e-06 |
| 307 | 307 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/145824.jpg` | 0.6834324002 | 0.6843307614 | 0.6856114864 | 0.0008983612061 | 0.002179086208 |
| 308 | 308 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/583270.jpg` | 0.3082137108 | 0.3090642393 | 0.3048119545 | 0.0008505284786 | 0.003401756287 |
| 309 | 309 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/36035.jpg` | 0.4739434719 | 0.4745452404 | 0.4730037451 | 0.0006017684937 | 0.0009397268295 |
| 310 | 310 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/94953.jpg` | 0.0725607127 | 0.07307505608 | 0.07143705338 | 0.0005143433809 | 0.001123659313 |
| 311 | 311 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/377296.jpg` | 0.9768377542 | 0.9773443937 | 0.9768295884 | 0.0005066394806 | 8.165836334e-06 |
| 312 | 312 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/836141.jpg` | 0.7937225699 | 0.7991544008 | 0.8050657511 | 0.005431830883 | 0.01134318113 |
| 313 | 313 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/25924.jpg` | 0.2456856966 | 0.2501679361 | 0.2546817958 | 0.004482239485 | 0.008996099234 |
| 314 | 314 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/471350.jpg` | 0.7536417246 | 0.7559276819 | 0.7576818466 | 0.002285957336 | 0.004040122032 |
| 315 | 315 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/270731.jpg` | 0.1594578922 | 0.1621865481 | 0.1598823816 | 0.002728655934 | 0.0004244893789 |
| 316 | 316 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/636145.jpg` | 0.8732340932 | 0.8737396002 | 0.8712730408 | 0.0005055069923 | 0.001961052418 |
| 317 | 317 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/365851.jpg` | 0.8451325893 | 0.8476407528 | 0.8485928774 | 0.002508163452 | 0.003460288048 |
| 318 | 318 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/498007.jpg` | 0.346890986 | 0.3509832621 | 0.3460042477 | 0.004092276096 | 0.0008867383003 |
| 319 | 319 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/12157.jpg` | 0.4351325929 | 0.4358659387 | 0.4258561134 | 0.000733345747 | 0.009276479483 |
| 320 | 320 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/298259.jpg` | 0.2825254202 | 0.286596477 | 0.289134562 | 0.004071056843 | 0.006609141827 |
| 321 | 321 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/808281.jpg` | 0.5819969773 | 0.5847344398 | 0.5982257128 | 0.002737462521 | 0.01622873545 |
| 322 | 322 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/505750.jpg` | 0.10446392 | 0.1044233516 | 0.1036331803 | 4.056841135e-05 | 0.0008307397366 |
| 323 | 323 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/215032.jpg` | 0.3678013086 | 0.3665262163 | 0.3651637137 | 0.001275092363 | 0.002637594938 |
| 324 | 324 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/486930.jpg` | 0.2937916815 | 0.2932944298 | 0.2921465039 | 0.000497251749 | 0.001645177603 |
| 325 | 325 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/906144.jpg` | 0.4565616548 | 0.4602305591 | 0.4667035937 | 0.003668904305 | 0.01014193892 |
| 326 | 326 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/31818.jpg` | 0.2457831353 | 0.2440738082 | 0.2367338389 | 0.001709327102 | 0.009049296379 |
| 327 | 327 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/822640.jpg` | 0.436950624 | 0.4378820956 | 0.4338279665 | 0.0009314715862 | 0.003122657537 |
| 328 | 328 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/252354.jpg` | 0.4773240685 | 0.4746201634 | 0.4778283834 | 0.002703905106 | 0.0005043148994 |
| 329 | 329 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/12327.jpg` | 0.4742773473 | 0.4727384746 | 0.4753144085 | 0.001538872719 | 0.001037061214 |
| 330 | 330 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/850739.jpg` | 0.1495979428 | 0.1519886553 | 0.1490797549 | 0.0023907125 | 0.0005181878805 |
| 331 | 331 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/933827.jpg` | 0.3634986579 | 0.3667604923 | 0.3571386933 | 0.003261834383 | 0.006359964609 |
| 332 | 332 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/564402.jpg` | 0.3609372079 | 0.3622273207 | 0.361558646 | 0.001290112734 | 0.0006214380264 |
| 333 | 333 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/379480.jpg` | 0.1627824754 | 0.1653398126 | 0.1647705734 | 0.002557337284 | 0.001988098025 |
| 334 | 334 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/192368.jpg` | 0.50995332 | 0.5117355585 | 0.5138679743 | 0.001782238483 | 0.003914654255 |
| 335 | 335 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/547448.jpg` | 0.9101051092 | 0.911254406 | 0.9105700254 | 0.001149296761 | 0.0004649162292 |
| 336 | 336 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/707468.jpg` | 0.5108876824 | 0.5147175789 | 0.5118857622 | 0.00382989645 | 0.0009980797768 |
| 337 | 337 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/525312.jpg` | 0.2558243275 | 0.2575687766 | 0.2555889487 | 0.001744449139 | 0.0002353787422 |
| 338 | 338 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/677973.jpg` | 0.6130326986 | 0.6161899567 | 0.6179099083 | 0.003157258034 | 0.004877209663 |
| 339 | 339 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/44812.jpg` | 0.469845593 | 0.471673727 | 0.4716343284 | 0.00182813406 | 0.00178873539 |
| 340 | 340 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/668568.jpg` | 0.6892416477 | 0.689676702 | 0.6890974045 | 0.0004350543022 | 0.0001442432404 |
| 341 | 341 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/652118.jpg` | 0.5873340964 | 0.5865261555 | 0.5709520578 | 0.0008079409599 | 0.01638203859 |
| 342 | 342 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/11521.jpg` | 0.1581092626 | 0.1601212174 | 0.1606715471 | 0.002011954784 | 0.00256228447 |
| 343 | 343 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/836656.jpg` | 0.5853969455 | 0.5884488821 | 0.5845338106 | 0.003051936626 | 0.000863134861 |
| 344 | 344 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/301720.jpg` | 0.7111591101 | 0.7119814754 | 0.7112290263 | 0.000822365284 | 6.991624832e-05 |
| 345 | 345 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/24667.jpg` | 0.2779279947 | 0.2796999216 | 0.2752174437 | 0.00177192688 | 0.002710551023 |
| 346 | 346 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/828675.jpg` | 0.8370312452 | 0.8405591249 | 0.8400632739 | 0.003527879715 | 0.003032028675 |
| 347 | 347 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/608742.jpg` | 0.7867593169 | 0.7870341539 | 0.7897785902 | 0.0002748370171 | 0.003019273281 |
| 348 | 348 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/798094.jpg` | 0.5325649977 | 0.5328814387 | 0.5310873389 | 0.0003164410591 | 0.001477658749 |
| 349 | 349 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/351006.jpg` | 0.1337694824 | 0.1356327832 | 0.1366894245 | 0.0018633008 | 0.002919942141 |
| 350 | 350 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/837859.jpg` | 0.3292471766 | 0.3294383287 | 0.3277032971 | 0.0001911520958 | 0.001543879509 |
| 351 | 351 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/526829.jpg` | 0.8863781095 | 0.8876128197 | 0.889836669 | 0.001234710217 | 0.003458559513 |
| 352 | 352 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/200601.jpg` | 0.324319005 | 0.3270547986 | 0.3310137391 | 0.002735793591 | 0.006694734097 |
| 353 | 353 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/835454.jpg` | 0.7926067114 | 0.7921997905 | 0.7869067192 | 0.0004069209099 | 0.00569999218 |
| 354 | 354 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/702636.jpg` | 0.8717889786 | 0.8732034564 | 0.8733385801 | 0.001414477825 | 0.001549601555 |
| 355 | 355 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/897286.jpg` | 0.9536569715 | 0.9534181952 | 0.9539029002 | 0.000238776207 | 0.0002459287643 |
| 356 | 356 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/503131.jpg` | 0.3602722585 | 0.3611925542 | 0.3580961525 | 0.0009202957153 | 0.002176105976 |
| 357 | 357 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/21217.jpg` | 0.2970694304 | 0.298138231 | 0.3002618849 | 0.001068800688 | 0.003192454576 |
| 358 | 358 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/69111.jpg` | 0.120586738 | 0.1220768988 | 0.1212808266 | 0.001490160823 | 0.0006940886378 |
| 359 | 359 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/215132.jpg` | 0.5843833685 | 0.585723877 | 0.5881886482 | 0.001340508461 | 0.003805279732 |
| 360 | 360 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/751386.jpg` | 0.9061866403 | 0.9055150747 | 0.9072173834 | 0.0006715655327 | 0.001030743122 |
| 361 | 361 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/6038.jpg` | 0.3974914849 | 0.3969393671 | 0.4030872285 | 0.0005521178246 | 0.005595743656 |
| 362 | 362 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/422767.jpg` | 0.6386547089 | 0.6410158873 | 0.6526684761 | 0.002361178398 | 0.01401376724 |
| 363 | 363 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/11362.jpg` | 0.5790733099 | 0.5804446936 | 0.5784657598 | 0.001371383667 | 0.0006075501442 |
| 364 | 364 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/348167.jpg` | 0.8744217753 | 0.8747515678 | 0.8731660247 | 0.0003297924995 | 0.001255750656 |
| 365 | 365 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/906646.jpg` | 0.9183100462 | 0.9184101224 | 0.9182918072 | 0.0001000761986 | 1.82390213e-05 |
| 366 | 366 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/166358.jpg` | 0.8218920231 | 0.8251737356 | 0.8266495466 | 0.003281712532 | 0.004757523537 |
| 367 | 367 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/674799.jpg` | 0.8445661664 | 0.8456371427 | 0.8450717926 | 0.001070976257 | 0.0005056262016 |
| 368 | 368 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/228217.jpg` | 0.5542171597 | 0.5567040443 | 0.5492158532 | 0.002486884594 | 0.005001306534 |
| 369 | 369 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/521326.jpg` | 0.9417808652 | 0.9424038529 | 0.9407764077 | 0.0006229877472 | 0.001004457474 |
| 370 | 370 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/547984.jpg` | 0.2799331546 | 0.2795899808 | 0.2778186202 | 0.0003431737423 | 0.002114534378 |
| 371 | 371 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/833192.jpg` | 0.6494524479 | 0.6508640051 | 0.6505402327 | 0.001411557198 | 0.001087784767 |
| 372 | 372 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/842818.jpg` | 0.9182009101 | 0.9183858037 | 0.917951107 | 0.0001848936081 | 0.0002498030663 |
| 373 | 373 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/44281.jpg` | 0.3644195199 | 0.3663653135 | 0.3613921404 | 0.001945793629 | 0.003027379513 |
| 374 | 374 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/314487.jpg` | 0.9345118999 | 0.93438375 | 0.9337137938 | 0.0001281499863 | 0.0007981061935 |
| 375 | 375 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/398290.jpg` | 0.5513474345 | 0.5517857075 | 0.549906373 | 0.000438272953 | 0.001441061497 |
| 376 | 376 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/450807.jpg` | 0.38413921 | 0.3833089471 | 0.3837405741 | 0.0008302628994 | 0.0003986358643 |
| 377 | 377 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/901503.jpg` | 0.7823035121 | 0.782528758 | 0.7816240788 | 0.0002252459526 | 0.0006794333458 |
| 378 | 378 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/476796.jpg` | 0.5743342042 | 0.5750249624 | 0.5705384612 | 0.0006907582283 | 0.003795742989 |
| 379 | 379 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/495892.jpg` | 0.7681235075 | 0.7694993615 | 0.77130723 | 0.001375854015 | 0.003183722496 |
| 380 | 380 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/108307.jpg` | 0.7580100894 | 0.7590614557 | 0.7588710785 | 0.001051366329 | 0.0008609890938 |
| 381 | 381 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/316362.jpg` | 0.8347740769 | 0.8355692029 | 0.8373689651 | 0.0007951259613 | 0.00259488821 |
| 382 | 382 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/221730.jpg` | 0.2237048447 | 0.2251187712 | 0.2207275629 | 0.001413926482 | 0.002977281809 |
| 383 | 383 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/242224.jpg` | 0.6706671119 | 0.6724230051 | 0.6719309092 | 0.00175589323 | 0.001263797283 |
| 384 | 384 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/207936.jpg` | 0.3596533537 | 0.3608240485 | 0.3584613502 | 0.001170694828 | 0.001192003489 |
| 385 | 385 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/522370.jpg` | 0.9442228675 | 0.9446910024 | 0.9452903271 | 0.0004681348801 | 0.001067459583 |
| 386 | 386 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/301291.jpg` | 0.8771032095 | 0.8776738644 | 0.8770002723 | 0.0005706548691 | 0.0001029372215 |
| 387 | 387 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/814881.jpg` | 0.4726332724 | 0.4739394784 | 0.4739465415 | 0.001306205988 | 0.001313269138 |
| 388 | 388 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/483966.jpg` | 0.7784980536 | 0.779286027 | 0.7799184918 | 0.0007879734039 | 0.00142043829 |
| 389 | 389 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/365.jpg` | 0.5133858323 | 0.5174583793 | 0.5216374397 | 0.004072546959 | 0.008251607418 |
| 390 | 390 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/328575.jpg` | 0.1723820269 | 0.1740341336 | 0.1743624657 | 0.001652106643 | 0.001980438828 |
| 391 | 391 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/823118.jpg` | 0.8404026031 | 0.8417970538 | 0.8421062231 | 0.001394450665 | 0.001703619957 |
| 392 | 392 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/280620.jpg` | 0.8045147061 | 0.8055343032 | 0.8056293726 | 0.001019597054 | 0.001114666462 |
| 393 | 393 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/590815.jpg` | 0.370642364 | 0.3759467602 | 0.3627195358 | 0.005304396152 | 0.007922828197 |
| 394 | 394 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/728203.jpg` | 0.4768811166 | 0.4782821238 | 0.4819400907 | 0.001401007175 | 0.005058974028 |
| 395 | 395 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/743401.jpg` | 0.1813894361 | 0.1836185753 | 0.1815875471 | 0.002229139209 | 0.0001981109381 |
| 396 | 396 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/422984.jpg` | 0.03644542396 | 0.03689233959 | 0.0375613831 | 0.0004469156265 | 0.001115959138 |
| 397 | 397 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/427219.jpg` | 0.6594464779 | 0.6599782705 | 0.6589386463 | 0.0005317926407 | 0.0005078315735 |
| 398 | 398 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/155497.jpg` | 0.6586582065 | 0.6663844585 | 0.6630209684 | 0.007726252079 | 0.004362761974 |
| 399 | 399 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/226144.jpg` | 0.5644127131 | 0.5666445494 | 0.5638333559 | 0.002231836319 | 0.0005793571472 |
| 400 | 400 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/529180.jpg` | 0.2089216709 | 0.2094878256 | 0.2088703662 | 0.0005661547184 | 5.130469799e-05 |
| 401 | 401 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/8490.jpg` | 0.2959329486 | 0.2958487272 | 0.2905982435 | 8.422136307e-05 | 0.005334705114 |
| 402 | 402 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/578555.jpg` | 0.6927322149 | 0.693020165 | 0.69112885 | 0.0002879500389 | 0.001603364944 |
| 403 | 403 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/75635.jpg` | 0.1128871292 | 0.1154341921 | 0.1152714565 | 0.002547062933 | 0.002384327352 |
| 404 | 404 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/16417.jpg` | 0.5754257441 | 0.5789275765 | 0.5805848837 | 0.003501832485 | 0.005159139633 |
| 405 | 405 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/664681.jpg` | 0.6067934632 | 0.609837234 | 0.6187286973 | 0.00304377079 | 0.01193523407 |
| 406 | 406 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/234328.jpg` | 0.9667479396 | 0.9669091702 | 0.9677227736 | 0.0001612305641 | 0.0009748339653 |
| 407 | 407 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/81924.jpg` | 0.5496578217 | 0.5539797544 | 0.5436694026 | 0.004321932793 | 0.005988419056 |
| 408 | 408 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/920318.jpg` | 0.4838152826 | 0.4859749377 | 0.4902479947 | 0.002159655094 | 0.006432712078 |
| 409 | 409 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/479323.jpg` | 0.8138906956 | 0.8149444461 | 0.8145361543 | 0.001053750515 | 0.0006454586983 |
| 410 | 410 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/546243.jpg` | 0.5593506098 | 0.5612258911 | 0.568092227 | 0.001875281334 | 0.008741617203 |
| 411 | 411 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/444033.jpg` | 0.9892511964 | 0.9894484282 | 0.9896789193 | 0.0001972317696 | 0.0004277229309 |
| 412 | 412 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/456125.jpg` | 0.8195289373 | 0.8208580017 | 0.8177486658 | 0.001329064369 | 0.00178027153 |
| 413 | 413 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/352902.jpg` | 0.2358355671 | 0.2355819792 | 0.2341463715 | 0.0002535879612 | 0.001689195633 |
| 414 | 414 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/131370.jpg` | 0.2362924814 | 0.2378064394 | 0.234333083 | 0.001513957977 | 0.001959398389 |
| 415 | 415 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/41148.jpg` | 0.5432005525 | 0.5435201526 | 0.5446692705 | 0.0003196001053 | 0.001468718052 |
| 416 | 416 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/793072.jpg` | 0.7188003063 | 0.7196006775 | 0.7201962471 | 0.00080037117 | 0.001395940781 |
| 417 | 417 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/791164.jpg` | 0.8853539228 | 0.8858436346 | 0.8857631087 | 0.0004897117615 | 0.0004091858864 |
| 418 | 418 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/95127.jpg` | 0.2338443846 | 0.2356057465 | 0.2318246067 | 0.001761361957 | 0.002019777894 |
| 419 | 419 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/576393.jpg` | 0.03250510618 | 0.03267114237 | 0.03204655647 | 0.0001660361886 | 0.0004585497081 |
| 420 | 420 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/816321.jpg` | 0.5598717928 | 0.5614540577 | 0.5616426468 | 0.0015822649 | 0.001770853996 |
| 421 | 421 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/525348.jpg` | 0.6660728455 | 0.6649680138 | 0.6694215536 | 0.001104831696 | 0.003348708153 |
| 422 | 422 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/875680.jpg` | 0.9806725383 | 0.9809505343 | 0.9807467461 | 0.0002779960632 | 7.420778275e-05 |
| 423 | 423 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/860749.jpg` | 0.661871016 | 0.6637166142 | 0.6660374403 | 0.001845598221 | 0.004166424274 |
| 424 | 424 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/42382.jpg` | 0.05603883415 | 0.05665529147 | 0.05383854359 | 0.0006164573133 | 0.002200290561 |
| 425 | 425 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/805456.jpg` | 0.4913393855 | 0.4942398667 | 0.4942151606 | 0.002900481224 | 0.002875775099 |
| 426 | 426 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/949690.jpg` | 0.03102507256 | 0.03138550743 | 0.03175412863 | 0.0003604348749 | 0.0007290560752 |
| 427 | 427 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/842275.jpg` | 0.1610985994 | 0.1613517553 | 0.1608204991 | 0.0002531558275 | 0.0002781003714 |
| 428 | 428 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/598090.jpg` | 0.2347759902 | 0.2354343683 | 0.2304991335 | 0.000658378005 | 0.00427685678 |
| 429 | 429 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/504759.jpg` | 0.1958815604 | 0.1968685836 | 0.1934671849 | 0.0009870231152 | 0.002414375544 |
| 430 | 430 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/684193.jpg` | 0.8180115819 | 0.8167821169 | 0.8154953718 | 0.001229465008 | 0.002516210079 |
| 431 | 431 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/145121.jpg` | 0.6338298917 | 0.6347385645 | 0.6291475296 | 0.0009086728096 | 0.00468236208 |
| 432 | 432 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/449976.jpg` | 0.9499734044 | 0.9502128959 | 0.9480069876 | 0.0002394914627 | 0.001966416836 |
| 433 | 433 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/216793.jpg` | 0.4883076549 | 0.4894313216 | 0.4915813804 | 0.001123666763 | 0.00327372551 |
| 434 | 434 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/950936.jpg` | 0.5703704357 | 0.5711252689 | 0.576343298 | 0.0007548332214 | 0.005972862244 |
| 435 | 435 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/928341.jpg` | 0.7787892222 | 0.7795028687 | 0.7783582807 | 0.0007136464119 | 0.0004309415817 |
| 436 | 436 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/13320.jpg` | 0.3961316049 | 0.3970158696 | 0.3985035121 | 0.0008842647076 | 0.002371907234 |
| 437 | 437 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/424492.jpg` | 0.2330977023 | 0.2346293032 | 0.235300675 | 0.001531600952 | 0.00220297277 |
| 438 | 438 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/396999.jpg` | 0.429161638 | 0.4319273531 | 0.4287712276 | 0.002765715122 | 0.0003904104233 |
| 439 | 439 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/713645.jpg` | 0.5960474014 | 0.5979779959 | 0.5990880728 | 0.001930594444 | 0.003040671349 |
| 440 | 440 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/517036.jpg` | 0.833912015 | 0.8362790346 | 0.835537374 | 0.002367019653 | 0.001625359058 |
| 441 | 441 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/68187.jpg` | 0.4751525223 | 0.477737397 | 0.4735338688 | 0.00258487463 | 0.001618653536 |
| 442 | 442 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/713289.jpg` | 0.8497187495 | 0.8502709866 | 0.850589335 | 0.0005522370338 | 0.0008705854416 |
| 443 | 443 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/451228.jpg` | 0.1998158991 | 0.2006627917 | 0.2001328766 | 0.0008468925953 | 0.0003169775009 |
| 444 | 444 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/353305.jpg` | 0.3293921649 | 0.3297784925 | 0.3282754719 | 0.0003863275051 | 0.00111669302 |
| 445 | 445 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/606417.jpg` | 0.933906734 | 0.9339131117 | 0.9337362051 | 6.377696991e-06 | 0.0001705288887 |
| 446 | 446 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/298560.jpg` | 0.174860701 | 0.1750854105 | 0.1746485978 | 0.0002247095108 | 0.0002121031284 |
| 447 | 447 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/552273.jpg` | 0.4552426636 | 0.4591418207 | 0.4588367939 | 0.003899157047 | 0.003594130278 |
| 448 | 448 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/868286.jpg` | 0.9082397819 | 0.9079282284 | 0.9070814848 | 0.0003115534782 | 0.001158297062 |
| 449 | 449 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/672103.jpg` | 0.753962636 | 0.7557206154 | 0.7542141676 | 0.001757979393 | 0.000251531601 |
| 450 | 450 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/303118.jpg` | 0.8516879082 | 0.8519482017 | 0.8496467471 | 0.0002602934837 | 0.00204116106 |
| 451 | 451 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/109864.jpg` | 0.6560145617 | 0.6582622528 | 0.6569117308 | 0.002247691154 | 0.0008971691132 |
| 452 | 452 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/679724.jpg` | 0.7265268564 | 0.7274329662 | 0.7291890383 | 0.0009061098099 | 0.002662181854 |
| 453 | 453 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/18661.jpg` | 0.5805609822 | 0.5813058615 | 0.5812786222 | 0.0007448792458 | 0.0007176399231 |
| 454 | 454 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/648924.jpg` | 0.8967772722 | 0.8966690302 | 0.8970990777 | 0.0001082420349 | 0.0003218054771 |
| 455 | 455 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/227985.jpg` | 0.2497891188 | 0.2506165206 | 0.2494569719 | 0.0008274018764 | 0.000332146883 |
| 456 | 456 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/588817.jpg` | 0.8728532791 | 0.8726568818 | 0.8681893945 | 0.0001963973045 | 0.00466388464 |
| 457 | 457 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/265298.jpg` | 0.6439238787 | 0.645344913 | 0.6460996866 | 0.001421034336 | 0.002175807953 |
| 458 | 458 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/233101.jpg` | 0.7910461426 | 0.7920976877 | 0.7884328365 | 0.001051545143 | 0.002613306046 |
| 459 | 459 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/925763.jpg` | 0.8657898903 | 0.8670266867 | 0.8648375273 | 0.001236796379 | 0.0009523630142 |
| 460 | 460 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/177712.jpg` | 0.5096399188 | 0.5102528334 | 0.5087994337 | 0.0006129145622 | 0.000840485096 |
| 461 | 461 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/43438.jpg` | 0.081632182 | 0.08169190586 | 0.08136833459 | 5.972385406e-05 | 0.0002638474107 |
| 462 | 462 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/611835.jpg` | 0.6027895808 | 0.6042174697 | 0.6013971567 | 0.00142788887 | 0.001392424107 |
| 463 | 463 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/737941.jpg` | 0.6164429188 | 0.6190046072 | 0.6183849573 | 0.002561688423 | 0.001942038536 |
| 464 | 464 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/128001.jpg` | 0.1696395129 | 0.1718216538 | 0.1690627784 | 0.002182140946 | 0.0005767345428 |
| 465 | 465 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/891813.jpg` | 0.6912597418 | 0.6895214319 | 0.6823092699 | 0.00173830986 | 0.008950471878 |
| 466 | 466 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/608974.jpg` | 0.4499165118 | 0.4508743286 | 0.449461937 | 0.0009578168392 | 0.0004545748234 |
| 467 | 467 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/554790.jpg` | 0.5607821345 | 0.5673223734 | 0.5714573264 | 0.006540238857 | 0.01067519188 |
| 468 | 468 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/92670.jpg` | 0.09839365631 | 0.1002796739 | 0.09788286686 | 0.001886017621 | 0.000510789454 |
| 469 | 469 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/596272.jpg` | 0.2790420651 | 0.2801250815 | 0.2788161337 | 0.001083016396 | 0.000225931406 |
| 470 | 470 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/606121.jpg` | 0.8658804297 | 0.8669612408 | 0.8652869463 | 0.001080811024 | 0.000593483448 |
| 471 | 471 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/126710.jpg` | 0.7467682362 | 0.7474325299 | 0.7478467822 | 0.000664293766 | 0.001078546047 |
| 472 | 472 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/418248.jpg` | 0.1009071246 | 0.102916345 | 0.1023360863 | 0.002009220421 | 0.001428961754 |
| 473 | 473 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/287987.jpg` | 0.5799077749 | 0.5836527348 | 0.5890715122 | 0.003744959831 | 0.009163737297 |
| 474 | 474 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/350157.jpg` | 0.1371019632 | 0.1391870826 | 0.1373718232 | 0.002085119486 | 0.0002698600292 |
| 475 | 475 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/198850.jpg` | 0.3525590897 | 0.3598398566 | 0.3572627008 | 0.007280766964 | 0.004703611135 |
| 476 | 476 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/98906.jpg` | 0.1275212318 | 0.1277298778 | 0.1245027706 | 0.000208646059 | 0.003018461168 |
| 477 | 477 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/878319.jpg` | 0.984924376 | 0.9851905107 | 0.984934032 | 0.0002661347389 | 9.655952454e-06 |
| 478 | 478 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/26172.jpg` | 0.3325927556 | 0.332855463 | 0.3327601552 | 0.0002627074718 | 0.0001673996449 |
| 479 | 479 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/732827.jpg` | 0.3790568411 | 0.3788704574 | 0.3733850121 | 0.0001863837242 | 0.005671828985 |
| 480 | 480 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/768104.jpg` | 0.5124635696 | 0.5157439113 | 0.5070738196 | 0.003280341625 | 0.005389750004 |
| 481 | 481 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/75823.jpg` | 0.5653221011 | 0.5663429499 | 0.5656129122 | 0.001020848751 | 0.0002908110619 |
| 482 | 482 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/754606.jpg` | 0.357770443 | 0.3565740883 | 0.3451284468 | 0.001196354628 | 0.01264199615 |
| 483 | 483 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/871420.jpg` | 0.804649055 | 0.804553628 | 0.8005134463 | 9.542703629e-05 | 0.004135608673 |
| 484 | 484 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/314314.jpg` | 0.7904983759 | 0.7907811403 | 0.7898496985 | 0.0002827644348 | 0.0006486773491 |
| 485 | 485 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/131372.jpg` | 0.4260832667 | 0.4279984236 | 0.4274868965 | 0.001915156841 | 0.00140362978 |
| 486 | 486 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/887255.jpg` | 0.9014088511 | 0.9037337899 | 0.8998783231 | 0.002324938774 | 0.001530528069 |
| 487 | 487 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/186022.jpg` | 0.8554279804 | 0.8549396992 | 0.8546384573 | 0.00048828125 | 0.0007895231247 |
| 488 | 488 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/252277.jpg` | 0.5963142514 | 0.5980724096 | 0.5974333286 | 0.001758158207 | 0.001119077206 |
| 489 | 489 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/120752.jpg` | 0.7410749197 | 0.7419857979 | 0.7420098782 | 0.0009108781815 | 0.0009349584579 |
| 490 | 490 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/200196.jpg` | 0.4610556066 | 0.4618604779 | 0.4648766518 | 0.0008048713207 | 0.00382104516 |
| 491 | 491 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/254081.jpg` | 0.611654222 | 0.6113805771 | 0.6110874414 | 0.0002736449242 | 0.0005667805672 |
| 492 | 492 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/593321.jpg` | 0.2904448807 | 0.2909072936 | 0.2945359647 | 0.0004624128342 | 0.004091084003 |
| 493 | 493 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/488202.jpg` | 0.603325665 | 0.611410141 | 0.6137448549 | 0.008084475994 | 0.01041918993 |
| 494 | 494 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/912209.jpg` | 0.1817883849 | 0.1815484315 | 0.1805827618 | 0.0002399533987 | 0.00120562315 |
| 495 | 495 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/429374.jpg` | 0.4694912136 | 0.4706333578 | 0.464413017 | 0.001142144203 | 0.005078196526 |
| 496 | 496 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/479001.jpg` | 0.8104609251 | 0.8131674528 | 0.8150308132 | 0.00270652771 | 0.004569888115 |
| 497 | 497 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/94614.jpg` | 0.6697099209 | 0.6730070114 | 0.6738833189 | 0.00329709053 | 0.004173398018 |
| 498 | 498 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/654045.jpg` | 0.8286501169 | 0.8305929899 | 0.8304188251 | 0.001942873001 | 0.001768708229 |
| 499 | 499 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/130642.jpg` | 0.6061483622 | 0.60521698 | 0.6015431881 | 0.0009313821793 | 0.004605174065 |
| 500 | 500 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/155217.jpg` | 0.269308418 | 0.2703430951 | 0.2683228254 | 0.001034677029 | 0.0009855926037 |
| 501 | 501 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/556965.jpg` | 0.2277598083 | 0.2281827927 | 0.2305737883 | 0.0004229843616 | 0.002813979983 |
| 502 | 502 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/541987.jpg` | 0.7569311857 | 0.7585073113 | 0.7561092377 | 0.001576125622 | 0.0008219480515 |
| 503 | 503 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/929321.jpg` | 0.54033041 | 0.5413684845 | 0.5333420038 | 0.001038074493 | 0.006988406181 |
| 504 | 504 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/85910.jpg` | 0.04780885205 | 0.04779618606 | 0.04610529914 | 1.266598701e-05 | 0.001703552902 |
| 505 | 505 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/6268.jpg` | 0.2649365962 | 0.2665011585 | 0.265940994 | 0.001564562321 | 0.001004397869 |
| 506 | 506 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/640027.jpg` | 0.7733125091 | 0.7735809088 | 0.7752010822 | 0.0002683997154 | 0.00188857317 |
| 507 | 507 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/801176.jpg` | 0.2357355952 | 0.2364234477 | 0.236942485 | 0.0006878525019 | 0.001206889749 |
| 508 | 508 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/356116.jpg` | 0.4391053915 | 0.4376579225 | 0.4318538904 | 0.001447468996 | 0.007251501083 |
| 509 | 509 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/671858.jpg` | 0.4752854705 | 0.4772100747 | 0.4778350294 | 0.001924604177 | 0.002549558878 |
| 510 | 510 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/742073.jpg` | 0.8607467413 | 0.8627194166 | 0.8633044958 | 0.001972675323 | 0.002557754517 |
| 511 | 511 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/288170.jpg` | 0.4794319868 | 0.484926939 | 0.4847999811 | 0.005494952202 | 0.005367994308 |
| 512 | 512 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/875668.jpg` | 0.2004937381 | 0.2009788305 | 0.196881175 | 0.0004850924015 | 0.003612563014 |
| 513 | 513 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/562213.jpg` | 0.6495420933 | 0.6473092437 | 0.6475526094 | 0.002232849598 | 0.001989483833 |
| 514 | 514 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/337497.jpg` | 0.3565896153 | 0.3573141396 | 0.3558413684 | 0.0007245242596 | 0.0007482469082 |
| 515 | 515 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/347028.jpg` | 0.4985007346 | 0.4983599484 | 0.4915880263 | 0.000140786171 | 0.006912708282 |
| 516 | 516 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/246267.jpg` | 0.6233863235 | 0.6247099638 | 0.6239461899 | 0.001323640347 | 0.0005598664284 |
| 517 | 517 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/935359.jpg` | 0.8415086269 | 0.8419398069 | 0.8382069468 | 0.0004311800003 | 0.003301680088 |
| 518 | 518 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/755061.jpg` | 0.9313211441 | 0.9325701594 | 0.9331805706 | 0.001249015331 | 0.001859426498 |
| 519 | 519 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/580488.jpg` | 0.3559812903 | 0.3570232689 | 0.3578328192 | 0.001041978598 | 0.001851528883 |
| 520 | 520 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/405118.jpg` | 0.8846144676 | 0.8845914006 | 0.8849810362 | 2.306699753e-05 | 0.0003665685654 |
| 521 | 521 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/272528.jpg` | 0.7918124199 | 0.793576777 | 0.7910286188 | 0.00176435709 | 0.0007838010788 |
| 522 | 522 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/246278.jpg` | 0.08341652155 | 0.08394870162 | 0.08458396792 | 0.0005321800709 | 0.001167446375 |
| 523 | 523 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/823757.jpg` | 0.8143737316 | 0.8149046302 | 0.8129119873 | 0.000530898571 | 0.001461744308 |
| 524 | 524 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/187378.jpg` | 0.09932655841 | 0.1010517478 | 0.1002441421 | 0.001725189388 | 0.000917583704 |
| 525 | 525 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/483998.jpg` | 0.0149457939 | 0.01514580753 | 0.01515283529 | 0.0002000136301 | 0.0002070413902 |
| 526 | 526 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/554072.jpg` | 0.3835951686 | 0.3796255291 | 0.3661668599 | 0.00396963954 | 0.01742830873 |
| 527 | 527 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/486581.jpg` | 0.1456787288 | 0.1461423188 | 0.1460657567 | 0.0004635900259 | 0.0003870278597 |
| 528 | 528 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/884759.jpg` | 0.3153607547 | 0.3143758476 | 0.3100825548 | 0.0009849071503 | 0.005278199911 |
| 529 | 529 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/775910.jpg` | 0.7787089348 | 0.7795607448 | 0.7784003615 | 0.0008518099785 | 0.000308573246 |
| 530 | 530 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/749141.jpg` | 0.6100396514 | 0.6116504669 | 0.612185359 | 0.001610815525 | 0.002145707607 |
| 531 | 531 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/512830.jpg` | 0.7784843445 | 0.7802653313 | 0.7808329463 | 0.001780986786 | 0.002348601818 |
| 532 | 532 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/179256.jpg` | 0.3260962069 | 0.3261769712 | 0.3196693957 | 8.076429367e-05 | 0.006426811218 |
| 533 | 533 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/92091.jpg` | 0.704739809 | 0.7073770761 | 0.7138388753 | 0.002637267113 | 0.009099066257 |
| 534 | 534 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/262859.jpg` | 0.831579566 | 0.8339495659 | 0.8349798918 | 0.002369999886 | 0.003400325775 |
| 535 | 535 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/522199.jpg` | 0.1322862506 | 0.1327188462 | 0.1331777871 | 0.0004325956106 | 0.0008915364742 |
| 536 | 536 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/368615.jpg` | 0.77645123 | 0.7772505283 | 0.7741077542 | 0.0007992982864 | 0.002343475819 |
| 537 | 537 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/313776.jpg` | 0.4912100434 | 0.4908912778 | 0.488260597 | 0.0003187656403 | 0.00294944644 |
| 538 | 538 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/623119.jpg` | 0.3831264079 | 0.3835157454 | 0.3755448759 | 0.0003893375397 | 0.007581532001 |
| 539 | 539 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/162483.jpg` | 0.4684382081 | 0.4795424044 | 0.488873899 | 0.01110419631 | 0.02043569088 |
| 540 | 540 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/742934.jpg` | 0.1477451921 | 0.1482357234 | 0.1474470496 | 0.0004905313253 | 0.0002981424332 |
| 541 | 541 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/691372.jpg` | 0.5604012609 | 0.5617527962 | 0.5520632267 | 0.00135153532 | 0.008338034153 |
| 542 | 542 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/637118.jpg` | 0.6173753738 | 0.6178118587 | 0.6159285903 | 0.0004364848137 | 0.001446783543 |
| 543 | 543 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/250949.jpg` | 0.5827233195 | 0.5850468874 | 0.5849738121 | 0.002323567867 | 0.002250492573 |
| 544 | 544 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/105729.jpg` | 0.8871734142 | 0.8875654936 | 0.8832221627 | 0.0003920793533 | 0.003951251507 |
| 545 | 545 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/480481.jpg` | 0.5168192983 | 0.5164190531 | 0.5083270073 | 0.0004002451897 | 0.008492290974 |
| 546 | 546 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/59199.jpg` | 0.459718734 | 0.4619588256 | 0.4623470604 | 0.002240091562 | 0.002628326416 |
| 547 | 547 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/479329.jpg` | 0.2853625119 | 0.2854041755 | 0.2847396731 | 4.16636467e-05 | 0.0006228387356 |
| 548 | 548 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/302555.jpg` | 0.8780683875 | 0.8789314032 | 0.878346324 | 0.0008630156517 | 0.0002779364586 |
| 549 | 549 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/891490.jpg` | 0.2394499779 | 0.2394378036 | 0.239267841 | 1.21742487e-05 | 0.0001821368933 |
| 550 | 550 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/153026.jpg` | 0.8197327256 | 0.8201600909 | 0.8195186257 | 0.000427365303 | 0.000214099884 |
| 551 | 551 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/156346.jpg` | 0.01445360389 | 0.01462750696 | 0.0142400926 | 0.0001739030704 | 0.0002135112882 |
| 552 | 552 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/419268.jpg` | 0.05076212063 | 0.0508797802 | 0.05096906424 | 0.0001176595688 | 0.0002069436014 |
| 553 | 553 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/837436.jpg` | 0.9018573165 | 0.902205646 | 0.9055531621 | 0.0003483295441 | 0.003695845604 |
| 554 | 554 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/674335.jpg` | 0.6726545095 | 0.6735094786 | 0.6677407026 | 0.0008549690247 | 0.004913806915 |
| 555 | 555 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/164752.jpg` | 0.9101054072 | 0.9099292755 | 0.9104753733 | 0.0001761317253 | 0.0003699660301 |
| 556 | 556 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/17902.jpg` | 0.3149997592 | 0.3179033399 | 0.3150840998 | 0.002903580666 | 8.434057236e-05 |
| 557 | 557 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/731319.jpg` | 0.1533348709 | 0.1550151706 | 0.1549754292 | 0.00168029964 | 0.001640558243 |
| 558 | 558 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/526341.jpg` | 0.7832944989 | 0.7833969593 | 0.7829322815 | 0.0001024603844 | 0.0003622174263 |
| 559 | 559 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/255042.jpg` | 0.8458922505 | 0.8463462591 | 0.8490381241 | 0.0004540085793 | 0.003145873547 |
| 560 | 560 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/365913.jpg` | 0.5891699195 | 0.5893622637 | 0.5891079903 | 0.0001923441887 | 6.192922592e-05 |
| 561 | 561 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/298648.jpg` | 0.3771034777 | 0.380630374 | 0.3717767298 | 0.003526896238 | 0.005326747894 |
| 562 | 562 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/354480.jpg` | 0.2849612832 | 0.2851383388 | 0.2821924388 | 0.0001770555973 | 0.002768844366 |
| 563 | 563 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/825809.jpg` | 0.185625419 | 0.1847468019 | 0.182368502 | 0.0008786171675 | 0.003256917 |
| 564 | 564 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/730538.jpg` | 0.3289130032 | 0.3277398348 | 0.3275215626 | 0.001173168421 | 0.00139144063 |
| 565 | 565 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/829049.jpg` | 0.8327478766 | 0.8324221373 | 0.8329373598 | 0.0003257393837 | 0.0001894831657 |
| 566 | 566 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/322797.jpg` | 0.1636449397 | 0.163469702 | 0.1617759913 | 0.0001752376556 | 0.00186894834 |
| 567 | 567 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/224239.jpg` | 0.3347263038 | 0.3372227848 | 0.3339378834 | 0.002496480942 | 0.0007884204388 |
| 568 | 568 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/457941.jpg` | 0.1146695763 | 0.1133286133 | 0.1095923781 | 0.001340962946 | 0.005077198148 |
| 569 | 569 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/600894.jpg` | 0.6500398517 | 0.6509740949 | 0.6485929489 | 0.0009342432022 | 0.001446902752 |
| 570 | 570 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/802121.jpg` | 0.8453882337 | 0.8464048505 | 0.8444555998 | 0.001016616821 | 0.0009326338768 |
| 571 | 571 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/9910.jpg` | 0.06011819467 | 0.062007837 | 0.06391590089 | 0.001889642328 | 0.003797706217 |
| 572 | 572 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/693580.jpg` | 0.646690011 | 0.6494750381 | 0.6510140896 | 0.002785027027 | 0.00432407856 |
| 573 | 573 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/570345.jpg` | 0.7788724899 | 0.7787897587 | 0.7795565128 | 8.273124695e-05 | 0.0006840229034 |
| 574 | 574 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/461013.jpg` | 0.9135860205 | 0.914305985 | 0.9129807949 | 0.0007199645042 | 0.000605225563 |
| 575 | 575 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/582176.jpg` | 0.8325423002 | 0.8335306048 | 0.8318654299 | 0.000988304615 | 0.0006768703461 |
| 576 | 576 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/347352.jpg` | 0.3247813582 | 0.3251398206 | 0.3308310509 | 0.0003584623337 | 0.006049692631 |
| 577 | 577 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/831690.jpg` | 0.09765698016 | 0.09907453507 | 0.09659217298 | 0.001417554915 | 0.001064807177 |
| 578 | 578 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/233333.jpg` | 0.1478269249 | 0.1530135274 | 0.1539857835 | 0.005186602473 | 0.006158858538 |
| 579 | 579 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/342616.jpg` | 0.9662637115 | 0.9668985009 | 0.9670390487 | 0.0006347894669 | 0.0007753372192 |
| 580 | 580 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/765938.jpg` | 0.5139251351 | 0.5133076906 | 0.5122129917 | 0.0006174445152 | 0.001712143421 |
| 581 | 581 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/863836.jpg` | 0.9943652153 | 0.9943270087 | 0.9944241047 | 3.82065773e-05 | 5.888938904e-05 |
| 582 | 582 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/24173.jpg` | 0.4493988752 | 0.451443851 | 0.448726356 | 0.002044975758 | 0.000672519207 |
| 583 | 583 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/681638.jpg` | 0.6488789916 | 0.6517167091 | 0.6493983269 | 0.002837717533 | 0.0005193352699 |
| 584 | 584 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/278623.jpg` | 0.5006998181 | 0.5055906773 | 0.5034964681 | 0.004890859127 | 0.002796649933 |
| 585 | 585 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/533158.jpg` | 0.7675497532 | 0.766857028 | 0.7698050141 | 0.0006927251816 | 0.002255260944 |
| 586 | 586 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/219020.jpg` | 0.4874811172 | 0.486689508 | 0.4833218157 | 0.0007916092873 | 0.004159301519 |
| 587 | 587 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/66435.jpg` | 0.6430038214 | 0.6445928812 | 0.6484563351 | 0.00158905983 | 0.005452513695 |
| 588 | 588 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/298138.jpg` | 0.6609530449 | 0.6632744074 | 0.6674993038 | 0.002321362495 | 0.006546258926 |
| 589 | 589 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/542625.jpg` | 0.8552516699 | 0.8578513265 | 0.8603186607 | 0.002599656582 | 0.005066990852 |
| 590 | 590 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/769299.jpg` | 0.6062435508 | 0.6054583192 | 0.5949229002 | 0.0007852315903 | 0.01132065058 |
| 591 | 591 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/40681.jpg` | 0.3353558183 | 0.3376127183 | 0.3389256895 | 0.002256900072 | 0.003569871187 |
| 592 | 592 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/664104.jpg` | 0.9995927215 | 0.9995969534 | 0.9996141791 | 4.231929779e-06 | 2.145767212e-05 |
| 593 | 593 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/12198.jpg` | 0.2481308728 | 0.246867612 | 0.2441549897 | 0.001263260841 | 0.003975883126 |
| 594 | 594 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/52497.jpg` | 0.3399608731 | 0.3398579359 | 0.3370494246 | 0.0001029372215 | 0.002911448479 |
| 595 | 595 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/111817.jpg` | 0.5147499442 | 0.5152452588 | 0.5142445564 | 0.0004953145981 | 0.0005053877831 |
| 596 | 596 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/160879.jpg` | 0.567432344 | 0.5662487745 | 0.5624785423 | 0.001183569431 | 0.004953801632 |
| 597 | 597 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/671203.jpg` | 0.8755675554 | 0.8760319352 | 0.8767081499 | 0.0004643797874 | 0.001140594482 |
| 598 | 598 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/11339.jpg` | 0.2547386587 | 0.2553409934 | 0.2579060793 | 0.0006023347378 | 0.003167420626 |
| 599 | 599 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/190649.jpg` | 0.5243800282 | 0.5279630423 | 0.5250525475 | 0.003583014011 | 0.000672519207 |
| 600 | 600 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/722398.jpg` | 0.6493262649 | 0.6507043839 | 0.6505322456 | 0.001378118992 | 0.001205980778 |
| 601 | 601 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/80871.jpg` | 0.1167922467 | 0.1167419329 | 0.1175046191 | 5.031377077e-05 | 0.0007123723626 |
| 602 | 602 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/236384.jpg` | 0.5463269949 | 0.5465077162 | 0.5473493338 | 0.000180721283 | 0.001022338867 |
| 603 | 603 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/159304.jpg` | 0.592869997 | 0.5956016779 | 0.591194272 | 0.00273168087 | 0.001675724983 |
| 604 | 604 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/803121.jpg` | 0.8189463019 | 0.8216351271 | 0.8244744539 | 0.00268882513 | 0.005528151989 |
| 605 | 605 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/882354.jpg` | 0.4511467516 | 0.4549444318 | 0.4443889856 | 0.00379768014 | 0.006757766008 |
| 606 | 606 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/442647.jpg` | 0.6703540683 | 0.6709210873 | 0.6668657064 | 0.0005670189857 | 0.003488361835 |
| 607 | 607 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/386739.jpg` | 0.2652712464 | 0.2644396424 | 0.2573722005 | 0.0008316040039 | 0.007899045944 |
| 608 | 608 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/611554.jpg` | 0.3744295835 | 0.3760016561 | 0.370097816 | 0.001572072506 | 0.004331767559 |
| 609 | 609 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/720045.jpg` | 0.1551363766 | 0.1546196789 | 0.1566847563 | 0.0005166977644 | 0.00154837966 |
| 610 | 610 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/275159.jpg` | 0.5765294433 | 0.5777730942 | 0.5758779049 | 0.001243650913 | 0.000651538372 |
| 611 | 611 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/691631.jpg` | 0.8342232108 | 0.8374252319 | 0.840714097 | 0.003202021122 | 0.006490886211 |
| 612 | 612 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/699960.jpg` | 0.3507075906 | 0.3541564643 | 0.3506441712 | 0.003448873758 | 6.341934204e-05 |
| 613 | 613 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/830622.jpg` | 0.5910811424 | 0.5917097926 | 0.5903145075 | 0.0006286501884 | 0.0007666349411 |
| 614 | 614 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/303066.jpg` | 0.8768761754 | 0.8770210147 | 0.8768548965 | 0.0001448392868 | 2.127885818e-05 |
| 615 | 615 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/115197.jpg` | 0.5662049055 | 0.5692384243 | 0.5683077574 | 0.003033518791 | 0.002102851868 |
| 616 | 616 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/92505.jpg` | 0.7651096582 | 0.7666021585 | 0.7707092762 | 0.001492500305 | 0.005599617958 |
| 617 | 617 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/96569.jpg` | 0.3245518208 | 0.3236109912 | 0.3250356317 | 0.0009408295155 | 0.0004838109016 |
| 618 | 618 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/552954.jpg` | 0.4172615111 | 0.419036746 | 0.4180454314 | 0.001775234938 | 0.0007839202881 |
| 619 | 619 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/30969.jpg` | 0.1512544304 | 0.1510658711 | 0.1485131979 | 0.0001885592937 | 0.002741232514 |
| 620 | 620 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/185884.jpg` | 0.1650853902 | 0.1661152542 | 0.1655216962 | 0.001029863954 | 0.0004363059998 |
| 621 | 621 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/540201.jpg` | 0.453755945 | 0.4556016028 | 0.457973063 | 0.001845657825 | 0.004217118025 |
| 622 | 622 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/799370.jpg` | 0.3920309544 | 0.3922370076 | 0.3939519823 | 0.000206053257 | 0.001921027899 |
| 623 | 623 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/141847.jpg` | 0.6777327061 | 0.6811774373 | 0.6747342348 | 0.003444731236 | 0.00299847126 |
| 624 | 624 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/806522.jpg` | 0.7943130136 | 0.7969844341 | 0.797924161 | 0.002671420574 | 0.003611147404 |
| 625 | 625 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/823360.jpg` | 0.4474096894 | 0.4499456286 | 0.4449581802 | 0.002535939217 | 0.002451509237 |
| 626 | 626 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/259758.jpg` | 0.9967445135 | 0.9967963696 | 0.9967190027 | 5.185604095e-05 | 2.551078796e-05 |
| 627 | 627 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/950581.jpg` | 0.2981198728 | 0.2978837192 | 0.294967562 | 0.0002361536026 | 0.003152310848 |
| 628 | 628 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/231867.jpg` | 0.1298371553 | 0.1325005591 | 0.1310513765 | 0.002663403749 | 0.00121422112 |
| 629 | 629 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/939529.jpg` | 0.3467992544 | 0.346109271 | 0.3465953171 | 0.0006899833679 | 0.0002039372921 |
| 630 | 630 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/391723.jpg` | 0.6565170288 | 0.6573857069 | 0.6618732214 | 0.000868678093 | 0.005356192589 |
| 631 | 631 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/163281.jpg` | 0.1878993213 | 0.1913025379 | 0.1919770688 | 0.0034032166 | 0.004077747464 |
| 632 | 632 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/653374.jpg` | 0.7788655758 | 0.7802321911 | 0.778783381 | 0.001366615295 | 8.219480515e-05 |
| 633 | 633 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/369277.jpg` | 0.9584843516 | 0.9586448073 | 0.9582955241 | 0.0001604557037 | 0.0001888275146 |
| 634 | 634 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/720429.jpg` | 0.4239031672 | 0.4250952601 | 0.4228684306 | 0.001192092896 | 0.001034736633 |
| 635 | 635 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/11026.jpg` | 0.7010503411 | 0.7021371126 | 0.7015115023 | 0.001086771488 | 0.0004611611366 |
| 636 | 636 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/673262.jpg` | 0.9400672913 | 0.9400529265 | 0.9408003092 | 1.436471939e-05 | 0.0007330179214 |
| 637 | 637 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/666235.jpg` | 0.8195776939 | 0.8213274479 | 0.8292002678 | 0.001749753952 | 0.009622573853 |
| 638 | 638 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/342512.jpg` | 0.699721992 | 0.7006275654 | 0.6928846836 | 0.0009055733681 | 0.006837308407 |
| 639 | 639 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/825659.jpg` | 0.7770815492 | 0.7793130279 | 0.7803927064 | 0.002231478691 | 0.003311157227 |
| 640 | 640 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/863568.jpg` | 0.397695303 | 0.3985354006 | 0.396918714 | 0.0008400976658 | 0.0007765889168 |
| 641 | 641 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/459789.jpg` | 0.1225985661 | 0.1227554902 | 0.1206526533 | 0.0001569241285 | 0.001945912838 |
| 642 | 642 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/128415.jpg` | 0.2102631032 | 0.2099787891 | 0.2041113526 | 0.0002843141556 | 0.006151750684 |
| 643 | 643 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/619493.jpg` | 0.5860561728 | 0.5879180431 | 0.5907970071 | 0.001861870289 | 0.004740834236 |
| 644 | 644 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/648746.jpg` | 0.5630113482 | 0.5657424331 | 0.5640006065 | 0.002731084824 | 0.0009892582893 |
| 645 | 645 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/607385.jpg` | 0.4049959481 | 0.4061566293 | 0.4040062428 | 0.001160681248 | 0.0009897053242 |
| 646 | 646 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/459575.jpg` | 0.3651275337 | 0.3657311797 | 0.368452549 | 0.00060364604 | 0.003325015306 |
| 647 | 647 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/82272.jpg` | 0.3984843791 | 0.400423944 | 0.3961593509 | 0.001939564943 | 0.002325028181 |
| 648 | 648 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/927617.jpg` | 0.2826890349 | 0.2842324078 | 0.2881282866 | 0.001543372869 | 0.005439251661 |
| 649 | 649 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/300954.jpg` | 0.08448547125 | 0.08345390856 | 0.0833722651 | 0.001031562686 | 0.001113206148 |
| 650 | 650 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/81179.jpg` | 0.2848007679 | 0.2881269157 | 0.2807712853 | 0.003326147795 | 0.004029482603 |
| 651 | 651 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/407520.jpg` | 0.2902773619 | 0.2910684943 | 0.295817405 | 0.0007911324501 | 0.005540043116 |
| 652 | 652 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/434376.jpg` | 0.5718275309 | 0.5721045732 | 0.5694743395 | 0.0002770423889 | 0.002353191376 |
| 653 | 653 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/419948.jpg` | 0.277664572 | 0.2778268754 | 0.2778633833 | 0.0001623034477 | 0.0001988112926 |
| 654 | 654 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/946117.jpg` | 0.9714096189 | 0.9712979198 | 0.9707875848 | 0.0001116991043 | 0.0006220340729 |
| 655 | 655 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/397509.jpg` | 0.8247259855 | 0.8255791664 | 0.8231577277 | 0.0008531808853 | 0.001568257809 |
| 656 | 656 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/72618.jpg` | 0.1215706468 | 0.1229641736 | 0.1219924018 | 0.001393526793 | 0.0004217550159 |
| 657 | 657 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/236968.jpg` | 0.5141009688 | 0.5193282962 | 0.5257070661 | 0.005227327347 | 0.01160609722 |
| 658 | 658 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/282998.jpg` | 0.7686514854 | 0.772606194 | 0.786747098 | 0.003954708576 | 0.01809561253 |
| 659 | 659 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/180995.jpg` | 0.4491705298 | 0.450167805 | 0.4545411468 | 0.0009972751141 | 0.005370616913 |
| 660 | 660 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/635660.jpg` | 0.4805376828 | 0.4813406467 | 0.4784509242 | 0.0008029639721 | 0.002086758614 |
| 661 | 661 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/131574.jpg` | 0.7148711085 | 0.7186602354 | 0.7191204429 | 0.003789126873 | 0.004249334335 |
| 662 | 662 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/761917.jpg` | 0.9826516509 | 0.9825855494 | 0.9819918275 | 6.610155106e-05 | 0.0006598234177 |
| 663 | 663 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/301725.jpg` | 0.7280279994 | 0.730148077 | 0.7255436182 | 0.00212007761 | 0.002484381199 |
| 664 | 664 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/334596.jpg` | 0.8005762696 | 0.8012674451 | 0.8015282154 | 0.0006911754608 | 0.0009519457817 |
| 665 | 665 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/181454.jpg` | 0.03980377689 | 0.04004001245 | 0.03909282759 | 0.000236235559 | 0.0007109493017 |
| 666 | 666 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/77181.jpg` | 0.8475296497 | 0.8472383022 | 0.8436307907 | 0.0002913475037 | 0.003898859024 |
| 667 | 667 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/8072.jpg` | 0.1586774886 | 0.1611676663 | 0.1601358801 | 0.002490177751 | 0.001458391547 |
| 668 | 668 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/385367.jpg` | 0.8573736548 | 0.8590214252 | 0.8554671407 | 0.001647770405 | 0.001906514168 |
| 669 | 669 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/104790.jpg` | 0.5079751015 | 0.5082583427 | 0.5073332787 | 0.000283241272 | 0.0006418228149 |
| 670 | 670 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/168238.jpg` | 0.3021293283 | 0.3046056926 | 0.3031549752 | 0.002476364374 | 0.001025646925 |
| 671 | 671 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/432011.jpg` | 0.8076614738 | 0.8095683455 | 0.8141970038 | 0.001906871796 | 0.00653553009 |
| 672 | 672 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/457873.jpg` | 0.2069126517 | 0.206042096 | 0.2070399076 | 0.0008705556393 | 0.0001272559166 |
| 673 | 673 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/788748.jpg` | 0.5059326291 | 0.5074490905 | 0.5064573288 | 0.001516461372 | 0.000524699688 |
| 674 | 674 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/342095.jpg` | 0.3854522407 | 0.3850584328 | 0.3772197068 | 0.000393807888 | 0.008232533932 |
| 675 | 675 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/778666.jpg` | 0.8124902844 | 0.8120664358 | 0.8121381998 | 0.000423848629 | 0.0003520846367 |
| 676 | 676 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/835140.jpg` | 0.9851343632 | 0.9850668311 | 0.9849936962 | 6.753206253e-05 | 0.0001406669617 |
| 677 | 677 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/51032.jpg` | 0.2441301197 | 0.2441425323 | 0.2427965254 | 1.241266727e-05 | 0.001333594322 |
| 678 | 678 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/824662.jpg` | 0.6889173388 | 0.6886576414 | 0.6885892153 | 0.0002596974373 | 0.0003281235695 |
| 679 | 679 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/154043.jpg` | 0.6854196191 | 0.6867300272 | 0.6900878549 | 0.001310408115 | 0.004668235779 |
| 680 | 680 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/298771.jpg` | 0.8073369265 | 0.8073055148 | 0.806604743 | 3.14116478e-05 | 0.0007321834564 |
| 681 | 681 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/429823.jpg` | 0.49106282 | 0.4922773838 | 0.4899785519 | 0.001214563847 | 0.001084268093 |
| 682 | 682 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/814723.jpg` | 0.1436629891 | 0.1431561559 | 0.1415636837 | 0.0005068331957 | 0.002099305391 |
| 683 | 683 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/432023.jpg` | 0.6819167733 | 0.6809589863 | 0.6804494858 | 0.0009577870369 | 0.00146728754 |
| 684 | 684 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/914704.jpg` | 0.402656883 | 0.4026238918 | 0.3961293697 | 3.299117088e-05 | 0.006527513266 |
| 685 | 685 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/627543.jpg` | 0.3642946184 | 0.3659247756 | 0.3697706461 | 0.001630157232 | 0.005476027727 |
| 686 | 686 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/142365.jpg` | 0.3436579406 | 0.3466849923 | 0.3500809073 | 0.003027051687 | 0.006422966719 |
| 687 | 687 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/684895.jpg` | 0.686060667 | 0.6874790192 | 0.6804689169 | 0.001418352127 | 0.005591750145 |
| 688 | 688 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/301537.jpg` | 0.6397910118 | 0.6415445805 | 0.6373782158 | 0.001753568649 | 0.002412796021 |
| 689 | 689 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/180511.jpg` | 0.8136476278 | 0.8148010373 | 0.8124796152 | 0.001153409481 | 0.001168012619 |
| 690 | 690 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/365427.jpg` | 0.3826881647 | 0.3820723891 | 0.3778990209 | 0.0006157755852 | 0.004789143801 |
| 691 | 691 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/638943.jpg` | 0.09074988961 | 0.09179302305 | 0.09261497855 | 0.001043133438 | 0.00186508894 |
| 692 | 692 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/101952.jpg` | 0.4089484811 | 0.4091367424 | 0.4073325098 | 0.0001882612705 | 0.001615971327 |
| 693 | 693 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/49013.jpg` | 0.3738625348 | 0.3758151531 | 0.3728263974 | 0.001952618361 | 0.001036137342 |
| 694 | 694 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/118162.jpg` | 0.2323016524 | 0.2322211117 | 0.2298939079 | 8.054077625e-05 | 0.002407744527 |
| 695 | 695 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/181564.jpg` | 0.3004769087 | 0.3009203076 | 0.3011484742 | 0.0004433989525 | 0.0006715655327 |
| 696 | 696 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/743592.jpg` | 0.9835022092 | 0.9835549593 | 0.9830505252 | 5.275011063e-05 | 0.0004516839981 |
| 697 | 697 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/954858.jpg` | 0.8160846829 | 0.8168181181 | 0.8165916204 | 0.000733435154 | 0.0005069375038 |
| 698 | 698 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/32812.jpg` | 0.5148155689 | 0.5148985386 | 0.5155861378 | 8.296966553e-05 | 0.0007705688477 |
| 699 | 699 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/690443.jpg` | 0.2050434351 | 0.2064927965 | 0.2052128762 | 0.001449361444 | 0.0001694411039 |
| 700 | 700 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/759919.jpg` | 0.3163296282 | 0.3178709149 | 0.3158653975 | 0.001541286707 | 0.0004642307758 |
| 701 | 701 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/95617.jpg` | 0.7052150965 | 0.7053890228 | 0.7098983526 | 0.0001739263535 | 0.004683256149 |
| 702 | 702 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/72769.jpg` | 0.7102956176 | 0.7092657685 | 0.7112271786 | 0.001029849052 | 0.0009315609932 |
| 703 | 703 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/821596.jpg` | 0.6996636987 | 0.6989164352 | 0.6976767778 | 0.0007472634315 | 0.001986920834 |
| 704 | 704 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/113386.jpg` | 0.8537176847 | 0.8541796803 | 0.8547520638 | 0.0004619956017 | 0.001034379005 |
| 705 | 705 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/638995.jpg` | 0.8840050101 | 0.884315908 | 0.8834239244 | 0.0003108978271 | 0.0005810856819 |
| 706 | 706 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/375824.jpg` | 0.5954093337 | 0.5933872461 | 0.5931628942 | 0.002022087574 | 0.002246439457 |
| 707 | 707 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/593569.jpg` | 0.7772877812 | 0.7777309418 | 0.7803208828 | 0.0004431605339 | 0.003033101559 |
| 708 | 708 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/41333.jpg` | 0.7499017715 | 0.7499394417 | 0.7496373653 | 3.76701355e-05 | 0.0002644062042 |
| 709 | 709 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/886463.jpg` | 0.375048846 | 0.3747804165 | 0.3747595549 | 0.0002684295177 | 0.0002892911434 |
| 710 | 710 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/234284.jpg` | 0.1076444089 | 0.1090361327 | 0.1066234633 | 0.001391723752 | 0.001020945609 |
| 711 | 711 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/947840.jpg` | 0.9605045915 | 0.960293591 | 0.9591653347 | 0.0002110004425 | 0.001339256763 |
| 712 | 712 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/521631.jpg` | 0.4785640836 | 0.4800458848 | 0.4746136069 | 0.001481801271 | 0.003950476646 |
| 713 | 713 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/609943.jpg` | 0.9446998835 | 0.9451850057 | 0.9453827143 | 0.0004851222038 | 0.0006828308105 |
| 714 | 714 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/3901.jpg` | 0.1481553018 | 0.1480577439 | 0.1451821923 | 9.755790234e-05 | 0.002973109484 |
| 715 | 715 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/98105.jpg` | 0.5605337024 | 0.5623658895 | 0.5645475388 | 0.001832187176 | 0.004013836384 |
| 716 | 716 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/503690.jpg` | 0.5182757378 | 0.5249148607 | 0.5264357328 | 0.006639122963 | 0.008159995079 |
| 717 | 717 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/658265.jpg` | 0.4928522706 | 0.4955890775 | 0.4937699437 | 0.00273680687 | 0.000917673111 |
| 718 | 718 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/230789.jpg` | 0.5132827163 | 0.515380621 | 0.5132316947 | 0.002097904682 | 5.102157593e-05 |
| 719 | 719 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/334622.jpg` | 0.5964999795 | 0.5977998972 | 0.5953714848 | 0.001299917698 | 0.00112849474 |
| 720 | 720 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/476658.jpg` | 0.6805908084 | 0.6848987341 | 0.6839470863 | 0.004307925701 | 0.003356277943 |
| 721 | 721 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/84858.jpg` | 0.5181447268 | 0.5203794241 | 0.522859633 | 0.002234697342 | 0.004714906216 |
| 722 | 722 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/57137.jpg` | 0.2090216279 | 0.2107062191 | 0.2095752507 | 0.001684591174 | 0.0005536228418 |
| 723 | 723 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/3768.jpg` | 0.217827484 | 0.2181173861 | 0.2193307728 | 0.000289902091 | 0.001503288746 |
| 724 | 724 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/58208.jpg` | 0.2522063255 | 0.2518180013 | 0.2500288486 | 0.0003883242607 | 0.002177476883 |
| 725 | 725 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/122467.jpg` | 0.2019254267 | 0.2022203356 | 0.200343281 | 0.0002949088812 | 0.001582145691 |
| 726 | 726 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/909930.jpg` | 0.74611485 | 0.7474658489 | 0.7500240803 | 0.001350998878 | 0.003909230232 |
| 727 | 727 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/949929.jpg` | 0.1009529009 | 0.1029998586 | 0.103643015 | 0.002046957612 | 0.002690114081 |
| 728 | 728 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/790388.jpg` | 0.7395303249 | 0.7395479679 | 0.7380795479 | 1.764297485e-05 | 0.001450777054 |
| 729 | 729 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/847569.jpg` | 0.9764031172 | 0.9766771197 | 0.9763831496 | 0.000274002552 | 1.9967556e-05 |
| 730 | 730 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/771272.jpg` | 0.7746109366 | 0.7752871513 | 0.7759580016 | 0.000676214695 | 0.001347064972 |
| 731 | 731 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/704096.jpg` | 0.4868001938 | 0.4867115617 | 0.4837049246 | 8.863210678e-05 | 0.003095269203 |
| 732 | 732 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/857121.jpg` | 0.9892373681 | 0.9893772006 | 0.9892656803 | 0.0001398324966 | 2.831220627e-05 |
| 733 | 733 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/96593.jpg` | 0.4211560488 | 0.4221027493 | 0.4206840098 | 0.000946700573 | 0.0004720389843 |
| 734 | 734 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/559952.jpg` | 0.4552325308 | 0.4554645419 | 0.4506123066 | 0.0002320110798 | 0.004620224237 |
| 735 | 735 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/956319.jpg` | 0.6258147955 | 0.6287115812 | 0.6284170747 | 0.002896785736 | 0.002602279186 |
| 736 | 736 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/299198.jpg` | 0.5965801477 | 0.5983775854 | 0.5939258337 | 0.001797437668 | 0.002654314041 |
| 737 | 737 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/818255.jpg` | 0.9929887652 | 0.9930901527 | 0.9931037426 | 0.0001013875008 | 0.0001149773598 |
| 738 | 738 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/347157.jpg` | 0.7383671403 | 0.7386933565 | 0.7306810021 | 0.0003262162209 | 0.007686138153 |
| 739 | 739 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/59038.jpg` | 0.4821770787 | 0.482794553 | 0.4847761691 | 0.0006174743176 | 0.002599090338 |
| 740 | 740 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/907810.jpg` | 0.4921519458 | 0.5015836954 | 0.519847393 | 0.009431749582 | 0.02769544721 |
| 741 | 741 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/479566.jpg` | 0.2893938422 | 0.2896899879 | 0.2867638767 | 0.0002961456776 | 0.002629965544 |
| 742 | 742 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/465558.jpg` | 0.172612071 | 0.17323111 | 0.1733925045 | 0.0006190389395 | 0.0007804334164 |
| 743 | 743 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/711543.jpg` | 0.910348773 | 0.9105805159 | 0.9130893946 | 0.0002317428589 | 0.002740621567 |
| 744 | 744 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/705166.jpg` | 0.9145709276 | 0.9151165485 | 0.9147873521 | 0.0005456209183 | 0.0002164244652 |
| 745 | 745 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/817184.jpg` | 0.9841214418 | 0.9842146039 | 0.9838196039 | 9.316205978e-05 | 0.0003018379211 |
| 746 | 746 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/66948.jpg` | 0.4635444283 | 0.4621003568 | 0.4488662779 | 0.001444071531 | 0.01467815042 |
| 747 | 747 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/65033.jpg` | 0.8070729375 | 0.8077466488 | 0.8087938428 | 0.0006737112999 | 0.001720905304 |
| 748 | 748 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/517029.jpg` | 0.9256910682 | 0.9263370037 | 0.9271815419 | 0.0006459355354 | 0.001490473747 |
| 749 | 749 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/405126.jpg` | 0.2373746634 | 0.2393414825 | 0.2350611389 | 0.001966819167 | 0.002313524485 |
| 750 | 750 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/167827.jpg` | 0.5769245625 | 0.5791115165 | 0.5701934695 | 0.002186954021 | 0.00673109293 |
| 751 | 751 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/59435.jpg` | 0.1044127196 | 0.1072025821 | 0.1101926565 | 0.002789862454 | 0.00577993691 |
| 752 | 752 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/23596.jpg` | 0.09898737818 | 0.1010900736 | 0.102069512 | 0.002102695405 | 0.00308213383 |
| 753 | 753 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/923688.jpg` | 0.6080469489 | 0.6086546779 | 0.6102334261 | 0.0006077289581 | 0.002186477184 |
| 754 | 754 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/465897.jpg` | 0.8601904511 | 0.8601480722 | 0.8591228127 | 4.237890244e-05 | 0.001067638397 |
| 755 | 755 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/143029.jpg` | 0.7435491085 | 0.7475898266 | 0.7422332168 | 0.004040718079 | 0.001315891743 |
| 756 | 756 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/699638.jpg` | 0.191773966 | 0.1935499907 | 0.1943075955 | 0.001776024699 | 0.002533629537 |
| 757 | 757 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/328431.jpg` | 0.473272115 | 0.473457247 | 0.4726143479 | 0.0001851320267 | 0.0006577670574 |
| 758 | 758 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/66399.jpg` | 0.6689611673 | 0.6702299118 | 0.6694431305 | 0.001268744469 | 0.0004819631577 |
| 759 | 759 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/13333.jpg` | 0.4995662868 | 0.4997984171 | 0.502617836 | 0.0002321302891 | 0.003051549196 |
| 760 | 760 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/253016.jpg` | 0.3856757581 | 0.3907741606 | 0.3984231651 | 0.0050984025 | 0.01274740696 |
| 761 | 761 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/145165.jpg` | 0.6116858721 | 0.6136366129 | 0.6105852127 | 0.001950740814 | 0.00110065937 |
| 762 | 762 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/597600.jpg` | 0.6013798714 | 0.60199368 | 0.600369215 | 0.0006138086319 | 0.001010656357 |
| 763 | 763 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/111542.jpg` | 0.9677583575 | 0.9678266048 | 0.9679739475 | 6.824731827e-05 | 0.0002155900002 |
| 764 | 764 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/626555.jpg` | 0.7206824422 | 0.7239249945 | 0.71448946 | 0.00324255228 | 0.006192982197 |
| 765 | 765 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/27198.jpg` | 0.7448040247 | 0.7461007833 | 0.7473955154 | 0.001296758652 | 0.002591490746 |
| 766 | 766 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/804719.jpg` | 0.3515900671 | 0.3516470194 | 0.3484310806 | 5.695223808e-05 | 0.003158986568 |
| 767 | 767 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/732172.jpg` | 0.3038721383 | 0.3045517504 | 0.3016830087 | 0.0006796121597 | 0.002189129591 |
| 768 | 768 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/543276.jpg` | 0.9066590667 | 0.9070794582 | 0.9059352279 | 0.0004203915596 | 0.0007238388062 |
| 769 | 769 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/207942.jpg` | 0.886380434 | 0.8865016699 | 0.8865833879 | 0.0001212358475 | 0.0002029538155 |
| 770 | 770 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/608669.jpg` | 0.3497631848 | 0.3465304673 | 0.3443021476 | 0.003232717514 | 0.005461037159 |
| 771 | 771 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/198.jpg` | 0.4278106093 | 0.4296393991 | 0.4286940396 | 0.001828789711 | 0.0008834302425 |
| 772 | 772 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/41041.jpg` | 0.8210069537 | 0.8212106228 | 0.8201533556 | 0.0002036690712 | 0.0008535981178 |
| 773 | 773 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/449543.jpg` | 0.9106448293 | 0.9121299386 | 0.9107934237 | 0.001485109329 | 0.0001485943794 |
| 774 | 774 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/28225.jpg` | 0.7940755486 | 0.7948509455 | 0.794047296 | 0.0007753968239 | 2.825260162e-05 |
| 775 | 775 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/843235.jpg` | 0.9589312673 | 0.9592483044 | 0.9588960409 | 0.0003170371056 | 3.522634506e-05 |
| 776 | 776 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/433186.jpg` | 0.6862385869 | 0.6876676679 | 0.6825867295 | 0.001429080963 | 0.003651857376 |
| 777 | 777 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/851201.jpg` | 0.600966692 | 0.5937984586 | 0.5785330534 | 0.007168233395 | 0.02243363857 |
| 778 | 778 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/321937.jpg` | 0.6089780331 | 0.6126126051 | 0.6054732203 | 0.003634572029 | 0.003504812717 |
| 779 | 779 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/562556.jpg` | 0.9841118455 | 0.9839178324 | 0.9838324785 | 0.0001940131187 | 0.0002793669701 |
| 780 | 780 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/361170.jpg` | 0.1900255531 | 0.1924203485 | 0.1957769096 | 0.002394795418 | 0.005751356483 |
| 781 | 781 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/471062.jpg` | 0.5100246668 | 0.5126039982 | 0.5117124319 | 0.002579331398 | 0.001687765121 |
| 782 | 782 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/825032.jpg` | 0.8988557458 | 0.8997172117 | 0.8985911012 | 0.0008614659309 | 0.0002646446228 |
| 783 | 783 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/668515.jpg` | 0.04103060067 | 0.04125019163 | 0.04189722985 | 0.0002195909619 | 0.0008666291833 |
| 784 | 784 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/124396.jpg` | 0.6976292729 | 0.7004412413 | 0.700592041 | 0.002811968327 | 0.002962768078 |
| 785 | 785 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/861940.jpg` | 0.7100531459 | 0.7121183872 | 0.710845232 | 0.002065241337 | 0.0007920861244 |
| 786 | 786 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/80582.jpg` | 0.6125564575 | 0.6162433624 | 0.61163342 | 0.003686904907 | 0.000923037529 |
| 787 | 787 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/208.jpg` | 0.3548820019 | 0.3538961411 | 0.3454830348 | 0.0009858608246 | 0.009398967028 |
| 788 | 788 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/699027.jpg` | 0.7065708637 | 0.7075428963 | 0.7050893903 | 0.000972032547 | 0.001481473446 |
| 789 | 789 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/775784.jpg` | 0.6675543189 | 0.6650922298 | 0.666349411 | 0.002462089062 | 0.001204907894 |
| 790 | 790 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/546150.jpg` | 0.7589683533 | 0.7597417831 | 0.7593349218 | 0.0007734298706 | 0.0003665685654 |
| 791 | 791 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/6251.jpg` | 0.4103806019 | 0.4108348787 | 0.4076686502 | 0.0004542768002 | 0.002711951733 |
| 792 | 792 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/243997.jpg` | 0.9220283031 | 0.922285974 | 0.9226346016 | 0.0002576708794 | 0.0006062984467 |
| 793 | 793 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/741832.jpg` | 0.9896894693 | 0.9899448156 | 0.9898284674 | 0.0002553462982 | 0.0001389980316 |
| 794 | 794 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/701119.jpg` | 0.562309742 | 0.564489603 | 0.5638344288 | 0.002179861069 | 0.001524686813 |
| 795 | 795 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/223656.jpg` | 0.1052084044 | 0.1051053852 | 0.1056578085 | 0.0001030191779 | 0.0004494041204 |
| 796 | 796 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/883631.jpg` | 0.147309944 | 0.149419114 | 0.1466620862 | 0.00210916996 | 0.0006478577852 |
| 797 | 797 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/307687.jpg` | 0.07709928602 | 0.07742639631 | 0.07574896514 | 0.0003271102905 | 0.001350320876 |
| 798 | 798 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/788290.jpg` | 0.6877131462 | 0.6918548346 | 0.6973959804 | 0.004141688347 | 0.009682834148 |
| 799 | 799 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/534415.jpg` | 0.9402143955 | 0.9400749207 | 0.9406903982 | 0.0001394748688 | 0.0004760026932 |
| 800 | 800 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/287755.jpg` | 0.5755459666 | 0.57534343 | 0.568087697 | 0.0002025365829 | 0.007458269596 |
| 801 | 801 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/816248.jpg` | 0.9292936921 | 0.9285033345 | 0.9305431843 | 0.0007903575897 | 0.001249492168 |
| 802 | 802 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/133095.jpg` | 0.3191810548 | 0.3206302524 | 0.3189969063 | 0.001449197531 | 0.00018414855 |
| 803 | 803 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/782663.jpg` | 0.7239893675 | 0.727262795 | 0.7298053503 | 0.003273427486 | 0.005815982819 |
| 804 | 804 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/24670.jpg` | 0.3245261014 | 0.3263085186 | 0.3233891428 | 0.001782417297 | 0.001136958599 |
| 805 | 805 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/273049.jpg` | 0.1078631058 | 0.1083132327 | 0.1059770882 | 0.0004501268268 | 0.001886017621 |
| 806 | 806 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/102523.jpg` | 0.3599532843 | 0.3596551418 | 0.3556592762 | 0.0002981424332 | 0.004294008017 |
| 807 | 807 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/716943.jpg` | 0.6319320798 | 0.6337957382 | 0.6317801476 | 0.001863658428 | 0.0001519322395 |
| 808 | 808 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/722552.jpg` | 0.4345273674 | 0.4355065823 | 0.4317876101 | 0.0009792149067 | 0.002739757299 |
| 809 | 809 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/589884.jpg` | 0.6381012797 | 0.6394259334 | 0.6374953985 | 0.001324653625 | 0.0006058812141 |
| 810 | 810 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/293539.jpg` | 0.4812795222 | 0.4806410074 | 0.4815515876 | 0.0006385147572 | 0.0002720654011 |
| 811 | 811 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/564348.jpg` | 0.5945912004 | 0.5972907543 | 0.5958891511 | 0.002699553967 | 0.001297950745 |
| 812 | 812 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/424171.jpg` | 0.4893737435 | 0.4901895523 | 0.4892313182 | 0.000815808773 | 0.0001424252987 |
| 813 | 813 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/403777.jpg` | 0.7062703967 | 0.7077593803 | 0.7088157535 | 0.001488983631 | 0.00254535675 |
| 814 | 814 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/433445.jpg` | 0.08121692389 | 0.08254116029 | 0.08154173195 | 0.001324236393 | 0.0003248080611 |
| 815 | 815 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/499338.jpg` | 0.4004805386 | 0.4055649936 | 0.4063103199 | 0.005084455013 | 0.005829781294 |
| 816 | 816 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/465001.jpg` | 0.2565826774 | 0.2566641271 | 0.2484777123 | 8.144974709e-05 | 0.008104965091 |
| 817 | 817 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/455060.jpg` | 0.7558828592 | 0.7570030689 | 0.7566438913 | 0.001120209694 | 0.0007610321045 |
| 818 | 818 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/33310.jpg` | 0.03043285012 | 0.03121263534 | 0.03196720406 | 0.0007797852159 | 0.001534353942 |
| 819 | 819 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/757821.jpg` | 0.7274807096 | 0.7275565267 | 0.729349792 | 7.581710815e-05 | 0.001869082451 |
| 820 | 820 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/947779.jpg` | 0.7152189612 | 0.7172498703 | 0.7155414224 | 0.002030909061 | 0.0003224611282 |
| 821 | 821 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/521274.jpg` | 0.6905794144 | 0.6900658607 | 0.6819554567 | 0.0005135536194 | 0.008623957634 |
| 822 | 822 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/276844.jpg` | 0.1123050526 | 0.1132732034 | 0.1124333963 | 0.0009681507945 | 0.0001283437014 |
| 823 | 823 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/42799.jpg` | 0.2501286268 | 0.2533901036 | 0.2502073646 | 0.003261476755 | 7.873773575e-05 |
| 824 | 824 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/696517.jpg` | 0.9293239713 | 0.9297586679 | 0.9292554855 | 0.0004346966743 | 6.848573685e-05 |
| 825 | 825 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/890997.jpg` | 0.4942839742 | 0.4973451793 | 0.4890052378 | 0.003061205149 | 0.005278736353 |
| 826 | 826 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/354623.jpg` | 0.5585600138 | 0.5603445768 | 0.5603510141 | 0.001784563065 | 0.001791000366 |
| 827 | 827 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/113678.jpg` | 0.4984139502 | 0.5000444651 | 0.502607584 | 0.00163051486 | 0.004193633795 |
| 828 | 828 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/604895.jpg` | 0.2366605997 | 0.237282902 | 0.2332615852 | 0.0006223022938 | 0.003399014473 |
| 829 | 829 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/504514.jpg` | 0.9202184081 | 0.9207032919 | 0.9217264056 | 0.0004848837852 | 0.001507997513 |
| 830 | 830 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/333040.jpg` | 0.8457465172 | 0.8455907106 | 0.8433557749 | 0.0001558065414 | 0.002390742302 |
| 831 | 831 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/44797.jpg` | 0.1623026729 | 0.1629449725 | 0.1632791311 | 0.0006422996521 | 0.0009764581919 |
| 832 | 832 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/699016.jpg` | 0.4092210233 | 0.4135493636 | 0.4148820937 | 0.004328340292 | 0.005661070347 |
| 833 | 833 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/848697.jpg` | 0.6525208354 | 0.6534665227 | 0.6520249248 | 0.000945687294 | 0.0004959106445 |
| 834 | 834 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/232841.jpg` | 0.212106809 | 0.2107562721 | 0.2122562528 | 0.001350536942 | 0.0001494437456 |
| 835 | 835 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/41410.jpg` | 0.9076743126 | 0.9079087377 | 0.9088499546 | 0.0002344250679 | 0.001175642014 |
| 836 | 836 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/561655.jpg` | 0.9228422046 | 0.9236393571 | 0.9227737188 | 0.0007971525192 | 6.848573685e-05 |
| 837 | 837 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/822642.jpg` | 0.04011297226 | 0.04013283551 | 0.0391798839 | 1.986324787e-05 | 0.0009330883622 |
| 838 | 838 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/11135.jpg` | 0.1443024576 | 0.1448170692 | 0.1431120485 | 0.0005146116018 | 0.001190409064 |
| 839 | 839 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/349192.jpg` | 0.5724770427 | 0.5739612579 | 0.570110321 | 0.00148421526 | 0.00236672163 |
| 840 | 840 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/423099.jpg` | 0.4005505443 | 0.404848218 | 0.4029128253 | 0.004297673702 | 0.002362281084 |
| 841 | 841 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/837148.jpg` | 0.6269917488 | 0.6280328631 | 0.6237359643 | 0.00104111433 | 0.003255784512 |
| 842 | 842 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/333490.jpg` | 0.2974122167 | 0.2980558574 | 0.2983365059 | 0.0006436407566 | 0.0009242892265 |
| 843 | 843 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/13448.jpg` | 0.2275959402 | 0.2295563668 | 0.2288367897 | 0.001960426569 | 0.001240849495 |
| 844 | 844 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/81723.jpg` | 0.815677464 | 0.8165003061 | 0.8142219186 | 0.0008228421211 | 0.001455545425 |
| 845 | 845 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/835286.jpg` | 0.7330085039 | 0.7347226143 | 0.7376998663 | 0.001714110374 | 0.004691362381 |
| 846 | 846 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/714461.jpg` | 0.2522128224 | 0.2545110583 | 0.2514688373 | 0.002298235893 | 0.0007439851761 |
| 847 | 847 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/99800.jpg` | 0.2405574024 | 0.2452139109 | 0.247539252 | 0.004656508565 | 0.00698184967 |
| 848 | 848 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/158116.jpg` | 0.8385989666 | 0.8396168947 | 0.8392293453 | 0.001017928123 | 0.0006303787231 |
| 849 | 849 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/632806.jpg` | 0.1886414587 | 0.1888762563 | 0.1891326606 | 0.0002347975969 | 0.0004912018776 |
| 850 | 850 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/228703.jpg` | 0.7493500113 | 0.7517486215 | 0.7560318708 | 0.002398610115 | 0.006681859493 |
| 851 | 851 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/852500.jpg` | 0.7870019078 | 0.7866412997 | 0.7858272195 | 0.0003606081009 | 0.001174688339 |
| 852 | 852 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/245842.jpg` | 0.6285799742 | 0.63142699 | 0.6262670159 | 0.002847015858 | 0.002312958241 |
| 853 | 853 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/441859.jpg` | 0.4483524561 | 0.4444523454 | 0.4433971643 | 0.003900110722 | 0.004955291748 |
| 854 | 854 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/818298.jpg` | 0.1279337257 | 0.1278279573 | 0.127631858 | 0.0001057684422 | 0.0003018677235 |
| 855 | 855 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/784086.jpg` | 0.7057539821 | 0.7050830722 | 0.7050116658 | 0.0006709098816 | 0.000742316246 |
| 856 | 856 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/677845.jpg` | 0.8349332809 | 0.8354086876 | 0.8344241977 | 0.0004754066467 | 0.000509083271 |
| 857 | 857 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/442076.jpg` | 0.499546349 | 0.5034998059 | 0.496404618 | 0.003953456879 | 0.003141731024 |
| 858 | 858 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/635494.jpg` | 0.2300767154 | 0.2304659784 | 0.2278357595 | 0.0003892630339 | 0.00224095583 |
| 859 | 859 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/491193.jpg` | 0.1725475639 | 0.1791965514 | 0.178484723 | 0.006648987532 | 0.005937159061 |
| 860 | 860 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/459976.jpg` | 0.819188416 | 0.8200944066 | 0.8171846867 | 0.0009059906006 | 0.002003729343 |
| 861 | 861 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/673021.jpg` | 0.4443483949 | 0.445841074 | 0.4539141357 | 0.001492679119 | 0.009565740824 |
| 862 | 862 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/31993.jpg` | 0.5452780128 | 0.5477864742 | 0.5594365597 | 0.002508461475 | 0.01415854692 |
| 863 | 863 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/716459.jpg` | 0.2555817068 | 0.2565846145 | 0.2591001093 | 0.001002907753 | 0.003518402576 |
| 864 | 864 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/657094.jpg` | 0.3642014861 | 0.3687285185 | 0.3690532148 | 0.004527032375 | 0.004851728678 |
| 865 | 865 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/824354.jpg` | 0.9437972903 | 0.9442638755 | 0.9438747764 | 0.0004665851593 | 7.748603821e-05 |
| 866 | 866 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/609411.jpg` | 0.2619985342 | 0.262348026 | 0.2629883587 | 0.0003494918346 | 0.0009898245335 |
| 867 | 867 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/14948.jpg` | 0.4342307746 | 0.4357465208 | 0.4309394062 | 0.001515746117 | 0.003291368484 |
| 868 | 868 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/543739.jpg` | 0.3027297556 | 0.3029878139 | 0.2970210612 | 0.0002580583096 | 0.005708694458 |
| 869 | 869 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/664622.jpg` | 0.3099443913 | 0.3116159439 | 0.3067675829 | 0.001671552658 | 0.003176808357 |
| 870 | 870 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/429325.jpg` | 0.5941563845 | 0.5952390432 | 0.5955427885 | 0.001082658768 | 0.001386404037 |
| 871 | 871 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/829164.jpg` | 0.9578409195 | 0.9599425793 | 0.9597877264 | 0.002101659775 | 0.001946806908 |
| 872 | 872 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/942936.jpg` | 0.8496898413 | 0.8509061337 | 0.8500945568 | 0.001216292381 | 0.000404715538 |
| 873 | 873 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/724870.jpg` | 0.8284676075 | 0.82880193 | 0.8290129304 | 0.0003343224525 | 0.0005453228951 |
| 874 | 874 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/7739.jpg` | 0.3890749216 | 0.3901746571 | 0.3892779946 | 0.001099735498 | 0.0002030730247 |
| 875 | 875 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/837785.jpg` | 0.7857602835 | 0.7866960764 | 0.781644702 | 0.000935792923 | 0.004115581512 |
| 876 | 876 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/210807.jpg` | 0.7543128133 | 0.7565878034 | 0.7582131028 | 0.002274990082 | 0.003900289536 |
| 877 | 877 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/358601.jpg` | 0.5141077042 | 0.5145394802 | 0.5037904978 | 0.0004317760468 | 0.01031720638 |
| 878 | 878 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/134910.jpg` | 0.1631859988 | 0.1629161686 | 0.1622941792 | 0.0002698302269 | 0.0008918195963 |
| 879 | 879 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/713165.jpg` | 0.4938174784 | 0.4973044991 | 0.5025850534 | 0.003487020731 | 0.008767575026 |
| 880 | 880 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/484693.jpg` | 0.3590474725 | 0.3596499562 | 0.3565016389 | 0.0006024837494 | 0.002545833588 |
| 881 | 881 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/791658.jpg` | 0.7278941274 | 0.7285974026 | 0.7288488746 | 0.0007032752037 | 0.0009547472 |
| 882 | 882 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/727096.jpg` | 0.9430776834 | 0.9429389238 | 0.9425053596 | 0.000138759613 | 0.0005723237991 |
| 883 | 883 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/734221.jpg` | 0.8453636765 | 0.8486258984 | 0.8429768085 | 0.003262221813 | 0.002386868 |
| 884 | 884 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/656396.jpg` | 0.3151507378 | 0.3183197081 | 0.3216343224 | 0.003168970346 | 0.006483584642 |
| 885 | 885 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/941531.jpg` | 0.4645567536 | 0.4688193798 | 0.4672186971 | 0.004262626171 | 0.002661943436 |
| 886 | 886 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/526731.jpg` | 0.5875274539 | 0.58776021 | 0.5834368467 | 0.0002327561378 | 0.004090607166 |
| 887 | 887 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/363899.jpg` | 0.7800602913 | 0.7816379666 | 0.7798939943 | 0.001577675343 | 0.0001662969589 |
| 888 | 888 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/412131.jpg` | 0.1157956868 | 0.1173926294 | 0.1178028956 | 0.001596942544 | 0.002007208765 |
| 889 | 889 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/494284.jpg` | 0.6513983011 | 0.6520287991 | 0.6522190571 | 0.0006304979324 | 0.0008207559586 |
| 890 | 890 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/240304.jpg` | 0.04204881564 | 0.04270864278 | 0.04231646284 | 0.000659827143 | 0.0002676472068 |
| 891 | 891 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/332225.jpg` | 0.106519185 | 0.1071080491 | 0.1063846275 | 0.0005888640881 | 0.0001345574856 |
| 892 | 892 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/722487.jpg` | 0.2950031161 | 0.2940424085 | 0.2930648327 | 0.0009607076645 | 0.001938283443 |
| 893 | 893 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/754947.jpg` | 0.6887729764 | 0.6920835972 | 0.6890041828 | 0.003310620785 | 0.0002312064171 |
| 894 | 894 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/936060.jpg` | 0.6915791631 | 0.6919029951 | 0.6920784712 | 0.0003238320351 | 0.0004993081093 |
| 895 | 895 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/319500.jpg` | 0.485535413 | 0.4868629575 | 0.4870672822 | 0.001327544451 | 0.001531869173 |
| 896 | 896 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/772483.jpg` | 0.7579056621 | 0.7589144707 | 0.7588707209 | 0.001008808613 | 0.0009650588036 |
| 897 | 897 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/82724.jpg` | 0.3062363863 | 0.3073992729 | 0.306075573 | 0.00116288662 | 0.0001608133316 |
| 898 | 898 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/606637.jpg` | 0.6434491277 | 0.6454532743 | 0.6450072527 | 0.002004146576 | 0.001558125019 |
| 899 | 899 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/625596.jpg` | 0.1546710134 | 0.1558169425 | 0.1535032094 | 0.001145929098 | 0.001167804003 |
| 900 | 900 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/95046.jpg` | 0.7287664413 | 0.7291080952 | 0.7265850306 | 0.0003416538239 | 0.002181410789 |
| 901 | 901 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/743701.jpg` | 0.08210745454 | 0.08306011558 | 0.08017791808 | 0.0009526610374 | 0.001929536462 |
| 902 | 902 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/352830.jpg` | 0.2938544452 | 0.2959245443 | 0.2973246276 | 0.002070099115 | 0.003470182419 |
| 903 | 903 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/348851.jpg` | 0.4988671541 | 0.4983639121 | 0.4941259623 | 0.0005032420158 | 0.004741191864 |
| 904 | 904 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/829320.jpg` | 0.5652944446 | 0.5707081556 | 0.5732389688 | 0.005413711071 | 0.007944524288 |
| 905 | 905 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/589353.jpg` | 0.6187640429 | 0.6203122735 | 0.6229439974 | 0.001548230648 | 0.004179954529 |
| 906 | 906 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/816894.jpg` | 0.3540515304 | 0.3561035097 | 0.3491907418 | 0.002051979303 | 0.004860788584 |
| 907 | 907 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/330464.jpg` | 0.5778291225 | 0.5794188976 | 0.578864336 | 0.001589775085 | 0.00103521347 |
| 908 | 908 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/135980.jpg` | 0.7087054253 | 0.7116190195 | 0.7095957994 | 0.002913594246 | 0.0008903741837 |
| 909 | 909 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/476723.jpg` | 0.8825847507 | 0.8823485374 | 0.8821257353 | 0.0002362132072 | 0.0004590153694 |
| 910 | 910 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/940564.jpg` | 0.7853348851 | 0.7855147719 | 0.7845894098 | 0.0001798868179 | 0.0007454752922 |
| 911 | 911 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/121042.jpg` | 0.7370412946 | 0.7385542393 | 0.73837924 | 0.001512944698 | 0.001337945461 |
| 912 | 912 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/33860.jpg` | 0.1827370673 | 0.1833405942 | 0.1809748709 | 0.0006035268307 | 0.001762196422 |
| 913 | 913 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/310847.jpg` | 0.1270985901 | 0.1278344542 | 0.1264337152 | 0.0007358640432 | 0.0006648749113 |
| 914 | 914 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/14850.jpg` | 0.3480232656 | 0.349214673 | 0.3442617655 | 0.001191407442 | 0.00376150012 |
| 915 | 915 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/543962.jpg` | 0.6354364753 | 0.6395460963 | 0.6391694546 | 0.004109621048 | 0.003732979298 |
| 916 | 916 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/782189.jpg` | 0.8281692863 | 0.8282341957 | 0.828433156 | 6.490945816e-05 | 0.0002638697624 |
| 917 | 917 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/721868.jpg` | 0.06079865992 | 0.06099549681 | 0.0597538203 | 0.0001968368888 | 0.001044839621 |
| 918 | 918 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/878421.jpg` | 0.5466470122 | 0.5443950891 | 0.5483854413 | 0.002251923084 | 0.00173842907 |
| 919 | 919 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/349134.jpg` | 0.8080252409 | 0.8096548915 | 0.8123612404 | 0.001629650593 | 0.004335999489 |
| 920 | 920 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/25253.jpg` | 0.4074502885 | 0.4074770212 | 0.3983591497 | 2.673268318e-05 | 0.00909113884 |
| 921 | 921 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/106053.jpg` | 0.5720495582 | 0.5706214905 | 0.5674866438 | 0.001428067684 | 0.004562914371 |
| 922 | 922 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/719161.jpg` | 0.6820598841 | 0.6842039824 | 0.6806452274 | 0.002144098282 | 0.001414656639 |
| 923 | 923 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/876115.jpg` | 0.4151569307 | 0.4153135419 | 0.4137177467 | 0.0001566112041 | 0.00143918395 |
| 924 | 924 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/763793.jpg` | 0.3573962748 | 0.3608718514 | 0.3635112345 | 0.003475576639 | 0.006114959717 |
| 925 | 925 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/587355.jpg` | 0.9764482379 | 0.9767684937 | 0.9767884016 | 0.0003202557564 | 0.0003401637077 |
| 926 | 926 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/171190.jpg` | 0.283874929 | 0.2877381146 | 0.2906649113 | 0.003863185644 | 0.006789982319 |
| 927 | 927 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/1912.jpg` | 0.6291245222 | 0.6306657195 | 0.6340346336 | 0.0015411973 | 0.004910111427 |
| 928 | 928 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/82196.jpg` | 0.830316484 | 0.8308900595 | 0.8303745389 | 0.0005735754967 | 5.805492401e-05 |
| 929 | 929 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/168326.jpg` | 0.9123945832 | 0.9137175083 | 0.9149355888 | 0.001322925091 | 0.002541005611 |
| 930 | 930 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/98002.jpg` | 0.1848208755 | 0.186111629 | 0.1854281574 | 0.001290753484 | 0.0006072819233 |
| 931 | 931 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/765543.jpg` | 0.2222503871 | 0.2239866108 | 0.2184541523 | 0.001736223698 | 0.003796234727 |
| 932 | 932 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/525298.jpg` | 0.5613203645 | 0.5612810254 | 0.5587751865 | 3.933906555e-05 | 0.002545177937 |
| 933 | 933 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/211943.jpg` | 0.2403656393 | 0.2430408597 | 0.240540415 | 0.00267522037 | 0.0001747757196 |
| 934 | 934 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/857635.jpg` | 0.968786478 | 0.9688034654 | 0.9683616161 | 1.698732376e-05 | 0.000424861908 |
| 935 | 935 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/562902.jpg` | 0.9148599505 | 0.9149963856 | 0.9147639871 | 0.0001364350319 | 9.596347809e-05 |
| 936 | 936 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/528474.jpg` | 0.1211985573 | 0.1224252135 | 0.1206843331 | 0.001226656139 | 0.0005142241716 |
| 937 | 937 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/522262.jpg` | 0.3268789053 | 0.3284974396 | 0.3246684074 | 0.001618534327 | 0.002210497856 |
| 938 | 938 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/150856.jpg` | 0.765732646 | 0.7694915533 | 0.7762117386 | 0.003758907318 | 0.0104790926 |
| 939 | 939 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/477502.jpg` | 0.148744002 | 0.1504305005 | 0.1479739249 | 0.001686498523 | 0.0007700771093 |
| 940 | 940 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/181775.jpg` | 0.3178024292 | 0.3200841546 | 0.3208147585 | 0.002281725407 | 0.00301232934 |
| 941 | 941 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/850066.jpg` | 0.9343515038 | 0.9347670078 | 0.9355033636 | 0.0004155039787 | 0.00115185976 |
| 942 | 942 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/401634.jpg` | 0.431412816 | 0.4321553409 | 0.4392960966 | 0.0007425248623 | 0.007883280516 |
| 943 | 943 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/7059.jpg` | 0.05721539259 | 0.05831836164 | 0.0572004281 | 0.00110296905 | 1.496449113e-05 |
| 944 | 944 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/8841.jpg` | 0.1287719309 | 0.1291505098 | 0.1277834922 | 0.0003785789013 | 0.0009884387255 |
| 945 | 945 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/346505.jpg` | 0.5316767097 | 0.5334554911 | 0.5357085466 | 0.001778781414 | 0.004031836987 |
| 946 | 946 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/809863.jpg` | 0.8061497211 | 0.8060358167 | 0.8036960959 | 0.0001139044762 | 0.002453625202 |
| 947 | 947 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/83724.jpg` | 0.02575290017 | 0.02584946528 | 0.02505468577 | 9.656511247e-05 | 0.0006982143968 |
| 948 | 948 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/267403.jpg` | 0.1876126677 | 0.1887750179 | 0.1886241287 | 0.001162350178 | 0.00101146102 |
| 949 | 949 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/946642.jpg` | 0.7033141851 | 0.7058347464 | 0.7061203718 | 0.002520561218 | 0.002806186676 |
| 950 | 950 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/467066.jpg` | 0.2772941887 | 0.2785198689 | 0.2741126418 | 0.001225680113 | 0.003181546926 |
| 951 | 951 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/594817.jpg` | 0.5731141567 | 0.5761586428 | 0.5691795945 | 0.003044486046 | 0.003934562206 |
| 952 | 952 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/158183.jpg` | 0.4221671522 | 0.4242535532 | 0.4209659696 | 0.002086400986 | 0.001201182604 |
| 953 | 953 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/181037.jpg` | 0.1550362408 | 0.1550174057 | 0.1554182917 | 1.883506775e-05 | 0.0003820508718 |
| 954 | 954 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/669446.jpg` | 0.07182215899 | 0.07356633246 | 0.07098602504 | 0.001744173467 | 0.0008361339569 |
| 955 | 955 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/819035.jpg` | 0.7024303675 | 0.7031606436 | 0.6926053166 | 0.0007302761078 | 0.009825050831 |
| 956 | 956 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/933187.jpg` | 0.6671301126 | 0.6681040525 | 0.6690111756 | 0.0009739398956 | 0.001881062984 |
| 957 | 957 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/483.jpg` | 0.6703243852 | 0.6721742749 | 0.6697564721 | 0.001849889755 | 0.0005679130554 |
| 958 | 958 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/342452.jpg` | 0.7334329486 | 0.7346323729 | 0.7297923565 | 0.001199424267 | 0.003640592098 |
| 959 | 959 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/839951.jpg` | 0.4110152721 | 0.4106359482 | 0.4105070531 | 0.0003793239594 | 0.0005082190037 |
| 960 | 960 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/392.jpg` | 0.02091924846 | 0.02120226808 | 0.02049063705 | 0.0002830196172 | 0.0004286114126 |
| 961 | 961 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/786252.jpg` | 0.3954715133 | 0.3950379193 | 0.3938190341 | 0.0004335939884 | 0.001652479172 |
| 962 | 962 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/577635.jpg` | 0.2242753357 | 0.226283282 | 0.2241327614 | 0.002007946372 | 0.0001425743103 |
| 963 | 963 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/270446.jpg` | 0.5344406962 | 0.533801198 | 0.5342603922 | 0.0006394982338 | 0.0001803040504 |
| 964 | 964 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/912244.jpg` | 0.9879392385 | 0.9878627658 | 0.9883148074 | 7.647275925e-05 | 0.0003755688667 |
| 965 | 965 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/34915.jpg` | 0.3653191328 | 0.3691056371 | 0.3619674146 | 0.003786504269 | 0.003351718187 |
| 966 | 966 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/820839.jpg` | 0.4020606577 | 0.4037300348 | 0.3993466198 | 0.001669377089 | 0.002714037895 |
| 967 | 967 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/125593.jpg` | 0.2880814672 | 0.2879344523 | 0.2863734961 | 0.0001470148563 | 0.001707971096 |
| 968 | 968 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/880215.jpg` | 0.3711121082 | 0.371163398 | 0.3654265106 | 5.128979683e-05 | 0.005685597658 |
| 969 | 969 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/255268.jpg` | 0.3441573381 | 0.3450899422 | 0.3468205333 | 0.0009326040745 | 0.002663195133 |
| 970 | 970 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/101602.jpg` | 0.08090335131 | 0.08182030916 | 0.07998362929 | 0.0009169578552 | 0.0009197220206 |
| 971 | 971 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/750640.jpg` | 0.9458081722 | 0.9459322095 | 0.9464304447 | 0.0001240372658 | 0.0006222724915 |
| 972 | 972 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/944348.jpg` | 0.3186172247 | 0.3176978827 | 0.3112430573 | 0.000919342041 | 0.007374167442 |
| 973 | 973 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/753819.jpg` | 0.5760800242 | 0.577177465 | 0.5781084895 | 0.00109744072 | 0.002028465271 |
| 974 | 974 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/817239.jpg` | 0.2391375303 | 0.2404417247 | 0.2405766249 | 0.001304194331 | 0.001439094543 |
| 975 | 975 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/50947.jpg` | 0.007263706531 | 0.007565278094 | 0.007885196246 | 0.0003015715629 | 0.0006214897148 |
| 976 | 976 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/433682.jpg` | 0.1309394538 | 0.131253019 | 0.1313432306 | 0.000313565135 | 0.0004037767649 |
| 977 | 977 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/439419.jpg` | 0.1810436249 | 0.1796985418 | 0.175173372 | 0.001345083117 | 0.005870252848 |
| 978 | 978 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/42193.jpg` | 0.4468712509 | 0.4515002966 | 0.4529597163 | 0.004629045725 | 0.006088465452 |
| 979 | 979 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/836899.jpg` | 0.9633457065 | 0.9640580416 | 0.9627049565 | 0.0007123351097 | 0.0006407499313 |
| 980 | 980 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/218615.jpg` | 0.3820803165 | 0.3840148747 | 0.3839584887 | 0.001934558153 | 0.001878172159 |
| 981 | 981 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/521752.jpg` | 0.8881598115 | 0.8912397623 | 0.8877563477 | 0.003079950809 | 0.0004034638405 |
| 982 | 982 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/581422.jpg` | 0.5208533406 | 0.5259159803 | 0.5255364776 | 0.005062639713 | 0.00468313694 |
| 983 | 983 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/382169.jpg` | 0.454778403 | 0.4562382996 | 0.4580725729 | 0.001459896564 | 0.003294169903 |
| 984 | 984 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/12170.jpg` | 0.4968094826 | 0.4970997572 | 0.5022226572 | 0.0002902746201 | 0.005413174629 |
| 985 | 985 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/786904.jpg` | 0.7438820004 | 0.7450628281 | 0.7448824644 | 0.001180827618 | 0.001000463963 |
| 986 | 986 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/65813.jpg` | 0.2315001488 | 0.2328089178 | 0.2308402956 | 0.001308768988 | 0.00065985322 |
| 987 | 987 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/642941.jpg` | 0.2587316334 | 0.2603513896 | 0.2605848312 | 0.001619756222 | 0.001853197813 |
| 988 | 988 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/12137.jpg` | 0.1036819369 | 0.1041759849 | 0.1014509574 | 0.0004940479994 | 0.002230979502 |
| 989 | 989 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/518585.jpg` | 0.8844474554 | 0.8853474259 | 0.8832832575 | 0.0008999705315 | 0.001164197922 |
| 990 | 990 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/651972.jpg` | 0.5826922059 | 0.5814962387 | 0.5811357498 | 0.001195967197 | 0.001556456089 |
| 991 | 991 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/802525.jpg` | 0.881528914 | 0.881292522 | 0.8837237358 | 0.0002363920212 | 0.002194821835 |
| 992 | 992 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/30101.jpg` | 0.1562894583 | 0.1586531401 | 0.1614458412 | 0.002363681793 | 0.005156382918 |
| 993 | 993 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/326630.jpg` | 0.3768764734 | 0.3763776422 | 0.3739052713 | 0.0004988312721 | 0.002971202135 |
| 994 | 994 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/606435.jpg` | 0.866216898 | 0.8663710356 | 0.8641818762 | 0.0001541376114 | 0.002035021782 |
| 995 | 995 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/199584.jpg` | 0.3188140988 | 0.3201996386 | 0.3160919547 | 0.00138553977 | 0.002722144127 |
| 996 | 996 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/532913.jpg` | 0.9920555949 | 0.9920434952 | 0.9916325212 | 1.209974289e-05 | 0.0004230737686 |
| 997 | 997 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/354340.jpg` | 0.7326488495 | 0.7315882444 | 0.7278643847 | 0.001060605049 | 0.004784464836 |
| 998 | 998 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/429501.jpg` | 0.6953528523 | 0.6960670948 | 0.6974490285 | 0.0007142424583 | 0.002096176147 |
| 999 | 999 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/251277.jpg` | 0.1825244725 | 0.1829456836 | 0.1831014603 | 0.0004212111235 | 0.0005769878626 |
| 1000 | 1000 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/729053.jpg` | 0.3477416039 | 0.3495579064 | 0.3422045708 | 0.001816302538 | 0.005537033081 |
| 1001 | 1001 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/4304.jpg` | 0.6263131499 | 0.6255556345 | 0.6252015829 | 0.0007575154305 | 0.00111156702 |
| 1002 | 1002 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/644255.jpg` | 0.9791339636 | 0.9794611931 | 0.979945302 | 0.0003272294998 | 0.0008113384247 |
| 1003 | 1003 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/860530.jpg` | 0.9139958024 | 0.9139922261 | 0.9139457345 | 3.576278687e-06 | 5.006790161e-05 |
| 1004 | 1004 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/458719.jpg` | 0.247942999 | 0.2487759888 | 0.2498134077 | 0.0008329898119 | 0.001870408654 |
| 1005 | 1005 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/787831.jpg` | 0.5299395919 | 0.5298065543 | 0.5267813802 | 0.0001330375671 | 0.003158211708 |
| 1006 | 1006 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/512503.jpg` | 0.744058907 | 0.7434897423 | 0.7389683723 | 0.000569164753 | 0.005090534687 |
| 1007 | 1007 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/660880.jpg` | 0.8813108802 | 0.8808479309 | 0.8786809444 | 0.000462949276 | 0.002629935741 |
| 1008 | 1008 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/883217.jpg` | 0.5150390863 | 0.5155400038 | 0.5105538368 | 0.0005009174347 | 0.004485249519 |
| 1009 | 1009 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/58488.jpg` | 0.1984613091 | 0.197781682 | 0.1963831484 | 0.0006796270609 | 0.002078160644 |
| 1010 | 1010 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/263737.jpg` | 0.181055814 | 0.1830142736 | 0.1828582734 | 0.001958459616 | 0.001802459359 |
| 1011 | 1011 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/127899.jpg` | 0.06500449777 | 0.0658159554 | 0.065289177 | 0.000811457634 | 0.000284679234 |
| 1012 | 1012 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/11220.jpg` | 0.1346177906 | 0.1371066123 | 0.1374990344 | 0.002488821745 | 0.002881243825 |
| 1013 | 1013 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/19140.jpg` | 0.1078000814 | 0.1083367467 | 0.1079674214 | 0.0005366653204 | 0.0001673400402 |
| 1014 | 1014 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/666046.jpg` | 0.7115294337 | 0.7111101151 | 0.7100166678 | 0.000419318676 | 0.001512765884 |
| 1015 | 1015 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/732898.jpg` | 0.3840721548 | 0.3846091032 | 0.3833059072 | 0.0005369484425 | 0.0007662475109 |
| 1016 | 1016 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/561840.jpg` | 0.9459347725 | 0.9466902018 | 0.9461472034 | 0.0007554292679 | 0.000212430954 |
| 1017 | 1017 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/53065.jpg` | 0.3442281187 | 0.3464029133 | 0.3370733857 | 0.002174794674 | 0.007154732943 |
| 1018 | 1018 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/125731.jpg` | 0.03193113953 | 0.03196663409 | 0.03222342581 | 3.549456596e-05 | 0.0002922862768 |
| 1019 | 1019 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/635171.jpg` | 0.9472613335 | 0.9475426078 | 0.9475449324 | 0.0002812743187 | 0.0002835988998 |
| 1020 | 1020 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/903358.jpg` | 0.8550171852 | 0.8552811146 | 0.8544870615 | 0.0002639293671 | 0.0005301237106 |
| 1021 | 1021 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/796219.jpg` | 0.3740433156 | 0.3743570745 | 0.3702800572 | 0.0003137588501 | 0.003763258457 |
| 1022 | 1022 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/822573.jpg` | 0.6436108947 | 0.6443171501 | 0.6392028928 | 0.0007062554359 | 0.0044080019 |
| 1023 | 1023 | `/home/omen_pc1/photo_score_project/data/raw/ava/images/780082.jpg` | 0.467829138 | 0.4693706334 | 0.4685163498 | 0.001541495323 | 0.0006872117519 |
