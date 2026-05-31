# Dynamic PTQ Extended Parity Report

## Summary
- **Reference FP16**: `/home/omen_pc1/photo_score_project/outputs/icaa_tf_native_tflite_fp16_20260516_144023/icaa_dat_tf_native_fp16.tflite`.
- **Dynamic PTQ**: `/home/omen_pc1/photo_score_project/outputs/icaa_ptq_candidates_20260601/icaa_dat_tf_native_dynamic_range_ptq.tflite`.
- **Image count**: 50.
- **Image selection source**: `deterministic_icaa17k`.
- **A-cut 50-image set found**: False.
- **Preprocessing**: RGB resize to `224x224`, `pixel / 255.0`, then ImageNet mean/std.
- **Passes model-level abs diff <= 0.001 with no NaN/range errors**: False.

## Overall Error Metrics
| Scope | MAE | MaxAE | Median AE | P90 AE | P95 AE | >0.001 | >0.005 | >0.01 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 0.003328498899936676 | 0.009905099868774414 | 0.0028647780418395996 | 0.006955149769783022 | 0.007731452584266663 | 81 | 25 | 0 |
| output_0 | 0.0039001697301864623 | 0.009905099868774414 | 0.0035609155893325806 | 0.007180136442184448 | 0.007756635546684265 | 43 | 16 | 0 |
| output_1 | 0.0027568280696868897 | 0.008145809173583984 | 0.0021817684173583984 | 0.005873370170593262 | 0.007356497645378112 | 38 | 9 | 0 |

## NaN And Range Checks
- **Candidate NaN count**: 0.
- **Reference NaN count**: 0.
- **Diff NaN count**: 0.
- **Candidate range errors outside [0,1]**: 0.
- **Reference range errors outside [0,1]**: 0.

## Interpreter Details
- **FP16 input details**: `[{'name': 'serving_default_image:0', 'index': 0, 'shape': [1, 224, 224, 3], 'dtype': 'float32', 'quantization': (0.0, 0)}]`.
- **FP16 output details**: `[{'name': 'StatefulPartitionedCall:0', 'index': 4763, 'shape': [1, 2], 'dtype': 'float32', 'quantization': (0.0, 0)}]`.
- **Dynamic PTQ input details**: `[{'name': 'serving_default_image:0', 'index': 0, 'shape': [1, 224, 224, 3], 'dtype': 'float32', 'quantization': (0.0, 0)}]`.
- **Dynamic PTQ output details**: `[{'name': 'StatefulPartitionedCall:0', 'index': 4284, 'shape': [1, 2], 'dtype': 'float32', 'quantization': (0.0, 0)}]`.

## Selection Details
- **Selection info**: `{'source': 'deterministic_icaa17k', 'official_test_csv_present': True, 'image_root': '/home/omen_pc1/photo_score_project/data/raw/icaa17k', 'acut_search': {'source': 'acut_manifest', 'manifest_path': '/home/omen_pc1/photo_score_project/outputs/test_vila_human_acut_review_stage5_vs_ensemble_20260423/ensemble_ranks_higher.csv', 'manifest_image_count': 50, 'candidate_manifest_count': 19, 'invalid_image_count': 1, 'invalid_images': [{'path': 'test_vila/IMG_20240519_215827_510.jpg', 'error': "cannot identify image file 'test_vila/IMG_20240519_215827_510.jpg'"}]}, 'invalid_image_count': 0, 'invalid_images': []}`.

## Threshold Interpretation
- The user-provided A-cut parity gate is stricter than the observed Dynamic PTQ errors if MaxAE exceeds `0.001`.
- This report does not evaluate final A-cut ranking because only ICAA model outputs are compared here.
