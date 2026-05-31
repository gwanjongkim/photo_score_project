# ICAA PTQ Review Report

## Decision
- **Final decision**: C. Dynamic PTQ fails the A-cut model-level parity threshold; move to QAT/student distillation.
- **Production readiness**: Not proven. Do not copy either candidate into Flutter.
- **Android experimental benchmark**: Not recommended yet because the Dynamic PTQ parity gate already fails.

## Confirmed Facts
- All new review files are under `outputs/icaa_ptq_candidates_20260601/review_20260601/`.
- FP16 reference: `outputs/icaa_tf_native_tflite_fp16_20260516_144023/icaa_dat_tf_native_fp16.tflite`.
- Dynamic PTQ candidate: `outputs/icaa_ptq_candidates_20260601/icaa_dat_tf_native_dynamic_range_ptq.tflite`.
- Full INT8 float-IO candidate: `outputs/icaa_ptq_candidates_20260601/icaa_dat_tf_native_full_integer_int8_float_io.tflite`.
- Preprocessing used by the review scripts is unchanged: RGB resize to `224x224`, `pixel / 255.0`, then ImageNet mean/std.
- A possible A-cut 50-image manifest was found at `outputs/test_vila_human_acut_review_stage5_vs_ensemble_20260423/ensemble_ranks_higher.csv`, but one referenced file was unreadable, so the extended comparison used 50 deterministic valid ICAA17K images.

## Dynamic PTQ Extended Parity
- Image count: 50.
- Output values compared: 100.
- Overall MAE: `0.003328498899936676`.
- Overall MaxAE: `0.009905099868774414`.
- Overall median AE: `0.0028647780418395996`.
- Overall P90 AE: `0.006955149769783022`.
- Overall P95 AE: `0.007731452584266663`.
- Count over `0.001`: 81 of 100 output values.
- Count over `0.005`: 25 of 100 output values.
- Count over `0.01`: 0 of 100 output values.
- NaN count: 0.
- Candidate range errors outside `[0,1]`: 0.
- Result against model-level abs diff `<= 0.001`: **fail**.

## Per-Output Metrics
| Output | MAE | MaxAE | Median AE | P90 AE | P95 AE | >0.001 | >0.005 | >0.01 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| output_0 | 0.0039001697301864623 | 0.009905099868774414 | 0.0035609155893325806 | 0.007180136442184448 | 0.007756635546684265 | 43 | 16 | 0 |
| output_1 | 0.0027568280696868897 | 0.008145809173583984 | 0.0021817684173583984 | 0.005873370170593262 | 0.007356497645378112 | 38 | 9 | 0 |

## WSL Latency
- This is local WSL TensorFlow Lite interpreter timing only. It is not Android/iOS latency and is not a production speed claim.
- Warmup runs per model: 8.
- Timed runs per model: 50.

| Model | Mean ms | Median ms | P90 ms | Max ms |
| :--- | ---: | ---: | ---: | ---: |
| FP16 reference | 331.652110934956 | 332.08576796459965 | 336.0989319276996 | 337.78566599357873 |
| Dynamic PTQ | 191.28270630375482 | 191.37078401399776 | 194.3571365380194 | 196.38122501783073 |

## Full INT8 Failure
- The INT8 float-IO model allocates and exposes float32 input/output.
- Minimal invokes failed for zeros, ones, random normal input, and the first real ICAA image.
- Exact repeated exception: `RuntimeError: tensorflow/lite/kernels/div.cc:242 data[i] != 0 was not true.Node number 593 (DIV) failed to invoke.`
- Failed node 593 is a quantized `DIV` op with int8 inputs and int8 output.
- Denominator tensor `tf_icaa17k_dat_1/stage2_1/attn_1_1/add_34` had valid quantization parameters but contained literal zero values at runtime.
- Quantization scan found no missing, zero, NaN, or Inf scales among quantized tensors.
- Conservative interpretation: this is a quantized graph numeric failure around attention normalization, not an external preprocessing or float32 IO contract failure.

## Assumptions And Unknowns
- Final A-cut score diff and ranking changes were not computed because this review compares ICAA model outputs only and does not run the full A-cut scoring pipeline.
- Since model-level output parity already fails the `0.001` gate, the missing final-score/ranking check does not change the conservative decision.
- It is not proven that representative calibration alone caused the INT8 failure. A different PTQ setting might change behavior, but the current evidence does not justify a new production-adjacent candidate.

## Generated Files
- `dynamic_ptq_extended_report.md`
- `dynamic_ptq_extended_comparison.csv`
- `dynamic_ptq_latency_report.md`
- `dynamic_ptq_extended_summary.json`
- `full_int8_failure_analysis.md`
- `full_int8_failure_analysis.json`
- `full_int8_op_inventory.csv`
- `next_action.md`
