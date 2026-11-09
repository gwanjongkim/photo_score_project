# Mobile Student IQA 14105 TFLite Export and Hybrid Design Report

## 1. Executive Summary
- Final decision: `STUDENT_TFLITE_READY_FOR_MOBILE_PROFILING_LOG_ONLY`.
- Run type: TFLite export/parity/profiling preparation only.
- Mobile status: log-only candidate, not production replacement.
- Runtime seconds: `32.51`.
- Parity sample rows: `300`.

## 2. Sources Inspected
- `/home/omen_pc1/photo_score_project/outputs/mobile_student_iqa_14105_full_controlled_20260604/best_by_val_mae_model.keras`
- `/home/omen_pc1/photo_score_project/outputs/student_distillation_manifest_14105_20260603/student_distillation_manifest.csv`
- `/home/omen_pc1/photo_score_project/outputs/mobile_student_iqa_14105_checkpoint_eval_20260604/report.md`
- `/home/omen_pc1/photo_score_project/outputs/mobile_student_iqa_14105_checkpoint_eval_20260604/checkpoint_metrics.csv`
- `/home/omen_pc1/photo_score_project/outputs/mobile_student_iqa_14105_checkpoint_eval_20260604/datasetwise_checkpoint_metrics.csv`
- `scripts/technical_iqa/test_tflite_parity.py` (old hard-coded parity script).
- `outputs/final_topiq_lite_mixed112_export_20260517/export_script.py` (prior builtin-only converter pattern).
- `scripts/technical_iqa/evaluate_student_iqa_14105_checkpoints.py` (metric/preprocess reference).

## 3. Export Configuration
- TensorFlow: `2.21.0`.
- Built with CUDA: `True`.
- GPUs visible: `[]`.
- Converter ops: `TFLITE_BUILTINS` only.
- FP16 conversion: `Optimize.DEFAULT` plus `supported_types=[tf.float16]`.

| model | path | mb | export_seconds |
| --- | --- | --- | --- |
| fp32 | /home/omen_pc1/photo_score_project/outputs/mobile_student_iqa_14105_tflite_export_20260604/student_iqa_14105_fp32.tflite | 23.683994 | 8.332864 |
| fp16 | /home/omen_pc1/photo_score_project/outputs/mobile_student_iqa_14105_tflite_export_20260604/student_iqa_14105_fp16.tflite | 11.963104 | 8.212343 |

## 4. TFLite Model Structure
| model | input_shape | input_dtype | output_shape | output_dtype | op_count | has_flex_ops | unique_ops |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp32 | [1, 256, 256, 3] | <class 'numpy.float32'> | [1, 1] | <class 'numpy.float32'> | 353 | False | ADD,CONV_2D,DELEGATE,DEPTHWISE_CONV_2D,FULLY_CONNECTED,LOGISTIC,MEAN,MUL,PACK,RESHAPE,SHAPE,STRIDED_SLICE,SUB |
| fp16 | [1, 256, 256, 3] | <class 'numpy.float32'> | [1, 1] | <class 'numpy.float32'> | 544 | False | ADD,CONV_2D,DELEGATE,DEPTHWISE_CONV_2D,DEQUANTIZE,FULLY_CONNECTED,LOGISTIC,MEAN,MUL,PACK,RESHAPE,SHAPE,STRIDED_SLICE,SUB |

## 5. Keras vs TFLite Parity
| model | dataset | n | mae_to_keras_0_100 | max_abs_diff_0_100 | mean_diff_0_100 | srcc_vs_keras | plcc_vs_keras | tflite_pred_min | tflite_pred_max | tflite_pred_mean | tflite_pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp32 | overall | 300 | 0.000005 | 0.000039 | 0.000001 | 1.000000 | 1.000000 | 11.600722 | 76.832086 | 56.688728 | 13.855952 |
| fp32 | flive | 100 | 0.000003 | 0.000012 | -0.000000 | 1.000000 | 1.000000 | 46.290708 | 71.674812 | 63.979497 | 3.615850 |
| fp32 | koniq10k | 100 | 0.000003 | 0.000018 | 0.000001 | 1.000000 | 1.000000 | 31.586599 | 71.036071 | 61.080051 | 5.331367 |
| fp32 | spaq | 100 | 0.000008 | 0.000039 | 0.000001 | 1.000000 | 1.000000 | 11.600722 | 76.832086 | 45.006637 | 18.043117 |
| fp16 | overall | 300 | 0.018522 | 0.142264 | -0.016724 | 0.999987 | 0.999999 | 11.565838 | 76.823699 | 56.672004 | 13.862649 |
| fp16 | flive | 100 | 0.011934 | 0.088888 | -0.011025 | 0.999988 | 0.999995 | 46.201819 | 71.675146 | 63.968473 | 3.620007 |
| fp16 | koniq10k | 100 | 0.011004 | 0.062805 | -0.009068 | 0.999952 | 0.999997 | 31.543803 | 71.020693 | 61.070982 | 5.334963 |
| fp16 | spaq | 100 | 0.032628 | 0.142264 | -0.030078 | 0.999964 | 0.999998 | 11.565838 | 76.823699 | 44.976558 | 18.043833 |

## 6. Model Size / Runtime Notes
| model | bytes | mb | runtime_seconds | avg_runtime_ms_per_image |
| --- | --- | --- | --- | --- |
| fp32 | 24834468 | 23.683994 | 4.799690 | 15.998968 |
| fp16 | 12544224 | 11.963104 | 4.847984 | 16.159945 |
| keras_reference | 37497194 | 35.760111 | 3.664770 | 12.215901 |

## 7. FLIVE Weakness and Hybrid Plan
- Student alone is not recommended as production replacement.
- Production/FLIVE robustness signal should remain active.
- Candidate deployment is log-only mobile profiling.
- Hybrid design note: `hybrid_runtime_design.md`.
- Do not alter Flutter production score yet.

## 8. Final Decision
STUDENT_TFLITE_READY_FOR_MOBILE_PROFILING_LOG_ONLY

## 9. Recommended Next Step
- Run mobile-device profiling/log-only parity against production technical score and FLIVE-heavy disagreement cases.
