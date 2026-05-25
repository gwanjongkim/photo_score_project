# RGNet Paper AADB TFLite Parity Report

- Created: 2026-05-14T12:00:51.737472+09:00
- Image source: manifest:data/processed/aadb/val.csv
- Sample count: 64
- Built-in DenseNet preprocessing detected: True
- External ImageNet mean/std preprocessing applied: False

## Metrics

| Model | Max abs diff | Mean abs diff | Pearson | Spearman | Ranking agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 TFLite | 4.470348358e-07 | 1.005828381e-07 | 1 | 1 | 1 |
| FP16 TFLite | 0.0005260109901 | 0.00012910855 | 0.9999988286 | 0.9999084249 | 0.9990079365 |

## Interpretation

- FP32 strict parity: PASS, max_abs_diff=4.470348358e-07
- FP16 ranking preservation: PASS for ranking-oriented review; mean difference and ranking/correlation targets are preserved.
- Ready for Flutter smoke test: True
- Ready to replace `rgnet_aadb_gpu.tflite`: False
