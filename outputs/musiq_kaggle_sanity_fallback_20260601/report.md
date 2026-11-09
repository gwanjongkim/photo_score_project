# Official Kaggle MUSIQ 100-image Sanity Benchmark Report (With Fallback)

## 1. Executive Summary
- Final decision: `MUSIQ_FAILURE_FIXED_AND_READY_FOR_FULL_BENCHMARK`.
- Evaluated 300 images and wrote 900 predictions.
- Elapsed seconds: 42.8.
- Scope stayed benchmark-only: no training, TFLite export, Flutter edits, production scoring edits, distillation, or local repo-practical MUSIQ.

## 2. Files Inspected
- Input manifest: `/home/omen_pc1/photo_score_project/outputs/research_protocol_iqa_manifests_20260530/benchmark_all_research_protocol_iqa.csv`.
- Output directory: `/home/omen_pc1/photo_score_project/outputs/musiq_kaggle_sanity_fallback_20260601`.
- Seed: `20260601`.
- Sample target: `100` readable images per dataset.
- TensorFlow: `2.21.0`.
- TensorFlow Hub: `0.16.1`.
- setuptools: `80.10.2`.

## 3. Failed Image Inspection
Refer to `outputs/musiq_kaggle_failure_diagnosis_20260601/failed_image_inspection.csv`.

## 4. Fallback Tests
Refer to `outputs/musiq_kaggle_failure_diagnosis_20260601/fallback_test_results.csv`.

## 5. Patch Decision
Patched with EXIF-transposed RGB re-encode fallback in `predict_one`.

## 6. Optional Rerun Results
This report is the result of the optional rerun.

## 7. Production Baseline Implication
| Dataset | Production SRCC | Best MUSIQ Variant | Best MUSIQ SRCC | Delta |
|---|---:|---|---:|---:|
| FLIVE | 0.574 | paq2piq | 0.4815 | -0.0925 |
| KonIQ-10k | 0.934 | koniq-10k | 0.9275 | -0.0065 |
| SPAQ | 0.873 | spaq | 0.9460 | 0.0730 |

## 8. Final Decision
`MUSIQ_FAILURE_FIXED_AND_READY_FOR_FULL_BENCHMARK`

## 9. Next Step
- Run the full benchmark for the best-performing variant(s).

## Detailed Metrics

| Variant | Dataset | n | SRCC | PLCC | Aligned MAE | Fallbacks | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| koniq-10k | FLIVE | 100 | 0.0589 | 0.1080 | 4.9286 | 1 | 0 |
| koniq-10k | KonIQ-10k | 100 | 0.9275 | 0.9547 | 2.2801 | 0 | 0 |
| koniq-10k | SPAQ | 100 | 0.8546 | 0.8552 | 8.2522 | 0 | 0 |
| spaq | FLIVE | 100 | 0.1688 | 0.2640 | 4.9019 | 1 | 0 |
| spaq | KonIQ-10k | 100 | 0.7504 | 0.7969 | 4.7921 | 0 | 0 |
| spaq | SPAQ | 100 | 0.9460 | 0.9441 | 5.4643 | 0 | 0 |
| paq2piq | FLIVE | 100 | 0.4815 | 0.5596 | 4.3793 | 1 | 0 |
| paq2piq | KonIQ-10k | 100 | 0.7998 | 0.8540 | 4.0037 | 0 | 0 |
| paq2piq | SPAQ | 100 | 0.8555 | 0.8419 | 9.0269 | 0 | 0 |

## Failure Cases

- None.
