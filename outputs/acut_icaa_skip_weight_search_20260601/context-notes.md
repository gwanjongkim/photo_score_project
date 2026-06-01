# A-cut ICAA Skip Weight Search Context Notes

- 2026-06-01: Scope is WSL benchmark-only. Do not modify Flutter production code, production metadata, model artifacts, A-cut technical scoring, KonIQ/FLIVE technical weights, or TOPIQ/C6.
- 2026-06-01: Android logs supplied by user indicate ICAA included average about 1594.5 ms/image, ICAA skip average about 548.4 ms/image, and ICAA inference cost about 1022.9 ms/image. These are user-provided speed facts, not remeasured in WSL.
- 2026-06-01: Need separate confirmed facts from assumptions. Existing production/reference aesthetic weights must be verified from code/docs before use.
- 2026-06-01: Memory says production A-cut selector path is `src/infer/select_best_shots.py` via `run_acut_pipeline.py`; separate mobile aesthetic policy files should not be conflated with this selector path.
- 2026-06-01: Accessible Flutter checkout was inspected read-only and did not expose an active ICAA aesthetic ensemble or the `POZY_ACUT_SKIP_ICAA_EXPERIMENT` flag. Treat the full ICAA reference weights as a WSL benchmark assumption unless the target Flutter branch confirms them.
- 2026-06-01: Selected source cache is `outputs/test_vila_aesthetic_direct_compare_20260504_135316/old_three_model_scores.csv` with 50 images and NIMA/RGNet/A-LAMP scores. No cached four-model NIMA/RGNet/A-LAMP/ICAA table was found.
- 2026-06-01: ICAA is scored in WSL from `outputs/icaa_tf_native_savedmodel_20260516_134439/saved_model`, using RGB 224x224 bilinear resize plus ImageNet mean/std normalization from the local ICAA export script. Output index 1/color is used as the active ICAA aesthetic signal for this benchmark.
- 2026-06-01: First benchmark run found one unreadable cached image, `test_vila/IMG_20240519_215827_510.jpg`. The script now records excluded images in `excluded_images.csv` and evaluates the readable subset instead of silently substituting data.
- 2026-06-01: Final benchmark evaluated 49 readable images. Proposed 0.20/0.30/0.50 scored MAE 0.020957, Spearman 0.951224, Top-10 overlap 8/10. Best selected ratio was NIMA 0.20, RGNet 0.51, A-LAMP 0.29 with MAE 0.017057, Spearman 0.983776, Top-10 overlap 9/10. Final decision B.
