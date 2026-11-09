# Research-Protocol Image-Level Test Evaluation Analysis

## 1. Summary
The image-level test evaluation (4,967 images) provides a high-confidence baseline for technical IQA performance. The `koniq_standalone` baseline is exceptionally accurate and well-calibrated on KonIQ and SPAQ, while `flive_standalone` is essential for the FLIVE dataset but exhibits significant positive bias on other datasets.

## 2. Inputs
- `outputs/research_protocol_image_test_eval_20260530/report.md`
- `outputs/research_protocol_image_test_eval_20260530/image_test_metrics.csv`
- `outputs/research_protocol_image_test_eval_20260530/score_range_checks.csv`
- `outputs/research_protocol_image_test_eval_20260530/runtime_summary.csv`
- `outputs/research_protocol_image_test_eval_20260530/image_test_predictions.csv`
- `outputs/research_protocol_image_test_eval_20260530/image_test_manifest.csv`
- `outputs/research_protocol_iqa_manifests_20260530/report.md`
- `outputs/research_protocol_bounded_eval_analysis_20260530/report.md`

## 3. Evaluation Scope
- **KonIQ-10k run_01 test**: 2,015 images (Source: `koniq10k_scores_and_distributions.csv`)
- **SPAQ official test**: 1,125 images (Source: `spaqTest.csv`)
- **FLIVE/PaQ-2-PiQ image-test**: 1,827 images (Source: IQA-Toolbox `meta_info_FLIVEDataset.csv`)
- **Total images**: 4,967
- **Patch rows**: Excluded
- **Baselines**: `koniq_standalone` (exports/tflite/koniq_mobile.tflite), `flive_standalone` (exports/tflite/flive_image_mobile.tflite)

## 4. Metrics by Dataset and Baseline

| Dataset | Baseline | SRCC | PLCC | MAE | RMSE |
|---------|----------|------|------|-----|------|
| KonIQ-10k | KonIQ | 0.937 | 0.952 | 5.18 | 6.86 |
| KonIQ-10k | FLIVE | 0.721 | 0.763 | 9.64 | 11.99 |
| SPAQ | KonIQ | 0.856 | 0.845 | 9.58 | 11.81 |
| SPAQ | FLIVE | 0.827 | 0.681 | 20.88 | 24.98 |
| FLIVE | FLIVE | 0.660 | 0.773 | 3.80 | 5.05 |
| FLIVE | KonIQ | 0.486 | 0.602 | 8.95 | 11.31 |

## 5. Score Range and Calibration Diagnosis
- **Prediction Validity**: All predictions for both baselines fall strictly within the 0..100 range.
- **KonIQ Standalone Calibration**: Exhibits excellent calibration across KonIQ and SPAQ. On SPAQ, the mean bias is nearly zero (-0.32 points).
- **FLIVE Standalone Calibration**: Shows significant positive bias on non-home datasets. On SPAQ, the mean is 18.4 points higher than ground truth (66.4 vs 47.9), and it fails to predict low scores (min 17.7 vs GT 4.8).
- **Cross-Dataset Generalization**: `koniq_standalone` generalizes significantly better to SPAQ than `flive_standalone`.

## 6. Interpretation
- **Winner on KonIQ**: `koniq_standalone` (0.937 SRCC).
- **Winner on SPAQ**: `koniq_standalone` (0.856 SRCC).
- **Winner on FLIVE**: `flive_standalone` (0.660 SRCC).
- **Combination Strategy**: The baselines have complementary strengths. The current production `technical_score` (KonIQ + FLIVE) should be evaluated next to determine if the combined ensemble provides a more robust signal across all three datasets than either standalone model.
- **Ensemble Potential**: A simple average or weighted combination is highly likely to benefit from `koniq_standalone`'s calibration and `flive_standalone`'s domain knowledge in FLIVE.

## 7. Runtime Interpretation
- Total evaluation of 4,967 images took 74.5 seconds (~15ms per image including I/O).
- Model invocation only is ~5ms per model.
- Scaling to larger datasets (e.g. 50k-100k) is feasible within reasonable timeframes (15-30 minutes).

## 8. Limitations
- **Image-level focus**: This analysis strictly covers full-size images. Patch-level performance on FLIVE is still unknown.
- **Production combination**: The actual `technical_score` (ensemble) has not yet been evaluated on this full test split.
- **C6/AADB/AVA**: These datasets were intentionally excluded to focus on the research-protocol technical IQA audit.

## 9. Next Action
**A. Evaluate current production technical_score combination on the same image-level test split**
We need to establish the baseline performance of the actual ensemble currently in production to determine the headroom for improvement.
