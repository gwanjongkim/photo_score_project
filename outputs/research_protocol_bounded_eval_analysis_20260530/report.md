# Research-Protocol Bounded Evaluation Analysis

## 1. Summary
Bounded evaluation results (300 rows) are interpretable and provide a clear signal for baseline comparison. The `koniq_standalone` baseline is highly accurate on KonIQ-10k (SRCC=0.938) and generalizes well to SPAQ. The `flive_standalone` baseline is weaker overall but remains the best option for the FLIVE dataset specifically. No critical technical failures (preprocessing/loading) were detected.

## 2. Inputs
- `outputs/research_protocol_bounded_eval_20260530/report.md`
- `outputs/research_protocol_bounded_eval_20260530/bounded_metrics.csv`
- `outputs/research_protocol_bounded_eval_20260530/runtime_summary.csv`
- `outputs/research_protocol_bounded_eval_20260530/bounded_predictions.csv`
- `outputs/research_protocol_bounded_eval_20260530/bounded_sample_manifest.csv`
- `outputs/research_protocol_iqa_manifests_20260530/report.md`

## 3. Metrics by Dataset and Baseline

| Dataset | Baseline | SRCC | PLCC | MAE | RMSE | Max AE |
|---------|----------|------|------|-----|------|--------|
| KonIQ-10k | KonIQ | 0.938 | 0.957 | 5.33 | 6.99 | 22.34 |
| KonIQ-10k | FLIVE | 0.778 | 0.814 | 9.37 | 11.69 | 27.46 |
| SPAQ | KonIQ | 0.854 | 0.831 | 9.28 | 11.75 | 37.70 |
| SPAQ | FLIVE | 0.810 | 0.664 | 20.76 | 24.52 | 53.41 |
| FLIVE | FLIVE | 0.570 | 0.713 | 4.43 | 5.94 | 20.77 |
| FLIVE | KonIQ | 0.455 | 0.674 | 8.91 | 10.88 | 27.72 |

## 4. Interpretation
- **KonIQ Performance**: `koniq_standalone` performs exceptionally well on its home dataset. SRCC of 0.938 confirms the model is correctly loaded and evaluated.
- **SPAQ Generalization**: `koniq_standalone` generalizes better to SPAQ (SRCC 0.854) than `flive_standalone` (SRCC 0.810).
- **FLIVE Performance**: `flive_standalone` is superior on FLIVE (SRCC 0.570 vs 0.455). The low absolute SRCC is expected due to the complexity of the FLIVE dataset.
- **Score Scale Issue**: `flive_standalone` shows a significantly high MAE (20.76) on SPAQ compared to `koniq_standalone` (9.28). This suggests a calibration or scale mismatch when applying the FLIVE model to SPAQ images.
- **Dataset Mismatch**: No catastrophic mismatch found, but `koniq_standalone` clearly struggles with the varied authentic distortions in FLIVE.

## 5. Runtime Interpretation
- Average prediction time (including I/O and prep) is ~7.5ms per image.
- Total 300 images took ~4.5 seconds.
- Evaluating the full test splits (approx. 5,000 images total across datasets) is estimated to take ~40-60 seconds, which is highly feasible.

## 6. Limitations
- **Sample Size**: Only 100 rows per dataset were evaluated. Statistical significance for small differences is low.
- **Image-level only**: FLIVE patches were excluded from this bounded run.
- **No Production Impact**: These results are for research/audit only. `technical_score` and C6 remain unchanged.
- **Model Selection**: This data is insufficient for final model selection but confirms baselines are functional.

## 7. Next Action
**B. Run full test split evaluation for image-level rows only**
The system is stable and runtime is efficient. Evaluating all test images (KonIQ run_01, SPAQ official test, FLIVE image test) will provide robust metrics for the baseline report.

## 8. Forbidden Actions
- No training was performed.
- No production scores were modified.
- C6 remains for debug/logging only.
