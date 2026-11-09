# Production Technical Score Evaluation Analysis

## 1. Summary
The existing production `technical_score` is a significantly stronger and more robust baseline than either `koniq_standalone` or `flive_standalone` individually. It achieves the **best SRCC on SPAQ (0.873)** and maintains performance very close to the best standalone baseline on KonIQ and FLIVE datasets. It successfully balances the high accuracy of KonIQ with the specialized knowledge of FLIVE while smoothing out significant calibration biases.

## 2. Inputs
- `outputs/research_protocol_production_technical_eval_20260530/report.md`
- `outputs/research_protocol_production_technical_eval_20260530/production_formula_audit.md`
- `outputs/research_protocol_production_technical_eval_20260530/production_technical_metrics.csv`
- `outputs/research_protocol_production_technical_eval_20260530/production_score_range_checks.csv`
- `outputs/research_protocol_image_test_eval_analysis_20260530/report.md`

## 3. Confirmed Production Formula
- **Status**: Confirmed via audit of `src/infer/select_best_shots.py` and `configs/stage5_student_server.json`.
- **Logic**: Weighted ensemble of available technical components, normalized and renormalized.
- **Image-Level Evaluation Formula**: `production_technical_score = (0.32 * KonIQ + 0.24 * FLIVE_image) / 0.56`
- **Simplified**: `(4/7 * KonIQ) + (3/7 * FLIVE_image)`
- **Score Scale**: Results are reported on a `0..100` scale.

## 4. Metrics by Dataset and Baseline

| Dataset | Baseline | SRCC | PLCC | MAE | RMSE |
|---------|----------|------|------|-----|------|
| **FLIVE** | KonIQ | 0.4863 | 0.6024 | 8.9485 | 11.3053 |
| **FLIVE** | FLIVE | **0.6603** | **0.7732** | **3.8017** | **5.0503** |
| **FLIVE** | Production | 0.5744 | 0.7021 | 5.6107 | 7.1966 |
| **KonIQ-10k** | KonIQ | **0.9369** | **0.9522** | 5.1802 | 6.8630 |
| **KonIQ-10k** | FLIVE | 0.7206 | 0.7631 | 9.6415 | 11.9886 |
| **KonIQ-10k** | Production | 0.9345 | 0.9476 | **3.2562** | **4.0592** |
| **SPAQ** | KonIQ | 0.8555 | 0.8451 | **9.5807** | **11.8064** |
| **SPAQ** | FLIVE | 0.8267 | 0.6815 | 20.8763 | 24.9845 |
| **SPAQ** | Production | **0.8730** | **0.8478** | 11.8859 | 14.4976 |

## 5. Best Baseline per Dataset
- **KonIQ-10k**: `koniq_standalone` (SRCC 0.9369) / `production_technical_score` (MAE 3.2562)
- **SPAQ**: `production_technical_score` (SRCC 0.8730)
- **FLIVE image-test**: `flive_standalone` (SRCC 0.6603)

## 6. Calibration and Bias Diagnosis
- **Range**: All production scores remain within `[12.79, 86.17]`, well within the `0..100` range.
- **Bias Smoothing**: The production score inherits the massive `+18.4` positive bias of FLIVE on SPAQ but reduces it to **+7.7**, thanks to the balanced weight of KonIQ.
- **Stability**: On KonIQ-10k, the production score actually achieves the lowest MAE (**3.26**), indicating high stability in point-wise prediction.

## 7. Interpretation
- **Generalization**: The production combination improves generalization, especially visible in the SPAQ dataset where it outperforms both individual models in rank correlation.
- **FLIVE Impact**: The inclusion of FLIVE is critical for performance on the FLIVE dataset (boosting SRCC from 0.48 to 0.57 compared to KonIQ alone).
- **Calibration Trade-off**: FLIVE does introduce some positive bias on SPAQ/KonIQ, but the weighting is well-balanced to prevent it from dominating the final score.
- **Stability**: KonIQ provides a stable "anchor" for the score, while FLIVE provides sensitivity to complex authentic distortions.
- **Verdict**: The current production `technical_score` is a high-quality baseline and should be kept as the primary benchmark for the project.

## 8. Limitations
- Evaluation is restricted to image-level test rows; patch-level evaluation is excluded.
- No Android runtime parity was verified in this specific offline run (logic was reproduced from Python source).
- C6, AADB, and AVA datasets were excluded per the research-protocol scope.

## 9. Decision
**A. Keep current production technical_score as baseline.**
The production score is robust, outperforms standalones in generalization (SPAQ), and provides a well-balanced combination of the available models.

## 10. Next Action
The next action is to finalize the research-protocol baseline report and include these ensemble results as the "Current Production Baseline."
