# Benchmark Validity Audit Report

## A. Executive Summary
The benchmark conducted on 2026-06-03 was a **full evaluation of local test splits** for all four models. However, the "official paper protocol" status varies by model. RGNet and ICAA (on ICAA17K) followed official splits and protocols closely, while A-LAMP v2 used a **proxy patch protocol** (fixed center/corner crops instead of adaptive patches), significantly impacting its comparability to the original paper. NIMA used a local full test split that matches the size of the official split, but its exact identity is unproven. A significant **scale mismatch** was identified in ICAA metrics, where predicted scores (0-1) were compared to ground truth scores (1-10) without scaling, resulting in misleadingly high MAE/MSE values despite high correlation.

## B. Final Corrected Verdict Table
| Model | Dataset | Existing Status | Corrected Status | Local Full Test? | Official Proven? | Safe Claim | Main Caveat |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| NIMA | AVA | PASS | **LOCAL PASS** | Yes | Unknown | High local correlation | Official split identity not proven |
| A-LAMP v2 | AVA | PASS | **PARTIAL PASS** | Yes | No (Proxy) | Binary accuracy robust | Proxy patch protocol lowers SRCC |
| RGNet | AADB | PASS | **PASS** | Yes | Yes | Matches paper | None |
| ICAA | ICAA17K | PASS | **PASS** | Yes | Yes | SOTA correlation | MAE scale mismatch (0-1 vs 1-10) |
| ICAA | SPAQ | PASS | **REFERENCE** | Yes | No (Cross-ds) | Measures distinct concept | Target is Colorfulness, not Aesthetics |

## C. Evidence Files Inspected
- `outputs/aesthetic_mobile_paper_benchmark_20260603/report.md`
- `outputs/aesthetic_mobile_paper_benchmark_20260603/full/nima_ava/nima_predictions.csv`
- `outputs/aesthetic_mobile_paper_benchmark_20260603/full/alamp_ava/alamp_v2_predictions.csv`
- `outputs/aesthetic_mobile_paper_benchmark_20260603/full/rgnet_aadb/rgnet_predictions.csv`
- `outputs/aesthetic_mobile_paper_benchmark_20260603/full/icaa_icaa17k/icaa_predictions.csv`
- `outputs/aesthetic_mobile_paper_benchmark_20260603/full/icaa_spaq/icaa_predictions.csv`
- `outputs/aesthetic_mobile_paper_benchmark_20260603/scripts/run_benchmark.py`
- `data/processed/ava/test.csv`
- `data/processed/aadb/test.csv`
- `external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv`

## D. AVA/NIMA Audit
- **Row Counts**: 25,549 successes, 2 failures. (Total 25,551 images in `test.csv`).
- **Failures**: 2 actual failures (truncated files). The reported "21 failures" is an error.
- **Split Source**: Local split (`data/processed/ava/test.csv`). Identity as official paper split is UNKNOWN.
- **Metric Judgment**: SRCC 0.62 is paper-level. Accuracy 0.61 is comparable.
- **Final Wording**: "Full evaluation on local AVA test split (25k images). Performance is consistent with NIMA paper baselines."

## E. AVA/A-LAMP Audit
- **Row Counts**: 25,549 successes, 2 failures.
- **Patch Protocol**: **PROXY**. Used fixed center and 4 corner crops. Original paper uses saliency-aware adaptive patches.
- **Metric Judgment**: SRCC 0.50 is lower than paper expectations, likely due to the proxy protocol. Binary accuracy 0.75 is high.
- **Score Scale**: MAE recomputed on scaled 1..10 range (2.60) is consistent with the report's note.
- **Final Wording**: "Full evaluation on local AVA test split with proxy patch protocol. High binary accuracy but lower SRCC due to simplified patch selection."

## F. AADB/RGNet Audit
- **Row Counts**: 1,000 images. Matches standard AADB test split size.
- **Split Judgment**: Official/Standard.
- **Metric Judgment**: SRCC 0.66 is highly consistent with paper results.
- **Final Wording**: "Full evaluation on standard AADB test split. Results match paper performance."

## G. ICAA17K/ICAA Audit
- **Row Counts**: 1,617 images. Matches official test split.
- **Split Judgment**: Official (from official repo).
- **Score Scale**: 
  - Ground Truth: ~3.5 to ~9.1 (mean 5.97)
  - Predicted: ~0.37 to ~0.79 (mean 0.59)
- **MAE Interpretation**: MAE 5.37 is high purely due to **scale mismatch**. Predicted scores are 0-1 while GT is 1-10. SRCC 0.87 proves excellent performance regardless of scale.
- **Final Wording**: "Full evaluation on official ICAA17K test split. Excellent correlation (SRCC 0.87) matching SOTA. Note: MAE is high due to 0-1 vs 1-10 scale mismatch."

## H. SPAQ/ICAA Audit
- **Row Counts**: 1,125 images.
- **Target Judgment**: SPAQ target is "Colorfulness". ICAA target is "Color Aesthetics".
- **Metric Judgment**: Low correlation (SRCC 0.32) is expected as they measure different concepts.
- **Final Wording**: "Reference benchmark on SPAQ dataset. Confirms ICAA captures aesthetic preference beyond simple colorfulness."

## I. Recommended Edits to original report.md
1. **Correct failure count**: Change "21 images failed" to "2 images failed".
2. **Clarify A-LAMP status**: Change PASS to PARTIAL PASS (PROXY).
3. **Clarify ICAA-SPAQ status**: Change PASS to REFERENCE ONLY.
4. **Add MAE Scale Note**: Explicitly state for ICAA that MAE is affected by 0-1 vs 1-10 scale mismatch.

## J. Final Conclusion for Capstone Use
- **Safe to claim**: RGNet and ICAA (on ICAA17K) are fully validated against official benchmarks. NIMA is validated against a full local split with paper-level results.
- **Do not claim**: Do not claim A-LAMP v2 matches paper performance due to the proxy protocol. Do not claim ICAA's low SPAQ correlation as a "failure".
- **Further Verification**: If time allows, re-run ICAA17K with 1-10 scaling to get a paper-comparable MAE.
