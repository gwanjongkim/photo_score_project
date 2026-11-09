# Aesthetic Benchmark CSV Audit Report

## Executive Summary
Audit of AVA and AADB benchmark CSVs for ensemble weight search.
VERDICT: **CSV_RELIABLE_WITH_CAVEATS**

## Findings
- File existence: Verified.
- Row counts: Match expected splits.
- Join keys: No duplicates, subset property holds.
- GT labels: Consistent scaling (AADB converted 0..1 to 1..10).
- Distributions: Predictions within expected ranges, no NaNs.
- Metric reproduction: Matches previous reports within tolerance.

## Key Discrepancy
The benchmark is technically reliable, but the weight selection is driven by SRCC on static datasets. The user intuition mismatch is likely due to model behavior differences on real-world photos vs benchmark photos, not CSV corruption.

## Outputs
- `file_inventory.csv`
- `row_count_summary.csv`
- `join_key_audit.csv`
- `score_distribution_summary.csv`
- `metric_reproduction_summary.csv`
- `leakage_audit.md`
- `recommended_next_action.md`
