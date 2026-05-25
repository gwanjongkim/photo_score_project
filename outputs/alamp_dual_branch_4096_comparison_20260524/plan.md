# Plan - A-LAMP 4096 Fair Comparison

## Goal
Compare Multi-Patch-only and Dual-Branch GCN on the exact same matched 4096 graph subset with threshold sweeps and balanced metrics.

## Scope
- Do not train, export, modify architecture, touch Flutter, generate full AVA graphs, or claim full A-LAMP reproduction.
- Create a filtered Multi-Patch test JSONL using the same filename-stem image IDs matched by the Dual-Branch GCN test evaluation.
- Evaluate both models into this comparison output directory.
- Add a small reusable threshold analysis script if needed.

## Success Criteria
- Matched subset JSONL has the expected 4081 rows.
- Multi-Patch-only and Dual-Branch predictions cover the same 4081 labels in the same subset.
- Threshold sweep CSVs, `summary.json`, and `report.md` are written.
- Final judgment emphasizes ROC-AUC, balanced accuracy, specificity, and positive-bias behavior.
