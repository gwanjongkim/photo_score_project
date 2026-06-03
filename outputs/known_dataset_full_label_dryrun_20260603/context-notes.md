# Phase 6 known-dataset full-label dry-run context notes

- Scope source is `outputs/hybrid_teacher_final_label_scope_decision_20260603/report.md`.
- Main labels include only KonIQ-10k, SPAQ, and FLIVE image rows with cached production scores.
- Unknown/fallback rows are excluded from the main label pool.
- `c6_parity` real-mobile rows are holdout only.
- `Test_weight` / product-gold rows are aesthetic/product validation only.
- `policy_f_shift_cap3` means KonIQ/SPAQ MDIQA targets receive a capped source mean shift; FLIVE production targets stay unchanged.
- Cached MDIQA file currently has 8999 unique KonIQ image paths and 1125 unique SPAQ image paths. This is below the expected possible base scope of 10073 and 11125.
- Because no new inference/full label generation is allowed, the dry-run should generate all valid cached rows and mark coverage as needing fix if full expected scope is unavailable.
- Script run result: 14105 valid rows total: FLIVE 3981, KonIQ 8999, SPAQ 1125.
- Safety result: severe FP 0, FP 0, FN 0.
- Scope result: no duplicate image_path, no unknown/fallback rows, FLIVE content_type image only.
- Range result: mean shift 12.2377, std ratio 2.0418, p95/p05 gap mismatch 14.9453, clipping 0.
- Final decision: KNOWN_DATASET_FULL_LABEL_NEEDS_COVERAGE_FIX.
