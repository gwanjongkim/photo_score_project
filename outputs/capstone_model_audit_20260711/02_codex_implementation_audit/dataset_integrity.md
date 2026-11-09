# Dataset integrity

## Independently executed checks

`run_bounded_checks.py` inspected the active core manifests. Full results are in `bounded_checks.json`.

| Dataset | Train/val/test rows | Exact path overlap | Missing files | Label range | Verdict |
|---|---:|---:|---:|---|---|
| KonIQ | 8058 / 1007 / 1008 | 0 / 0 / 0 | 0 | 3.91–88.39 | Exact-path split integrity MATCH; content/near duplicates UNVERIFIED |
| FLIVE image | 31848 / 3981 / 3981 | 0 / 0 / 0 | 0 | 0–94.33 | Exact-path split integrity MATCH; official FLIVE partition MISMATCH |
| AADB | 7612 / 846 / 1000 | 0 / 0 / 0 | 0 | 0–1 | Official test preserved; train/val totals differ from paper 8500/500 because missing images/filtering and 10% internal split |

## Split-generation evidence

- KonIQ `make_koniq_csv.py:31-36` assigns `mos_col` repeatedly for any column containing `mos`; if both `MOS` and `MOS_zscore` exist, the last column wins. Current manifest values near 0–100 show ordinary MOS was used in the generated files, but the builder is order-sensitive and not reproducible by schema name. `:58-63` creates random 80/10/10 with seed 42, not KonCept512’s official split.
- FLIVE `make_flive_csv.py:40-46` randomly creates 80/10/10 with seed 42 separately for images/patches. This is not the official PaQ-2-PiQ split and weakens paper-comparable claims.
- AADB `make_aadb_csv.py:53-69` reads official MAT train/test lists and carves 10% of train as validation with seed 42. The test list is structurally preserved. The observed 7612+846 indicates 42 official-train entries were absent or filtered from the expected 8500.

## Unverified integrity risks

Content-hash duplicate detection, perceptual near-duplicate detection, label conflicts across renamed files, and cross-manifest reuse in later teacher/student datasets were not executed in this bounded pass. They must not be reported clean. Student datasets especially require grouping by source image before splitting; inventory evidence says the 50k line uses grouping, but raw target provenance and test reuse remain unresolved.

The technical prediction table inspected here has 4,967 rows but no ground-truth column. It cannot establish dataset integrity or accuracy independently.
