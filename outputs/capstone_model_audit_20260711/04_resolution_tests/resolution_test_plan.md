# Resolution Test Plan

## Objective

Adjudicate every factual disagreement between the primary implementation audit in `02_codex_implementation_audit/` and the independent cross-check in `03_independent_crosscheck/`. The work will use existing repository artifacts and the minimum non-destructive tests needed to resolve disputed claims. It will not retrain models, alter production source files, or overwrite prior audit evidence.

## Assumptions and scope decisions

- The continuation request named `01_paper_baseline/` and `01_paper_baseline_claude_crosscheck/`, but both paths are absent. The existing `01_paper_baseline_codex/` directory is treated as the paper baseline, and `03_independent_crosscheck/` is the available independent review.
- “Every disagreement” includes explicit disputes, score/classification differences, and material factual challenges that would change a model verdict or presentation-safe claim.
- Resolution tests may read production code, datasets, checkpoints, prediction tables, and exports, but all new scripts and outputs must remain in `04_resolution_tests/`.
- Existing reports are evidence pointers, not authoritative conclusions. Final scores will be assigned independently after adjudication.
- If a claim cannot be resolved with existing evidence and a bounded test, it will remain `UNRESOLVED` or `PARTIALLY_RESOLVED`; missing evidence will not be inferred.

## Success criteria

1. Build a complete, uniquely identified dispute register covering the primary audit and independent cross-check.
2. For each dispute, record the exact claim, prior evidence, feasibility of a bounded test, command/output or reason not tested, and one required resolution status.
3. Run bounded checks for the prioritized technical areas wherever the required artifact exists.
4. Produce all requested files under `04_resolution_tests/` and `05_final_report/`.
5. Give every in-scope model four independently reasoned scores, a classification, and a confidence level.
6. Separate proven facts, incomplete evidence, interpretations, unsupported assumptions, and unknowns.
7. Validate report completeness, tabular schemas, referenced paths, and hashes of final evidence files.

## Execution plan

1. Inventory evidence and models → verify all four evidence sources and enumerate report claims.
2. Extract disagreements → verify every cross-check dispute maps to a primary-audit claim or score.
3. Design minimal tests → verify each test is non-destructive, bounded, and writes only here.
4. Execute and capture evidence → verify commands, exit status, stdout/stderr, and failed attempts are recorded.
5. Adjudicate model verdicts → verify scores are evidence-backed rather than averaged.
6. Write final reports → verify all requested sections and presentation-claim fields exist.
7. Run final consistency checks → verify manifests, hashes, cross-references, and no production files changed.

## Constraints

- No full retraining.
- No production source changes.
- No mutation of prior audit directories.
- No suppression of failed tests.
- No claim promoted beyond its strongest reproducible evidence.
