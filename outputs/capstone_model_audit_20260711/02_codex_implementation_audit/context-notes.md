# Implementation audit context notes

- Existing reports are leads only, never authoritative evidence.
- No production source edits and no full retraining are permitted.
- Executable tests must be bounded and deterministic.
- Positive statuses require direct code, artifact, manifest, hash, or runtime evidence.
- Failed and unexecuted tests remain visible.
- Requested paper baseline path was absent; available `01_paper_baseline_codex/` was used and disclosed.
- Bounded TFLite tests used seed 20260711 and validate interfaces, not model accuracy.
- Core manifest checks covered exact path duplicates and missing files only; content and near duplicates remain unresolved.
