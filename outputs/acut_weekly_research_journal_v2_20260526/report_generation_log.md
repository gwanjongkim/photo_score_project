# Report Generation Log (Research Journal v2)

## Search Strategy
- **Method**: Broad searching using `find` and `grep` across the entire repository.
- **Tools**: `xargs grep` for efficient large-scale text searching.
- **Topics**: Gemma, VILA, Firebase, Aesthetic (NIMA, RGNet, A-LAMP), Technical (TOPIQ, FLIVE), Mobile Deployment.

## Data Extraction
- **Dates**: Priority given to filenames (YYYYMMDD), then file content (YYYY-MM-DD), then modification time as a fallback.
- **Metrics**: Extracted directly from `report.md`, `handoff.md`, `top_k_summary.json`, and `summary.md` files.
- **Verification**: Cross-referenced `top_k_summary.json` with `run_acut_pipeline.py` to understand model ensemble composition.

## File Statistics
- **Potential Evidence Files**: 3365 identified.
- **Relevant Files found**:
  - Aesthetic: 1657
  - Deployment: 783
  - Firebase: 223
  - Gemma: 204
  - Technical: 690
  - VLM/Reasoning: 992
- **Evidence Indexed**: 2532 files.

## Self-check results
- **Did we search for Gemma/LiteRT?**: Yes, found 282-sample adapter and latency harness.
- **Did we search for Firebase?**: Yes, found cloud worker architecture and emulator smoke tests.
- **Did we search for VILA?**: Yes, found VILA-lite and NVILA-8B reasoning evidence.
- **Did we search beyond outputs/experiments?**: Yes, searched `docs`, `src`, `scripts`, `configs`.
- **Did each weekly file include evidence paths?**: Yes, included in per-week markdown files.
- **Did we avoid unsupported “완벽 재현” claims?**: Yes, used terms like "confirmed fact" and "paper-oriented approximation".
- **Did we mark uncertain dates and missing evidence?**: Yes, used "Unknown" and "reliability" flags in CSV.

## Known Missing Areas
- **Stage 5 full training logs**: Some epoch-level logs were not found; summary metrics used instead.
- **Firestore production deployment**: Only emulator and local worker smoke tests confirmed.
