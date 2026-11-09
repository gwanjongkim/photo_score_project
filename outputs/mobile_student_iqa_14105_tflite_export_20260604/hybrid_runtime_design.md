# Hybrid Runtime Design Note

## Scope
- Phase 12 is export/parity/profiling preparation only.
- Student TFLite is a log-only mobile profiling candidate.
- This is not a production replacement for `technical_score`.

## Evidence
- Phase 11 best checkpoint is strong overall and on KonIQ/SPAQ.
- FLIVE remains weak: test MAE/SRCC/PLCC 8.9626/0.5176/0.5168.
- Severe FP was 0, but FLIVE ranking/absolute alignment is not production-safe alone.

## Candidate Runtime Plan
- Keep current production/FLIVE robustness signal active.
- Add student TFLite only behind telemetry/log-only switch.
- Log student score, production technical score, image metadata, model version, runtime ms, and disagreement bands.
- Review disagreement cases before any hybrid weighting.

## Non-goals
- Do not alter Flutter production scoring in Phase 12.
- Do not replace `technical_score` with student model.
- Do not ship final model assets from this phase.
