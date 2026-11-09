# Phase 18 Context Notes

- Scope: weighted B0 smoke only. No B1/B2, no TFLite export, no Flutter edits, no production `technical_score`, no commit.
- Phase 17 recommended W5-lite weighting: FLIVE 0.30, KonIQ-10k 0.35, SPAQ 0.35.
- Current smoke trainer has no sample-weight output from `make_dataset`; controlled runner imports smoke utilities.
- Implementation choice: add optional training-only sample weights to smoke `make_dataset`, pass weights only for training dataset in controlled runner, leave validation/test/official prediction unweighted.
- First sandboxed `.venv_tf` run failed before training with TensorFlow GPU count `0` and `failed call to cuInit: UNKNOWN ERROR (100)`.
- Escalated `.venv_tf` run saw one TensorFlow GPU and completed weighted smoke in `239.80` seconds.
- W5-lite smoke was numerically healthy but official Track B Overall SRCC was `0.530639`, far below unweighted controlled B0 `0.703290` and 14k student `0.743600`.
- Final decision: `WEIGHTED_B0_SMOKE_NEEDS_WEIGHT_TUNING`; keep B1/B2 paused and do not run controlled weighted rerun yet.
- Handoff: `/tmp/phase18_weighted_b0_smoke_handoff_20260609.md`.
- Architecture note: `/tmp/architecture-review-20260609-phase18.html`.
