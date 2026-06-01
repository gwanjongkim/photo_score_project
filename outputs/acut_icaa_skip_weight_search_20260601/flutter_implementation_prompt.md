# Flutter Implementation Prompt

You are working in the Flutter project. Do not change A-cut technical scoring, KonIQ/FLIVE weights, model assets, or production metadata unrelated to the ICAA skip experiment.

Goal:
- Make ICAA skipped by default for A-cut aesthetic scoring if the target branch has the ICAA path.
- Keep ICAA available behind a debug or override flag if possible.
- Set the non-ICAA aesthetic weights to NIMA=0.200, RGNet=0.510, A-LAMP=0.290.
- Preserve all technical scoring behavior.
- Run `flutter analyze`.
- Document the exact files changed and the before/after scoring contract.

Context:
- WSL benchmark output: `/home/omen_pc1/photo_score_project/outputs/acut_icaa_skip_weight_search_20260601/report.md`.
- Benchmark reference assumed NIMA 0.10, RGNet 0.40, A-LAMP 0.30, ICAA 0.20 with ICAA `color` output.
- Accessible Flutter checkout inspected from WSL did not contain the ICAA active ensemble, so first verify the actual target branch implements `POZY_ACUT_SKIP_ICAA_EXPERIMENT` or equivalent before editing.

Strict boundaries:
- Do not modify technical score formulas.
- Do not reintroduce TOPIQ/C6.
- Do not copy or replace model files unless explicitly requested.
- Do not claim production readiness without Flutter-side validation and ranking checks.
