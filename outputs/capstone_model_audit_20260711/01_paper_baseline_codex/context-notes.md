# Phase 2 context notes

## Scope decisions

- This phase defines a normative paper baseline before local fidelity evaluation.
- The Phase 1 inventory is used only to identify names, datasets, and relevant research families.
- Local source code, checkpoints, metrics, and the interrupted Claude output are not evidence for paper requirements.
- No final fidelity score will be assigned.
- Paper, supplement, official code, dataset documentation, and pretrained-model documentation are kept distinct.
- A source is marked verified only after it is opened in this phase.
- Source disagreements are preserved rather than silently resolved.
- New files are written only under `outputs/capstone_model_audit_20260711/01_paper_baseline_codex/`.
- Inventory names were frozen as NIMA, RGNet, A-LAMP, ICAA/DAT, MUSIQ, KonIQ/KonCept512, FLIVE/PaQ-2-PiQ, MDIQA, TOPIQ, and custom Stage 5/guard/calibration labels.
- FLIVE is a dataset associated with PaQ-2-PiQ; “KonIQ Mobile” is an adaptation label rather than the paper-defined KonCept512 architecture.
- Missing fields are marked NR or UNVERIFIED rather than populated from local code.
- The incomplete Claude output directory was not read.
