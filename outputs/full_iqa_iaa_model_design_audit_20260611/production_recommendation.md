# Production Recommendation — A-CUT IQA/IAA Model Design

## Final verdict
`FULL_MODEL_DESIGN_AUDIT_SUPPORTS_CURRENT_KONIQ_FLIVE_AND_NIMA_RGNET_ALAMP_PRODUCTION`

## Technical IQA
- **Keep KonIQ+FLIVE as the production technical_score.**
  - SRCC 0.7168, PLCC 0.7874, MAE 12.0, **0 severe false positives** on the 4,967-image official test split.
  - KonIQ is the strongest hard-FP guard; FLIVE adds diversity but is only safe inside the ensemble.
  - No replacement is justified: the only higher-SRCC options are blocked by safety (TOPIQ+FLIVE 0.8069, 3 severe FP) or runtime (MDIQA performance-ceiling reference / strong reference teacher 0.8227, transformer, not mobile).

## Aesthetic IAA
- **Keep the submitted Flutter app ensemble: NIMA + RGNet + Mobile A-LAMP v2.**
  - Submitted production defaults: **NIMA 0.40 / RGNet 0.45 / A-LAMP 0.15**.
  - (Note: Offline search recommendation was NIMA 0.37 / RGNet 0.41 / A-LAMP 0.22).
  - Held-out combined SRCC 0.6596; 5-fold std 0.0022 (stable). The three models are complementary across AVA and AADB.
  - Total aesthetic latency ≈ 528 ms, within budget.

## Research / reference only (do NOT integrate into Flutter production)
- MDIQA performance-ceiling reference / strong reference teacher — supervision only (not mobile).
- MDIQA student — No-Go, archived; absent from Mac assets.
- ICAA — separate color/research subscore; weight 0.0 in fusion; absent from Mac assets.
- TOPIQ / TOPIQ+FLIVE — log-only shadow candidate (present on Mac assets but not in production ensemble).
- TechIQA-Guard v1 — internal reference (weakened safety guard).

## Explicit do-not-do
- Do not deploy or TFLite-export the MDIQA student.
- Do not add ICAA to the weighted production ensemble.
- Do not promote TOPIQ+FLIVE beyond log-only without on-device runtime + severe-FP gating evidence.
- Do not collapse technical and aesthetic into a single model.

## Decision basis (one line each)
- Safety first: severe-FP behaviour, not raw SRCC, governs the technical choice.
- Runtime first: per-photo latency × model count governs the aesthetic choice; ICAA fails it.
- Evidence first: replacement requires both official benchmark AND on-device runtime/safety proof.
