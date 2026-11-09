# Full IQA/IAA Model Design Audit and Evaluation

_Audit date: 2026-06-13. Scope: read-only design audit of the A-CUT / Pozy on-device
best-shot selection system, covering both the technical IQA stack and the aesthetic
IAA stack. No training, no new experiments, no code/Flutter/production changes, no
TFLite export, no commits. All metrics below are quoted from existing local reports;
nothing is fabricated. Where a number could not be confirmed in this checkout it is
explicitly flagged._

## 1. Executive Summary

- **The overall model-design strategy was sound.** Separating technical image
  quality (IQA) from aesthetic image quality (IAA), building each from validated
  mobile models, and combining them by late fusion is a reasonable and defensible architecture for
  an on-device best-shot selector under mobile constraints.
- **Technical IQA production recommendation: keep KonIQ+FLIVE.** It scores
  SRCC 0.7168 with **0 severe false positives** and MAE 12.0 on the 4,967-image
  official test split — the safest behaviour of any candidate on hard low-quality
  images, which is the property that matters most for best-shot ranking.
- **Aesthetic IAA production recommendation: keep NIMA + RGNet + Mobile A-LAMP v2.**
  The submitted Flutter app uses NIMA 0.40 / RGNet 0.45 / A-LAMP 0.15 as the default aesthetic weights. 
  The offline weight-search recommendation was NIMA 0.37 / RGNet 0.41 / A-LAMP 0.22, but the 
  submitted app should be reported using its actual Flutter defaults. The ensemble remains 
  stable and genuinely complementary across AVA and AADB.
- **MDIQA mobile student is a confirmed No-Go (research/reference only).** The
  teacher is a strong reference teacher (SRCC 0.8227, 0 severe FP) but the best 50k B0 student reached
  only 0.7033 — below both the prior 14k student (0.7436) and production (0.7168) —
  and every corrective attempt (weighting, ranking loss, label scaling) regressed
  to 0.48–0.54. `MDIQA_STUDENT_FINAL_NO_GO_RESEARCH_REFERENCE_ONLY` is justified.
- **ICAA exclusion from the production weighted ensemble is justified.** ICAA is an
  excellent color-aesthetics reference (ICAA17K SRCC 0.8791) but adds essentially
  nothing to the rank ensemble (Group B with ICAA was −0.0033 combined SRCC vs
  Group A) and is ~1.1 s/image on device — roughly the cost of the entire rest of
  the aesthetic ensemble combined. Keep it as a separate research/color subscore.
- **One honest caveat for the technical stack:** a benchmark-superior alternative
  exists. TOPIQ+FLIVE (60/40) reaches SRCC 0.8069 and MDIQA teacher 0.8227, both
  above production's 0.7168. They are correctly held at **log-only / teacher-only**
  status because TOPIQ+FLIVE introduces 3 severe FPs (vs production's 0) and MDIQA
  is not mobile-runnable. Production is the *safest*, not the *highest-benchmark*,
  technical model — this trade-off is defensible but must be stated explicitly in
  the thesis.
- **The current app model design is defensible for a capstone.** It is a
  safety-first, mobile-feasible, explainable ensemble whose every inclusion and
  exclusion is backed by an official-protocol benchmark and a documented decision.
- **No model should be added or removed before capstone finalization.** The
  production set is correct as-is. MDIQA student, ICAA-in-ensemble, TOPIQ+FLIVE, and
  TechIQA-Guard v1 should all remain outside the Flutter production path.

**Final audit verdict:**
`FULL_MODEL_DESIGN_AUDIT_SUPPORTS_CURRENT_KONIQ_FLIVE_AND_NIMA_RGNET_ALAMP_PRODUCTION`

## 2. Original System Goal

A-CUT / Pozy performs **on-device best-shot selection**: given a burst or album of
user photos, rank them and surface the best one. The product question is not "can we
match an academic SRCC leaderboard" but:

> "Can the app reliably rank a user's photos on-device using a practical combination
> of technical quality and aesthetic quality models?"

Two distinct judgments are required and they are not interchangeable:

- **Technical IQA** — is the image *well captured*? (blur, noise, exposure,
  compression). A sharp, clean photo of a boring subject scores high here.
- **Aesthetic IAA** — is the image *well composed / pleasing*? (composition,
  color harmony, subject interest). A beautiful but slightly soft photo scores
  high here.

Multiple models are used per axis because no single mobile model is simultaneously
accurate, fast, stable, and well-calibrated across the heterogeneous datasets that
define each axis (KonIQ/SPAQ/FLIVE for IQA; AVA/AADB/ICAA17K for IAA). An ensemble
buys cross-dataset robustness and reduces the chance that one model's blind spot
becomes a user-visible ranking error.

**Why benchmark score alone is insufficient:** on mobile the binding constraints are
runtime (per-image latency × number of models × number of photos), stability (no
collapse, no NaNs, low variance under small input changes), **severe false-positive
risk** (never rank a technically broken photo as "best"), explainability (the debug
UI must show per-model sub-scores so a bad rank can be diagnosed), and production
safety (don't replace a validated stack without runtime *and* benchmark evidence). A
model that wins SRCC by 0.05 but adds severe FPs, triples latency, or cannot be
quantized to TFLite is a net loss for this product.

## 3. Evidence Sources Inspected

### Technical IQA
- `outputs/official_protocol_technical_signal_benchmark_20260604/report.md` and
  `official_signal_metrics_overall.csv`, `official_signal_recommendation.json`
- `outputs/official_protocol_mdiqa_signal_benchmark_20260604/report.md` and
  `official_mdiqa_metrics_overall.csv`
- `outputs/mdiqa_student_final_no_go_report_20260611/` (report.md, summary_table.csv,
  decision_matrix.csv, research_takeaways.md, next_actions.csv)
- Student phase reports (present, aggregated by the no-go report):
  `mobile_student_iqa_50k_controlled_b0_20260608`,
  `..._weighted_b0_smoke_w5v2_ifreq_20260609`,
  `..._ranking_b0_smoke_r2_lambda010_margin005_20260609`,
  `..._label_scaled_b0_smoke_s2_koniq_x2_20260609`,
  `mdiqa_student_pre_final_audit_20260609`,
  `mobile_student_iqa_14105_checkpoint_eval_20260604`
- `outputs/final_research_protocol_technical_iqa_baseline_report_20260530/final_decision.md`
- `outputs/final_c6_topiq_koniq_technical_score_decision_20260527/`
- `outputs/final_techiqa_guard_v1_nogo_20260522/report.md`

### Aesthetic IAA
- `outputs/aesthetic_mobile_paper_benchmark_20260603/report.md`
- `outputs/aesthetic_ensemble_weight_search_complete_20260603/report.md` and
  `final_recommendation.json`
- `outputs/aesthetic_ensemble_production_precheck_20260603/production_precheck_report.md`
- `outputs/acut_icaa_skip_weight_search_20260601/report.md`
- `outputs/icaa_int8_qat_audit_20260601/report.md`,
  `outputs/icaa_ptq_candidates_20260601/`

### Ensemble / Flutter
- `outputs/aesthetic_ensemble_weight_search_complete_20260603/final_recommendation.json`
  (calibrated weights, ICAA decision)
- **Mac Asset Verification**: Flutter project assets on Mac (`pozy_app/assets/models/`) 
  were verified to contain the production KonIQ/FLIVE and NIMA/RGNet/A-LAMP models. 
  TOPIQ is present as a shadow asset. ICAA and MDIQA student models are absent, 
  consistent with production decisions.
- **Submitted app parity checks**: `pubspec.yaml` declares `assets/models/`, the TOPIQ shadow flag default was verified as false, and ICAA/MDIQA student are not part of the active production evaluation contract.

### Mobile runtime
- `outputs/alamp_gpu_acceleration_audit_20260603/report.md` (A-LAMP CPU/XNNPACK,
  GPU delegate unavailable, expensive adaptive-patch preprocessing)
- `outputs/icaa_int8_qat_audit_20260601/report.md` (no INT8/QAT model exists yet;
  PTQ feasible but parity-risky)
- Representative per-image device timings (NIMA ~177 ms, RGNet ~302 ms,
  A-LAMP ~49 ms, ICAA ~1.1 s, KonIQ/FLIVE ~25 ms each) are taken from
  **representative Android/project logs**. Note that a single consolidated timing 
  report was not found; end-to-end app latency can include image loading and 
  preprocessing (e.g., A-LAMP's adaptive patch preprocessing is a separate CPU cost).

### Current Flutter App Asset Snapshot
A manual audit of the Mac-side Flutter project assets (`/Users/gwanjung_mac/StudioProjects/pozy_app/assets/models`) confirmed the following:
- **Production Technical IQA assets present**: `koniq_mobile.tflite`, `flive_image_mobile.tflite` (and respective metadata).
- **Production Aesthetic IAA assets present**: `nima_mobile_fixed_preproc_fp16.tflite`, `rgnet_pil_resize_aadb_fp16.tflite`, `mobile_alamp_v2_fp16.tflite` (and respective metadata).
- **TOPIQ assets present**: `topiq_lite_mixed112_frozen_fp16.tflite` exists in the folder but is classified as a **shadow/log-only** candidate, as it is not part of the active production ensemble contract.
- **ICAA asset absent**: No ICAA TFLite model is bundled in the current production asset folder.
- **MDIQA student absent**: No MDIQA student TFLite model is bundled, supporting the `NO_GO` decision.
- **Other assets present**: Non-ensemble models like `fastscnn_cityscapes_float16.tflite`, `lighting_model.tflite`, `yolo11n.tflite`, and `yolov8n-pose_float16.tflite` are present but do not impact the A-CUT IQA/IAA production ensemble.

## 4. Technical IQA Model Assessment

Official 4,967-image test split (KonIQ-10k 2,015 + SPAQ 1,125 + FLIVE 1,827).

| Signal | SRCC | PLCC | MAE | Severe FP |
|---|---|---|---|---|
| production KonIQ+FLIVE (`production_existing`) | 0.7168 | 0.7874 | 12.00 | **0** |
| KonIQ alone | 0.6901 | 0.7582 | 19.68 | 0 |
| FLIVE alone | 0.6543 | 0.6801 | 11.03 | 189 |
| 14k student | 0.7436 | 0.7782 | 9.39 | 3 |
| TOPIQ alone | 0.7975 | 0.8717 | 7.67 | 2 |
| TOPIQ+FLIVE 60/40 | 0.8069 | 0.8694 | 7.62 | 3 |
| MDIQA teacher (`mdiqa_selected`) | 0.8227 | 0.8222 | 8.41 | **0** |
| 50k B0 student (best) | 0.7033 | — | — | — |

**KonIQ** — *Role:* in-the-wild authentic-distortion expert; the strong "guard"
component (lowest score on hard FPs, 47.84 mean in the TechIQA hard-FP study).
*Evidence:* alone 0.6901 SRCC, 0 severe FP. *Risk:* high MAE (19.68) / pessimistic
scale alone; narrow output range. *Recommendation:* **keep as production component.**

**FLIVE** — *Role:* large-scale crowd-sourced diversity / robustness component.
*Evidence:* alone 0.6543 SRCC but **189 severe FPs** and a large positive bias —
unsafe standalone. *Risk:* over-optimistic on low-quality images. *Recommendation:*
**keep only inside the ensemble**, where KonIQ corrects its bias.

**KonIQ+FLIVE (production)** — *Role:* the production technical_score. *Evidence:*
0.7168 SRCC, **0 severe FP**, well-calibrated (final-decision report: SPAQ 0.873,
KonIQ rank retention 99.7%, FLIVE bias reduced +18.4 → +7.7). *Risk:* lowest SRCC of
the high-end candidates; gives up benchmark points for safety. *Recommendation:*
**keep as production.** It is the only stack combining acceptable accuracy with zero
severe FPs and 25 ms/model latency.

**MDIQA teacher** — *Role:* performance-ceiling reference / strong reference teacher.
*Evidence:* best SRCC 0.8227 with 0 severe FP; KonIQ 0.9311, SPAQ 0.9136 datasetwise.
*Risk:* PyTorch transformer backbone — not real-time mobile; heavy. *Recommendation:*
**research/reference only (teacher-only).** Do not deploy.

**MDIQA student (50k B0 line)** — *Role:* attempted mobile replacement.
*Evidence:* best 0.7033, below 14k (0.7436) and production (0.7168); corrective
variants 0.5163 / 0.4803 / 0.5404. Pipeline, data, and architecture audits all
passed, so the gap is intrinsic to the distillation, not a bug. *Risk:* would lower
production accuracy and add severe FPs (14k student already had 3). *Recommendation:*
**No-Go, research/reference only.** Do not convert to TFLite or integrate.

**TOPIQ / TOPIQ+FLIVE** — *Role:* benchmark-superior research candidate.
*Evidence:* TOPIQ+FLIVE 60/40 = 0.8069 SRCC (+0.090 over production) but **3 severe
FPs** vs production's 0; decision file: `production_replacement: false`,
`TOPIQ_FLIVE_REMAINS_BEST_LOG_ONLY_CANDIDATE`. *Risk:* severe FPs are exactly the
user-visible failure best-shot selection must avoid; on-device latency/battery of the
TOPIQ backbone not yet profiled. *Recommendation:* **log-only / shadow signal**, not
production, until the severe-FP behaviour is gated and runtime is profiled.
TOPIQ is either a feature-flagged shadow/log-only path or a post-submission 
working-tree addition, depending on the exact submitted build. In either case, 
it is not part of the default production technical score. 


**14k student** — stronger than the 50k line (0.7436) but 3 severe FPs and still no
runtime/safety advantage over production. *Recommendation:* **research reference.**

**Is there a technical model that should replace production now?** No. The two models
that beat production on SRCC are blocked by safety (TOPIQ+FLIVE: severe FPs) or
runtime (MDIQA: not mobile). Replacement is not justified without combined official
benchmark **and** on-device runtime/severe-FP evidence.

## 5. Aesthetic IAA Model Assessment

| Model | Dataset | SRCC | PLCC | Acc | Representative device latency |
|---|---|---|---|---|---|
| NIMA | AVA (25k) | 0.6220 | 0.6340 | 0.6161 | ~177 ms |
| Mobile A-LAMP v2 | AVA (proxy 5-crop) | 0.5067 | 0.5052 | 0.7536 | ~49 ms |
| RGNet | AADB (1k) | 0.6647 (TFLite) | 0.6708 | — | ~302 ms |
| ICAA | ICAA17K (1.6k) | 0.8791 | 0.8958 | — | ~1.1 s |

*Note: Timings are based on representative Android/project logs. Actual latency depends on image loading and preprocessing.*

Cross-dataset baselines from the weight search (calibrated) show the
complementarity clearly: NIMA AVA 0.623 / AADB 0.439; RGNet AVA 0.196 / AADB 0.668;
A-LAMP AVA 0.579 / AADB 0.274; ICAA AVA 0.336 / AADB 0.285.

**NIMA** — *Role:* general AVA aesthetic backbone (composition/overall appeal).
*Benchmark:* 0.6220 SRCC, slightly above the original paper. *Speed:* ~177 ms,
acceptable. *Complementarity:* strongest single model on AVA; weak on AADB — pairs
well with RGNet. *Risk:* distribution output needs the documented 1..10 → unit
mapping; mild calibration risk handled by the fitted calibration. *Interpretation
value:* high (recognizable "aesthetic score"). *Recommendation:* **production.**

**RGNet** — *Role:* attribute/relation aesthetic model, AADB-trained.
*Benchmark:* AADB 0.6647 TFLite (baseline 0.6819). *Speed:* ~302 ms (slowest
production aesthetic model but tolerable). *Complementarity:* dominates AADB
(0.668) where NIMA/A-LAMP are weak — the single biggest reason the ensemble
generalizes; it receives the **highest submitted app weight (0.45)**, while the offline search recommendation also assigned it the highest weight (0.41). *Risk:* poor on AVA alone
(0.196) — must never be used standalone. *Recommendation:* **production.**

**Mobile A-LAMP v2** — *Role:* layout/patch-aware aesthetic model.
*Benchmark:* AVA proxy SRCC 0.5067 but **classification accuracy 0.7536** — useful
for the binary good/bad decision even if rank correlation is modest. *Speed:*
~49 ms model, but adaptive-patch preprocessing is the real CPU cost (~0.1–0.2 s,
see GPU audit). *Complementarity:* adds composition/layout signal orthogonal to
NIMA/RGNet. *Risk:* proxy 5-crop protocol (not full adaptive-salience paper
protocol) likely caps its SRCC; preprocessing cost on low-end CPUs. *Recommendation:*
**production, with the preprocessing cost tracked.**

**ICAA** — *Role:* color-harmony / SOTA color-aesthetics reference.
*Benchmark:* ICAA17K 0.8791 — by far the strongest single aesthetic model, but on a
narrow color-aesthetics target (SPAQ-colorfulness only 0.3201 by design). *Speed:*
~1.1 s/image — ~3× the entire rest of the aesthetic ensemble; TFLite GPU delegate
unavailable, no INT8/QAT model exists yet. *Complementarity:* adds almost nothing to
the rank ensemble — Group B (with ICAA) was **−0.0033 combined SRCC** vs Group A.
*Risk:* latency dominates the per-photo budget; heterogeneous target (color, not
overall aesthetics) can distort the combined ranking if force-weighted.
*Recommendation:* **research / separate color subscore only**, weight 0.0 in the
production fusion.

**Is NIMA + RGNet + A-LAMP a reasonable production ensemble?** Yes — it is fast
(~528 ms total aesthetic), stable (5-fold SRCC std 0.0022), complementary across
AVA/AADB, and explainable.

**Is ICAA exclusion justified?** Yes — on both grounds: ~0 rank benefit and ~1.1 s
latency.

**Are the current weights reasonable?** Yes. The submitted app uses NIMA 0.40 / 
RGNet 0.45 / A-LAMP 0.15. For comparison, the offline search recommendation was 
NIMA 0.37 / RGNet 0.41 / A-LAMP 0.22. Both are three-model NIMA/RGNet/A-LAMP 
ensembles with ICAA excluded, and the submitted defaults remain close to the 
searched optimum. Since the app has already been submitted, this audit 
documents the implementation as-is rather than prescribing further code changes.

**Heterogeneous-dataset red flags (AVA/AADB/ICAA17K):** real and worth documenting.
The datasets use different scales and different notions of "aesthetic" (AVA = overall
appeal 1–10; AADB = attribute-based 0–1; ICAA17K = color aesthetics). The weight
search correctly (a) calibrates each model with slopes/intercepts fit only on the
train fold, (b) converts AADB 0–1 → 1–10 before fusion, and (c) optimizes a combined
70/30 AVA/AADB objective rather than a single dataset. This is the right mitigation;
the residual risk is that the **combined SRCC ~0.66 is a within-benchmark proxy, not
a measurement on real Pozy user photos.**

**Color aesthetics in production?** Keep it out of the live ranking on speed grounds,
but it is reasonable to surface ICAA (or a cheaper color proxy) **as an explanation /
debug subscore only**, not as a weighted ranking term.

## 6. Ensemble Design Review

- **Late fusion of IQA and IAA is correct.** Technical and aesthetic quality are
  different questions with different ground truth; collapsing them into one model (as
  TechIQA-Guard v1 tried for the technical side) caused **single-head interference** —
  the "pretty but technically broken" images leaked high scores. Keeping the axes
  separate and combining scores late preserves both signals and is debuggable.
- **Do not collapse technical and aesthetic early.** A single end-to-end "best photo"
  regressor would be difficult to calibrate, difficult to explain, and would hide
  exactly the severe-FP failures that KonIQ currently guards against.
- **Model diversity is doing real work.** Each aesthetic model owns a dataset where
  the others are weak (RGNet→AADB, NIMA→AVA, A-LAMP→layout); each technical model
  covers the other's blind spot (KonIQ guards FLIVE's optimism). Diversity here is
  not redundancy.
- **Calibration is necessary and already handled.** Aesthetic models live on
  different scales; the production path applies fitted linear calibration before
  weighted fusion. Technical scores are kept on a fixed 0..100 scale. This must stay
  in sync between the audit constants and the Flutter implementation.
- **Severe false positives are the dominant risk** and the production choices are
  explicitly optimized for them (KonIQ+FLIVE 0 severe FP; TOPIQ+FLIVE rejected for 3).
- **Explainability / debug UI:** late fusion lets the app show per-model sub-scores,
  which is essential for diagnosing a bad rank and for the capstone demonstration.
  Recommend the debug UI continue to expose NIMA/RGNet/A-LAMP and (optionally) ICAA
  and TOPIQ+FLIVE as **log-only** sub-scores.

## 7. Mobile Deployment Review

- **TFLite readiness:** NIMA, RGNet, A-LAMP v2 are already FP16 TFLite and
  identity-matched to production-ready artifacts (production precheck). KonIQ/FLIVE
  technical models are deployed. ICAA exists only in research artifacts as an FP16 TFLite candidate at ~1.1 s and is absent from the submitted app assets; no INT8/QAT production asset is used.
- **Runtime budget:** per photo ≈ technical (~50 ms for KonIQ+FLIVE) + aesthetic
  (NIMA 177 + RGNet 302 + A-LAMP 49 ≈ 528 ms). Adding ICAA (~1.1 s) would roughly
  triple the aesthetic budget and dominate album-scale ranking — disqualifying.
- **CPU/GPU fallback:** TFLite GPU delegate is **not available** in the audit
  environment (A-LAMP GPU audit); production runs CPU + XNNPACK. A-LAMP's
  adaptive-patch preprocessing is a CPU hotspot independent of the model itself.
- **iOS implications:** INT8/CoreML acceleration profiles differ from Android; any
  future ICAA INT8 path must keep float32 in/out interfaces and be parity-checked on
  iOS before use (ICAA QAT audit §7).
- **Memory/size:** three small aesthetic CNNs + two technical models is modest;
  adding heavy research backbones (MDIQA transformer, ICAA) is the main size/latency
  risk.
- **Why slow research models are excluded:** MDIQA (transformer, not mobile) and ICAA
  (~1.1 s, GPU delegate unavailable) violate the runtime/stability constraints with
  no commensurate ranking benefit. The **production-vs-research boundary** is the
  right organizing principle: deploy only models proven on *both* official benchmark
  and on-device runtime/safety.

## 8. Rejected / Research-Only Models

- **MDIQA student (50k B0 line):** No-Go. Best 0.7033 < production 0.7168 < 14k
  0.7436 < teacher 0.8227; all corrective variants regressed (0.48–0.54). Root cause
  is distillation capacity + cross-dataset label compression + weaker FLIVE
  alignment, not an implementation bug. Archive; do not deploy.
- **MDIQA teacher:** not rejected — retained as the performance-ceiling teacher;
  excluded from production only because the transformer backbone is not mobile-feasible.
- **ICAA (in the weighted ensemble):** excluded. Strong color reference (0.8791) but
  ~0 rank gain (−0.0033 combined SRCC) and ~1.1 s latency. Keep as separate
  research/color subscore.
- **TOPIQ / TOPIQ+FLIVE:** not production. Benchmark-superior (0.8069) but introduces
  3 severe FPs and is runtime-unprofiled on device → log-only candidate only.
- **TechIQA-Guard v1 (single-head technical replacement):** No-Go. Over-scored
  confirmed hard-FP images (68.07 vs production 61.02; KonIQ 47.84 strongest guard),
  i.e. it weakened the safety guard. Single-head interference confirmed.

## 9. Final Production Model Recommendation

**Technical IQA — keep KonIQ+FLIVE.** Safest (0 severe FP), well-calibrated,
fast (~25 ms/model). No justified replacement.

**Aesthetic IAA — keep the submitted Flutter app ensemble: NIMA + RGNet + Mobile A-LAMP v2**, using the submitted default weights NIMA 0.40 / RGNet 0.45 / A-LAMP 0.15. ICAA remains inactive / not part of the active production ensemble.

**Research / reference only (do NOT integrate into Flutter production):**
- MDIQA teacher (performance ceiling)
- MDIQA student outputs (archived)
- ICAA (separate color subscore / reference)
- TOPIQ and TOPIQ+FLIVE variants (log-only shadow at most)
- TechIQA-Guard v1 (internal reference)

**Do not integrate:** MDIQA student, ICAA-in-ensemble, unsafe TOPIQ variants,
TechIQA-Guard v1.

## 10. Thesis / Capstone Writing Notes (요약)

- **모델 앙상블을 사용한 이유:** 단일 모바일 모델로는 정확도·속도·안정성·교정을
  KonIQ/SPAQ/FLIVE(기술) 및 AVA/AADB/ICAA17K(미학)의 이질적 데이터셋 전반에서
  동시에 만족시킬 수 없다. 각 모델이 서로의 약한 데이터셋을 보완하므로 앙상블이
  교차 데이터셋 견고성을 확보한다(예: RGNet은 AADB, NIMA는 AVA에서 강함).
- **모바일 제약이 중요한 이유:** 사진 한 장당 (모델 수 × 지연시간)이 앨범 단위
  랭킹 비용을 결정한다. 따라서 벤치마크 점수보다 지연시간·안정성·심각 오탐 억제·
  배포 가능성이 실제 적용 여부를 좌우한다. ICAA(~1.1s)와 MDIQA(트랜스포머)는 이 
  제약으로 제외된다.
  (대표 지연시간은 프로젝트 컨텍스트 Android 로그를 기준으로 하며, 실제 앱 지연시간은 
  이미지 로딩 및 전처리 비용을 포함한다.)
- **강력한 연구 모델을 배포하지 않은 이유:** MDIQA teacher는 SRCC 0.8227과 심각 오탐 0건을 기록한 강력한 연구용 기준 모델이며, TOPIQ+FLIVE도 SRCC 0.8069로 프로덕션
  KonIQ+FLIVE(0.7168)보다 높은 벤치마크 성능을 보였다. 그러나 MDIQA는 PyTorch 트랜스포머 기반으로
  실시간 모바일 실행이 어렵고, TOPIQ+FLIVE는 심각 오탐 3건을 유발하므로 production 통합 대상으로 채택하지 않았다.
- **부정적 결과(Negative Result)의 가치:** MDIQA 학생 No-Go와 TechIQA-Guard
  No-Go는 기존 검증된 스택(KonIQ+FLIVE)의 견고함을 데이터로 입증하고, 검증되지
  않은 교체로 인한 위험과 자원 낭비를 방지한 의사결정 사례다.
- **최종적으로 방어 가능한 아키텍처:** 기술 IQA(KonIQ+FLIVE)와 미학 IAA(NIMA +
  RGNet + A-LAMP v2)를 분리해 각각 산출하고, 교정 후 후기 융합(late fusion)으로
  결합하며, 디버그 UI에서 모델별 서브 스코어를 노출한다. ICAA/TOPIQ는 로그·참조용.
- **최종 제출 앱 구현:** 최종 제출 앱은 IAA 모델 구성으로 NIMA, RGNet, Mobile A-LAMP v2를 
  사용하며, Flutter 기본 가중치는 NIMA 0.40 / RGNet 0.45 / A-LAMP 0.15이다. 
  별도의 오프라인 weight search에서는 0.37 / 0.41 / 0.22 조합이 추천되었지만, 
  본 보고서의 제출 앱 기준 설명에서는 실제 Flutter 기본값을 기준으로 서술한다. 

## 11. Final Recommendation

`FULL_MODEL_DESIGN_AUDIT_SUPPORTS_CURRENT_KONIQ_FLIVE_AND_NIMA_RGNET_ALAMP_PRODUCTION`
