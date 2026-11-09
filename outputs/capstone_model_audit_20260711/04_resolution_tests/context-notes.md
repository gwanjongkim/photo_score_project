# Resolution Audit Context Notes

## 2026-07-11 — Initial scope decisions

- The user named `01_paper_baseline/`, but the filesystem contains `01_paper_baseline_codex/`. The latter is used as the intended evidence source; this discrepancy will appear in limitations.
- Existing evidence directories `00` through `03` are read-only for this task. All generated scripts, raw outputs, and adjudication artifacts will stay in `04_resolution_tests/`; synthesized reports will stay in `05_final_report/`.
- The initial Git check shows the entire capstone audit output tree as untracked. This means Git cannot establish a clean tracked baseline for those artifacts; file hashes and scoped before/after inventories will be used instead.
- Final model scores will be reconstructed from evidence. Prior numeric scores are inputs to dispute identification, not values to average.
- Absence of evidence will be recorded as an unknown or unsupported assumption, never as confirmation.

## 2026-07-11 — Post-test adjudication decisions

- The cross-check correctly identified missing external MobileNetV2 normalization and the residual fixed-NIMA normalization defect. The primary audit's claim that modern MobileNetV2 embeds the rescale is false for the serialized checkpoints.
- Practical RGNet's duplicate scaling is directly present, but “vanishing activations/gradients” is not directly measured. The finding is a confirmed contract defect with partially resolved impact.
- ICAA was split into the official/TF-native model line and its FP16 mobile artifact. `src/models/icaa_dat.py` absence does not justify `NOT_IMPLEMENTED`.
- FLIVE image, FLIVE patch, and combined manifests are scored separately. Image parent overlap is clean, exact byte duplicates are not; patch parent leakage is severe.
- Default and calibrated aesthetic ensembles are separate scorecard rows. The default normalizes values; the calibrated candidate reproduces offline metrics but uses its test fold for final selection and exports stale/mismatched AADB calibration values.
- Direct metric replay is accepted when raw prediction/ground-truth rows reproduce stored values. It does not establish official paper-protocol comparability or active runtime use.
- Same-input parity is tied to the precise source model. A TFLite that matches a float32 rebuild from `best.weights.h5` is not described as a direct export of a divergent mixed-precision `final_model.keras`.
- The final scorecard covers every phase-00 inventory line and splits previously conflated lineages. No score is an average of the two auditors.

## 2026-07-13 — Continuation and finalization decisions

- Continuation began from the interrupted unsafe-claims review. Completed resolution scripts/results were preserved; no bounded inference, metric, hash, overlap, or parity test was regenerated merely to rewrite it.
- The two additional baseline paths named in the continuation request were checked. `01_paper_baseline/` and `01_paper_baseline_claude_crosscheck/` are absent; `01_paper_baseline_codex/` and `03_independent_crosscheck/` are the actual baseline/cross-check sources used.
- The final dispute register contains 38 exact claims: 1 `RESOLVED_CONFIRMED`, 17 `RESOLVED_REFUTED`, 13 `PARTIALLY_RESOLVED`, and 7 `UNRESOLVED`. Composite and subjective point-score disputes were not promoted beyond the evidence.
- The final scorecard contains 33 rows. The former aggregate technical-family row is split into the default five-signal selector, the measured image-only 4/7–3/7 reconstruction, and the mobile helper's 0.60/0.40 technical subcomponent/fallback. The helper's complete parent policy is 0.55 AADB aesthetic plus 0.45 technical.
- The calibrated aesthetic ensemble changed from the interrupted draft's `PRODUCTION_ADAPTED_MODEL` to `EXPERIMENTAL_PROTOTYPE` because no runtime consumes its config and selection touched test metrics. Pairwise AADB changed from `PRODUCTION_ADAPTED_MODEL` to `PAPER_INSPIRED_IMPLEMENTATION`, with PF 20 limited to its generic shared Siamese/pairwise concept rather than full AADB-paper identity.
- Claim registers and the paper-to-code matrix now use the controlled evidence vocabulary. `STRONG_BUT_INCOMPLETE` rows always require qualification and are never presented as direct proof. Lineage-only `UNKNOWN_LINK`, external-evidence `NOT_AVAILABLE`, and claim-state `UNVERIFIED` are defined separately.
- A bounded source-safety check rehashed all 77 unique architecture, training, data-pipeline, export, runtime-inference, evaluation, and configuration paths captured in phase 00. Missing files, hash mismatches, and duplicate inventory conflicts were all zero. This proves no drift from that snapshot, not global Git cleanliness before phase 00.
- A preliminary validator pass exposed one wildcard evidence locator in D16. That failed pass is retained; five concrete verification JSON paths replaced the wildcard. The strengthened pre-closeout run then passed with 38 disputes, 33 score rows, 33 verdicts, 43 paper-to-code rows, 37 defensible claims, 50 unsafe claims, 9 bounded result JSONs, 184 evidence rows, and 77 unchanged source/config paths.
- The manifest is a live file-hash snapshot. It intentionally excludes itself and `validation_results.json` to avoid recursive hashes. Final closeout freezes this note, checklist, and command log first, then reruns source drift, manifest generation, and validation without further manifest-covered edits.
