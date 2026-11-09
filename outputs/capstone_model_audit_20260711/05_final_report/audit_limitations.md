# Audit Limitations

## Scope boundary

This is a repository-snapshot adjudication dated 2026-07-11. It establishes what can be proven from the supplied files and bounded non-destructive tests. It is not a new training study, external application audit, hardware qualification, security review, or independent reproduction from raw papers and pristine datasets.

The continuation request named `01_paper_baseline/` and `01_paper_baseline_claude_crosscheck/`, but neither directory existed in this checkout. The available `01_paper_baseline_codex/` directory was treated as the intended baseline, while `03_independent_crosscheck/` supplied the independent review. Those substitutions are explicit and may matter if separate baseline packages were expected.

## Evidence limitations

| Limitation | Effect on verdict | Status |
|---|---|---|
| Many original training commands, resolved configs, stdout logs, environments, and seeds are absent. | Current code and checkpoints can be called compatible, but exact trainer-to-checkpoint lineage cannot be proven. | **UNKNOWN** for affected historical runs. |
| Important checkpoints and processed data are ignored or untracked; the entire audit output tree was initially untracked. | Git history cannot establish artifact creation, immutability, or a clean baseline. Current SHA-256 values establish only snapshot identity. | **PROVEN limitation.** |
| The research repository is not the external Flutter/application repository. | Active assets, Dart preprocessing, output-index handling, weights, and currently shipped formula cannot be verified. | **UNKNOWN deployment state.** |
| A running server process and resolved live configuration were not supplied. | `configs/stage5_reference.json` proves a repository default, not the identity of a live service. | **UNKNOWN active runtime.** |
| Paper access and source retrieval were restricted to the supplied evidence set. | Conflicting source summaries—especially NIMA/AVA split wording—remain unresolved, and inaccessible official repositories/supplements cannot be used to fill gaps. | **PARTIALLY RESOLVED.** |
| The audit did not recreate raw AVA preprocessing from its original annotation source. | Current AVA manifests and distribution normalization are checked; raw-source-to-manifest lineage remains incomplete. | **UNKNOWN_LINK.** |
| Exact-content hashing was bounded. Full KonIQ, FLIVE image, and AADB checks ran; full AVA hashing over 255,508 manifest rows did not. | AVA exact-content duplication is unknown. Perceptual near-duplicate detection is unknown for every dataset. | **BLOCKED by bounded-test scope.** |
| Cross-manifest reuse was deeply checked only for the core FLIVE lines. | Parent leakage in other teacher/student or mixed-dataset manifests may remain undetected. | **UNKNOWN.** |
| No retraining or corrective preprocessing ablation was permitted. | Tensor-contract defects are proven; their causal effect on accuracy, convergence, or ranking cannot be quantified. | **Not tested by design.** |
| No complete model suite was re-evaluated from images. | Metric replays validate supplied prediction tables, not every evaluator, decoder, augmentation, or checkpoint. | **PARTIAL execution evidence.** |
| The MDIQA 900-prediction sanity benchmark was not rerun. | Its raw predictions had zero recorded failures, hashes matched, and all nine group metrics replayed; checkpoint inference remains accepted as supplied bounded evidence rather than a fresh benchmark. | **PARTIALLY VERIFIED.** |
| The ICAA strict-load smoke used one CPU forward input. | It proves checkpoint/architecture compatibility and `[1,2]` output shape, not training provenance or dataset accuracy. Full accuracy claims rely on raw 1,617-row prediction replay. | **PARTIAL execution evidence.** |
| Direct `final_model.keras`–TFLite replay used two real images per model for six model pairs. | It exposes lineage differences but is too small to bound worst-case conversion error. Existing Mobile A-LAMP evidence is stronger at 1,024 rows. | **PARTIAL parity evidence.** |
| Every TFLite was not deterministically reconverted. | Several exports are supported as rebuilds from `best.weights.h5`; exact converter flags, package versions, and source serialization remain model-specific. | **PARTIALLY RESOLVED.** |
| No Android/iOS/Flutter execution or device profiling was performed. | Latency, memory, delegate compatibility, energy, thermal behavior, and on-device numerical parity are unknown. | **UNVERIFIED.** |
| The calibrated ensemble was recomputed from existing aligned predictions, not from fresh component inference. | The arithmetic, calibration, weights, and held-out metrics are proven for those tables; upstream model/runtime identity inherits the source tables' lineage limits. | **PARTIAL lineage.** |
| Logistic PLCC fitting was diagnostic only. | It demonstrates numeric sensitivity but cannot establish a method requirement without a method-specific source. | **No universal fitted-PLCC verdict.** |
| Rubric scores are adjudicative judgments. | Paper Fidelity, Experimental Validity, Production Suitability, and Traceability are not measured physical quantities and have no meaningful confidence interval. | **Explicit scoring limitation.** |

## Tests deliberately not performed

- Full retraining or fine-tuning of any model.
- Production-source, checkpoint, export, dataset, or external-app modification.
- Corrective preprocessing ablations for technical MobileNetV2, NIMA, RGNet, or A-LAMP.
- Full AVA content hashing or any perceptual-near-duplicate scan.
- Deterministic reconversion of every TFLite from every possible source checkpoint.
- External network retrieval to settle absent paper details.
- On-device deployment, latency, memory, thermal, or delegate testing.
- Recovery of historical commands from checkpoint bytes, which is not technically possible.

## Execution caveats

All purpose-built resolution scripts exited zero. TensorFlow reported CUDA-initialization warnings and used CPU/XNNPACK where applicable; this does not invalidate the inspected graph contracts or output replays, but it means the audit does not certify GPU-specific numerical behavior.

One malformed diagnostic command, `./ .venv_gpu/bin/python --version`, exited 126 because `./` is a directory. It had no evidentiary consequence; all actual tests used `./.venv_gpu/bin/python` successfully. An initial metric check compared two distinct protocol tables and was superseded by a corrected table-specific recomputation. A preliminary closeout validation failed on a wildcard evidence locator and then passed after concrete paths replaced it. One read-only inventory probe assumed a nonexistent `.tsv` filename before locating the actual `.md` inventory. All executed failures, superseded detectors, and corrections are retained in `04_resolution_tests/failed_tests.md` and `commands_executed.txt`.

## Interpretation boundaries

Directly proven statements apply to the exact current file paths, bytes, data rows, code, and runtime environment inspected. They do not automatically transfer to older runs, a submitted application, another clone, or future replacement artifacts.

Absence of a file is evidence only about this checkout. It is not evidence that an external experiment never occurred. Conversely, the presence of a trainer, paper-named class, TFLite file, or handoff document is not proof that the artifact was trained by that code, implements the official paper, or is active in production.

No score was obtained by mechanically averaging the primary and independent audits. Factual disagreements were tested where possible; the remaining numerical rubric differences are judgment under explicit uncertainty. A `LOW` or `UNVERIFIED` verdict should therefore be read as an evidence limit, not as proof that the model has no value.

The evidence manifest is a file-level snapshot. Its `model_or_scope` field uses model-family labels where safely inferable and phase/protocol scope elsewhere; the detailed per-model associations and missing transitions remain in `final_lineage_report.md`. The manifest intentionally excludes itself and `04_resolution_tests/validation_results.json` to avoid recursive hashes; the validator recomputes every listed live file's size and SHA-256.

## Remaining material unknowns

1. Which exact artifacts, preprocessing contracts, and ensemble weights are active in the external product.
2. The original execution lineage for many local checkpoints and probe models.
3. Model accuracy after correcting the proven preprocessing defects.
4. Generalization after parent-grouped, content-deduplicated splits, especially for FLIVE patch.
5. Performance on a genuinely locked holdout after ensemble/model selection.
6. Full AVA content/perceptual duplication and broader cross-dataset reuse.
7. Complete original-paper parity where primary sources or official split files remain unavailable.
8. Converter reproducibility and worst-case Python–TFLite parity for every candidate.
9. On-device feasibility of the 175,885,416-byte ICAA artifact and all other candidate models.

These unknowns are carried into the classifications and confidence levels; none is silently treated as a pass.
