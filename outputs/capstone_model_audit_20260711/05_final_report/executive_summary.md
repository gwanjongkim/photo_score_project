# Executive Summary

## Final adjudication

The repository contains a substantive, working multi-model IQA/IAA engineering program: architectures, checkpoints, raw predictions, exports, evaluation artifacts, server configuration, and several candidate handoff bundles are all present. The bounded resolution tests independently loaded models, invoked TFLite files, recomputed metrics, reconstructed ensembles, checked hashes, and inspected dataset contracts.

That evidence does **not** support a blanket claim that the portfolio consists of paper-faithful reproductions or that the checked artifacts are the models currently deployed in an external application. The strongest defensible description is a mixture of paper-inspired implementations, production-adapted server components, externally sourced official checkpoints, mobile candidates, experiments, and explicitly rejected lines. No model line is certified here as a complete paper reproduction under the original paper's architecture, training, split, preprocessing, and evaluation protocol.

The final scorecard therefore scores each model independently. It does not average the two prior auditors' scores, and it separates lineages that the prior reports had conflated, including FLIVE image versus patch, the default five-signal technical selector versus the measured image-only 4/7–3/7 reconstruction versus the mobile helper's 0.60/0.40 technical subcomponent/fallback, practical versus paper-v1/PIL RGNet, legacy versus fixed NIMA, official ICAA versus its mobile export, default versus calibrated aesthetic ensembles, and the two MDIQA student programs.

## Decision

| Intended use | Final decision | Reason |
|---|---|---|
| Present the repository as a broad paper-reproduction suite | **NO-GO** | Exact official protocols and historical training lineage are missing for multiple lines, and defining paper components differ in several local implementations. |
| Present it as an engineering capstone that evaluates and adapts multiple IQA/IAA approaches | **GO, with qualifications** | The code, artifacts, hashes, bounded execution, and several raw metric replays directly support this description. |
| Claim the default server bundle is validated production quality | **NO-GO** | It references legacy NIMA, several preprocessing contracts are defective, and the active external runtime was not supplied. |
| Claim mobile readiness from the existence of TFLite files | **NO-GO** | Conversion evidence is model-specific; app wiring, Dart preprocessing, on-device latency, thermal behavior, and active asset identity remain unverified. |
| Report exact local benchmark results | **GO, protocol-specific** | Use only replayed results, name the table/split/target/scale, and disclose overlap, selection, and preprocessing limitations. |

## Directly proven facts

1. **Technical metric reconstruction is possible.** `production_technical_predictions.csv` contains the ground-truth column `quality_score_0_100`. Across 4,967 rows, the reconstructed 4/7 KonIQ–3/7 FLIVE-image score is exact to `7.11e-15` and yields SRCC `0.795784`, raw PLCC `0.860414`, MAE `6.076832`, and RMSE `8.563833`. A different 4,967-row official-signal snapshot independently yields SRCC `0.716815` and PLCC `0.787413`; the two tables are not the same protocol snapshot.
2. **Several stored model metrics replay.** Fixed NIMA on 2,000 rows yields SRCC `0.589398`, PLCC `0.592904`, MAE `0.658118`, and binary accuracy `0.6025`. RGNet-PIL on the 1,000-row AADB test yields SRCC `0.664790`, PLCC `0.670798`, MAE `0.121894`, and RMSE `0.152153`. ICAA on 1,617 rows yields color SRCC `0.879062`, PLCC `0.895772`, MAE `0.021371`, and MOS SRCC `0.905809`. All nine supplied MDIQA 100-image variant/dataset groups replay to floating-point precision.
3. **The calibrated aesthetic ensemble's reported numbers replay.** With weights NIMA `0.37`, RGNet `0.41`, A-LAMP `0.22`, and ICAA `0.00`, the held-out results are AVA SRCC `0.653648`, AADB SRCC `0.673337`, and 70/30 combined SRCC `0.659554`.
4. **Its model-selection protocol is not a pristine final holdout.** Calibration is fit on the training portion, but final variant/preparation selection and the stable-pool construction consult the test metric. The exported AVA calibration parameters match the recomputation exactly; the exported AADB parameters do not.
5. **Dataset integrity differs by line.** FLIVE image manifests have no parent-ID overlap, but contain nine byte-identical cross-split groups; every group has conflicting labels. FLIVE patch train overlaps 10,452 of 10,753 test parents (`97.2008%`), and the combined FLIVE manifests overlap 13,589 of 13,652 test parents (`99.5385%`). AADB has four byte-identical cross-split groups, all with conflicting labels. KonIQ has zero exact-path and zero detected exact-byte overlap in the bounded full-content test.
6. **Current KonIQ manifests use `MOS_zscore`.** All 10,073 manifest rows match the raw `MOS_zscore` field exactly and not the raw `MOS` field. This establishes the current data contract, not the absent historical checkpoint invocation.
7. **High-impact preprocessing defects are serialized in current artifacts.** The three technical MobileNetV2 checkpoints receive `[0,1]` tensors, contain no internal rescaling, and omit the expected external `[-1,1]` preprocessing. Fixed NIMA removed its duplicate rescale but still supplies `[0,1]` to EfficientNetV2 with preprocessing disabled, whose documented contract is `[-1,1]`. Legacy NIMA and practical RGNet receive `[0,1]` and then apply an embedded `1/255` rescale before normalization.
8. **ICAA is implemented locally.** The official DAT checkpoint strict-loads all 487 keys with no missing or unexpected keys and forwards to shape `[1,2]`. A TF-native port, SavedModel lineage, and FP16 TFLite exist. The FP16 artifact is exactly `175,885,416` bytes with SHA-256 `10ae68a3283bdabf05ecfa8b679e1e89865f3c960687ce5495500455bbd111ff`; the prior `11.9 MB` statement is false.
9. **The ICAA mobile-lab contract is wrong.** The port output is `[MOS, color]`; validated color uses output index 1 and external ImageNet normalization, while the lab YAML uses index 0 and `[0,1]`. On one real image, changing only that preprocessing contract changed an output by as much as `0.221268`.
10. **The default aesthetic selector does normalize ranges.** It maps values such as NIMA `5.5` to `0.5` before weighting. Its value-inferred dispatch is nevertheless unsafe at the boundary: `1.05` maps to `1.0`, while `1.06` maps to `0.00667`. Practical A-LAMP has a linear output head, so out-of-range values are possible.
11. **Export parity is source-specific.** Same-tensor replay from six named `final_model.keras` files to TFLite succeeded, but maximum differences ranged from `0.000251` for fixed NIMA to `0.442081` for KonIQ. Several existing TFLites are better supported as exports of float32 rebuilds from `best.weights.h5`, not direct descendants of the differently serialized final model. Mobile A-LAMP has stronger 1,024-row conversion evidence, with FP16 Pearson `0.999895`; that is conversion parity, not model accuracy or Flutter parity.

## Strong but incomplete evidence

- Compatible trainers, model definitions, checkpoints, manifests, and output artifacts support many proposed lineage chains. Missing original commands, resolved configurations, environments, seeds, and stdout logs prevent those chains from being called fully proven.
- The practical RGNet contains genuine region graph construction and graph convolution, while the paper-v1 family contains DenseNet121, ASPP/DenseASPP, graph layers, and LSE aggregation. This supports “paper-inspired” or “substantially faithful” descriptions at the appropriate variant level, not equivalence to an official implementation.
- ICAA's strict official-weight load, TF port, 1,617-row metric replay, and prior 64-image SavedModel/FP16 parity provide strong implementation evidence. The faulty lab contract and 175.9 MB artifact prevent a mobile-production conclusion.
- Hash-identical copies were confirmed for major handoff artifacts, including legacy NIMA, fixed NIMA FP16, practical RGNet, RGNet-PIL FP16, practical A-LAMP, Mobile A-LAMP FP16, and ICAA FP16. Copy identity does not prove active runtime use.

## Reasonable interpretations

- The preprocessing defects are likely to reduce transfer-learning efficiency or change predictions materially. Their exact causal effect on accuracy is not measured because retraining and corrective ablations were outside scope.
- The nine FLIVE-image and four AADB duplicate groups weaken clean-holdout claims, but their exact aggregate metric effect was not isolated. The FLIVE-patch parent overlap is severe enough that patch-line generalization metrics should not be treated as independent test evidence.
- Model files with reproducible export parity and documented handoff metadata are legitimate **production candidates**. They are not proven deployed models until the external application and runtime contracts are inspected.

## Unsupported assumptions

- That the existence of a paper-inspired module, model name, or dataset establishes paper fidelity.
- That a compatible current trainer generated a historical checkpoint.
- That TFLite existence proves mobile deployment, acceptable on-device performance, or correct Dart preprocessing.
- That conversion parity is accuracy, or that a small sanity benchmark is a paper-result reproduction.
- That the calibrated ensemble configuration is used by the default server or submitted application. No production source reads its exact path.
- That a single “production score” applies to all technical or aesthetic policies. The repository contains distinct default, image-only, mobile-helper, and calibrated formulas; the mobile helper's complete default formula is a 0.55/0.45 aesthetic/technical mix, not merely its 0.60/0.40 technical subcomponent.

## Remaining unknowns

- The currently running server configuration, external Flutter asset hashes, Dart tensor preparation, and submitted-app formula.
- Exact historical training commands and environments for many checkpoints, including the checkpoint target selected for KonIQ.
- The complete AVA exact-content/perceptual overlap profile and perceptual near-duplicate rates for all datasets.
- The accuracy change after correcting MobileNetV2, fixed-NIMA, practical-RGNet, and practical-A-LAMP preprocessing.
- Exact original-paper split wording where the supplied primary-source summaries conflict, most notably NIMA/AVA.
- Deterministic converter provenance for every TFLite and on-device latency, memory, and thermal performance.

## Capstone-safe conclusion

The defensible capstone claim is that the project **implemented, adapted, benchmarked, converted, and comparatively audited a broad set of image-quality and aesthetic-scoring approaches, producing several reproducible local results and deployment candidates while identifying material dataset, preprocessing, lineage, and runtime-contract limitations**. Claims of exact paper reproduction, production deployment, or universally valid benchmark superiority require the qualifications and future work listed in the accompanying reports.
