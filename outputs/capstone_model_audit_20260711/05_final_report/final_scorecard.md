| Model | Paper Fidelity | Experimental Validity | Production Suitability | Traceability | Classification | Confidence |
|---|---:|---:|---:|---:|---|---|
| KonIQ Mobile | 5 | 48 | 70 | 68 | PRODUCTION_ADAPTED_MODEL | HIGH |
| FLIVE Image Mobile | 5 | 42 | 69 | 67 | PRODUCTION_ADAPTED_MODEL | HIGH |
| FLIVE Patch | 5 | 18 | 42 | 58 | PRODUCTION_ADAPTED_MODEL | HIGH |
| Default five-signal technical selector | 0 | 35 | 60 | 68 | PRODUCTION_ADAPTED_MODEL | HIGH |
| Image-only 4/7–3/7 technical reconstruction | 0 | 67 | 58 | 75 | PRODUCTION_ADAPTED_MODEL | HIGH |
| Mobile-helper 0.60/0.40 technical subcomponent/fallback | 0 | 25 | 45 | 48 | PRODUCTION_ADAPTED_MODEL | MEDIUM |
| TOPIQ-lite mixed112 | 30 | 50 | 48 | 56 | PAPER_INSPIRED_IMPLEMENTATION | MEDIUM |
| C6 TOPIQ-KonIQ | 0 | 60 | 12 | 66 | EXPERIMENTAL_PROTOTYPE | HIGH |
| TOPIQ RankCalib | 20 | 46 | 18 | 52 | EXPERIMENTAL_PROTOTYPE | MEDIUM |
| MUSIQ AADB practical | 32 | 35 | 44 | 58 | PAPER_INSPIRED_IMPLEMENTATION | MEDIUM |
| Local MUSIQ technical probes | 20 | 25 | 20 | 38 | EXPERIMENTAL_PROTOTYPE | MEDIUM |
| Kaggle MUSIQ variants | 55 | 25 | 5 | 25 | UNVERIFIED | LOW |
| MDIQA teacher | 78 | 56 | 18 | 72 | SUBSTANTIALLY_FAITHFUL | MEDIUM |
| MDIQA 14k student | 5 | 44 | 28 | 63 | EXPERIMENTAL_PROTOTYPE | HIGH |
| MDIQA 50k student | 5 | 48 | 12 | 67 | EXPERIMENTAL_PROTOTYPE | HIGH |
| TechIQA-Guard v1 | 0 | 43 | 8 | 62 | EXPERIMENTAL_PROTOTYPE | HIGH |
| DistortionGuard-IQA | 0 | 50 | 8 | 65 | EXPERIMENTAL_PROTOTYPE | HIGH |
| NIMA AVA legacy | 55 | 28 | 35 | 70 | EXPERIMENTAL_PROTOTYPE | HIGH |
| NIMA AVA fixed candidate | 68 | 63 | 70 | 80 | SUBSTANTIALLY_FAITHFUL | HIGH |
| RGNet AADB practical | 32 | 36 | 48 | 70 | PAPER_INSPIRED_IMPLEMENTATION | HIGH |
| RGNet paper-v1 family | 65 | 56 | 25 | 58 | PAPER_INSPIRED_IMPLEMENTATION | MEDIUM |
| RGNet PIL-resize AADB candidate | 62 | 67 | 65 | 82 | PAPER_INSPIRED_IMPLEMENTATION | HIGH |
| A-LAMP AADB practical | 35 | 35 | 48 | 68 | PAPER_INSPIRED_IMPLEMENTATION | HIGH |
| Mobile A-LAMP v2 AVA | 33 | 68 | 82 | 84 | PRODUCTION_ADAPTED_MODEL | HIGH |
| A-LAMP teacher variants | 45 | 40 | 8 | 48 | EXPERIMENTAL_PROTOTYPE | MEDIUM |
| ICAA17K official DAT and TF-native port | 88 | 79 | 15 | 76 | SUBSTANTIALLY_FAITHFUL | HIGH |
| ICAA FP16 mobile artifact | 84 | 78 | 25 | 86 | SUBSTANTIALLY_FAITHFUL | HIGH |
| Stage 5 AADB aesthetic student | 0 | 35 | 30 | 64 | EXPERIMENTAL_PROTOTYPE | HIGH |
| Stage5 default aesthetic ensemble | 0 | 38 | 55 | 68 | PRODUCTION_ADAPTED_MODEL | HIGH |
| Calibrated NIMA-RGNet-A-LAMP ensemble | 0 | 66 | 42 | 76 | EXPERIMENTAL_PROTOTYPE | MEDIUM |
| AADB composition regressor | 0 | 35 | 52 | 58 | PRODUCTION_ADAPTED_MODEL | HIGH |
| Pairwise AADB comparator | 20 | 38 | 50 | 55 | PAPER_INSPIRED_IMPLEMENTATION | MEDIUM |
| VILA/Gemma multimodal scorers | 0 | 20 | 8 | 20 | UNVERIFIED | LOW |

# Final Scorecard

## How to read the scores

The four scores are independent judgments, not averages of the two earlier audits. They measure different things.

- **Paper Fidelity** asks whether the implemented architecture, objective, data protocol, preprocessing, and output semantics match the named paper. A zero for a project-specific ensemble, guard, or student means that no audited paper-reproduction claim applies; it does not mean the code is absent.
- **Experimental Validity** rewards clean splits, correct tensor/label contracts, replayable predictions and metrics, and untouched holdouts. It penalizes leakage, target ambiguity, preprocessing defects, test-guided selection, and missing raw evaluations.
- **Production Suitability** rewards a coherent input/output contract, bounded artifact size, export evidence, robust policy semantics, and a repository runtime link. It does not treat a TFLite file alone as proof of deployment.
- **Traceability** rewards immutable hashes, raw predictions, exact manifests, recorded commands, and a named checkpoint-to-export chain. Missing historical invocations and unavailable external app state remain penalties.

Confidence describes confidence in the evidence boundary, classification, and overall score bands—not exact point-score precision and not the model's predictions. A `HIGH` verdict can therefore coexist with an unresolved narrow point-score disagreement when the material evidence and classification are stable.

## Decisive scoring rules

1. A genuine but materially adapted paper mechanism is `PAPER_INSPIRED_IMPLEMENTATION`, not an exact reproduction.
2. A local port of an official checkpoint can be `SUBSTANTIALLY_FAITHFUL` when architecture and output semantics are verified, even when original training was not reproduced.
3. A repository-specific model or policy can be `PRODUCTION_ADAPTED_MODEL` only when its runtime role is directly evidenced. Active use in an external submitted app is still not inferred.
4. A rejected line remains an `EXPERIMENTAL_PROTOTYPE` even if its experiment was executed well.
5. `UNVERIFIED` is used where the audited identity depends on external handles, conflated model families, or evidence not captured in the repository.

## Material departures from both earlier audits

- The technical policies are scored separately. The image-only production table has ground truth and its 4,967-row metrics plus 4/7–3/7 formula were recomputed. Those metrics do not transfer to the default five-signal selector or the mobile helper's 0.60/0.40 technical subcomponent/fallback; the helper's complete default policy mixes AADB aesthetic and technical components at 0.55/0.45.
- FLIVE patch receives the lowest validity score among active technical components because 97.2008% of test parents occur in patch training. FLIVE image is separate: parent overlap is zero, but nine exact-content cross-split duplicate groups have conflicting labels.
- Fixed NIMA is stronger than legacy NIMA, but it is not called fully corrected: its EfficientNetV2 backbone has preprocessing disabled while the dataset still supplies `[0,1]` rather than the expected `[-1,1]`.
- Practical RGNet and paper-v1/PIL RGNet are scored separately. Practical RGNet has a real graph but omits defining paper components and has an active input-range defect; paper-v1/PIL has DenseNet/ASPP/graph/LSE evidence and replayable 1,000-row metrics.
- ICAA is not `NOT_IMPLEMENTED`. The official checkpoint strict-loads, a TF-native port exists, output order is `[MOS,color]`, 1,617-row metrics replay, and the FP16 copies hash identically. Deployment suitability remains low because the artifact is 175,885,416 bytes and the lab contract uses the wrong normalization and output index.
- The default and calibrated aesthetic ensembles are different lineages. The calibrated result replays, but final selection consulted the test metric, AADB calibration values drift in the exported config, and no default runtime consumes that config.

## Overall adjudication

No audited line qualifies as `PAPER_FAITHFUL_REPRODUCTION`. The strongest paper-grounded evidence belongs to the official ICAA DAT/TF-native line, its FP16 conversion, the MDIQA teacher, and fixed NIMA, all with explicit qualifications. The strongest mobile-candidate engineering package is Mobile A-LAMP v2, followed by fixed NIMA and RGNet PIL-resize; none has proven parity through the unavailable external Flutter runtime.
