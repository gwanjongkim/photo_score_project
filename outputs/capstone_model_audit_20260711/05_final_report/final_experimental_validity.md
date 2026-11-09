# Final Experimental Validity

## Assessment standard

Experimental validity is assessed independently from paper fidelity and deployment suitability. A metric is accepted as **recomputed** only when raw predictions and paired ground truth reproduce it. Reproducibility of a number does not by itself establish a clean split, correct preprocessing, paper-protocol equivalence, unbiased model selection, or active deployment.

Evidence labels in this report have the following meaning.

- **PROVEN** — directly executed or exactly reconstructed from the supplied repository snapshot.
- **STRONG** — multiple mutually consistent artifacts support the finding, but an execution or provenance link is missing.
- **INTERPRETATION** — reasonable consequence of proven facts, not itself measured.
- **UNSUPPORTED** — contradicted by evidence or asserted without the required evidence.
- **UNVERIFIED** — the evidence set cannot determine the answer.

## Dataset integrity and split independence

| Dataset/line | Rows train/val/test | Exact-path overlap | Content/parent result | Experimental verdict |
|---|---:|---:|---|---|
| KonIQ | 8,058 / 1,007 / 1,008 | 0 | 0 exact-byte cross-split groups among all 10,073 files checked | **PROVEN clean for exact paths/bytes.** Perceptual near-duplicates and official-split equivalence remain unknown. |
| FLIVE image | 31,848 / 3,981 / 3,981 | 0 | 0 parent-ID overlap; 9 exact-byte cross-split groups, all with conflicting MOS | **PARTIALLY VALID.** The blanket parent-leakage allegation is false for this line, but the holdout is not content-clean. |
| FLIVE patch | 95,544 / 11,943 / 11,943 | Not the right unit | Train shares 10,452 of 10,753 test parents (`97.2008%`) and 10,530 of 10,823 validation parents (`97.2928%`) | **INVALID as an independent parent-image holdout.** Patch metrics cannot support unseen-image generalization. |
| Combined FLIVE `all` | 127,392 / 15,924 / 15,924 | Not the right unit | Train shares 13,589 of 13,652 test parents (`99.5385%`) | **INVALID as an independent parent-image holdout.** |
| AADB | 7,612 / 846 / 1,000 | 0 | 4 exact-byte cross-split groups, all with conflicting labels | **PARTIALLY VALID.** The listed official test structure is retained, but the observed data are not content-disjoint. |
| AVA | 204,406 / 25,551 / 25,551 | 0 | Full exact-content hashing not run; distribution labels sum to 1 within `1.0e-15` | **UNVERIFIED for content duplication.** The local seed-42 80/10/10 split cannot be presented as the paper split. |

Cross-task FLIVE reuse is also material. FLIVE patch training contains parents for 3,940 of 3,981 image-test rows (`98.9701%`); FLIVE image training contains parents for 8,632 of 10,753 patch-test parents (`80.2753%`). Experiments that train across these lines must group by parent before splitting.

The duplicate-label audit found that all nine FLIVE-image byte-duplicate groups disagree in MOS, with within-group differences from `1.2712` to `12.6498` points. All four AADB groups disagree by `0.025` to `0.15`. These are not harmless repeated rows with identical targets.

## Label scales and target semantics

| Dataset/model | Proven current target/output scale | Required qualification |
|---|---|---|
| KonIQ manifests | `3.9118–88.3889`; all 10,073 labels exactly match raw `MOS_zscore` | Call this the project's `MOS_zscore`-derived 0–100-like target, not ordinary raw MOS. Historical checkpoint CLI remains unknown. |
| FLIVE image | `0–94.3332` MOS-like scalar | State that the local split is custom and contains nine cross-split exact duplicates. |
| FLIVE patch | `14.6903–100` patch MOS-like scalar | Patch score is not an image score until an aggregation policy is named. |
| AADB | `0–1` scalar | Do not mix directly with 1–10 or 0–100 outputs. |
| AVA | mean score `1.8090–8.6`; ten-bin distributions sum to 1 | NIMA expected score is computed over bins 1 through 10. Mobile A-LAMP instead predicts a thresholded binary probability. |
| Practical RGNet | sigmoid scalar, nominally `0–1` | Its AADB use does not make AADB attribute losses an RGNet paper requirement. |
| Practical A-LAMP | linear scalar | `[0,1]` is a training tendency, not a guaranteed output bound; downstream scale inference must handle outliers explicitly. |
| ICAA TF/TFLite | vector `[MOS, color]` | Validated color is index 1. Index 0 is MOS, not color. |

## Preprocessing validity

### Proven contract defects

| Line | Loader/runtime tensor | Serialized transform | Backbone contract | Result |
|---|---|---|---|---|
| KonIQ, FLIVE image, FLIVE patch MobileNetV2 | `[0,1]` | No Rescaling or Normalization | External MobileNetV2 preprocessing produces `[-1,1]` | **MISMATCH.** The primary audit's claim of embedded `1/127.5,-1` rescaling is refuted. |
| Fixed NIMA EfficientNetV2 | `[0,1]` | No Rescaling or Normalization; `include_preprocessing=False` | Installed Keras contract is `[-1,1]` when preprocessing is disabled | **MISMATCH.** Duplicate scaling was removed, but the corrected path is still not coherent. |
| Legacy NIMA | `[0,1]` | Embedded `1/255` then ImageNet Normalization | Embedded branch is configured as though input were byte-scaled | **MISMATCH.** Input is under-scaled before normalization. |
| Practical RGNet | `[0,1]` | Embedded `1/255` then ImageNet Normalization | Same mismatch as legacy NIMA | **MISMATCH.** The intermediate range is `[0,0.00392]`; downstream activation/accuracy impact was not measured. |
| Practical A-LAMP | `[0,1]` native-size inputs | Global branch contains `1/255` plus ImageNet Normalization; output is linear | Branch contracts are not consistently documented | **MISMATCH/RISK.** A corrective ablation was not run. |
| Mobile A-LAMP v2 | raw `[0,255]` trainer inputs | Embedded MobileNetV3 `1/127.5,-1` transforms | Embedded preprocessing enabled | **MATCH in the inspected training/model path.** |
| ICAA FP16 validated path | External ImageNet normalization | No substitute `[0,1]` contract is validated | Official/port transform | **MATCH only for the validated path.** Lab YAML's `[0,1]` path is a confirmed mismatch. |

These findings prove tensor-contract errors. The claim that any one defect caused a specific metric loss, vanishing gradient, or rank degradation is an **INTERPRETATION**, because no corrected retraining or controlled inference ablation was authorized.

## Output semantics and aggregation

- **PROVEN.** Technical MobileNetV2 models have unbounded linear scalar heads trained against 0–100-like targets.
- **PROVEN.** NIMA emits a ten-bin softmax distribution whose expected score uses bins 1–10; an all-uniform distribution maps to `5.5`.
- **PROVEN.** Practical RGNet emits a sigmoid scalar; practical A-LAMP emits a linear scalar.
- **PROVEN.** ICAA emits `[MOS, color]`. On one real image, ImageNet-normalized inference produced `[0.655066, 0.589033]`, whereas `[0,1]` inference produced `[0.433797, 0.598445]`; the maximum component change was `0.221268`.
- **PROVEN.** The FLIVE patch server path runs five `224×224` crops and emits mean/min/std. A real checked example produced predictions `[75.875, 75.25, 75.9375, 74.375, 75.3125]`, hence mean `75.35`, minimum `74.375`, and standard deviation `0.562639`.
- **INTERPRETATION.** Five-crop mean/min/std is a project heuristic. It is not evidence of PaQ-2-PiQ's learned patch-to-image aggregation.

## Metric recomputation

### Accepted repository-specific results

| Result set | n | SRCC | PLCC | MAE | RMSE | Status and limit |
|---|---:|---:|---:|---:|---:|---|
| Production technical table, 4/7 KonIQ + 3/7 FLIVE image | 4,967 | `0.795784` | `0.860414` raw | `6.076832` | `8.563833` | **PROVEN.** Formula max error `7.11e-15`; valid only for this table/protocol snapshot. |
| Official technical signal table | 4,967 | `0.716815` | `0.787413` raw | `11.995573` | `14.019543` | **PROVEN.** Distinct rows/ground truth from the production table; do not merge or choose between them without naming the protocol. |
| NIMA fixed AVA evaluation | 2,000 | `0.589398` | `0.592904` | `0.658118` | `0.803679` | **PROVEN replay.** Binary accuracy is `0.6025`; local split/preprocessing limitations remain. |
| RGNet-PIL AADB test | 1,000 | `0.664790` | `0.670798` | `0.121894` | `0.152153` | **PROVEN replay.** Four AADB exact duplicate groups and paper-protocol differences remain. |
| ICAA color full test | 1,617 | `0.879062` | `0.895772` | `0.021371` | `0.029794` | **PROVEN replay.** This is color output/index 1 under the validated contract. |
| ICAA MOS full test | 1,617 | `0.905809` | `0.910948` | `0.037506` | `0.053655` | **PROVEN replay.** This is MOS output/index 0, not the color score. |

For MDIQA, nine supplied groups—three checkpoint variants across FLIVE, KonIQ10k, and SPAQ, each with `n=100`—reproduced stored SRCC, PLCC, and raw-MAE values with maximum discrepancies of `1.11e-16`, `3.33e-16`, and `1.78e-15`, respectively. That validates the stored arithmetic and supplied prediction files. It remains a bounded sanity benchmark, not a full paper-result reproduction. FLIVE results in these groups are especially weak (`SRCC 0.0279–0.0698`), while same-domain KonIQ/SPAQ results are much higher; the sample size is too small for broad comparative conclusions.

### PLCC adjudication

Raw PLCC on the production technical table is `0.860414`. A diagnostic four-parameter fit raises it to `0.864333` and reduces MAE from `6.076832` to `5.881258`. This proves that fitting changes the number. It does **not** prove that every cited method requires logistic fitting. The supplied paper baseline requires raw Pearson unless a method-specific source explicitly specifies a nonlinear mapping. Therefore the cross-check's blanket assertion that all raw PLCC values are invalid is **UNSUPPORTED**.

## Ensemble validity

### Technical ensemble

The 4/7–3/7 image-only reconstruction is exact for the inspected production table. It must not be conflated with either the default five-signal server selector or the mobile helper's 0.60/0.40 technical subcomponent/fallback within its 0.55/0.45 aesthetic/technical parent policy.

The default selector does normalize common scales before weighting and renormalizes weights when signals are missing. The claim that it directly weights raw 0–1, 1–10, and 0–100 values is **REFUTED**. Its range selection is still unsafe: a linear-head value of `1.05` maps to `1.0`, while `1.06` changes branch and maps to `0.00667`. That discontinuity prevents a clean semantic-validity verdict until every model declares an explicit output contract.

### Calibrated aesthetic ensemble

| Dataset | Test rows | SRCC | PLCC raw | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| AVA | 5,109 | `0.653648` | `0.665540` | `0.460510` | `0.587886` |
| AADB, ground truth mapped from 0–1 to 1–10 | 200 | `0.673337` | `0.668978` | `1.339797` | `1.654691` |
| Combined 70/30 SRCC | — | `0.659554` | — | — | — |

These metrics and weights NIMA/RGNet/A-LAMP/ICAA = `0.37/0.41/0.22/0.00` replay exactly. Calibration uses the 60% training subset, which is valid as far as that step goes. However, variant/preparation selection and stable-pool construction consult the 20% test result. Consequently:

- the reported test result is **reproducible**;
- it is **not an untouched confirmatory holdout**;
- confidence intervals or claims of generalization need a new locked test set;
- the standalone audit config is **not a faithful AADB reconstruction contract**, because all four AADB calibration pairs differ from recomputation, while AVA parameters match exactly;
- active default/server/application use is **UNVERIFIED**, because no production source reads the exact config path.

## Conversion evidence is not experimental accuracy

Direct same-tensor replay from six `final_model.keras` files to named TFLites succeeded on two real manifest-ordered images per model.

| Model | Maximum absolute difference | Interpretation |
|---|---:|---|
| Fixed NIMA | `0.000251` | Tight direct final-model parity on the bounded sample. |
| Practical A-LAMP | `0.009665` | Small scalar difference, but only two images. |
| Legacy NIMA | `0.027810` | Material distribution-component difference; source lineage must be named. |
| Practical RGNet | `0.027605` | Material scalar difference relative to its 0–1 output. |
| FLIVE image | `0.054344` | Moderate difference on 0–100 scale; only two images. |
| KonIQ | `0.442081` | Largest checked difference on 0–100 scale; only two images. |

The existing Mobile A-LAMP v2 1,024-row conversion table is stronger: FP32 Pearson `0.999983`, FP16 Pearson `0.999895`, FP16 Spearman `0.999865`, and FP16 maximum error `0.027695`. These are valid conversion-parity statistics. They do not establish AVA accuracy, mobile preprocessing correctness, on-device numerical parity, latency, or production suitability.

## Model-line validity conclusions

| Line | What the evidence supports | What it does not support |
|---|---|---|
| KonIQ / FLIVE image technical regressors | Reproducible current table metrics and real server/export artifacts | Paper-comparable accuracy or a correct pretrained-backbone input contract |
| FLIVE patch / combined FLIVE | Existence and exact five-crop server behavior | Independent held-out parent-image generalization |
| TOPIQ-lite, C6, RankCalib | Bounded candidate/research evaluations and explicit C6 no-go | A single combined performance claim or deployed replacement |
| MUSIQ practical/probes | Local MUSIQ-style experiments and server feature use | Official MUSIQ reproduction or equivalence of probes to external Kaggle variants |
| MDIQA official checkpoints | Hash-stable checkpoints and reproducible 100-image sanity arithmetic | Full benchmark reproduction, local training lineage, or mobile suitability |
| MDIQA students, TechIQA-Guard, DistortionGuard | Genuine experiments with explicit evaluation/no-go evidence | Successful production replacement; rejected lines must remain rejected |
| NIMA fixed | Correct 10-bin output semantics and reproducible 2,000-row metrics | Correct EfficientNetV2 input preprocessing or paper-split equivalence |
| Practical RGNet | Real graph modules and server/export artifacts | Defining RGNet-paper equivalence or a correct input contract |
| RGNet paper-v1/PIL | Defining paper-inspired components and reproducible 1,000-row AADB metrics | Official implementation identity or untouched paper protocol |
| Practical / Mobile A-LAMP | Real five-patch attention implementations; strong 1,024-row mobile conversion evidence for v2 | Original A-LAMP mechanism equivalence or accuracy inferred from parity |
| ICAA official/TF-native | Official strict weight load, working port, reproducible full-test metrics | Correct use by the supplied mobile-lab config or practical mobile deployability |
| Default Stage5 ensemble | Actual default server configuration and explicit score normalization | Use of the newer calibrated weights or certification of the legacy bundle |
| Calibrated aesthetic ensemble | Exact offline metric reconstruction | Untouched test validation or active runtime use |

## Final validity verdict

The audit accepts the replayed metrics as facts about their named local tables. It rejects using those facts as unqualified evidence of paper reproduction, unseen-parent generalization for FLIVE patch, unbiased final selection for the calibrated ensemble, or production correctness. Future claims become experimentally defensible only after contract-correct retraining/evaluation, group-disjoint deduplication, a locked final test set, and immutable run manifests that bind code, data, weights, preprocessing, predictions, and metrics.
