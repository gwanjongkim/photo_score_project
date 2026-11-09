# Resolution Test Results

## Outcome

The bounded evidence refutes several high-impact claims in each prior audit. It also confirms important defects that the primary audit left as risks rather than facts. No full training, production-source modification, replacement export, or external-app mutation was performed.

Status meanings in `resolved_disputes.tsv` apply to the exact claim stated in that row.

- `RESOLVED_CONFIRMED` — the stated claim is directly supported.
- `RESOLVED_REFUTED` — the stated claim is directly contradicted or falsely attributed.
- `PARTIALLY_RESOLVED` — the factual core is resolved but impact, provenance, or numeric judgment remains open.
- `UNRESOLVED` — supplied evidence cannot determine the claim.

## Decisive adjudications

| ID | Result | Status |
|---|---|---|
| D01 | MobileNetV2 checkpoints contain no internal rescale; `[0,1]` is fed where pretrained MobileNetV2 expects external `[-1,1]`. | RESOLVED_REFUTED for the primary mechanism |
| D02 | Fixed NIMA removed duplicate scaling but still lacks the `[-1,1]` transform required by EfficientNetV2 with preprocessing disabled. | RESOLVED_REFUTED for “coherent fixed path” |
| D03 | Practical RGNet has an active under-scaling contract defect; `[0,0.0039]` is only the intermediate range and causal impact remains unmeasured. | RESOLVED_CONFIRMED for the exact defect claim |
| D04 | Primary used “functionally equivalent” only for the narrow existence of graph reasoning, not the whole RGNet implementation. | RESOLVED_REFUTED |
| D05 | A real local ICAA TF-native port exists and the official checkpoint strict-loads and forwards. | RESOLVED_REFUTED for NOT_IMPLEMENTED |
| D06 | ICAA is `[MOS,color]`; validated color index 1/ImageNet preprocessing conflicts with lab index 0/zero-to-one, refuting the claim that semantics are wholly unknown. | RESOLVED_REFUTED |
| D07 | ICAA FP16 is 175,885,416 bytes, not 11.9 MB; canonical and candidate copy are byte-identical. | RESOLVED_REFUTED for 11.9 MB |
| D08 | FLIVE image parent IDs do not overlap, but nine exact byte duplicates do. FLIVE patch/all parent leakage is severe. | PARTIALLY_RESOLVED |
| D09 | Local AVA is seed-42 80/10/10 and cannot be called the paper protocol under either supplied paper interpretation. | PARTIALLY_RESOLVED |
| D10 | Current KonIQ manifests exactly use `MOS_zscore`, refuting the ordinary-MOS claim; exact historical checkpoint CLI remains a separate unknown. | RESOLVED_REFUTED |
| D11 | Logistic fitting changes PLCC slightly but is not universally mandated by the supplied primary-source baseline. | RESOLVED_REFUTED |
| D12 | AADB attributes are requirements of the AADB ranking network, not RGNet merely because RGNet evaluates on AADB. | RESOLVED_REFUTED |
| D13 | Default Stage5 normalizes score ranges before weighting; its value-inferred dispatch is nevertheless discontinuous. | RESOLVED_REFUTED for “no normalization” |
| D14 | Both auditors missed `quality_score_0_100`; all four technical metrics and the formula are independently reproducible. | RESOLVED_REFUTED |
| D15 | Compatible code and filenames do not prove exact historical execution; trainer-to-checkpoint lineage remains unknown. | RESOLVED_REFUTED |
| D16 | The blanket direct-`final_model.keras` descendant claim is false; older exports identify `best.weights.h5` rebuilds, while exact build lineage remains model-specific. | RESOLVED_REFUTED |
| D17 | Mobile A-LAMP has real 1,024-row conversion evidence, refuting the blanket 32/50-image claim; Dart parity remains a separate unknown. | RESOLVED_REFUTED |
| D18 | Repository default config uses legacy NIMA; a currently running server or mobile app was not supplied. | UNRESOLVED |
| D19 | A-LAMP mechanism presence/absence is resolved; whether a score is 30 or 40 remains rubric judgment. | PARTIALLY_RESOLVED |

## New findings missed by both auditors

1. `production_technical_predictions.csv` contains ground truth. On 4,967 paired rows the current table yields SRCC `0.795784`, PLCC `0.860414`, MAE `6.076832`, and RMSE `8.563833`; its 4/7–3/7 reconstruction is exact to `7.11e-15`.
2. FLIVE image manifests contain nine byte-identical cross-split groups and AADB contains four. Every group has conflicting labels.
3. The calibrated three-model ensemble's published held-out metrics replay exactly, but final variant/preparation selection uses the test metric. The “test” fold is therefore not a pristine final holdout.
4. The exported aesthetic ensemble config reproduces AVA calibration parameters exactly but not AADB parameters. It is not a faithful standalone reconstruction contract for the reported AADB result.
5. Practical A-LAMP has a linear head. The selector's value-inferred scale routing changes abruptly from `1.05 → 1.0` to `1.06 → 0.00667`, so slight out-of-range outputs can be misinterpreted.
6. Direct `final_model.keras` versus TFLite parity is materially looser than prior exporter-rebuild checks for several legacy models. Lineage must name `best.weights.h5` rebuilds where that is the actual export source.

## Test suite summary

| Test | Scope | Result |
|---|---|---|
| DATA-001 | Split paths, labels, FLIVE parents, cross-split byte hashes | PASS; material overlaps found |
| PREPROC-001 | Serialized Rescaling/Normalization and head semantics | PASS; no inspection failures |
| METRIC-002 | Technical metrics, raw/fitted PLCC, ensemble reconstruction | PASS; metrics reproduced and protocol flaws exposed |
| HASH-001 | Artifact hashes, copy equality, runtime references | PASS |
| PARITY-001 | Same tensors through six Keras/TFLite pairs | PASS execution; mixed parity outcomes |
| ICAA-001 | Strict official DAT checkpoint load and forward | PASS |
| CONTRACT-001 | ICAA contract and live FLIVE patch aggregation | PASS |
| DUPLICATE-001 | Labels on exact byte duplicates | PASS; all tested groups conflict |
| REPLAY-001 | NIMA/RGNet/ICAA/MDIQA metrics and Mobile A-LAMP parity | PASS; stored values reproduced |
| ARCH-001 | Defining paper-to-code component assertions | PASS after one corrected static detector rerun |

## Limit of resolution

The tests resolve repository facts. They cannot prove active external deployment, reconstruct absent historical commands, quantify the causal accuracy cost of preprocessing defects without retraining, or turn subjective 0–100 rubric scores into measured quantities. Those boundaries are carried into the final reports rather than hidden.
