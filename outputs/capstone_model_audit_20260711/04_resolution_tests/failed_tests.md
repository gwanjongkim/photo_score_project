# Failed, Superseded, and Blocked Resolution Tests

## Executed command failures

| ID | Command | Result | Consequence |
|---|---|---|---|
| DIAG-ERR-001 | `./ .venv_gpu/bin/python --version` | Exit 126: `./` is a directory. This was an accidental malformed environment diagnostic appended after CSV header inspection. | No evidence impact. All resolution scripts used the explicit `./.venv_gpu/bin/python` path successfully. |
| VALIDATION-PRE-001 | `./.venv_gpu/bin/python outputs/capstone_model_audit_20260711/04_resolution_tests/validate_audit_outputs.py` | Exit 1: dispute D16 cited the non-literal wildcard `exports/tflite/*.verify.json`, which failed exact evidence-path resolution. Every other preliminary check passed. | The D16 row was repaired to cite five concrete verification JSON files; the validator is rerun after finalization. |
| INSPECT-ERR-001 | `head`/`wc` on `00_inventory/model_inventory.tsv` | The assumed TSV name did not exist; the actual inventory is `00_inventory/model_inventory.md`. The surrounding read-only inspection continued and its final shell status was zero. | No evidence impact. The actual Markdown inventory was located and read in full immediately afterward. |

No purpose-built resolution script exited non-zero. TensorFlow emitted CUDA initialization warnings and used CPU/XNNPACK; every affected test exited zero and wrote its result file.

## Superseded test logic

| ID | Test | Problem found | Correction |
|---|---|---|---|
| METRIC-001 | First run of `test_metrics_and_ensembles.py` | It compared metrics recomputed from `production_technical_predictions.csv` against a stored metrics row generated from the distinct `official_signal_predictions.csv` protocol snapshot. The join itself proved the tables were not row-identical. | METRIC-002 recomputed each table against its own ground truth and compares the stored row only to `official_signal_predictions.csv`. The corrected output is `metric_and_ensemble_results.json`. |
| PREPROC-DETECT-001 | First static source-contract run in `test_preprocessing_and_outputs.py` | The detector searched for `convert_image_dtype(image, tf.float32)`, while the actual NIMA dataset source uses the variable name `img`. It therefore emitted a false `false` value for one static flag. | PREPROC-002 changed only the detector token and reran the complete script. The final JSON records `nima_dataset_convert_image_dtype_float32: true`; serialized-model findings did not change. |
| ARCH-DETECT-001 | First static LSE check in `test_architecture_claims.py` | The detector looked for a generic LSE spelling and missed the custom `RegionScoreAggregation(aggregation="lse")` implementation in `rgnet_paper_v1.py`. | ARCH-002 checks the actual class/config pattern and reran the script. The final JSON records paper-v1 LSE as `true`. |
| DATA-TEXT-001 | First AVA bounded-scope reason in `test_dataset_and_contracts.py` | The explanatory text stated `255,555` rows instead of the observed `255,508`; computed counts and executed checks were unaffected. | DATA-002 corrected the explanatory count and reran the script. The final JSON states `255,508`; all evidence findings remain unchanged. |

The first output was overwritten only inside `04_resolution_tests/`; its command and reason for supersession remain recorded in `commands_executed.txt`.

## Bounded tests not executed or unable to close the claim

| ID | Proposed test | Status/reason | Consequence |
|---|---|---|---|
| BLOCK-001 | Full AVA byte-hash duplicate audit across 255,508 manifest rows | Not run to preserve the “small, bounded test” constraint. Exact-path overlap and full label-distribution checks were run. | AVA content/perceptual duplication remains unknown. |
| BLOCK-002 | Retrain or ablate corrected MobileNetV2/NIMA/RGNet preprocessing | Prohibited full retraining and production-source mutation. | Tensor contracts are resolved; causal accuracy loss remains unquantified. |
| BLOCK-003 | Verify active Flutter assets, Dart preprocessing, and current server process | External checkout/build/runtime was not supplied. | External mobile and live-server deployment identity remains `UNRESOLVED`. |
| BLOCK-004 | Recover original training CLI, environment, seed, and target argument from checkpoint bytes | Historical execution records are absent and cannot be reconstructed non-destructively. | Trainer-to-checkpoint links remain compatible/inferred rather than proven. |
| BLOCK-005 | Resolve NIMA paper split wording from a newly retrieved paper | The adjudication stayed within the supplied evidence set, whose baseline and cross-check conflict. | Exact 80/20 versus 250k/5k wording remains unresolved; the local 80/10/10 evaluation is non-equivalent under either interpretation. |
| BLOCK-006 | Re-run the 435.9-second MDIQA 900-prediction benchmark | Not necessary: supplied raw predictions had zero recorded failures, hashes matched, and all nine stored group metrics replayed to floating-point precision. | Checkpoint/inference evidence is accepted for the bounded audit, not as a new full benchmark. |
| BLOCK-007 | Deterministically reconvert every TFLite from every final Keras checkpoint | Would create large redundant exports; prior exporters identify `best.weights.h5` rebuild sources. | Same-input replay resolves final-model parity only partially; exact converter-build provenance remains model-specific. |
| BLOCK-008 | On-device latency/thermal profiling for ICAA and other candidates | No Android/Flutter runtime in scope. | Production suitability uses artifact size/contracts and remains conservative. |

## Inconclusive but useful outputs

- Core final-model-to-TFLite replay executed successfully but was not uniformly close. This is a result, not a test failure. It proves that several TFLites should be linked to exporter rebuilds from `best.weights.h5`, not casually to `final_model.keras`.
- Diagnostic logistic fitting executed successfully. It demonstrates a numerical change, but it cannot establish a paper requirement absent a method-specific source.
