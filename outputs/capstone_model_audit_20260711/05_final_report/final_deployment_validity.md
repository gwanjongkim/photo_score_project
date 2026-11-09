# Final Deployment Validity

## Deployment verdict

The repository proves a Python/Keras Stage5 configuration and multiple runnable TFLite artifacts. It does not prove a currently running production server, a byte-identical external Flutter asset, Dart preprocessing, device latency, memory use, or end-to-end application behavior. Therefore, “repository server model,” “mobile candidate,” and “deployed mobile model” must remain separate labels.

The immediate deployment decisions are:

- **KEEP WITH QUALIFICATION** — repository-local server paths that are directly configured and whose limitations are disclosed.
- **CANDIDATE ONLY** — runnable exports with useful parity/handoff evidence but no external-runtime proof.
- **NO-GO** — artifacts with a known contract break, explicit rejection, or no deployment chain.
- **UNVERIFIED** — external or optional paths for which runtime evidence is insufficient.

## Runtime layers

| Layer | Directly proven | Not proven |
|---|---|---|
| Repository server configuration | `configs/stage5_reference.json` names composition, KonIQ, FLIVE image/patch, legacy NIMA, practical A-LAMP, practical MUSIQ, practical RGNet, and pairwise Keras checkpoints | A current process uses that file; operational availability, latency, and monitoring |
| Repository inference implementation | Bundle loading, five-crop FLIVE patch aggregation, score normalization, weighting, pairwise/diversity options | Correctness under every missing-signal combination; production traffic behavior |
| Repository TFLite execution | All six resolution-test TFLites loaded and invoked; ICAA strict load/FP16 invoke succeeded | External app decoder, resize, channel order, threading, delegates, and postprocessing |
| Mobile handoff copies | Several candidate copies are byte-identical to canonical artifacts | Exact same bytes inside an external Flutter APK/IPA or active runtime |
| Calibrated ensemble | Offline scores, formula, and held-out metrics replay | Production source consumption; pristine final holdout; faithful AADB calibration config |

## Model-by-model deployment disposition

| Model line | Repository role | Deployable artifact/contract evidence | Final disposition | Evidence level and reason |
|---|---|---|---|---|
| KonIQ Mobile | Default Python server technical signal; TFLite handoff | Runnable FP32 TFLite, scalar raw MOS-like output, divide by 100 postprocess | KEEP WITH QUALIFICATION on server; CANDIDATE ONLY on mobile | STRONG artifact evidence; preprocessing is mismatched to pretrained MobileNetV2 and direct final-Keras/TFLite max difference is `0.442081` on two images |
| FLIVE Image Mobile | Default server technical signal; TFLite handoff | Runnable FP32 TFLite, scalar raw MOS-like output | KEEP WITH QUALIFICATION on server; CANDIDATE ONLY on mobile | STRONG; normalization mismatch and 9 conflicting-label byte-duplicate groups; direct max difference `0.054344` |
| FLIVE Patch | Default server guard signal | Keras only; five crops produce mean/min/std | KEEP WITH QUALIFICATION on server; NO-GO for mobile claim | PROVEN server contract; 97.20% test-parent overlap with train and no TFLite/mobile chain |
| Default technical ensemble | Hard-coded selector policy over auto-resolved technical signals | Source weights `0.32/0.24/0.22/0.12/0.10` with renormalization | KEEP WITH QUALIFICATION for repository server | PROVEN source/model-bundle resolution; JSON `selector_args` are not applied by the loader, value-inferred dispatch is discontinuous, and live runtime is unknown |
| Image-only 4/7–3/7 technical reconstruction | Default-selector two-signal fallback plus offline table | `(0.32K+0.24F)/0.56`, exactly reconstructs stored predictions | PROVEN two-signal fallback behavior; offline evaluation only, not proven standalone deployment | PROVEN formula/metrics on the named table; uses only two signals and live occurrence is unknown |
| Mobile-helper 0.60/0.40 technical subcomponent/fallback | Mobile-oriented Python helper whose parent policy is 0.55 AADB aesthetic plus 0.45 technical | Source proves the technical component is 0.60 KonIQ plus 0.40 FLIVE Image and becomes final when aesthetic output is absent | UNVERIFIED for external mobile deployment | PROVEN repository code and fallback; no policy-specific metrics or external app implementation |
| TOPIQ-lite mixed112 | Shadow/log-only candidate | FP32/FP16 exports and parity artifacts | CANDIDATE ONLY | STRONG export evidence; historical trainer-option drift and no active runtime |
| C6 TOPIQ-KonIQ | Rejected replacement experiment | Output-scoped capped formula only | NO-GO | STRONG explicit rejection; absent production implementation |
| TOPIQ RankCalib | Experimental calibration/ranking | Experiment checkpoints/results | UNVERIFIED/EXPERIMENTAL | PARTIAL lineage; no active runtime contract |
| Practical MUSIQ AADB | Default server bundle signal, despite AADB semantics | Keras multi-input/token assembly | KEEP WITH STRONG QUALIFICATION on server | PROVEN config; technical use of an AADB-trained model is a semantic caveat; no core mobile contract |
| Local MUSIQ probes | Experimental technical signals | Probe TFLites exist | NO-GO for production claim | PARTIAL export lineage; exact subset/export invocations unknown |
| Kaggle MUSIQ variants | External sanity benchmark | External model handles only | UNVERIFIED for deployment | No local checkpoint, export, or runtime |
| MDIQA teacher | Teacher/reference | Local PyTorch checkpoints and benchmark wrappers | NO-GO for mobile; experimental server/reference only | STRONG checkpoint/evaluation evidence; no export/runtime chain |
| MDIQA 14k student | Rejected mobile student | FP32/FP16 TFLite and parity exist | NO-GO | Explicit final rejection overrides artifact availability |
| MDIQA 50k student | Rejected larger student | Keras/evaluation only | NO-GO | Explicit rejection; export intentionally stopped |
| TechIQA-Guard | Rejected replacement | Keras/evaluation only | NO-GO | Explicit no-go and no export |
| DistortionGuard | Rejected guard line | Keras/evaluation only | NO-GO | B3 export and Flutter integration explicitly rejected |
| NIMA legacy | Default server aesthetic component; legacy TFLite copies | Runnable `[1,10]` distribution export | NO-GO for new deployment; legacy server use requires remediation | PROVEN default reference and byte-identical copies; known duplicate/under-scaling defect; direct max distribution difference `0.027810` |
| NIMA fixed | Candidate replacement | FP32/FP16, `[1,10]`, fixed-Keras/FP32 max difference `0.000251` on two images | CANDIDATE ONLY | STRONG parity and byte identity; duplicate scaling is removed but required `[-1,1]` transform remains absent; not default runtime |
| Practical RGNet | Default server aesthetic component; TFLite copies | Runnable 256×256 scalar sigmoid `[0,1]` | KEEP ONLY WITH STRONG QUALIFICATION on server; CANDIDATE ONLY on mobile | PROVEN config/copies; active preprocessing defect; direct max difference `0.027605` |
| RGNet paper-v1 family | Experimental structural branch | Keras teachers/ablations | EXPERIMENTAL | No active runtime; model family should not be substituted silently for practical RGNet |
| RGNet PIL candidate | Mobile candidate | FP16 file and parity/mobile-like outputs | CANDIDATE ONLY | STRONG chain and byte-identical handoff copy; external runtime unknown |
| Practical A-LAMP | Default server aesthetic component; TFLite copies | Two inputs, five patches plus global image, linear scalar output | KEEP WITH QUALIFICATION on server; CANDIDATE ONLY on mobile | PROVEN config/copies; direct max difference `0.009665`; linear output can leave expected range |
| Mobile A-LAMP v2 | Mobile candidate | 2.93 MB FP16, two inputs, `[0,255]`, sigmoid binary probability; 1,024-row FP16 Pearson `0.999895` | CANDIDATE ONLY, strongest compact mobile artifact | STRONG export/parity; external Dart runtime and patch-generation parity unknown |
| A-LAMP teacher variants | Experimental teachers/rejected branches | Keras experiments only | NO-GO for deployment claim | No active runtime; some branches explicitly failed/rejected |
| ICAA17K DAT PyTorch | Official reference/teacher | 348.9 MB checkpoint strict-loads and forwards `[1,2]` | REFERENCE ONLY | PROVEN load; not a mobile artifact |
| ICAA FP16 | Mobile candidate with broken lab contract | 175,885,416-byte FP16, input `[1,224,224,3]`, output `[1,2]`, SavedModel/FP16 max difference `0.002588` on 64 images | NO-GO until contract and size are addressed | PROVEN artifact/parity; lab uses wrong zero-to-one/index 0 instead of validated ImageNet/index 1; active app unknown |
| Stage5 AADB student | Historical rejected candidate | 3.41 MB FP32 scalar TFLite | NO-GO | Artifact exists, but later rejection is the current decision and runtime is unproven |
| Default Stage5 aesthetic ensemble | Repository bundle plus hard-coded selector policy | Composition + legacy NIMA + practical A-LAMP/RGNet; pairwise is CLI-gated, and MUSIQ belongs to the technical component | KEEP WITH QUALIFICATION for repository server only | PROVEN source/config wiring; JSON `selector_args` are not applied, and the policy inherits legacy/component preprocessing defects |
| Calibrated 0.37/0.41/0.22 ensemble | Offline candidate | Formula and published metrics replay exactly | NO-GO as a production claim; experimental candidate only | STRONG offline evidence, but selection uses test metrics, AADB config calibration drifts, and no runtime reads the config |
| AADB composition regressor | Default server component | Keras scalar model | KEEP WITH QUALIFICATION on server | PROVEN config; exact historical invocation and mobile path are absent |
| Pairwise AADB comparator | Optional server reranker | Keras checkpoint and validation-reference config | KEEP ONLY when explicitly enabled and documented | PROVEN configuration; no mobile path and exact original invocation absent |
| VILA/Gemma | Optional teacher/rerank/explanation | Local/API inference paths and cached outputs | UNVERIFIED/OPTIONAL | Not a core mobile model; availability, latency, and external dependencies vary |

## Exact mobile contracts that are safe to state

| Artifact | Input | Output | Required postprocess | Contract status |
|---|---|---|---|---|
| `koniq_mobile.tflite` | RGB NHWC float32 `[1,224,224,3]`, repository metadata says `/255` | `[1,1]` raw MOS-like technical score | `clip(score/100,0,1)` if a unit score is needed | Artifact contract PROVEN; pretrained-backbone normalization scientifically defective |
| `flive_image_mobile.tflite` | Same as KonIQ | `[1,1]` raw MOS-like technical score | `clip(score/100,0,1)` | Artifact contract PROVEN; same normalization defect |
| `nima_mobile.tflite` legacy | RGB NHWC float32 `[1,224,224,3]`, `/255` | `[1,10]` softmax over scores 1..10 | `sum(p[i]*(i+1))`; optional `(mean-1)/9` | Contract PROVEN; legacy preprocessing path |
| NIMA fixed FP16 | RGB NHWC float32 `[1,224,224,3]`, current metadata says `/255` | `[1,10]` softmax | Same expected-score calculation | Contract PROVEN; residual pretrained-backbone normalization mismatch |
| Practical RGNet TFLite | RGB NHWC float32 `[1,256,256,3]`, `/255` | `[1,1]` sigmoid score | Squeeze; optional clip `[0,1]` | Contract PROVEN; embedded under-scaling defect |
| Practical A-LAMP TFLite | Global `[1,384,384,3]` and patches `[1,5,224,224,3]`, float32 `/255` | `[1,1]` linear scalar | Squeeze; range must be calibrated/checked, not inferred solely from value | Shape/head PROVEN; external patch generation UNVERIFIED |
| Mobile A-LAMP v2 FP16 | Full image `[1,384,384,3]`, five patches `[1,5,224,224,3]`, float32 `[0,255]` | `[1,1]` sigmoid probability for AVA mean >5 | Squeeze probability | Export contract STRONG; external patch/Dart parity NOT_AVAILABLE and UNVERIFIED |
| ICAA FP16 | RGB float32 `[1,224,224,3]`, external ImageNet normalization | `[1,2]` ordered `[MOS,color]` | Use index 1 for color aesthetic | Validated contract PROVEN; existing lab config REFUTED |
| Stage5 student FP32 | RGB NHWC float32 `[1,224,224,3]`, `/255` | `[1,1]` distilled aesthetic score `[0,1]` | Squeeze | Metadata contract PROVEN; model rejected |

## Direct parity evidence

The resolution test used the same Python-created tensors for Keras and TFLite on two real images per model. Execution passed for all six tested pairs.

| Pair | Max absolute difference | Interpretation |
|---|---:|---|
| KonIQ `final_model.keras` vs repository TFLite | 0.442081 | Materially looser than a normal conversion check; TFLite metadata names a rebuild from best weights |
| FLIVE image final Keras vs repository TFLite | 0.054344 | Same sibling-lineage caveat |
| Legacy NIMA final Keras vs repository TFLite distribution | 0.027810 | Same sibling-lineage caveat; compare output vectors, not only expected score |
| NIMA fixed final Keras vs FP32 TFLite distribution | 0.000251 | Close on the two tested images; full distribution/device parity still needed |
| Practical RGNet final Keras vs repository TFLite | 0.027605 | Same sibling-lineage caveat |
| Practical A-LAMP final Keras vs repository TFLite | 0.009665 | Same sibling-lineage caveat |

Additional existing evidence is stronger in sample count for two candidates: Mobile A-LAMP FP16 has 1,024-row Pearson `0.999895`, and ICAA SavedModel-to-FP16 has maximum absolute difference `0.002588` on 64 real images. Neither test establishes Dart/mobile-decoder parity.

## Blocking defects before any mobile-production claim

1. Bind each release to a source-weight hash, export command, TFLite hash, and external app asset hash.
2. Correct and retrain or revalidate the MobileNetV2/EfficientNetV2 input contracts; do not repair only the app-side tensor while retaining weights trained under a different contract.
3. Replace the ICAA weight-lab contract with ImageNet normalization and output index 1, then rerun end-to-end parity; assess whether a 175.9 MB asset is acceptable on target devices.
4. Test the exact Dart JPEG decoder, resize, RGB ordering, multi-input patch generator, interpreter/delegate, and postprocess against frozen Python tensors.
5. Benchmark latency, peak memory, binary size, warm/cold start, thermal behavior, and failure handling on declared device tiers.
6. Remove value-inferred scale dispatch from the selector or make model scale explicit; the observed `1.05 → 1.0` and `1.06 → 0.00667` mapping is unsafe.
7. Do not promote the calibrated ensemble until a validation-only selection protocol, a frozen untouched test set, faithful calibration config, and a real runtime consumer exist.

## Safe final wording

Safe: “The repository contains runnable server checkpoints and several verified TFLite candidates with documented tensor contracts.”

Unsafe: “These models are deployed in the app” or “Python-to-mobile parity is proven.”

Required qualification: “External app byte identity, Dart preprocessing, device parity, and active runtime use were not available to this audit.”
