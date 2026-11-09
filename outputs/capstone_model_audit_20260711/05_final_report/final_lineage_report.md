# Final Model Lineage Report

## Verdict

Repository-local server references are directly provable for the Stage5 Keras bundle. Several TFLite copies are byte-identical to their canonical artifacts. Those facts do **not** prove that an external Flutter application contains or executes the same bytes. For the older `exports/tflite/` models, metadata identifies a rebuild from `best.weights.h5`; direct comparisons to `final_model.keras` therefore test sibling artifacts rather than a parent-child export and show materially different outputs in several cases.

## Lineage-link status definitions

- These labels describe provenance transitions, not the controlled claim-evidence levels used in the claim registers.
- **PROVEN** — direct code/config reference, executed load/invoke, or independently recomputed hash proves the link.
- **STRONG** — aligned metadata, source, and run artifacts support the link, but an immutable build invocation is missing.
- **PARTIAL** — endpoints exist and some transitions are grounded, with at least one material missing link.
- **INFERRED** — naming or chronology suggests the link without direct evidence.
- **UNKNOWN_LINK** — no supplied evidence proves the provenance transition.
- **NOT_AVAILABLE** — the necessary external evidence source, such as the submitted Flutter build, was not supplied.
- **UNVERIFIED** — the claim cannot be determined from the available evidence; this is a claim state rather than a positive lineage link.

No transition is upgraded solely from a filename or modification time.

## Hash-anchored artifacts

| Model/artifact | Bytes | SHA-256 | What the hash proves | Status |
|---|---:|---|---|---|
| KonIQ `best.weights.h5` | 48,574,040 | `c1a8775e2c77f23843a322439038c1be38f19fdb51316357632c9e7b4e77dc8a` | Identity of the documented rebuild source weights | PROVEN |
| KonIQ `final_model.keras` | 48,725,759 | `932099ed846e7467ff8be1d3e9a010bbc902eccc48aae2677d58f4b29d085546` | Identity of the default server checkpoint | PROVEN |
| KonIQ `koniq_mobile.tflite` | 9,545,852 | `67faf88050c73ba77cc76e93f02b854095e483b8dff72577cb03f55199b8be56` | Identity of the repository export | PROVEN |
| FLIVE image `best.weights.h5` | 48,574,040 | `59df74ee17196e5747c2d2c560ae70a79c7fb95447ba45535e2dd544c71acd8b` | Identity of the documented rebuild source weights | PROVEN |
| FLIVE image `final_model.keras` | 48,725,759 | `e3af51c6eae714d66c2d1dbe6838099cecb56b3247f04e688550aa47290d982a` | Identity of the default server checkpoint | PROVEN |
| FLIVE image `flive_image_mobile.tflite` | 9,545,852 | `58aee1e11f6ec2baf6f9705c457b3bd2fa3e22e7151582ebae3e68c838f5ebdb` | Identity of the repository export | PROVEN |
| FLIVE patch `final_model.keras` | 48,725,759 | `f1b5f886a6fe01cab7dbac1cae8faed5c5d2678f73d6609078571962b8bfa177` | Identity of the server-only patch checkpoint | PROVEN |
| NIMA legacy `final_model.keras` | 125,352,765 | `bb20e6e06840a2d01caf42bf9e233358dbe92515187ca5e5de362b65164ae66c` | Identity of the checkpoint in the default server config | PROVEN |
| NIMA legacy `nima_mobile.tflite` | 24,765,068 | `48ed5a9f1e82e3f162c75ac6f6931004041736a4641002d282408d8fecf8800a` | All four audited repository copies are byte-identical | PROVEN |
| NIMA fixed `final_model.keras` | 125,347,783 | `513bab97d3bad2bd9cd03922264d5cba74cd9a5a5136c1e35e2a61e1ed9e7002` | Identity of the fixed candidate checkpoint | PROVEN |
| NIMA fixed FP16 | 12,495,220 | `ccbb51123f8414fdb40e2ba6a9c4a3b7be11cbf00e0bd14442923da0de46d255` | Canonical and `forWeights2` copies are byte-identical | PROVEN |
| Practical RGNet `final_model.keras` | 131,951,528 | `49b7f0fa768fa85068d4b4c69633588dc49158728c9119f97a161953404937b8` | Identity of the default server checkpoint | PROVEN |
| Practical RGNet TFLite | 26,085,072 | `b81461fe49d1ff4ff83ccccdc8bbdee958558552472f4d5db74a6217e24cdeb2` | All three audited repository copies are byte-identical | PROVEN |
| RGNet PIL FP16 | 38,934,148 | `1d3b19a937581947400248d8a29e0e3125877bb89d9833cd42dcfe3cb48e7418` | Canonical and `forWeights2` copies are byte-identical | PROVEN |
| Practical A-LAMP `final_model.keras` | 191,908,754 | `9ad9199a844138f1e38ee2805e2111e3891d9574f36ad3420bb995d604ee68e8` | Identity of the default server checkpoint | PROVEN |
| Practical A-LAMP TFLite | 39,158,340 | `8b579a0cabc3e51a548f8e3a25ab4ee9f8ecc016632764e11b38ff9a58f288c7` | All three audited repository copies are byte-identical | PROVEN |
| Mobile A-LAMP v2 FP16 | 2,930,400 | `70cbc5c75d5468a3bfd825047f83eda32f4a8da36150f6905013e967c90efa43` | Parity-bundle and `forWeights2` copies are byte-identical | PROVEN |
| Official ICAA checkpoint | 348,884,485 | `063059b0b62aa3e5e352b1bd642ac6cd1bbaf1c729343787c87168ba1524f282` | Identity of the strict-loaded PyTorch weight | PROVEN |
| ICAA FP16 TFLite | 175,885,416 | `10ae68a3283bdabf05ecfa8b679e1e89865f3c960687ce5495500455bbd111ff` | Canonical and `forWeights2` copies are byte-identical; the 11.9 MB claim is false | PROVEN |
| Stage5 student `final_model.keras` | 17,939,905 | `3d8850d996a6376dded969bce17b7afddc4241fceccdd3602d881ec2ddc95a42` | Identity of the full AADB student checkpoint | PROVEN |
| Stage5 student FP32 TFLite | 3,411,060 | `0e983dc26ee0ecc2f1140d6108e68b0fe5a96f095831db50c501b6b499d63940` | Identity of the rejected mobile handoff artifact | PROVEN |

The hashes above prove file identity only. They do not prove the training data, command, or active external runtime unless a separate row below does so.

## Final lineage chains

### Technical IQA

| Model line | Evidence-backed chain | Link strength | Unproven or broken transition |
|---|---|---|---|
| KonIQ Mobile | Raw table → current seed-42 manifests using `MOS_zscore` **[PROVEN]** → generic trainer is compatible **[STRONG]** → checkpoint **[PARTIAL]** → rebuild metadata names `best.weights.h5` → TFLite **[STRONG]** → default Keras server config **[PROVEN]** | PARTIAL overall | Exact historical trainer command/target argument; cryptographic build record; external app asset/runtime |
| FLIVE Image Mobile | Raw image labels → custom seed-42 image manifests **[PROVEN]** → compatible trainer **[STRONG]** → checkpoint **[PARTIAL]** → rebuild from named best weights → TFLite **[STRONG]** → default Keras server config **[PROVEN]** | PARTIAL overall | Exact training invocation; byte-identical external asset; external decoder/runtime |
| FLIVE Patch | Raw patch labels → patch manifests **[PROVEN]** → compatible trainer/checkpoint **[PARTIAL]** → default config and five-crop mean/min/std server path **[PROVEN]** | PARTIAL overall | Severe parent leakage limits scientific lineage; no export/mobile link exists |
| Default technical ensemble | Keras KonIQ + FLIVE image + FLIVE patch + practical MUSIQ → `stage5_reference.json` bundle resolution **[PROVEN]** → hard-coded selector weights with available-signal renormalization **[PROVEN]** | PROVEN for repository source/config | JSON `selector_args` are not applied by `configure_aesthetic_scoring`; current server/app use is UNVERIFIED; image-only 4/7–3/7 and the helper's 0.60/0.40 technical subcomponent/fallback are different policies |
| Image-only 4/7–3/7 technical reconstruction | Default selector weights `0.32/0.24` → two-available-signal renormalization to `4/7` and `3/7` **[PROVEN]** → 4,967-row prediction/ground-truth table and exact formula/metric replay **[PROVEN]** | PROVEN for formula and named table | The table is an offline local protocol snapshot; occurrence as a standalone live policy and external app use are UNVERIFIED |
| Mobile-helper 0.60/0.40 technical subcomponent/fallback | `predict_mobile_aesthetic.py` default parent policy mixes AADB aesthetic/technical at `0.55/0.45` **[PROVEN]** → technical component uses KonIQ/FLIVE Image at `0.60/0.40` **[PROVEN]** → technical score becomes final when aesthetic output is absent **[PROVEN]** | PROVEN for repository source | No policy-specific metric table, external Flutter implementation, or active runtime; it is not the helper's entire parent policy |
| TOPIQ-lite mixed112 | Mixed manifests → compatible trainer/artifacts **[PARTIAL]** → output-scoped export/parity **[STRONG]** → shadow/log-only candidate | PARTIAL | Preserved command uses an `--init_weights` option absent from the current parser; no active runtime |
| C6 | Cached TOPIQ/KonIQ predictions → capped formula → parity/decision reports → rejection | STRONG | No production source implements C6; active deployment is absent by design |
| RankCalib | Stage-specific source/configs → experiment outputs | PARTIAL | No unified immutable run-to-model record or active runtime |
| MUSIQ AADB practical | AADB manifests → practical source/trainer → Keras checkpoint **[PARTIAL]** → model-bundle resolution and technical `musiq_score` use **[PROVEN]** | PARTIAL | Exact original invocation is absent; no mobile export/runtime is established; AADB semantics are a technical-signal caveat |
| Local MUSIQ technical probes | Bounded technical subsets → local probe checkpoints/TFLites/results **[PARTIAL]** | PARTIAL | Exact subset/training/export invocations are absent; probes are not the practical AADB server model or official Kaggle MUSIQ |
| Kaggle MUSIQ | External model handles → sanity-run outputs | PARTIAL | No local checkpoint, export, or runtime artifact |
| MDIQA teacher | Official-family source/config → downloaded checkpoints → local benchmark wrappers/results | STRONG from checkpoint onward | Official training data/command to downloaded bytes is an UNKNOWN_LINK; no mobile export |
| MDIQA 14k student | Cached teacher predictions → manifest → local trainer → Keras → FP32/FP16 TFLite/parity → final no-go | STRONG | External runtime absent; line is explicitly rejected |
| MDIQA 50k student | Teacher manifest → grouped/stratified local splits → trainer/variants → Keras/evaluation → final no-go | STRONG | No TFLite was produced; line is explicitly rejected |
| TechIQA-Guard | Constructed datasets → local trainer/checkpoints → evaluation → no-go | STRONG | No export/runtime link; rejection is the terminal state |
| DistortionGuard | Synthetic Stage A → authentic Stage B/B2/B3 → evaluations → export/Flutter rejection | STRONG | No TFLite/runtime link; rejection is the terminal state |

### Aesthetic and color IAA

| Model line | Evidence-backed chain | Link strength | Unproven or broken transition |
|---|---|---|---|
| NIMA legacy | Processed AVA → historical training **[UNKNOWN_LINK]** → legacy checkpoint → default Stage5 Keras reference **[PROVEN]**; named best weights → rebuild TFLite **[STRONG]** | PARTIAL | Training invocation absent; preprocessing defective; external runtime UNVERIFIED |
| NIMA fixed | Current AVA manifests → fixed source/trainer → checkpoint **[PARTIAL]** → fixed FP32/FP16 artifacts and 2,000-row replay **[STRONG]** → `forWeights2` copy **[PROVEN]** | PARTIAL | Raw AVA annotation and exact run command are UNKNOWN_LINK; no active default config; external runtime UNVERIFIED |
| Practical RGNet | AADB official-list builder → manifests **[PROVEN]** → compatible trainer → checkpoint **[PARTIAL]** → default server reference **[PROVEN]**; named best weights → TFLite rebuild **[STRONG]** | PARTIAL | Exact invocation is an UNKNOWN_LINK; preprocessing defect; external runtime UNVERIFIED |
| RGNet paper-v1 family | AADB manifests → dedicated architecture/config/trainer/evaluator → teacher and ablation checkpoints/results **[PARTIAL/STRONG by variant]** | PARTIAL overall | No single canonical family winner, deterministic export chain, or active runtime applies to the entire family |
| RGNet PIL-resize candidate | AADB manifests → dedicated PIL config/trainer/evaluator → checkpoint and raw predictions → dedicated FP16 export/parity → byte-identical `forWeights2` copy | STRONG | Active Flutter/runtime reference is NOT_AVAILABLE and UNVERIFIED |
| Practical A-LAMP | AADB manifests/patch builder → compatible trainer → checkpoint **[PARTIAL]** → default server reference **[PROVEN]**; rebuild from best weights → TFLite **[STRONG]** | PARTIAL | Exact train invocation is an UNKNOWN_LINK; final-model/TFLite sibling mismatch; external runtime UNVERIFIED |
| Mobile A-LAMP v2 | AVA manifests → saliency/patch JSONL **[PARTIAL]** → explicit trainer → `best_val_auc_model.keras` → exact export command → FP32/FP16 and 1,024-row parity → byte-identical `forWeights2` copy | STRONG | Exact full patch-generation command is an UNKNOWN_LINK; Dart/app runtime is NOT_AVAILABLE and UNVERIFIED |
| A-LAMP teacher variants | Dedicated architecture/trainer/evaluator/config → experiment outputs | PARTIAL/STRONG by variant | Failed/rejected branches do not join production; no runtime reference |
| ICAA17K official DAT and TF-native port | Official ICAA17K split/code **[STRONG]** → official checkpoint training **[UNKNOWN_LINK]** → strict local load/forward **[PROVEN]** → TF weight mapping and SavedModel/full-test outputs **[STRONG]** | STRONG from official checkpoint onward | Exact official training invocation/revision is an UNKNOWN_LINK; complete layer-by-layer framework equivalence is UNVERIFIED |
| ICAA FP16 mobile artifact | TF-native SavedModel → FP16 conversion and 64-image parity **[STRONG]** → canonical TFLite → byte-identical `forWeights2` copy **[PROVEN]** | STRONG | Lab config uses wrong preprocessing/index; 175,885,416-byte device feasibility and external runtime are UNVERIFIED |
| Stage5 student | AADB manifests → default teacher bundle → teacher-label generation **[PARTIAL]** → student config/trainer → checkpoint/export **[STRONG]** → later rejection | PARTIAL | Teacher-label provenance is an UNKNOWN_LINK; external runtime UNVERIFIED; rejection is current project status |
| AADB composition | AADB manifests → compatible generic trainer → checkpoint → default server config | PARTIAL | Exact original invocation absent; no mobile link established |
| Pairwise AADB | Pair manifests/trainer → checkpoint → default config → optional rerank when enabled | PARTIAL | Exact original invocation absent; no mobile link; runtime use depends on configuration |
| Default aesthetic ensemble | Composition + legacy NIMA + practical A-LAMP + practical RGNet → bundle auto-resolution **[PROVEN]** → hard-coded aesthetic selector weights **[PROVEN]**; pairwise model loads only when CLI-enabled | PROVEN for repository source/config | Practical MUSIQ belongs to the technical component; JSON `selector_args` are not applied here; this is not the calibrated candidate; current live process is UNVERIFIED |
| Calibrated three-model ensemble | Reused aligned AVA/AADB predictions → deterministic weight search → 0.37/0.41/0.22 recommendation/config | STRONG for offline analysis | Selection consults test metrics; AADB config calibration drifts from replay; no source reads the exact config path; active runtime is UNVERIFIED |
| VILA/Gemma | Model/API inference paths → cached teacher/rerank/explanation outputs | PARTIAL | No core mobile IQA/IAA lineage or required runtime is proven |

## Export-parent adjudication

The canonical `exports/tflite/` metadata for KonIQ, FLIVE image, legacy NIMA, practical RGNet, and practical A-LAMP states `rebuild` or `rebuild_from_weights`, naming `best.weights.h5`. Consequently:

1. The rebuild source and TFLite association is **STRONG**, not cryptographically PROVEN, because the original build was not reproduced into the same hash during this audit.
2. `final_model.keras` and TFLite are sibling products that can contain different training states or numeric policies.
3. Resolution-test same-tensor differences against `final_model.keras`—KonIQ `0.442081`, FLIVE image `0.054344`, legacy NIMA distribution `0.027810`, practical RGNet `0.027605`, and practical A-LAMP `0.009665` maximum absolute difference on two real images—must not be described as conversion error alone.
4. Fixed NIMA FP32 showed `0.000251` maximum absolute difference on two images. Mobile A-LAMP’s stored 1,024-row FP16 evidence reports Pearson `0.999895`; its export command names `best_val_auc_model.keras` explicitly.

## Runtime lineage decision

| Runtime claim | Final status | Basis |
|---|---|---|
| Repository default Stage5 config names the listed Keras checkpoints | PROVEN | Direct config parse and source references |
| Repository selector contains default technical and aesthetic policies | PROVEN | Direct source/static resolution test |
| Calibrated 0.37/0.41/0.22 ensemble is the default runtime | REFUTED | No production source reads the audited config path |
| Fixed NIMA replaces legacy NIMA in the default config | REFUTED | Default config still names `checkpoints/nima_ava_gpu/final_model.keras` |
| `forWeights2` copies match canonical NIMA-fixed, RGNet-PIL, Mobile-A-LAMP, and ICAA files | PROVEN | Independently recomputed SHA-256 equality |
| Any external Flutter app contains those exact bytes | NOT_AVAILABLE / UNVERIFIED | External repository/assets were not supplied |
| Any current server process is executing `stage5_reference.json` | UNVERIFIED | No process/runtime-state evidence was supplied |
| ICAA lab configuration implements the validated model contract | REFUTED | Lab uses zero-to-one/index 0; validated contract is ImageNet normalization/index 1 |

## Remaining lineage requirements

An auditable release needs a signed or hashed manifest binding dataset manifests, source revision, resolved command/config, environment, random seed, source checkpoint hash, exporter command, TFLite hash, same-input parity set, and external app asset hash. Until that exists, file-presence and copy-equality claims are safe; end-to-end training or deployment identity claims are not.
