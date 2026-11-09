# Final Paper-to-Code Matrix

## Scope and controlled evidence levels

This matrix adjudicates implementation identity, not merely whether a model was trained on a paper-associated dataset. The paper baseline used here is `01_paper_baseline_codex/`; the requested `01_paper_baseline/` directory was absent. Paper requirements are treated as essential only where the baseline identifies them as such.

- **DIRECTLY_PROVEN** — the exact bounded component fact is demonstrated by source, a serialized graph, checkpoint load, or an executed resolution test.
- **STRONGLY_SUPPORTED** — multiple independent evidence lines support the claim without a material contradiction, but the proof is not end to end.
- **STRONG_BUT_INCOMPLETE** — substantial implementation evidence exists, but a necessary protocol, identity, or lineage link is missing.
- **WEAKLY_SUPPORTED** — only limited, indirect, or identity-incomplete evidence exists.
- **UNSUPPORTED**, **REFUTED**, and **UNVERIFIED** retain the definitions in the final claim registers.

`MATCH` below means the named component is present. It does not by itself establish a paper-faithful reproduction.

## Component-level adjudication

| Model line | Paper or reference requirement | Local implementation evidence | Evidence level | Final component finding |
|---|---|---|---|---|
| KonIQ Mobile | KonCept512 uses Inception-ResNet-v2 at 512×384 | `technical_regressor.py` and the serialized checkpoint use MobileNetV2 at 224×224 | DIRECTLY_PROVEN | MISMATCH. This is a KonIQ-trained mobile adaptation, not KonCept512. |
| KonIQ Mobile | Paper dataset/label protocol | Current manifests reproduce raw `MOS_zscore` exactly for all 10,073 rows; builder is a seed-42 custom 80/10/10 split | DIRECTLY_PROVEN for current manifests; UNVERIFIED for historical CLI | MISMATCH to an exact paper protocol. The checkpoint target argument and original command are absent. |
| KonIQ Mobile | Backbone preprocessing | Loader supplies RGB float `[0,1]`; serialized MobileNetV2 has no `Rescaling`; pretrained MobileNetV2 expects external `[-1,1]` preprocessing | DIRECTLY_PROVEN | MISMATCH. The prior claim of embedded rescaling is refuted. |
| FLIVE Image Mobile | PaQ-2-PiQ ResNet-18 global/local ROI formulation | Generic MobileNetV2 scalar regressor; no ROI pooling or joint local/global path | DIRECTLY_PROVEN | MISSING. Call it FLIVE-trained, not PaQ-2-PiQ. |
| FLIVE Patch | Learned picture/patch relationship and image aggregation | Five deterministic crops are scored independently; server uses mean/min/std | DIRECTLY_PROVEN by live inference | MISMATCH. Mean/min/std is an application heuristic, not PaQ-2-PiQ aggregation. |
| FLIVE image/patch | Clean evaluation separation | Image parent IDs do not overlap, but 9 cross-split byte-duplicate groups exist; patch test parents overlap train by 97.20% | DIRECTLY_PROVEN | MISMATCH for clean experimental protocol, especially the patch line. |
| NIMA legacy | Ten-bin ordered rating distribution and softmax | Serialized Keras/TFLite output `[1,10]`, sums to one | DIRECTLY_PROVEN | MATCH. |
| NIMA fixed | Ten-bin distribution, expected score over 1..10, squared CDF EMD | `nima_distribution.py`, architecture test, and replayed predictions | DIRECTLY_PROVEN | MATCH for head, score semantics, and loss. |
| NIMA fixed | Paper backbone | EfficientNetV2B0 replaces paper-evaluated backbones | DIRECTLY_PROVEN | INTENTIONAL_ADAPTATION; it must be named as such. |
| NIMA fixed/legacy | Paper evaluation split | Local seed-42 80/10/10 manifests are 204,406/25,551/25,551; supplied paper interpretations conflict but neither is this protocol | DIRECTLY_PROVEN locally; UNVERIFIED for disputed paper wording | MISMATCH to an exact paper reproduction. |
| NIMA fixed | No preprocessing mismatch | Duplicate scaling was removed, but `include_preprocessing=False` receives `[0,1]` where installed EfficientNetV2 documentation requires `[-1,1]` | DIRECTLY_PROVEN | MISMATCH. “Fixed” means the duplicate rescale was removed, not that the ImageNet contract is fully correct. |
| NIMA legacy | No duplicate/under-scaling | Serialized model has `Rescaling(1/255)` plus ImageNet `Normalization`, while the audited loader supplies `[0,1]` | DIRECTLY_PROVEN | MISMATCH. This remains a legacy path. |
| RGNet practical | Region graph and graph convolution | `RegionGraphBuilder`, all-pairs cosine adjacency, and two real `GraphConvolution` layers | DIRECTLY_PROVEN | MATCH only for the narrow claim that relational graph reasoning exists. |
| RGNet practical | Dilated DenseNet-121 FCN plus DenseASPP rates 3/6/12/18 | EfficientNetV2B0 plus a 1×1 projection; no DenseASPP | DIRECTLY_PROVEN | MISSING. |
| RGNet practical | Learned feature projections for graph affinity | Parameter-free cosine similarity instead of learned phi/theta projections | DIRECTLY_PROVEN | MISMATCH. |
| RGNet practical | Equation-4 region aggregation, including LSE `r=4` | Learned attention pooling and a sigmoid scalar head | DIRECTLY_PROVEN | MISMATCH. |
| RGNet practical | Coherent pretrained-backbone input | Loader `[0,1]` enters embedded `Rescaling(1/255)` and ImageNet normalization | DIRECTLY_PROVEN | MISMATCH. The intermediate range is `[0,0.0039]`; the accuracy impact is not isolated. |
| RGNet paper-v1 family | DenseNet-121, ASPP/DenseASPP, graph, LSE option | Static architecture resolution test finds all four | DIRECTLY_PROVEN for component presence | STRONG structural alignment, but not an official implementation or proven paper-protocol reproduction. |
| RGNet paper-v1 PIL candidate | Paper-v1 architecture plus AADB evaluation/export | Dedicated trainer/evaluator, raw predictions, FP16 export, and parity bundle exist | STRONG_BUT_INCOMPLETE | Paper-inspired candidate with materially better traceability than practical RGNet; official-code identity remains unproven. |
| A-LAMP practical | Five adaptive patches selected with saliency, diversity, and overlap constraints | Five patches and a heuristic saliency selector are present | DIRECTLY_PROVEN | PARTIAL for patch count; MISMATCH for paper selection optimization. |
| A-LAMP practical | Local/global object-attribute graphs | Coordinate/mask layout cues and convolutional layout branch; no object graph | DIRECTLY_PROVEN | MISSING. |
| A-LAMP practical | Shared VGG16 columns and orderless statistical aggregation | EfficientNetV2/MobileNetV2 branches with learned patch attention | DIRECTLY_PROVEN | MISMATCH; learned attention is an adaptation, not the paper mechanism. |
| A-LAMP practical | Output semantics | Serialized top-level head is linear, one scalar | DIRECTLY_PROVEN | Application-scale scalar; not inherently bounded to `[0,1]`. |
| Mobile A-LAMP v2 | Paper A-LAMP identity | Two MobileNetV3Small backbones, five patches, global image, layout attention, sigmoid binary head | DIRECTLY_PROVEN | Mobile production adaptation, not paper A-LAMP reproduction. |
| A-LAMP teacher variants | Multi-patch teacher behavior | Dedicated model, trainer, evaluator, and paper-benchmark config exist | STRONG_BUT_INCOMPLETE | Paper-inspired experimental teachers; object-graph and exact paper aggregation equivalence are not proven. |
| ICAA/DAT official line | Official DAT checkpoint can be loaded by its architecture | Strict load succeeds for 487 keys with no missing or unexpected keys; CPU forward emits `[1,2]` | DIRECTLY_PROVEN | MATCH at official checkpoint/code level. The cross-check’s NOT_IMPLEMENTED claim is refuted. |
| ICAA/DAT TF-native port | Delegate Transformer and dual MOS/color heads | `experiments/icaa_tf_native/tf_models/tf_dat_model.py` contains the port; architecture test confirms delegate port and two heads | DIRECTLY_PROVEN | MATCH for port existence and output structure. |
| ICAA/DAT TF-native port | Learned color-space component and exact PyTorch equivalence | Mapping/export stages and parity artifacts exist, but the audit did not replay every intermediate tensor or all learned-weight mappings | STRONG_BUT_INCOMPLETE | Substantial implementation evidence, not a complete independent proof of mathematical equivalence. |
| ICAA/DAT target | Color-aesthetic output | Port order is `[MOS,color]`; validated color index is 1 | DIRECTLY_PROVEN | MATCH when index 1 and the validated preprocessing are used. |
| MUSIQ practical | Native/multiscale ARP 224/384, 32×32 tokens, spatial/scale embeddings | Native-size multi-scale tokenization and scale/position features exist | DIRECTLY_PROVEN | PARTIAL functional match. |
| MUSIQ practical | Paper-size transformer and CLS-token aggregation | Local default is embed-dim 128/depth 4 and masked-mean scalar aggregation | DIRECTLY_PROVEN | MISMATCH. This is explicitly a practical MUSIQ-style approximation. |
| Local MUSIQ probes | Official MUSIQ model identity | Local small probe checkpoints and exports, with unknown exact subset/export invocations | DIRECTLY_PROVEN for local artifacts; UNVERIFIED for official identity | Experimental local probes only. |
| Kaggle MUSIQ variants | Official external model behavior | External-handle sanity benchmark outputs exist; no local official checkpoint/export/runtime | WEAKLY_SUPPORTED | External sanity evidence, not a local reproduction or deployment lineage. |
| MDIQA teacher | Shared representation, dimension heads, fused overall quality, staged official training | Official-family source/configs and downloaded checkpoints are present; local sanity outputs replay | STRONG_BUT_INCOMPLETE | Substantially aligned teacher/reference line; local training-to-checkpoint lineage is UNKNOWN. |
| MDIQA 14k/50k students | Preserve MDIQA multidimensional identity | EfficientNetV2 scalar students distill teacher outputs | DIRECTLY_PROVEN | MISMATCH as MDIQA reproduction; valid only as student/distillation experiments. |
| TOPIQ-lite mixed112 | CFANet top-down semantic-to-distortion cross-scale attention | Lightweight EfficientNetV2B0 cross-scale guided attention | STRONG_BUT_INCOMPLETE | Paper-inspired adaptation, not full official TOPIQ. |
| C6 | Standalone paper identity | Repository-specific capped TOPIQ/KonIQ score policy | DIRECTLY_PROVEN | NOT_APPLICABLE to paper fidelity; experimental application policy. |
| TOPIQ RankCalib | TOPIQ-style multilevel semantic-to-distortion guidance | EfficientNetV2 feature pyramid, low/mid/high guided attention, and later calibration/ranking stages | DIRECTLY_PROVEN for local components | PARTIAL paper relationship; lightweight TOPIQ-inspired experimental model, not full CFANet. |
| AADB pairwise comparator | Shared pairwise comparison plus the paper’s rating anchor, within-rater sampling, 11 attributes, and content-adaptive branches | Local code proves a shared EfficientNetV2 Siamese encoder and ternary relative-aesthetics head; rating/within-rater/attribute/content stack and paper evaluation are not established | WEAKLY_SUPPORTED for pairwise concept; UNVERIFIED for paper protocol | Lightweight paper-inspired comparator, not the full AADB ranking method. |
| AADB composition regressor | Standalone paper identity | Generic AADB-trained scalar regressor | DIRECTLY_PROVEN | NOT_APPLICABLE; production-adapted dataset model. |
| TechIQA-Guard / DistortionGuard | Paper identity | Repository-defined guard architectures and no-go reports | DIRECTLY_PROVEN | NOT_APPLICABLE; experimental/rejected project models. |
| Stage 5 student and ensembles | Standalone paper identity | Distillation and application fusion policies | DIRECTLY_PROVEN | NOT_APPLICABLE; evaluate on experiment and deployment evidence only. |
| VILA/Gemma scorers | Core IQA/IAA paper identity | Optional teacher/rerank/explanation paths | WEAKLY_SUPPORTED | Adjacent multimodal experiments, not core paper reproductions. |

## Model-identity conclusions

| Model family | Highest defensible paper relationship | Evidence level | Qualification required |
|---|---|---|---|
| KonIQ/FLIVE mobile regressors | Dataset-trained mobile adaptations | DIRECTLY_PROVEN | Do not use KonCept512 or PaQ-2-PiQ as implementation identities. |
| NIMA fixed | Substantially NIMA-structured model with a backbone and protocol adaptation | STRONG_BUT_INCOMPLETE | State the EfficientNetV2 backbone, local 80/10/10 split, and unresolved normalization defect. |
| NIMA legacy | NIMA-structured legacy artifact | DIRECTLY_PROVEN | State that the default server config still points to the legacy preprocessing path. |
| Practical RGNet | RGNet-inspired graph regressor | STRONG_BUT_INCOMPLETE | Limit similarity to the existence of region-graph reasoning; disclose absent DenseASPP, learned affinity, and LSE. |
| RGNet paper-v1/PIL | Closer paper-inspired reconstruction | STRONG_BUT_INCOMPLETE | Do not call it official or exact without official-code and protocol parity. |
| Practical/mobile A-LAMP | Multi-patch, layout-aware A-LAMP-inspired adaptations | STRONG_BUT_INCOMPLETE | Disclose heuristic patch selection, no object graph, and changed aggregation. |
| ICAA official checkpoint and TF-native conversion | Substantially faithful official-checkpoint port/conversion lineage | STRONG_BUT_INCOMPLETE | Distinguish official checkpoint use from local retraining and correct the `[MOS,color]` contract. |
| Practical MUSIQ | MUSIQ-style multi-scale transformer approximation | STRONG_BUT_INCOMPLETE | Disclose shallow dimensions and masked-mean aggregation. |
| MDIQA teacher | Official-family teacher/reference implementation | STRONG_BUT_INCOMPLETE | Do not claim local reproduction training; checkpoints are downloaded artifacts. |
| TOPIQ-lite | Lightweight TOPIQ-inspired implementation | STRONG_BUT_INCOMPLETE | Do not equate with full CFANet. |
| Students, guards, ensembles, adjacent scorers | Repository-specific experimental or production adaptations | DIRECTLY_PROVEN / STRONG_BUT_INCOMPLETE | No paper-reproduction claim is supported. |

## Final boundary

No audited local training lineage proves an exact, end-to-end paper-faithful reproduction including architecture, preprocessing, split, original training command, and reported evaluation protocol. The strongest paper-linked evidence is the official ICAA checkpoint plus TF-native conversion, the MDIQA official-family teacher artifacts, NIMA’s defining head/loss, and the RGNet paper-v1 structural branch. Each still requires the qualifications above.
