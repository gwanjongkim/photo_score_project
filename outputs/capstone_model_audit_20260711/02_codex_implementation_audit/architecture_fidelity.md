# Architecture fidelity

## Technical IQA

KonIQ mobile and FLIVE image mobile share `build_technical_regressor` (`src/models/technical_regressor.py:3-17`): ImageNet MobileNetV2, GAP, dropout .3, Dense-128 ReLU, scalar linear output. This is a valid compact regressor but lacks KonCept512’s Inception-ResNet-v2/512×384 identity and PaQ-2-PiQ’s ROI/global-local paths. Status: **MISSING** essential paper architectures; **PRODUCTION_ADAPTED_MODEL**.

The technical ensemble is an application policy rather than a paper model. Code and inventory show at least three formulas: default five-signal server weights, a renormalized two-image-model benchmark, and a 0.60/0.40 mobile helper. These are not interchangeable. Status: paper fidelity **NOT_APPLICABLE**, implementation identity **UNVERIFIED** unless the exact policy/config is named.

TOPIQ-lite contains cross-scale guided attention on EfficientNetV2B0 according to inventory source evidence, but it is not the full official ResNet50 CFANet. C6 and RankCalib are additional calibration/distillation designs. Classification: **PAPER_INSPIRED_IMPLEMENTATION**.

## Aesthetic IAA

The fixed NIMA source has direct positive evidence for its essential head and loss. Changing the backbone is allowed only as an explicit adaptation. The legacy export also has the correct output shape but is not evidence that the fixed preprocessing checkpoint was deployed.

RGNet practical has a real dense graph. `RegionGraphBuilder` L2-normalizes nodes, computes all-pairs cosine similarity and softmax adjacency (`src/models/rgnet.py:55-70`); two `GraphConvolution` blocks aggregate neighbors (`:76-105,137-139`). This supports **FUNCTIONALLY_EQUIVALENT** for “relational reasoning exists,” not for exact RGNet. DenseNet-121, DenseASPP and Eq. 4 LSE are absent.

Practical A-LAMP accepts global `[384,384,3]` and five `[224,224,3]` patches and uses shared MobileNetV2 patch encoding plus learned attention (`src/models/alamp.py:83-142`). It replaces paper patch optimization and object attribute graphs with heuristics, coordinate/valid-mask cues, and a convolutional layout branch. Mobile A-LAMP v2’s exported two-input contract confirms multi-patch behavior, but not paper selection fidelity.

The local practical MUSIQ is explicitly named `musiq_practical` (`src/models/musiq.py:108-116`). Its positional Dense embedding and scale embedding are reasonable approximations, while masked mean pooling and shallow/small transformer are structural changes.

ICAA remains **UNVERIFIED**: the attempted inspection of `src/models/icaa_dat.py` failed because the file does not exist. Official upstream code presence and a checkpoint do not prove the local TensorFlow/mobile port’s head semantics.

## Teacher-student and rejected lines

MDIQA official code includes weighted spatial score aggregation and multidimensional fusion. The 14k/50k students, TechIQA-Guard and DistortionGuard are separately designed student/guard models, not MDIQA architecture reproductions. Their names and teacher provenance must be described as distillation/experimental lineage. Rejected/no-go status is supported as project state only when runtime/export absence is also checked; it is not a paper verdict.
