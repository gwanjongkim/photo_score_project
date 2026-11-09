# Paper-to-code matrix

| Model requirement | Status | Direct local evidence | Finding |
|---|---|---|---|
| NIMA 10-bin distribution | MATCH | `src/models/nima_distribution.py:58-70`, `Dense(10, softmax)`; legacy TFLite runtime `[1,10]`, sum 1.0 in `bounded_checks.json` | Implemented. |
| NIMA expected score | MATCH | `distribution_mean_score`, `src/models/nima_distribution.py:83-85` | Exact 1..10 expectation. |
| NIMA squared CDF EMD r=2 | MATCH | `emd_loss`, lines 73-80 | Formula matches baseline. |
| NIMA paper backbone | INTENTIONAL_ADAPTATION | lines 12-16, 35-70 | EfficientNetV2B0 replaces VGG/Inception/MobileNet variants. |
| NIMA no duplicate rescale | MATCH for fixed source | `include_preprocessing=False`, lines 40-54; dataset `[0,1]`, `ava_distribution_dataset.py:45-60` | Fixed path avoids internal rescaling; legacy model remains separate. |
| RGNet actual region graph | FUNCTIONALLY_EQUIVALENT | `src/models/rgnet.py:55-70`, cosine adjacency; lines 76-105 GCN; lines 132-144 graph path | Genuine relational path exists. |
| RGNet DenseNet-121 + DenseASPP | MISMATCH | `src/models/rgnet.py:32-52,126-135` | EfficientNetV2B0 + 1×1 projection; no DenseASPP rates 3/6/12/18. |
| RGNet learned projected affinity | MISMATCH | lines 61-70 | Fixed cosine similarity, not learned phi/theta projections. |
| RGNet paper aggregation | MISMATCH | lines 141-150 | Learned attention + sigmoid replaces per-region LSE `r=4`. |
| A-LAMP five patches | MATCH | `src/models/alamp.py:92-100`; TFLite input `[1,5,224,224,3]` | Five-patch contract proven. |
| A-LAMP adaptive saliency/diversity selection | MISMATCH | source doc lines 13-16 admits deterministic heuristic | No paper optimization proven. |
| A-LAMP object/global attribute graph | MISSING | `LayoutCueAugmentation`, lines 52-70 and conv layout branch 109-118 | Coordinate/mask cues replace object graph. |
| A-LAMP learned patch weighting | INTENTIONAL_ADAPTATION | lines 123-131 | Learned attention is useful but not the paper’s orderless statistics. |
| ICAA Delegate Transformer | UNVERIFIED | expected `src/models/icaa_dat.py` absent; official vendored/upstream artifacts only | Filename/checkpoint not enough. |
| ICAA output heads/scales | UNVERIFIED | no executed local port/head inspection completed | MOS/attribute interpretation cannot be certified. |
| MUSIQ native/multi-scale patch behavior | FUNCTIONALLY_EQUIVALENT | `src/models/musiq.py:55-87`; inputs include patches, positions, scale IDs | General mechanism present. |
| MUSIQ paper dimensions/CLS aggregation | MISMATCH | `src/models/musiq.py:55-65,91-106` | D128/depth4/heads4/masked mean vs D384/depth14/heads6/CLS. |
| KonCept512 architecture | MISSING | `src/models/technical_regressor.py:3-17` | MobileNetV2 scalar model is KonIQ-trained, not KonCept512. |
| PaQ-2-PiQ global/local ROI behavior | MISSING | same generic scalar model; `technical_regressor.py:3-17` | FLIVE image model is not PaQ-2-PiQ. |
| MDIQA multidimensional teacher | MATCH at code-family level | `MDIQA/MDIQA/archs/topiq_shared_fusion_arch.py`, weighted multidimensional fusion; official option files in inventory | Architecture family present; checkpoint-to-local-training link unknown. |
| Stage 5 paper identity | NOT_APPLICABLE | no baseline paper identity; scalar `[1,1]` TFLite verified | Evaluate as production/distillation artifact only. |
