# Production adaptation boundaries

Mobile deployment may change engineering, not silently change the scientific identity.

| Method | Must remain for a paper-method claim | Reasonably adaptable for mobile | Rename/reframe if changed |
|---|---|---|---|
| NIMA | ordered score distribution, softmax, expected score, squared EMD training, declared dataset/split | backbone (paper itself evaluates MobileNet), quantization, operator fusion, fixed input size if preprocessing equivalence is tested | scalar-regression head or MSE-only training is “NIMA-inspired,” not faithful NIMA |
| RGNet | region graph, learned similarity edges, graph convolution, region aggregation, task-specific loss | lighter feature encoder/DenseASPP widths, graph approximation, quantization, input resolution after ablation | removing graph reasoning is not RGNet; replacing it with pooling is an RGNet-inspired regressor |
| A-LAMP | multi-patch bag, adaptive saliency/diversity selection, layout-aware object/global graph, joint image output | fewer patches/objects, lighter shared encoder/detector, cached selection, quantization | generic attention pooling or five fixed/random crops is not A-LAMP |
| AADB ranking | rating anchor, pairwise ranking and within-rater sampling; attribute supervision for full model | lighter backbone, reduced content branches, deployment scalar head | no ranking/attributes means an AADB-trained regressor, not the full paper model |
| ICAA/DAT | delegate attention, learned color-space segmentation, correct color-aesthetic target/head | fewer delegates/blocks, lighter patch embedding, quantization | ordinary Swin/ViT scalar aesthetic model is not DAT/ICAA method |
| MUSIQ | native/multi-scale ARP behavior, patch tokens, HSE/SCE, transformer image aggregation | reduce scales/patch cap/layers only with explicit ablation label; quantize | fixed square single image is not full MUSIQ; use “MUSIQ-single-inspired” where appropriate |
| KonCept512 | 512×384 Inception-ResNet-v2 and official KonIQ protocol for exact claim | smaller input/backbone for a clearly named KonIQ-trained mobile adaptation | “KonIQ Mobile” is a deployment model, not KonCept512 |
| PaQ-2-PiQ | joint/global-local region formulation and FLIVE full/patch protocol | lightweight encoder/ROI implementation, quantization | image-only scalar model is “FLIVE-trained,” not PaQ-2-PiQ |
| MDIQA | named dimension heads, shared/fused overall quality, staged official training | student backbone/distillation, head pruning only if output semantics retained | generic teacher-student scalar IQA is not MDIQA |
| TOPIQ | top-down cross-scale attention from semantics to distortion features | lighter backbone/attention widths and quantization | simple multiscale concatenation is TOPIQ-inspired only |

For every adaptation record teacher/source checkpoint hash, pre/post metrics on the same immutable test set, latency/device, output-scale mapping, preprocessing parity, and whether the paper name is used as lineage (“derived from”) or identity (“implements”).
