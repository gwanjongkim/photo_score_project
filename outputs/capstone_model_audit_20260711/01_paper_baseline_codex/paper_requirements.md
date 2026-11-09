# Normative paper requirements

This document defines paper baselines independently of the local implementation. `NR` means not reported in the opened primary source; it is not permission to invent a value.

## NIMA

Identity: *NIMA: Neural Image Assessment*, Hossein Talebi and Peyman Milanfar, IEEE Transactions on Image Processing 27(8), 2018; arXiv v2. [S01]

| Requirement | Normative specification | Class | Evidence/status |
|---|---|---|---|
| Target/head | Softmax probability mass function over the ordered rating bins; AVA has 10 bins (1–10), so output shape is `[B,10]`, sums to one. | ESSENTIAL_FOR_PAPER_FIDELITY | S01 Sec. II-C, Fig. 8; VERIFIED_PRIMARY_SOURCE, HIGH |
| Image score | Expected score `sum(s * p_s)`; distribution variance/std is also available. | ESSENTIAL_FOR_PAPER_FIDELITY | S01 Sec. II-B; VERIFIED_PRIMARY_SOURCE, HIGH |
| Loss | Normalized squared EMD between predicted and target CDFs, `r=2` (Eq. 1). | ESSENTIAL_FOR_PAPER_FIDELITY | S01 Sec. II-C/Eq. 1; VERIFIED_PRIMARY_SOURCE, HIGH |
| Backbone | Paper evaluates VGG16, Inception-v2, and MobileNet, ImageNet initialized; a faithful claim must name the variant. | IMPORTANT | S01 Sec. II-B and Table I; VERIFIED_PRIMARY_SOURCE, HIGH |
| Input | One rescaled image, not a crop ensemble. Exact tensor size and backbone-specific pixel normalization are not recoverable from rendered equations/text and must be tied to the chosen pretrained backbone. | IMPORTANT | S01 Sec. III-A; VERIFIED_PRIMARY_SOURCE, MEDIUM |
| Split | Random 80/20 train/test for each dataset in the authors’ experiment; the paper does not publish a reusable AVA filename list or validation split. | ESSENTIAL_FOR_PAPER_FIDELITY | S01 Sec. III; VERIFIED_PRIMARY_SOURCE, HIGH |
| Optimization | TensorFlow; SGD; momentum 0.9; dropout 0.75 at final baseline layer; distinct backbone/head LRs; both decay by 0.95 every 10 epochs. Exact rendered LR values and epoch count are unresolved. | IMPORTANT | S01 Sec. III; VERIFIED_PRIMARY_SOURCE, MEDIUM |
| Evaluation | AVA: accuracy at mean-score cutoff 5, LCC/PLCC and SRCC for predicted mean and std, and EMD. Cross-dataset evaluation is explicitly reported. No logistic fit is described. | ESSENTIAL_FOR_PAPER_FIDELITY | S01 Tables I, III–IV; VERIFIED_PRIMARY_SOURCE, HIGH |

## AADB ranking network (the attribute/ranking baseline relevant to RGNet claims)

Identity: *Photo Aesthetics Ranking Network with Attributes and Content Adaptation*, Shu Kong, Xiaohui Shen, Zhe Lin, Radomir Mech, Charless Fowlkes, ECCV 2016. [S03]

| Requirement | Normative specification | Class | Evidence/status |
|---|---|---|---|
| Dataset/labels | AADB: 10,000 Flickr photos; five raters/image; mean aesthetic score plus 11 named photographic attributes and anonymized rater identity; source ratings rescaled to `[0,1]`. | ESSENTIAL_FOR_PAPER_FIDELITY | S03 Sec. 3/4.1; VERIFIED_PRIMARY_SOURCE, HIGH |
| Split | 8,500 train, 500 validation, 1,000 test. | ESSENTIAL_FOR_PAPER_FIDELITY | S03 Sec. 5.1; VERIFIED_PRIMARY_SOURCE, HIGH |
| Architecture | AlexNet regression base; Siamese shared network for pairs; attribute branch; optional content-adaptive branches gated by softmax weights from `k=10` fc7 k-means content groups. | ESSENTIAL_FOR_PAPER_FIDELITY | S03 Fig. 3, Secs. 4.1–4.4; VERIFIED_PRIMARY_SOURCE, HIGH |
| Losses | Euclidean rating regression anchors score; margin pairwise ranking loss (Eq. 1); Euclidean attribute loss on AADB; joint training. Within-rater pair sampling is central evidence, not generic random pairs. | ESSENTIAL_FOR_PAPER_FIDELITY | S03 Secs. 4.1–4.3; VERIFIED_PRIMARY_SOURCE, HIGH |
| Input/optimization | Warp RGB image to 256×256; random 227×227 crop; Caffe; initial LR `1e-4`, periodic ×0.1; momentum .9; weight decay `1e-5`. Batch/epochs NR. | IMPORTANT | S03 Sec. 4.5; VERIFIED_PRIMARY_SOURCE, HIGH |
| Evaluation | AADB test SRCC; AVA SRCC and binary accuracy with threshold selected on validation; cross-dataset matrix. No logistic fitting. | ESSENTIAL_FOR_PAPER_FIDELITY | S03 Sec. 5.2; Tables 2,3,7; VERIFIED_PRIMARY_SOURCE, HIGH |

## RGNet

Identity: *Composition-Aware Image Aesthetics Assessment*, Dong Liu, Rohit Puri, Nagendra Kamath, Subhabrata Bhattacharya, WACV 2020. [S04]

| Requirement | Normative specification | Class | Evidence/status |
|---|---|---|---|
| Encoder | ImageNet DenseNet-121 converted to FCN; last two pool layers removed; following convolutions dilated 2 and 4; 1/8-resolution feature map. DenseASPP has four cascaded 64-channel atrous layers at rates 3,6,12,18 and concatenates them. | ESSENTIAL_FOR_PAPER_FIDELITY | S04 Sec. 3.1/Fig. 4; VERIFIED_PRIMARY_SOURCE, HIGH |
| Graph | Every spatial feature location is a region node; fully connected edges use learned projected inner-product similarity `phi(x_i)^T theta(x_j)` (Eq. 1); graph convolution reasons over the region graph. A model without this relational path is not RGNet. | ESSENTIAL_FOR_PAPER_FIDELITY | S04 Sec. 3.2/Eqs. 1–3; VERIFIED_PRIMARY_SOURCE, HIGH |
| Head/aggregation | Per-region class scores aggregated to image scores by the paper’s LSE-style Eq. 4 with validated `r=4`, then two-class softmax (Eq. 5); AADB replaces classification with scalar regression and MSE. | ESSENTIAL_FOR_PAPER_FIDELITY | S04 Secs. 3.2, 4.1; VERIFIED_PRIMARY_SOURCE, HIGH |
| Input/augmentation | RGB resized 300×300; horizontal flip p=.5; scale 1.05–1.25 then random 300 crop. Pixel range/mean/std/channel order NR. | IMPORTANT | S04 Sec. 3.3; VERIFIED_PRIMARY_SOURCE, HIGH |
| Training | End-to-end; Adam; batch 32 on four GPUs; 80 epochs; LR `1e-4*(1-epoch/maxEpoch)^0.9`; weight decay `1e-5`; BN. AVA BCE, AADB MSE. | IMPORTANT | S04 Sec. 3.3 and Sec. 4.1; VERIFIED_PRIMARY_SOURCE, HIGH |
| Protocol | AVA ~230k/~20k literature split, labels thresholded at 5; accuracy and AP. AADB 8500/500/1000; SRCC. No attribute supervision, pairwise ranking, PLCC, logistic fit, or checkpoint criterion is described for RGNet. | ESSENTIAL_FOR_PAPER_FIDELITY | S04 Sec. 4.1/Tables 4–5; VERIFIED_PRIMARY_SOURCE, HIGH |

## A-LAMP

Identity: *A-Lamp: Adaptive Layout-Aware Multi-Patch Deep Convolutional Neural Network for Photo Aesthetic Assessment*, Shuang Ma, Jing Liu, Chang Wen Chen, CVPR 2017. [S05]

| Requirement | Normative specification | Class | Evidence/status |
|---|---|---|---|
| Multi-patch input | Five 224×224×3 patches per image, selected jointly from the original arbitrary-size image. | ESSENTIAL_FOR_PAPER_FIDELITY | S05 Sec. 3.1 and Sec. 4; VERIFIED_PRIMARY_SOURCE, HIGH |
| Patch selection | Optimize saliency, pattern diversity (edge/chrominance Gaussian distributions and Wasserstein distance), and spatial non-overlap/distance; not random crops. | ESSENTIAL_FOR_PAPER_FIDELITY | S05 Sec. 3.1.1/Eqs. 1–4; VERIFIED_PRIMARY_SOURCE, HIGH |
| Patch network | Five shared-weight VGG16 columns initialized from ImageNet. Orderless elementwise statistics aggregate patch features (paper prose/equation says max+mean, footnote says max+min—recorded as disagreement), followed by two FC layers. | ESSENTIAL_FOR_PAPER_FIDELITY | S05 Sec. 3.1.2/Fig. 3; VERIFIED_PRIMARY_SOURCE, HIGH |
| Layout path | Salient-object detector; four local object nodes plus global scene node; fully connected local and global attribute graphs encode geometry/appearance; graph vector is fused with multi-patch features. This is not a generic learned attention block. | ESSENTIAL_FOR_PAPER_FIDELITY | S05 Sec. 3.2/Eqs. 6–8/Fig. 4; VERIFIED_PRIMARY_SOURCE, HIGH |
| Output/training | Image-level binary aesthetic prediction; train multi-patch subnet then fuse/fine-tune. Caffe; base LR .01, weight decay `1e-5`, momentum .9; 20–30 epochs reported for New-MP-Net. Batch, schedule, normalization, checkpoint criterion NR. | IMPORTANT | S05 Sec. 4; VERIFIED_PRIMARY_SOURCE, MEDIUM |
| Protocol | AVA binary categorization; accuracy and F-measure; A-LAMP Table 2 = 82.5%/0.92. The paper does not report SRCC/PLCC or score-distribution training. | ESSENTIAL_FOR_PAPER_FIDELITY | S05 Table 2; VERIFIED_PRIMARY_SOURCE, HIGH |

## ICAA / Delegate Transformer (DAT)

Identity: *Thinking Image Color Aesthetics Assessment: Models, Datasets and Benchmarks*, Shuai He et al., ICCV 2023. [S06,S07]

| Requirement | Normative specification | Class | Evidence/status |
|---|---|---|---|
| Task/labels | ICAA is color-aesthetics assessment. ICAA17K has ~17k images across 30 color combinations/types and >50 scenes; manual color-aesthetic rating and annotator confidence `[0,1]`. The main paper also evaluates a holistic-aesthetic auxiliary task. | ESSENTIAL_FOR_PAPER_FIDELITY | S06 Sec. 4/Fig. 3; VERIFIED_PRIMARY_SOURCE, HIGH |
| Backbone | Four-stage hierarchy: 4×4 convolution patch embedding; stages I–II use Swin blocks; stages III–IV use Delegate Transformer blocks (9 then 1 in Fig. 2). | ESSENTIAL_FOR_PAPER_FIDELITY | S06 Sec. 3/Fig. 2; VERIFIED_PRIMARY_SOURCE, HIGH |
| DAT | Local attention plus delegate attention; learned offsets sample interest points by bilinear interpolation; multihead attention allocates global attention through delegates. | ESSENTIAL_FOR_PAPER_FIDELITY | S06 Sec. 3.1/Fig. 2; VERIFIED_PRIMARY_SOURCE, HIGH |
| Color-space module | Learnable centers and widths create soft piecewise-linear memberships over `kappa` color spaces (Eq. 5), then output head predicts score. | ESSENTIAL_FOR_PAPER_FIDELITY | S06 Sec. 3.2/Eq. 5; VERIFIED_PRIMARY_SOURCE, HIGH |
| Heads | The paper distinguishes `Ours*1` (color score only) and `Ours*2` (joint holistic aesthetic prediction). It does **not** establish a fixed multi-head vector whose dimensions are the 30 color types, nor does confidence become a prediction head. Exact official-code return arity and label scale remain unresolved pending file-level inspection. | IMPORTANT | S06 Table 4 caption; S07 `ICAA17K_code`; VERIFIED_PRIMARY_SOURCE/OFFICIAL_CODE, MEDIUM |
| Training/protocol | Official README requires AVA-pretrained weights before ICAA17K/SPAQ tuning and runs NNI via `config.yml`/`main_nni.py`. Main paper reports SRCC and LCC and epoch counts, but exact split, input normalization, optimizer, batch and checkpoint rule are not stated clearly enough in opened primary text. | IMPORTANT | S06 Sec. 5.1/Tables 2–6; S07 README; VERIFIED_OFFICIAL_CODE, MEDIUM |

## MUSIQ

Identity: *MUSIQ: Multi-Scale Image Quality Transformer*, Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, Feng Yang, ICCV 2021. [S08,S09]

| Requirement | Normative specification | Class | Evidence/status |
|---|---|---|---|
| Input | Three scales by default: native-resolution image plus aspect-ratio-preserving resized variants with long side 224 and 384; no square warp or crop ensemble. Single-scale is a separately named ablation. | ESSENTIAL_FOR_PAPER_FIDELITY | S08 Sec. 4.2; VERIFIED_PRIMARY_SOURCE, HIGH |
| Tokens | Split each scale into 32×32 patches; convolutional patch encoder; hash-based 2D spatial embedding (grid 10); scale embedding; `[CLS]` token yields image-level representation. Training caps native-scale patches at 512. | ESSENTIAL_FOR_PAPER_FIDELITY | S08 Fig. 2/Secs. 3.1–3.4, 4.2; VERIFIED_PRIMARY_SOURCE, HIGH |
| Transformer | D=384, 14 layers, MLP 1152, 6 heads, ~27M parameters. | IMPORTANT | S08 Sec. 4.2; VERIFIED_PRIMARY_SOURCE, HIGH |
| Output/loss | One image MOS for PaQ-2-PiQ/KonIQ/SPAQ with L1; AVA 10-bin distribution with EMD `r=2`. No patch-to-image averaging at inference—the transformer directly aggregates all tokens. | ESSENTIAL_FOR_PAPER_FIDELITY | S08 Sec. 3.4 and Sec. 4.2; VERIFIED_PRIMARY_SOURCE, HIGH |
| Training | ImageNet pretrain 300 epochs, Adam β=(.9,.999), batch 4096, wd .1, cosine from .001. Fine-tune SGD+momentum/cosine: PaQ 10 ep LR .0002; KonIQ 30/.0001; SPAQ 30/.0001; AVA 20/.12. Batch AVA 512, KonIQ 96, others 128. | IMPORTANT | S08 Sec. 4.2; VERIFIED_PRIMARY_SOURCE, HIGH |
| Protocol | KonIQ: random 80/20, 10 runs; others literature splits. SRCC and PLCC, no stated logistic fitting. PaQ reports full-size val/test separately; cross-dataset claims require matching the paper table. | ESSENTIAL_FOR_PAPER_FIDELITY | S08 Secs. 4.1–4.3/Tables 1–4; VERIFIED_PRIMARY_SOURCE, HIGH |

## KonIQ-10k / KonCept512

Identity: *KonIQ-10k: An Ecologically Valid Database for Deep Learning of Blind Image Quality Assessment*, Vlad Hosu, Hanhe Lin, Tamas Sziranyi, Dietmar Saupe, IEEE TIP 29, 2020. [S10,S11]

| Requirement | Normative specification | Class | Evidence/status |
|---|---|---|---|
| Dataset | 10,073 authentically distorted YFCC100M images; canonical full images 1024×768, smaller release 512×384; five-point ACR MOS plus z-scored MOS and rating histograms. | ESSENTIAL_FOR_PAPER_FIDELITY | S10 dataset sections; S11 score schema; VERIFIED_PRIMARY_SOURCE/DATASET_DOCUMENTATION, HIGH |
| KonCept512 | Inception-ResNet-v2 quality regressor at 512×384; scalar MOS output. A “KonIQ Mobile” architecture is not defined by this paper. | ESSENTIAL_FOR_PAPER_FIDELITY | S10 model section/abstract; VERIFIED_PRIMARY_SOURCE, HIGH |
| Protocol | The paper’s published KonCept512 experiment uses its prescribed train/validation/test split and reports SROCC/PLCC plus cross-dataset LIVE-in-the-Wild SROCC. Exact filenames must come from official split metadata, not a fresh random split. | ESSENTIAL_FOR_PAPER_FIDELITY | S10 experiment tables; VERIFIED_PRIMARY_SOURCE, MEDIUM |
| Preprocessing/calibration | Preserve aspect ratio/history of standardized dataset crops; paper model operates at 512×384. Exact pixel normalization, augmentation, optimizer schedule and logistic fitting require code-level confirmation and are unresolved here. | IMPORTANT | S10/S11; VERIFIED_PRIMARY_SOURCE, MEDIUM |

## FLIVE / PaQ-2-PiQ

Identity: *From Patches to Pictures (PaQ-2-PiQ): Mapping the Perceptual Space of Picture Quality*, Zhenqiang Ying et al., CVPR 2020. [S12,S13]

| Requirement | Normative specification | Class | Evidence/status |
|---|---|---|---|
| Dataset | FLIVE contains about 39.8k authentically distorted Flickr pictures plus about 120k separately rated patches, ~4M judgments. Image MOS and patch MOS are distinct supervision. | ESSENTIAL_FOR_PAPER_FIDELITY | S12 abstract/dataset section; VERIFIED_PRIMARY_SOURCE, HIGH |
| Model | PaQ-2-PiQ uses a ResNet-18-based region/ROI quality model with global-to-local and local-to-global paths; predicts global picture quality and local patch/quality-map values. A generic scalar MobileNet trained only on image MOS is not PaQ-2-PiQ. | ESSENTIAL_FOR_PAPER_FIDELITY | S12 method/Figs. 3–5; VERIFIED_PRIMARY_SOURCE, HIGH |
| Sampling/aggregation | Full picture and rated patch regions participate; ROI pooling/region prediction ties local predictions to the full-image representation. Image prediction is learned, not arithmetic averaging of independently sampled patches. | ESSENTIAL_FOR_PAPER_FIDELITY | S12 method; VERIFIED_PRIMARY_SOURCE, HIGH |
| Protocol | Use official FLIVE train/validation/test partition; full-size test pictures have at least one dimension >640. Report SRCC and PLCC separately for full-image and patch sets as applicable. | ESSENTIAL_FOR_PAPER_FIDELITY | S12 experiments; S08 Table 1 corroborates full-size test property; VERIFIED_PRIMARY_SOURCE, HIGH |
| Numeric recipe | Exact resize/crop normalization, optimizer, schedule and calibration must be taken from S13’s training implementation; not all were visible in the opened README. | IMPLEMENTATION_DETAIL | S13; VERIFIED_OFFICIAL_CODE, LOW |

## MDIQA

Identity: *MDIQA: Unified Image Quality Assessment for Multi-dimensional Evaluation and Restoration*, Shunyu Yao et al., arXiv:2508.16887v1, 2025. [S14,S15]

| Requirement | Normative specification | Class | Evidence/status |
|---|---|---|---|
| Method | A shared representation with multiple dimension-specific heads predicts interpretable technical/aesthetic dimensions and fuses them for overall IQA; supports NR/FR use and restoration loss. | ESSENTIAL_FOR_PAPER_FIDELITY | S14 abstract/method; S15 architecture diagram; VERIFIED_PRIMARY_SOURCE/OFFICIAL_CODE, MEDIUM |
| Training stages | Official code explicitly separates shared backbone/head training (`train_shared.py`) from overall/cross-dataset IQA (`train_MDIQA.py`); configurations under `options_MDIQA/aesthetic/` and `benchmark_experiments/` are part of the reproducible recipe. | ESSENTIAL_FOR_PAPER_FIDELITY | S15 README commands; VERIFIED_OFFICIAL_CODE, HIGH |
| Dependencies | Official setup requires modified `open_clip` ModifiedResNet return behavior. Omission changes the representation consumed by MDIQA heads. | IMPORTANT | S15 README setup; VERIFIED_OFFICIAL_CODE, HIGH |
| Exact heads/labels | The opened abstract/README does not fully expose the precise count, label scales, losses, optimizer, input normalization or patch protocol. These are `UNVERIFIED`, not inferred from local MDIQA copies. | IMPORTANT | S14/S15; UNVERIFIED, LOW |
| Student variants | Official MDIQA paper/repo baseline does not establish the inventory’s MobileNetV3 student variants as the paper method. They can only claim distillation/adaptation after separately documenting teacher outputs and losses. | PRODUCTION_ADAPTABLE | absence from S14/S15; VERIFIED_OFFICIAL_CODE, MEDIUM |

## TOPIQ

Identity: *TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment*, Chaofeng Chen et al., IEEE TIP 2024 / arXiv:2308.03060. [S16,S17]

| Requirement | Normative specification | Class | Evidence/status |
|---|---|---|---|
| Architecture | CFANet with ResNet-50 backbone; high-level semantic features propagate top-down into low-level distortion features through cross-scale attention. FR and NR variants are distinct. | ESSENTIAL_FOR_PAPER_FIDELITY | S16 Sec. III; VERIFIED_PRIMARY_SOURCE, HIGH |
| Head/output | NR variant produces a scalar image quality prediction from fused multi-scale features. Dataset-specific training/evaluation settings must name the variant and checkpoint. | ESSENTIAL_FOR_PAPER_FIDELITY | S16 Sec. III; S17 `pyiqa/archs/topiq_arch.py`; VERIFIED_PRIMARY_SOURCE/OFFICIAL_CODE, HIGH |
| Recipe/protocol | Exact resize/crop, normalization, loss, optimizer and dataset splits are delegated to paper Sec. IV and official `pyiqa` configs. They were not all captured in the opened page and remain unresolved rather than borrowed from local code. | IMPORTANT | S16/S17; UNVERIFIED, LOW |

## Stage 5 and other project-specific teacher–student labels

No published method called “Stage 5 student,” “C6,” “RankCalib,” “TechIQA-Guard,” or “DistortionGuard” was identified in the primary-source search. Therefore there is no paper-fidelity baseline for those names. A scientifically auditable distillation claim would at minimum require a named teacher checkpoint, teacher output semantics, student architecture, temperature/soft-target definition, hard-label term, weighting, identical data/split provenance, and teacher-versus-student evaluation. These are methodological disclosure requirements, not proof that Hinton et al.’s particular recipe was followed. Status: UNVERIFIED; confidence HIGH that no identity can be assigned from the inventory name alone.
