# Implementation audit summary

## Scope and baseline

The requested `01_paper_baseline/` directory does not exist. This audit used the available independent baseline `01_paper_baseline_codex/` and records this as a traceability limitation. Existing project reports were used only to locate raw artifacts; their conclusions were not accepted without code, artifact, manifest, hash, or runtime corroboration.

## Principal conclusions

- **NIMA fixed candidate** implements the essential distribution head, expected score, and CDF EMD. Its EfficientNetV2B0 backbone is an intentional paper adaptation. The old deployed/exported artifact is traceable to the legacy checkpoint and is affected by the documented preprocessing-generation split; the fixed export is not proven active in runtime.
- **RGNet practical** contains real relational graph construction and graph convolution (`RegionGraphBuilder`, `GraphConvolution`), so it is more than an EfficientNet scalar approximation. It still replaces the paper’s DenseNet-121/DenseASPP/learned projected affinity/LSE head and 300-pixel recipe, making it paper-inspired, not faithful.
- **A-LAMP practical and Mobile v2** really accept five patches plus a global image and use learned patch attention. They omit the paper’s saliency/diversity optimization and object/global attribute graph. The A-LAMP name overstates paper fidelity unless qualified as mobile/practical adaptation.
- **KonIQ and FLIVE image mobile models** are MobileNetV2 scalar regressors trained on those datasets, not KonCept512 or PaQ-2-PiQ. Their production suitability is materially stronger than their paper fidelity.
- **MUSIQ practical** implements multi-scale patch tokens and a transformer, but uses 16 patches/scale, D=128, four layers, masked mean pooling and scalar MSE—substantial deviations from paper MUSIQ.
- **MDIQA teacher** preserves official multidimensional CFANet/fusion code and configurations; locally trained lineage is not established. Mobile students are separate distilled regressors and were not proven deployed.
- **ICAA** has official upstream code/checkpoints and a port lineage lead, but no local source file at the expected `src/models/icaa_dat.py`; exact local head semantics and active deployment remain unverified.
- **Technical and aesthetic ensembles** have executable formulas, but several inputs have incompatible semantic scales. Weight-selection provenance exists in artifacts, while active mobile/runtime application is incomplete or policy-dependent.

## Executable evidence

`run_bounded_checks.py` wrote `bounded_checks.json`. Seven selected TFLite files loaded and completed deterministic CPU inference with finite outputs. The legacy NIMA output shape was `[1,10]` and summed to exactly 1.0. KonIQ, FLIVE image, AADB practical RGNet, practical A-LAMP, Mobile A-LAMP v2 and Stage 5 student returned scalar outputs. Core split manifests contained no exact-path overlap and no missing files. These tests do **not** prove accuracy or Python/TFLite parity except where separately supported by raw parity artifacts.

## Overall risk

The dominant audit risk is claim inflation: dataset-trained mobile adaptations are repeatedly named after research methods whose essential architecture/protocol is absent. The second risk is lineage fragmentation between fixed candidates, legacy default runtime paths, exports, and external mobile assets. The third is metric non-recomputability where published tables lack ground truth or immutable sample manifests.
