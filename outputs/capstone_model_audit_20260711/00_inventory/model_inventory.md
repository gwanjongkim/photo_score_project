# Model inventory

## Scope and interpretation

This is a repository-evidence inventory, not a paper-fidelity verdict. A model is listed as a project model when the repository contains at least one of the following: an in-project architecture plus run artifact, a checkpoint plus evaluation output, a default runtime reference, or a documented mobile handoff. The large vendored `MDIQA/pyiqa/` model zoo is recorded separately; files present only as third-party library code are not treated as trained capstone models without a local checkpoint or run association.

Classifications are preliminary and artifact-specific.

- `PRODUCTION` means a default repository runtime/config directly consumes the artifact, or the repository's technical mobile contract directly identifies it.
- `PRODUCTION_CANDIDATE` means the model has export, parity, benchmark, or handoff evidence but the active end runtime is not proven in this checkout.
- `EXPERIMENTAL` means a bounded benchmark, teacher, probe, or research branch exists without an active production link.
- `REJECTED` means an explicit later decision closes or excludes that model line.
- `LEGACY` means a superseded artifact remains referenced or retained.
- `UNKNOWN_ORIGIN` means role or provenance cannot be grounded beyond a claim artifact.

The authoritative per-file paths, sizes, mtimes, hashes, evidence, confidence, and classifications are in `artifact_manifest.tsv` and `artifact_hashes.tsv`.

## Summary

| Model or line | Domain | Preliminary status | Confidence | Repository-grounded role |
|---|---|---|---|---|
| KonIQ Mobile | Technical IQA | PRODUCTION | HIGH | Default server component and documented mobile technical model |
| FLIVE Image Mobile | Technical IQA | PRODUCTION | HIGH | Default server component and documented mobile technical model |
| FLIVE Patch | Technical IQA | PRODUCTION, server only | HIGH | Five-crop mean/min guard in the default Python bundle |
| Technical ensembles | Technical IQA | PRODUCTION plus candidate variants | HIGH | Default five-signal server selector; separate image-only and mobile policies |
| TOPIQ-lite mixed112 | Technical IQA | PRODUCTION_CANDIDATE | HIGH | Exported shadow/log-only candidate |
| C6 TOPIQ-KonIQ | Technical IQA | REJECTED as replacement | HIGH | Output-scoped capped candidate retained as log-only evidence |
| TOPIQ RankCalib | Technical IQA | EXPERIMENTAL | HIGH | Multi-stage calibration/ranking experiments |
| MUSIQ AADB practical | Cross-domain | PRODUCTION, server only | HIGH | Default server bundle emits `musiq_score`; not a mobile model |
| Local MUSIQ probes | Technical IQA | EXPERIMENTAL | HIGH | Local 1,024-row probe models, explicitly not official Kaggle MUSIQ |
| Kaggle MUSIQ variants | Technical IQA | EXPERIMENTAL | HIGH | External-model sanity benchmark only |
| MDIQA teacher | Technical IQA | EXPERIMENTAL | HIGH | Teacher/performance reference, no mobile runtime |
| MDIQA 14k and 50k students | Technical IQA | REJECTED | HIGH | Distillation lines closed by final no-go report |
| TechIQA-Guard v1 | Technical IQA | REJECTED | HIGH | Single-head replacement line closed before export |
| DistortionGuard-IQA | Technical IQA | REJECTED for B3 | HIGH | Stage A/B research line; B3 export and Flutter integration rejected |
| NIMA AVA legacy | Aesthetic IAA | LEGACY | HIGH | Still referenced by default server config but superseded by preprocessing repair |
| NIMA AVA fixed | Aesthetic IAA | PRODUCTION_CANDIDATE | HIGH | Fixed checkpoint, TFLite, parity, and handoff bundle |
| RGNet AADB practical | Aesthetic IAA | PRODUCTION | HIGH | Default server aesthetic component |
| RGNet paper-v1 family | Aesthetic IAA | EXPERIMENTAL | HIGH | Paper-inspired teacher/ablation family |
| RGNet PIL-resize AADB | Aesthetic IAA | PRODUCTION_CANDIDATE | HIGH | Complete train/eval/export/parity candidate lineage |
| A-LAMP AADB practical | Aesthetic IAA | PRODUCTION | HIGH | Default server aesthetic component |
| Mobile A-LAMP v2 AVA | Aesthetic IAA | PRODUCTION_CANDIDATE | HIGH | Full AVA binary classifier with FP16 handoff candidate |
| A-LAMP teacher variants | Aesthetic IAA | EXPERIMENTAL or REJECTED | HIGH | Multi-patch teacher retained; failed graph/fusion branches archived |
| ICAA17K DAT | Color IAA | EXPERIMENTAL | HIGH | Official checkpoint and TF-native port |
| ICAA FP16 mobile artifact | Color IAA | PRODUCTION_CANDIDATE with contract conflict | MEDIUM | Local lab candidate; active app link unproven |
| Stage 5 AADB aesthetic student | Aesthetic IAA | REJECTED | HIGH | Historical mobile candidate later classified as rejected |
| Stage5 default aesthetic ensemble | Aesthetic IAA | PRODUCTION, server | HIGH | Old AADB/NIMA/A-LAMP/RGNet/pairwise bundle loaded by default |
| Calibrated NIMA-RGNet-A-LAMP ensemble | Aesthetic IAA | PRODUCTION_CANDIDATE | HIGH | Offline aligned-prediction search; runtime application is UNKNOWN_LINK |
| AADB composition regressor | Aesthetic IAA | PRODUCTION, server | HIGH | Default `aadb_model` in Stage5 bundle |
| Pairwise AADB comparator | Aesthetic IAA | PRODUCTION, optional server rerank | HIGH | Default checkpoint/config exists; rerank disabled unless enabled |
| VILA/Gemma multimodal scorers | Aesthetic/explanation | EXPERIMENTAL or optional | MEDIUM | Teacher, rerank, and explanation paths; not core IQA/IAA mobile assets |

## Technical IQA records

### KonIQ Mobile

- Architecture implementation. `src/models/technical_regressor.py` builds ImageNet MobileNetV2, global average pooling, dropout 0.3, Dense 128 ReLU, and a linear scalar head.
- Training. `src/train/train_regression.py --task technical` is the closest direct trainer. It defines Adam 1e-4, MSE, MAE, best-`val_loss` checkpointing, and early stopping with patience 3. No scheduler is defined. The original invocation and stdout log are absent.
- Data and split. `src/preprocess/make_koniq_csv.py` creates random 80/10/10 splits with seed 42. Manifests are `data/processed/koniq10k/{train,val,test}.csv`. The builder's MOS-column loop appears to select the later `MOS_zscore` column in the current raw CSV, which must remain a forensic caveat.
- Preprocessing and augmentation. `src/datasets/csv_dataset.py` uses float RGB in [0,1]; train resizes to 256, random-crops 224, and horizontally flips; validation directly resizes to 224.
- Model artifacts. `checkpoints/technical_koniq_gpu/` contains best weights, final Keras, and SavedModel. `exports/tflite/koniq_mobile.tflite` has metadata and a passing verification record.
- Evaluation and raw predictions. Important evidence includes `scripts/eval_final_topiq_candidates_vs_existing_technical.py`, `outputs/research_protocol_production_technical_eval_20260530/production_technical_predictions.csv`, and the official technical signal benchmark.
- Runtime. `configs/stage5_reference.json` points to the Keras checkpoint; `src/infer/run_acut_pipeline.py`, `predict_quality_bundle.py`, and `select_best_shots.py` consume its score. The mobile contract is documented in `docs/tflite_mobile_handoff.md`.
- Lineage caveat. Exact training CLI and byte identity between the April verification record, the May-mtime TFLite, and an external submitted app remain `UNKNOWN_LINK`.

### FLIVE Image Mobile

- Architecture, trainer, objective, and optimizer are shared with KonIQ through `technical_regressor.py` and `train_regression.py`.
- Data and split. The original lineage uses `data/raw/paq2piq/labels_image.csv` and `src/preprocess/make_flive_csv.py`, which creates a custom random 80/10/10 split with seed 42 at `data/processed/paq2piq/image_{train,val,test}.csv`.
- Preprocessing and augmentation match the generic CSV pipeline: [0,1] RGB, train resize/crop/flip, direct eval resize.
- Artifacts. `checkpoints/technical_flive_image_gpu/` contains best weights, Keras, and SavedModel. `exports/tflite/flive_image_mobile.tflite` has metadata and a passing source/TFLite verification record.
- Evaluation and runtime share the KonIQ benchmark and default Stage5 runtime paths.
- Lineage caveat. Later toolbox-standardized FLIVE manifests are evaluation evidence and cannot be the March checkpoint's training split.

### FLIVE Patch

- Architecture and trainer are the same generic MobileNetV2 regressor.
- Dataset manifests are `data/processed/paq2piq/patch_{train,val,test}.csv`, built by `make_flive_csv.py`.
- Checkpoints are under `checkpoints/technical_flive_patch_gpu/`.
- Runtime preprocessing is explicit in `predict_quality_bundle.py`: five crops, resize to 224, then mean/min/std aggregation.
- No TFLite, ONNX, or PyTorch export was found. Patch checkpoint to mobile asset is `UNKNOWN_LINK`; repository notes keep repeated patch inference off device.

### Technical ensemble variants

Three distinct runtime/evaluation formulas must not be merged.

1. Default Python selector. `select_best_shots.py` configures KonIQ 0.32, FLIVE image 0.24, FLIVE patch mean 0.22, patch min 0.12, and MUSIQ 0.10, renormalizing over available signals.
2. Image-only benchmark reconstruction. When only KonIQ and FLIVE image exist, the effective blend is `(0.32*K + 0.24*F)/0.56`, or 4/7 and 3/7.
3. Mobile-oriented Python helper. `src/infer/predict_mobile_aesthetic.py` separately uses a 0.60/0.40 KonIQ/FLIVE policy.

The official benchmark evidence is under `outputs/official_protocol_technical_signal_benchmark_20260604/`, including manifest, raw predictions, metrics, safety cases, candidate ranking, and recommendation.

### TOPIQ-lite mixed112, RankCalib, and C6

- Architecture. `src/models/topiq_lite.py` uses EfficientNetV2B0 with cross-scale guided attention; `src/models/topiq_rankcalib.py` defines later calibration/ranking variants.
- Training. `src/train/train_topiq_lite.py` uses Adam and Huber loss; the RankCalib line has stage-specific trainers under `src/train/train_topiq_rankcalib_stage*.py`.
- Data. `scripts/create_mixed_topiq_datasets.py` creates SPAQ:KonIQ:FLIVE 1:1:2 manifests such as `data/processed/topiq_replacement/mixed_112_train.csv`.
- Preprocessing. 384x384 `resize_with_pad`, raw float [0,255], relying on embedded EfficientNetV2 preprocessing. The direct trainer has no separate augmentation.
- Export/evaluation. `outputs/final_topiq_lite_mixed112_export_20260517/` contains FP32/FP16 TFLite and parity outputs. Raw comparative predictions and reports live in `outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/`.
- C6. Output-scoped code evaluates `min(0.7*TOPIQ + 0.3*KonIQ, KonIQ + 8)`; `outputs/final_c6_topiq_koniq_technical_score_decision_20260527/report.md` rejects replacement. No default production source implements C6.
- Source drift. The preserved training command refers to `--init_weights`, which is absent from the current trainer parser. Historical run to current code is therefore `UNKNOWN_LINK`.

### MUSIQ practical model and probes

- Architecture. `src/models/musiq.py` explicitly describes a repository-practical MUSIQ-style approximation using multi-scale patches and a transformer.
- Training. `src/train/train_musiq.py` defines Adam 1e-4, MSE, MAE, best-`val_loss`, early stopping patience 3, and no scheduler.
- Data/preprocessing. `src/datasets/native_size_dataset.py` maintains aspect ratio at 224/384/512 scales, extracts limited 32x32 tokens, uses [0,1] RGB, and random horizontal flip in training.
- Practical AADB artifacts. `checkpoints/musiq_aadb_gpu/` and `musiq_train.log` exist. The default Stage5 config loads this Keras model and the selector assigns `musiq_score` to the technical component. Its AADB training semantics make that role a documented design caveat.
- Technical probes. `outputs/probe1024_musiq_{koniq,spaq,flive}/` and `exports/tflite/musiq_*_probe1024/` contain local Keras/SavedModel/TFLite probe artifacts. Exact subset-builder, trainer invocation, and exporter are `UNKNOWN_LINK`.
- Official Kaggle variants. The sanity runners use external handles rather than local model files. Their outputs are sanity evidence only; no checked-in export or runtime path exists.

### MDIQA teacher

- Architecture. `MDIQA/MDIQA/archs/topiq_backbone_head_arch.py` defines shared extractors and `topiq_shared_fusion_arch.py` defines `MultiDimFusionWeightedMLP`. The curated hash manifest includes the principal multidimensional architecture/model files.
- Training/config. Entries are `MDIQA/MDIQA/train_MDIQA.py` and `train_shared.py`; official benchmark configs are under `MDIQA/MDIQA/options_MDIQA/benchmark_experiments/`. They specify 384 input, random crop and horizontal flip, AdamW 1e-5 with weight decay 1e-5, cosine annealing, and NiN/MSE objectives.
- Data. MDIQA includes KonIQ/SPAQ metadata and split material under `MDIQA/datasets/`.
- Checkpoints. Important local PyTorch artifacts include `mdiqa_shared_5t_4a_mbv3_koniq_seed1.pth`, `mdiqa_shared_5t_4a_mbv3_koniq_xd.pth`, `net_best.pth`, technology heads/backbone, and the PARA aesthetic/color checkpoint.
- Evaluation. Repository wrappers under `scripts/technical_iqa/run_mdiqa_*.py` produce manifests, predictions, failures, and reports. `outputs/mdiqa_official_variant_sanity_20260601/` is a representative complete bundle.
- Export/runtime. No TFLite/ONNX/mobile runtime link was found. Checkpoints are downloaded official artifacts; official local data to weights is `UNKNOWN_LINK`.

### MDIQA mobile students

- 14,105-row line. Scripts `train_mobile_student_iqa_14105_*.py` build an EfficientNetV2B0 regressor with embedded preprocessing, direct 256 resize, Huber loss, Adam, head/finetune phases, and no separate scheduler. Manifests and Keras outputs exist, and `outputs/mobile_student_iqa_14105_tflite_export_20260604/` contains FP32/FP16 TFLite plus parity.
- 50k line. `train_mobile_student_iqa_50k_controlled.py` and related smoke/weighting/ranking/scaling scripts use a grouped/stratified split from the 50k teacher manifest. `outputs/mobile_student_iqa_50k_controlled_b0_20260608/` contains train/val/test manifests, Keras models, history, metrics, safety outputs, and raw official predictions.
- Final status. `outputs/mdiqa_student_final_no_go_report_20260611/report.md` closes the full student line. No 50k TFLite was produced; export/runtime is intentionally absent.

### TechIQA-Guard v1

- Architecture. `src/models/techiqa_guard.py` implements an EfficientNetV2B0 multi-scale attention and defect-aware scalar head.
- Training/data. `src/train/train_techiqa_guard.py`, `src/data/build_techiqa_guard_dataset.py`, and `data/processed/techiqa_guard/` cover train/validation and hard-false-positive material. The later run uses Adam, Huber, PLCC, and false-positive terms with partial unfreeze and frozen batch normalization.
- Evaluation. `scripts/eval_techiqa_guard_v1_stage1.py` and output-scoped evaluation directories preserve predictions and summaries.
- Decision/export. `outputs/final_techiqa_guard_v1_nogo_20260522/report.md` closes the line and forbids mobile export. No TFLite lineage exists.

### DistortionGuard-IQA

- Architecture. `src/models/distortionguard.py` defines multi-head Stage A and scalar Stage B models on EfficientNetV2B0.
- Data/augmentation. `src/data/generate_distortionguard_synthetic.py` creates synthetic distortion types/severities and pair manifests. Stage B uses authentic manifests at 384 `resize_with_pad`, raw [0,255], without separate random image augmentation.
- Objectives. Stage A combines distortion classification, severity Huber, and pair ranking; Stage B combines Huber and PLCC. Adam is used; B3 applies FLIVE sample weight in the Huber term.
- Training/evaluation. Stage-specific trainers and `scripts/eval_distortionguard_stageB_authentic.py` exist. B3 artifacts include best/final Keras, training log, summary, transfer report, model summary, and trainable-layer list.
- Export/runtime. Later diagnosis reports reject B3 export and Flutter integration. No TFLite/ONNX/mobile asset exists.

## Aesthetic IAA records

### NIMA AVA

Two artifact states must be separated.

- Legacy state. `checkpoints/nima_ava_gpu/` and `exports/tflite/nima_mobile.tflite` are referenced by the default Stage5 server bundle. `outputs/nima_preprocessing_fix_20260525/report.md` identifies double rescaling, so these files are classified `LEGACY` even though the old server config still points to them.
- Fixed candidate. `src/models/nima_distribution.py` uses EfficientNetV2B0 with `include_preprocessing=False`, GAP, dropout, Dense, 10-bin softmax, EMD loss, and expected-score metrics. `src/train/train_nima.py` uses Adam, checkpointing, and early stopping without a scheduler.
- Split/preprocessing. `src/preprocess/make_ava_csv.py` implements a seed-42 random split. `src/datasets/ava_distribution_dataset.py` uses train resize-min-256, random 224 crop and flip; eval uses deterministic center crop. The expected raw AVA annotation file is absent, so raw source to current processed CSV is `UNKNOWN_LINK`.
- Evaluation/export. `src/infer/predict_nima.py` implements single-image center-crop inference. Fixed Keras, FP32/FP16 TFLite, parity, metrics, diagnostics, and metadata are under `checkpoints/nima_ava_gpu_fixed_preproc_20260525/` and `outputs/nima_fixed_preproc_eval_20260525/`. The report-generating bulk evaluator was not found.
- Mobile candidate. The FP16 file and metadata are copied to `forWeights2/models/aesthetic/`; no active Flutter runtime is proven.

### RGNet practical and paper-v1

- Practical architecture. `src/models/rgnet.py` is a repository RGNet-style EfficientNetV2B0/region-graph approximation. `src/train/train_rgnet.py` uses Adam, MSE, MAE, checkpointing, and early stopping.
- Data. `src/preprocess/make_aadb_csv.py` uses AADB's MAT train/test lists and splits 10% of official train for validation with seed 42. The manifests are `data/processed/aadb/{train,val,test}.csv`.
- Practical runtime. `checkpoints/rgnet_aadb_gpu/` is loaded by the default Stage5 config. TFLite metadata/verification exist in `exports/tflite/`.
- Paper-v1 architecture. `src/models/rgnet_paper_v1.py` uses DenseNet121, ASPP or cascaded DenseASPP, graph layers, and mean/max/LSE aggregation. It is paper-inspired, not an official implementation.
- Paper-v1 train/eval. `src/train/train_rgnet_paper_v1_aadb.py` and `src/eval/evaluate_rgnet_paper_v1_aadb.py` support TF/PIL preprocessing, flip/scale-crop options, Adam or weight decay, MSE/MAE, polynomial LR, checkpointing, and early stopping.
- PIL candidate. `outputs/rgnet_v1_aadb_pil_resize_retrain_20260524/` contains summary, history, checkpoints, evaluations, and predictions. The paired TFLite directory contains FP32/FP16, parity, metadata, and mobile-like predictions. The FP16 copy is in `forWeights2`; Flutter activation is `UNKNOWN_LINK`.

### A-LAMP practical

- Architecture. `src/models/alamp.py` combines an EfficientNetV2B0 global branch, MobileNetV2 patch branch, layout cue branch, learned patch attention, and scalar output.
- Training. `src/train/train_alamp.py` uses `native_size_dataset`, Adam, MSE, MAE, checkpointing, and early stopping.
- Evaluation/inference/export. `src/infer/predict_alamp.py`, `src/export/export_alamp_tflite.py`, and `verify_alamp_tflite.py` cover runtime and mobile conversion.
- Artifacts/runtime. `checkpoints/alamp_aadb_gpu/` and `exports/tflite/alamp_aadb_gpu.tflite` exist. The default Stage5 config directly loads the Keras checkpoint.

### Mobile A-LAMP v2 AVA

- Architecture. `src/models/mobile_alamp_v2.py` uses two MobileNetV3Small backbones, five 224 patches, one 384 global image, layout self-attention, fusion, and sigmoid output.
- Training. `src/train/train_mobile_alamp_v2.py` uses Adam 1e-4, binary crossentropy, accuracy/AUC, class weights, and best-`val_auc` checkpointing. No scheduler or early stopping is defined.
- Data. AVA labels are thresholded at mean >5. The full patch manifests are under `outputs/alamp_v4_full_ava_20260517/subsets/`. No separate image augmentation beyond shuffle is evident in the trainer.
- Artifacts. `outputs/mobile_alamp_v2_full_ava_20260519/` contains best/final Keras, training log, summary, predictions, and reports. `tools/export_mobile_alamp_v2_tflite.py` creates FP32/FP16 artifacts; the FP16 candidate is copied to `forWeights2`.
- Caveats. Exact saliency and V4 patch-generation commands are absent. The run report retains smoke-template wording that conflicts with its full-run summary. Flutter runtime is `UNKNOWN_LINK`.

### A-LAMP research teachers

- Multi-patch teacher. `src/models/alamp_multipatch_teacher.py`, the matching trainer/evaluator, and `configs/paper_benchmarks/alamp_multipatch_teacher_ava.yaml` define a standalone teacher baseline.
- Graph-lite, MPNet, dual-branch, and paper-fusion implementations and configs are also present under `src/models/alamp_*`, `src/train/`, `src/eval/`, and `configs/paper_benchmarks/`.
- Existing teacher-decision reports retain multi-patch as a research teacher and close several dual-branch/graph/fusion branches. None has a default production runtime reference.

### ICAA / ICAA17K

- Official architecture and data. `external/icaa_official_repo/ICAA17K_code/models/dat.py` and related blocks implement DAT. `dataset.py` uses resize 256, random flip/crop 224, and ImageNet normalization for training; evaluation directly resizes 224. The official train/test CSVs and 348,884,485-byte PyTorch checkpoint exist.
- Training. `main_nni.py` shows Adam and MSE, but current training calls and scheduler use are commented, and `config.yml` points to a missing trial script. Official local invocation is `UNKNOWN_LINK`.
- Evaluation. `scripts/evaluate_icaa_official_weight.py` strict-loads the checkpoint.
- TF-native port. `experiments/icaa_tf_native/tf_models/tf_dat_model.py` plus staged parity/export scripts map the PyTorch model to SavedModel and FP16 TFLite. Canonical output is `outputs/icaa_tf_native_tflite_fp16_20260516_144023/icaa_dat_tf_native_fp16.tflite`.
- Predictions/logs. SavedModel and FP16 directories contain reports, logs, and raw predictions; `outputs/icaa_performance_reverification_20260525/` contains full predictions and metrics.
- Mobile/lab conflict. The canonical and `forWeights2` files are both 175,885,416 bytes and hash-identical in this audit. The verified export applies ImageNet normalization outside the model and treats output index 1 as color, while `forWeights2/configs/aesthetic_weight_lab.yaml` specifies zero-to-one and index 0. Therefore lab-to-validated-runtime is `UNKNOWN_LINK`.
- Rejected accelerations. PTQ and QAT reports document parity/runtime failure or infeasibility; those branches are `REJECTED`.

### Stage 5 AADB aesthetic student

- Architecture. `src/distill/modeling.py` builds MobileNetV2 alpha 0.5, GAP, dropout, Dense 128, and a sigmoid scalar.
- Training/evaluation/export. `src/distill/train_student.py`, `evaluate_student.py`, `export_student_tflite.py`, and `verify_student_tflite.py` form the code path.
- Config and objective. `configs/student_distill_full_aadb.json` defines direct bilinear 224, [0,1], no horizontal augmentation, Huber delta 0.05, AdamW 1e-4 with weight decay 1e-5, ReduceLROnPlateau, and validation-Spearman selection.
- Data/teacher target. AADB train/val manifests feed `outputs/distill_full_aadb/teacher_labels_trainval.csv`; the target is the Stage5 aesthetic component, not the whole selector score.
- Artifacts. Full student checkpoints and evaluation live under `checkpoints/student_full_aadb/`; an earlier bounded lineage is under `outputs/distill_exec_20260411/`. `exports/tflite/stage5_student_full_aadb_aesthetic_mobile_fp32.tflite` has metadata and verification.
- Runtime/status. The handoff calls it an app candidate but lists Flutter integration as future work. Later thesis/design reports reject it. No active Flutter reference was found.

## Aesthetic ensembles and adjacent models

### Default Stage5 server ensemble

- Configuration. `configs/stage5_reference.json`.
- Components. AADB composition, KonIQ, FLIVE image, FLIVE patch, legacy NIMA, practical A-LAMP, practical MUSIQ, practical RGNet, and pairwise AADB.
- Runtime. `src/infer/stage5_reference.py` resolves the paths; `run_acut_pipeline.py` enables the bundle; `predict_quality_bundle.py` emits scores; `select_best_shots.py` performs weighted late fusion.
- Aesthetic weights. AADB 0.28, NIMA 0.24, A-LAMP 0.22, RGNet 0.18, pairwise 0.08. Aesthetic/technical component mix is 0.6/0.4.
- Important caveat. The default uses the legacy NIMA and is not the same implementation as the June calibrated three-model mobile recommendation.

### Calibrated three-model candidate

- Inputs. Existing aligned AVA/AADB predictions under `outputs/aesthetic_cross_predictions_for_weight_search_20260603/`.
- Search. `outputs/aesthetic_ensemble_weight_search_complete_20260603/run_weight_search.py` produces all grid results, bootstrap/CV tables, and `final_recommendation.json`.
- Recommendation. NIMA 0.37, RGNet 0.41, A-LAMP 0.22, ICAA 0.
- Config. `outputs/aesthetic_ensemble_weight_search_complete_20260603_audit/aesthetic_ensemble_config.json` records weights and dataset-specific calibration.
- Runtime. No default production source reads that config. Existing reports disagree about submitted app weights and contracts. Search output to active runtime remains `UNKNOWN_LINK`.

### AADB composition regressor and pairwise comparator

- AADB composition. `src/models/aesthetic_nima.py` supplies an EfficientNetV2B0 sigmoid regressor trained through `train_regression.py --task aadb`; `checkpoints/composition_aadb_gpu` and `exports/tflite/aadb_mobile.tflite` exist. The default Stage5 bundle directly loads the Keras checkpoint.
- Pairwise comparator. `src/models/pairwise_comparator.py`, `src/train/train_pairwise.py`, `checkpoints/pairwise_aadb_gpu`, and AADB validation references form the pairwise path. The checkpoint is in the default config, but reranking is disabled unless explicitly enabled.

### VILA and Gemma adjacent scorers

- VILA is supported by `src/infer/run_acut_pipeline.py`, `src/vila/`, and optional rerank/explanation arguments. Large checkpoints exist under `checkpoints/`.
- Gemma teacher/distillation code and configs live under `src/gemma_distill/` and `configs/gemma4_*`.
- These systems are teacher, explanation, or optional reranking infrastructure rather than the core mobile IQA/IAA model assets inventoried above. Representative source/config artifacts are hashed, while their large model-weight shards are recorded by the repository tree rather than the curated 289-artifact hash set.

## Vendored model-zoo boundary

`MDIQA/pyiqa/archs/` contains many additional IQA implementations, including TOPIQ, MUSIQ, NIMA, MANIQA, HyperIQA, PaQ-2-PiQ, DBCNN, CLIPIQA, QualiCLIP, ARNIQA, LIQE, NIQE, BRISQUE, WaDIQaM, and numerous full-reference metrics. Their presence proves bundled third-party source availability, not that the capstone trained, evaluated, exported, or deployed each architecture. With no model-specific local checkpoint/run association, they remain library inventory rather than separately classified capstone model lineages.

## Cross-cutting provenance findings

- Core production checkpoints and processed manifests are ignored/untracked, while several TFLite contracts and source files are tracked. Git provenance is therefore incomplete for reconstructing training.
- Important original checkpoints often lack an exact command, resolved config, and stdout log.
- Later evaluation manifests must not be back-projected into earlier checkpoint training lineages.
- The research repository is not the primary Flutter repository. Mobile handoff and manual external-app reports are evidence, but external asset byte identity and active Dart wiring are not directly proven here.
- Existing reports contain contradictory production weights, model sizes, and runtime-contract statements. Those are inventoried in `existing_claims_inventory.md` and `unresolved_inventory_questions.md`.
