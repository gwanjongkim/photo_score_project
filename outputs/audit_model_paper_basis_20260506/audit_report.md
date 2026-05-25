# A-cut Image Quality and Aesthetic Model Paper-Basis Audit Report

## 1. Investigation Environment

- WSL path: `/home/omen_pc1/photo_score_project`
- Working directory at start: `/home/omen_pc1/photo_score_project`
- Host: `DESKTOP-3T40JVV`
- User: `omen_pc1`
- Audit date/time command output: `Wed May  6 14:13:54 KST 2026`
- `python --version`: `/bin/bash: line 1: python: command not found`
- `python3 --version`: `Python 3.12.3`
- TFLite inspection interpreter used: `/home/omen_pc1/photo_score_project/.venv_gpu/bin/python`
- GPU command output: `nvidia-smi` showed NVIDIA-SMI `590.48.01`, CUDA `13.1`, and one NVIDIA GPU.
- Discovered `photo_score_project` directories:
  - `/home/omen_pc1/photo_score_project`
  - `/home/omen_pc1/photo_score_project/photo_score_project`
- Primary Flutter repo used for app integration evidence: `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app`
  - `git remote -v`: `https://github.com/Mobile-Capstone-HS/pozy_app.git`
  - `git branch --show-current`: `feat/acut`
  - `git rev-parse HEAD`: `8d31cb394fb83baaf43ed84582b0068ae9c1b20e`
  - `git log -1 --oneline`: `8d31cb3 Merge pull request #30 from Mobile-Capstone-HS/fix/#10`
  - `git status --short | wc -l`: `626`
  - Status is dirty. The first status lines include modified `.env.example`, `.gitignore`, `.metadata`, `README.md`, Android files, Flutter files, and model metadata files.
- Secondary Flutter checkout found but not used as primary because it is not on the requested branch:
  - `/home/omen_pc1/pozy_app`
  - branch `feature/firebase-acut-merge`
  - HEAD `99921ebbe43d4f1686e2cf6f7442639d19f89897`
- Inaccessible or absent candidate paths:
  - `/Users/gwanjung_mac/StudioProjects/pozy_app`: not visible from this environment.
- Important command classes used: environment discovery, `find`, `rg`, `nl -ba`, `git`, `du`, TensorFlow Lite model inspection through `.venv_gpu/bin/python`.

## 2. Confirmed Model Summary

| Category | Model | Actual file | Used by Flutter app | WSL creation evidence | Paper-basis judgment |
|---|---|---|---|---|---|
| Aesthetic | NIMA | WSL: `/home/omen_pc1/photo_score_project/exports/tflite/nima_mobile.tflite`; Flutter: `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/assets/models/nima_mobile.tflite` | Yes. Active ensemble contract at `lib/feature/a_cut/layer/inference/aesthetic_model_contract.dart:405-416`, included at `:513-518`. | `src/models/nima_distribution.py`, `src/train/train_nima.py`, AVA manifests, checkpoint `checkpoints/nima_ava_gpu/*`, metadata and verify JSON. | Close to direct paper implementation. |
| Aesthetic | RGNet | WSL: `/home/omen_pc1/photo_score_project/exports/tflite/rgnet_aadb_gpu.tflite`; Flutter: `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/assets/models/rgnet_aadb_gpu.tflite` | Yes. Active ensemble contract at `aesthetic_model_contract.dart:418-431`, included at `:513-518`. | `src/models/rgnet.py`, `src/train/train_rgnet.py`, AADB manifests, checkpoint `checkpoints/rgnet_aadb_gpu/*`, metadata and verify JSON. | Close to RGNet paper implementation by the user rule because graph convolution over a region graph is implemented; architecture is a practical approximation. |
| Aesthetic | A-LAMP / ALAMP | WSL: `/home/omen_pc1/photo_score_project/exports/tflite/alamp_aadb_gpu.tflite`; Flutter: `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/assets/models/alamp_aadb_gpu.tflite` | Yes. Active ensemble contract at `aesthetic_model_contract.dart:433-446`, included at `:513-518`. | `src/models/alamp.py`, `src/train/train_alamp.py`, `src/datasets/native_size_dataset.py`, AADB manifests, checkpoint `checkpoints/alamp_aadb_gpu/*`, metadata and verify JSON. | Paper-inspired implementation, not direct A-LAMP. |
| Technical quality | FLIVE | WSL: `/home/omen_pc1/photo_score_project/exports/tflite/flive_image_mobile.tflite`; Flutter: `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/assets/models/flive_image_mobile.tflite` | Yes. Default technical contract at `aesthetic_model_contract.dart:379-390`, used by `photo_evaluation_service.dart:61-66`. | `src/models/technical_regressor.py`, `src/train/train_regression.py`, `src/preprocess/make_flive_csv.py`, processed PaQ-2-PiQ/FLIVE-named manifests, checkpoint `checkpoints/technical_flive_image_gpu/*`, metadata and verify JSON. | Dataset-based model, not direct paper architecture. |
| Technical quality | KonIQ | WSL: `/home/omen_pc1/photo_score_project/exports/tflite/koniq_mobile.tflite`; Flutter: `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/assets/models/koniq_mobile.tflite` | Yes. Default technical contract at `aesthetic_model_contract.dart:366-377`, used by `photo_evaluation_service.dart:61-66`. | `src/models/technical_regressor.py`, `src/train/train_regression.py`, `src/preprocess/make_koniq_csv.py`, KonIQ manifests, checkpoint `checkpoints/technical_koniq_gpu/*`, metadata and verify JSON. | Dataset-based model, not direct paper architecture. |
| Candidate / idea | MUSIQ | WSL Keras/checkpoint only: `/home/omen_pc1/photo_score_project/checkpoints/musiq_aadb_gpu/*`; no MUSIQ `.tflite` found. | No. No MUSIQ Flutter asset or code reference found in the primary Flutter repo. | `src/models/musiq.py`, `src/train/train_musiq.py`, `src/datasets/native_size_dataset.py`, AADB manifests, checkpoint `checkpoints/musiq_aadb_gpu/*`. | Paper-inspired WSL implementation exists, but MUSIQ is not used by the current Flutter app. |

Additional model artifacts exist but are outside the six requested paper-basis candidates: `composition_aadb_gpu.tflite`, `stage5_student_full_aadb_aesthetic_mobile_fp32.tflite`, `student_conservative_aesthetic_full_aadb_20260423_fp16_builtin.tflite`, `lighting_model.tflite`, `fastscnn_cityscapes_float16.tflite`, `yolo11n.tflite`, and `yolov8n-pose_float16.tflite`.

## 3. Flutter App Integration

- `pubspec.yaml:25-26` declares `flutter_litert` and `flutter_litert_flex`.
- `pubspec.yaml:57-62` registers `assets/models/`, so every file under `assets/models/` is registered as a Flutter asset.
- Primary Flutter app model files under `assets/models/` include:
  - `nima_mobile.tflite`
  - `rgnet_aadb_gpu.tflite`
  - `alamp_aadb_gpu.tflite`
  - `koniq_mobile.tflite`
  - `flive_image_mobile.tflite`
  - `composition_aadb_gpu.tflite`
  - `stage5_student_full_aadb_aesthetic_mobile_fp32.tflite`
  - `student_conservative_aesthetic_full_aadb_20260423_fp16_builtin.tflite`
  - unrelated lighting, segmentation, and YOLO models.
- `android/app/src/main/assets/` contains only `yolo11n.tflite` and `yolov8n-pose_float16.tflite`; the A-cut quality models are Flutter assets under `assets/models/`.

Actual loading code:

- `TfliteInterpreterManager` loads models with `Interpreter.fromAsset` at `lib/feature/a_cut/layer/inference/tflite_interpreter_manager.dart:45`.
- It adds a Flex delegate only when a contract sets or metadata resolves `useFlexDelegate` at `tflite_interpreter_manager.dart:36-43`.
- Active NIMA, RGNet, A-LAMP, KonIQ, and FLIVE assets inspected locally allocate successfully without Select TF ops.

Actual evaluation flow in the primary `feat/acut` checkout:

- `SinglePhotoEvalScreen` defaults to `HybridPhotoEvaluationService` at `lib/screen/single_photo_eval_screen.dart:45-47`.
- `ACutResultScreen` also uses on-device scoring through `OnDeviceImageScoreService(evaluationService: HybridPhotoEvaluationService())` at `lib/screen/a_cut_result_screen.dart:39-41`.
- `OnDeviceImageScoreService` reads each selected asset's bytes and calls the evaluation service at `lib/feature/a_cut/layer/scoring/image_scoring_service.dart:73-82`.
- `HybridPhotoEvaluationService` states that scoring is on-device and explanations are separate at `lib/feature/a_cut/layer/evaluation/hybrid_photo_evaluation_service.dart:12-22`.
- `HybridPhotoEvaluationService.evaluate` calls the scorer first at `hybrid_photo_evaluation_service.dart:42-43`, then calls an explainer with precomputed scores at `:45-54`. Explanation failure is non-fatal at `:64-68`.

Aesthetic ensemble method:

- `activeAestheticEnsembleContracts` contains NIMA, RGNet, and A-LAMP at `aesthetic_model_contract.dart:513-518`.
- `AestheticEnsembleScoringService` defaults to those active contracts at `lib/feature/a_cut/layer/evaluation/aesthetic_ensemble_scoring_service.dart:20`.
- Default aesthetic weights are explicit at `lib/feature/a_cut/model/aesthetic_ensemble_weights.dart:1-10`:
  - NIMA: `0.10`
  - RGNet: `0.50`
  - A-LAMP: `0.40`
  - Sum: `1.00`
- The weights are normalized in `aesthetic_ensemble_weights.dart:74-92`.
- The weighted aesthetic score is computed at `aesthetic_ensemble_weights.dart:62-71`.
- `AestheticEnsembleScoringService` requires all three model scores for `finalAestheticScore`; if any of NIMA, RGNet, or A-LAMP fails, final aesthetic is `null` at `aesthetic_ensemble_scoring_service.dart:56-66`.

Technical quality model usage:

- `defaultTechnicalModelContracts` contains KonIQ and FLIVE at `aesthetic_model_contract.dart:448-451`.
- KonIQ contract weight is `0.6` at `aesthetic_model_contract.dart:366-377`.
- FLIVE contract weight is `0.4` at `aesthetic_model_contract.dart:379-390`.
- `OnDevicePhotoEvaluationService` constructs a technical-only `TfliteAestheticService` using those contracts at `photo_evaluation_service.dart:61-66`.
- `TfliteAestheticService._blend` divides weighted sum by total available weight at `tflite_aesthetic_service.dart:656-671`.

Final score behavior:

- `OnDevicePhotoEvaluationService` computes final score as technical-only if aesthetic score is unavailable, otherwise `0.5 * technical + 0.5 * aesthetic` at `photo_evaluation_service.dart:98-104`.
- If the aesthetic ensemble cannot run, it catches the error and adds a warning at `photo_evaluation_service.dart:85-92`.
- If an individual aesthetic model fails, the ensemble service catches that model error and records a warning at `aesthetic_ensemble_scoring_service.dart:35-43`.
- If all technical models fail, `TfliteAestheticService.evaluate` throws `No technical quality model could be executed` at `tflite_aesthetic_service.dart:99-108`.
- If one technical model fails but the other succeeds, `TfliteAestheticService.evaluate` continues with available technical details because each model run is caught at `tflite_aesthetic_service.dart:83-96` and blend re-normalizes by available weight.
- `MockPhotoEvaluationService` exists at `photo_evaluation_service.dart:19-54`, but default app construction uses `HybridPhotoEvaluationService`, not the mock.

Metadata behavior:

- FLIVE and KonIQ metadata JSON files are present in Flutter assets.
- Active NIMA, RGNet, and A-LAMP `.metadata.json` files are not present in Flutter `assets/models/` based on local `find` output, so the app uses contract fallback values for those models. Metadata load failure returns a warning, not an exception, at `tflite_model_metadata_loader.dart:70-74`.
- WSL metadata files do exist for NIMA, RGNet, and A-LAMP under `/home/omen_pc1/photo_score_project/exports/tflite/`.

## 4. WSL Model Creation Trace

### NIMA

- Dataset manifest:
  - `/home/omen_pc1/photo_score_project/data/processed/ava/train_cleaned.csv`
  - `/home/omen_pc1/photo_score_project/data/processed/ava/val_cleaned.csv`
  - Header includes `vote_1` through `vote_10`, `dist_1` through `dist_10`, and `mean_score`.
- Dataset creation evidence:
  - `src/preprocess/make_ava_csv.py:3-24` documents AVA output files and histogram fields.
  - `src/preprocess/make_ava_csv.py:118-130` computes vote distributions and mean score.
- Training script:
  - `src/train/train_nima.py:27-37` defines train/val CSV and output arguments.
  - `src/train/train_nima.py:42-63` loads AVA distribution datasets.
  - `src/train/train_nima.py:75-80` builds NIMA and compiles EMD loss.
  - `src/train/train_nima.py:82-104` saves checkpoints and exports the final model.
  - `src/train.sh:43-48` runs `src/train/train_nima.py` against cleaned AVA CSVs.
- Checkpoint:
  - `/home/omen_pc1/photo_score_project/checkpoints/nima_ava_gpu/best.weights.h5`
  - `/home/omen_pc1/photo_score_project/checkpoints/nima_ava_gpu/final_model.keras`
- Export:
  - `/home/omen_pc1/photo_score_project/exports/tflite/nima_mobile.tflite`
  - `/home/omen_pc1/photo_score_project/exports/tflite/nima_mobile.metadata.json`
  - `/home/omen_pc1/photo_score_project/exports/tflite/nima_mobile.verify.json`
- Flutter copy path:
  - `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/assets/models/nima_mobile.tflite`
  - Exact copy command is not verifiable from local evidence.
- Validation metrics:
  - TFLite/source parity verification is present in `nima_mobile.verify.json`.
  - Training validation curves or saved training history for this checkpoint are not verifiable from local evidence.

### RGNet

- Dataset manifest:
  - `/home/omen_pc1/photo_score_project/data/processed/aadb/train.csv`
  - `/home/omen_pc1/photo_score_project/data/processed/aadb/val.csv`
  - Header includes `image_path,score,dataset,split,relative_path`.
- Training script:
  - `src/train/train_rgnet.py:25-35` defines train/val CSV, target column, and image size.
  - `src/train/train_rgnet.py:40-54` builds regression datasets.
  - `src/train/train_rgnet.py:56-61` compiles with MSE and MAE.
  - `src/train/train_rgnet.py:63-88` saves checkpoint and final model.
  - `src/sequential_train.sh:54-67` shows the RGNet AADB training invocation.
- Checkpoint:
  - `/home/omen_pc1/photo_score_project/checkpoints/rgnet_aadb_gpu/best.weights.h5`
  - `/home/omen_pc1/photo_score_project/checkpoints/rgnet_aadb_gpu/final_model.keras`
- Export:
  - `/home/omen_pc1/photo_score_project/exports/tflite/rgnet_aadb_gpu.tflite`
  - `/home/omen_pc1/photo_score_project/exports/tflite/rgnet_aadb_gpu.metadata.json`
  - `/home/omen_pc1/photo_score_project/exports/tflite/rgnet_aadb_gpu.verify.json`
  - Flex variant also exists: `/home/omen_pc1/photo_score_project/exports/tflite/rgnet_aadb_gpu_flex.tflite`
- Flutter copy path:
  - `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/assets/models/rgnet_aadb_gpu.tflite`
  - Exact copy command is not verifiable from local evidence.
- Validation metrics:
  - TFLite/source parity verification is present in `rgnet_aadb_gpu.verify.json`.
  - Training validation curves or saved training history for this checkpoint are not verifiable from local evidence.

### A-LAMP / ALAMP

- Dataset manifest:
  - `/home/omen_pc1/photo_score_project/data/processed/aadb/train.csv`
  - `/home/omen_pc1/photo_score_project/data/processed/aadb/val.csv`
- Training script:
  - `src/train/train_alamp.py:25-38` defines train/val CSV and target arguments.
  - `src/train/train_alamp.py:43-63` builds A-LAMP datasets.
  - `src/train/train_alamp.py:65-74` builds and compiles MSE/MAE training.
  - `src/train/train_alamp.py:87-91` saves and exports.
  - `src/sequential_train.sh:39-52` shows the A-LAMP AADB training invocation.
- Checkpoint:
  - `/home/omen_pc1/photo_score_project/checkpoints/alamp_aadb_gpu/best.weights.h5`
  - `/home/omen_pc1/photo_score_project/checkpoints/alamp_aadb_gpu/final_model.keras`
- Export:
  - `/home/omen_pc1/photo_score_project/exports/tflite/alamp_aadb_gpu.tflite`
  - `/home/omen_pc1/photo_score_project/exports/tflite/alamp_aadb_gpu.metadata.json`
  - `/home/omen_pc1/photo_score_project/exports/tflite/alamp_aadb_gpu.verify.json`
- Flutter copy path:
  - `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/assets/models/alamp_aadb_gpu.tflite`
  - Exact copy command is not verifiable from local evidence.
- Validation metrics:
  - TFLite/source parity verification is present in `alamp_aadb_gpu.verify.json`.
  - Training validation curves or saved training history for this checkpoint are not verifiable from local evidence.

### FLIVE

- Dataset manifest:
  - `/home/omen_pc1/photo_score_project/data/processed/paq2piq/image_train.csv`
  - `/home/omen_pc1/photo_score_project/data/processed/paq2piq/image_val.csv`
  - Header includes `image_path,mos,dataset,is_patch,relative_path`.
- Dataset creation evidence:
  - `src/preprocess/make_flive_csv.py:5-8` uses raw path `data/raw/paq2piq` and processed path `data/processed/paq2piq`.
  - `src/preprocess/make_flive_csv.py:12-24` writes image-level labels with dataset name `flive_image`.
  - `src/data/build_flive_csv.py:36-46` also maps `labels_image.csv` to `dataset=flive_image`.
- Training script:
  - `src/train/train_regression.py:23-31` defines arguments for generic regression.
  - `src/train/train_regression.py:54-58` chooses technical model construction.
  - `src/train/train_regression.py:59-63` compiles with MSE and MAE.
- Checkpoint:
  - `/home/omen_pc1/photo_score_project/checkpoints/technical_flive_image_gpu/best.weights.h5`
  - `/home/omen_pc1/photo_score_project/checkpoints/technical_flive_image_gpu/final_model.keras`
- Export:
  - `/home/omen_pc1/photo_score_project/exports/tflite/flive_image_mobile.tflite`
  - `/home/omen_pc1/photo_score_project/exports/tflite/flive_image_mobile.metadata.json`
  - `/home/omen_pc1/photo_score_project/exports/tflite/flive_image_mobile.verify.json`
- Flutter copy path:
  - `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/assets/models/flive_image_mobile.tflite`
  - `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/assets/models/flive_image_mobile.metadata.json`
  - Exact copy command is not verifiable from local evidence.

### KonIQ

- Dataset manifest:
  - `/home/omen_pc1/photo_score_project/data/processed/koniq10k/train.csv`
  - `/home/omen_pc1/photo_score_project/data/processed/koniq10k/val.csv`
  - Header includes `image_path,mos,dataset,is_patch,relative_path`.
- Dataset creation evidence:
  - `src/preprocess/make_koniq_csv.py:5-8` defines raw and processed paths.
  - `src/preprocess/make_koniq_csv.py:27-41` detects image and MOS columns.
  - `src/preprocess/make_koniq_csv.py:43-55` writes dataset `koniq10k` with MOS.
  - `src/preprocess/make_koniq_csv.py:58-67` writes train/val/test splits.
- Training script:
  - `src/train/train_regression.py:23-31`, `:54-63`, and `:89-93`.
- Checkpoint:
  - `/home/omen_pc1/photo_score_project/checkpoints/technical_koniq_gpu/best.weights.h5`
  - `/home/omen_pc1/photo_score_project/checkpoints/technical_koniq_gpu/final_model.keras`
- Export:
  - `/home/omen_pc1/photo_score_project/exports/tflite/koniq_mobile.tflite`
  - `/home/omen_pc1/photo_score_project/exports/tflite/koniq_mobile.metadata.json`
  - `/home/omen_pc1/photo_score_project/exports/tflite/koniq_mobile.verify.json`
- Flutter copy path:
  - `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/assets/models/koniq_mobile.tflite`
  - `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/assets/models/koniq_mobile.metadata.json`
  - Exact copy command is not verifiable from local evidence.

### MUSIQ

- Dataset manifest:
  - `/home/omen_pc1/photo_score_project/data/processed/aadb/train.csv`
  - `/home/omen_pc1/photo_score_project/data/processed/aadb/val.csv`
- Training script:
  - `src/train/train_musiq.py:33-49` defines target column, scale sizes, patch size, and patches per scale.
  - `src/train/train_musiq.py:55-73` builds MUSIQ datasets.
  - `src/train/train_musiq.py:75-89` builds and compiles MSE/MAE training.
  - `src/train/train_musiq.py:102-106` fits, saves, and exports a Keras model.
  - `src/sequential_train.sh:24-37` shows the MUSIQ AADB training invocation.
- Checkpoint:
  - `/home/omen_pc1/photo_score_project/checkpoints/musiq_aadb_gpu/best.weights.h5`
  - `/home/omen_pc1/photo_score_project/checkpoints/musiq_aadb_gpu/final_model.keras`
- Export:
  - No MUSIQ `.tflite`, metadata JSON, or verify JSON found from local `find` commands.
- Flutter copy path:
  - No MUSIQ Flutter asset or code usage found.

## 5. Paper Comparison

### 5.1 NIMA

Paper checklist:

- 10 score buckets.
- Softmax distribution output.
- EMD or CDF-based loss.
- AVA, TID2013, or LIVE human rating distribution data.
- ImageNet-pretrained CNN backbone.
- Final score computed as expectation of the predicted distribution.

Local evidence:

- `src/models/nima_distribution.py:27-40` builds an ImageNet-pretrained EfficientNetV2B0 backbone and a `Dense(10, activation="softmax")` output.
- `src/models/nima_distribution.py:43-50` implements EMD loss through CDFs using `tf.cumsum`.
- `src/models/nima_distribution.py:53-55` computes distribution mean with weights `1..10`.
- `src/datasets/ava_distribution_dataset.py:65-73` detects `dist_`, `score_`, or `vote_` 10-bin labels.
- `src/datasets/ava_distribution_dataset.py:88-104` normalizes distributions and computes mean score.
- `src/train/train_nima.py:75-80` compiles with `emd_loss`.
- TFLite inspection confirms `/assets/models/nima_mobile.tflite` has input `[1,224,224,3]` and output `[1,10]`.
- Flutter normalizes distribution score by expectation at `aesthetic_model_contract.dart:339-362`.

Matching points:

- 10-bin distribution output exists.
- Softmax output exists.
- CDF/EMD loss exists.
- AVA distribution data exists.
- ImageNet-pretrained CNN backbone exists.
- Expected-value scoring exists in WSL and Flutter.

Differences:

- The backbone is EfficientNetV2B0, not one of the original NIMA examples such as MobileNet, VGG, or Inception. This is still an ImageNet-pretrained CNN backbone.

Final judgment:

- Close to direct NIMA implementation.

### 5.2 RGNet

Paper checklist:

- DenseNet-121 or FCN feature encoder.
- Local region feature map.
- Region composition graph.
- Feature similarity adjacency matrix.
- Graph convolution.
- Region-level map aggregated into image-level score.
- AVA or AADB aesthetic dataset evidence.

Local evidence:

- `src/models/rgnet.py:32-52` uses EfficientNetV2B0 as the feature encoder.
- `src/models/rgnet.py:55-70` builds a similarity adjacency matrix from L2-normalized region nodes.
- `src/models/rgnet.py:76-105` implements graph convolution.
- `src/models/rgnet.py:126-152` builds feature maps, region nodes, region graph, GCN blocks, attention fusion, and sigmoid image score.
- `src/train/train_rgnet.py:40-61` trains on AADB regression data with MSE/MAE.
- TFLite inspection confirms `/assets/models/rgnet_aadb_gpu.tflite` has graph-related ops including `BATCH_MATMUL`, `L2_NORMALIZATION`, `MATRIX_DIAG`, and `SOFTMAX`.

Matching points:

- Local region feature map exists.
- Region graph and similarity adjacency matrix exist.
- Graph convolution exists.
- AADB aesthetic dataset evidence exists.

Differences:

- The local model uses EfficientNetV2B0, not DenseNet-121.
- The local implementation is documented as a practical RGNet-style approximation, not a line-by-line reproduction.

Final judgment:

- Close to RGNet paper implementation under the audit rule because graph convolution over a region composition graph exists.

### 5.3 A-LAMP / ALAMP

Paper checklist:

- Adaptive patch selection.
- Saliency-map-based patch selection.
- Diversity or overlap/spatial-distance constraint.
- Multi-patch subnet.
- 5 x 224 x 224 patch bag or equivalent.
- VGG16 shared-column CNN.
- Layout-aware subnet.
- Object/global attribute graph.
- Multi-patch feature aggregation.

Local evidence:

- `src/models/alamp.py:52-67` adds layout cues.
- `src/models/alamp.py:83-89` defines a shared patch encoder based on MobileNetV2.
- `src/models/alamp.py:92-99` defines a 384 global input and `[5,224,224,3]` patch input.
- `src/models/alamp.py:101-118` builds global and layout branches.
- `src/models/alamp.py:120-130` applies patch encoding and patch attention.
- `src/models/alamp.py:133-142` fuses global, layout, and patch features into scalar score.
- `src/datasets/native_size_dataset.py:60-81` computes saliency from edge, local variance, and color variance.
- `src/datasets/native_size_dataset.py:84-153` proposes adaptive boxes and applies non-max suppression.
- `src/datasets/native_size_dataset.py:156-175` prepares padded global view and 5 cropped patches.
- TFLite inspection confirms `/assets/models/alamp_aadb_gpu.tflite` has inputs `[1,384,384,3]` and `[1,5,224,224,3]`.

Matching points:

- Adaptive saliency-like patch selection exists in WSL training/input code.
- Five-patch bag exists.
- Multi-patch branch and patch aggregation exist.
- Layout/global branch exists.

Differences:

- No local evidence of VGG16 shared-column CNN; the patch encoder uses MobileNetV2.
- No local evidence of the paper's object/global attribute graph.
- Flutter preprocessing for A-LAMP does not match WSL adaptive saliency patch selection. Flutter uses fixed anchor crops at `image_preprocessor.dart:114-181`.
- Flutter global input uses direct square resize at `image_preprocessor.dart:84-89`, while WSL A-LAMP input uses aspect-preserving `resize_with_pad` at `src/datasets/native_size_dataset.py:43-45`.

Final judgment:

- Paper-inspired implementation, not direct A-LAMP.

### 5.4 FLIVE / KonIQ

Dataset or architecture distinction:

- Local FLIVE and KonIQ models are not implemented as named paper architectures.
- Both use the same generic technical regressor architecture.

Local evidence:

- `src/models/technical_regressor.py:3-17` defines a MobileNetV2 ImageNet backbone, global average pooling, dropout, dense layer, and scalar output.
- `src/train/train_regression.py:54-63` builds either AADB or technical regressor and compiles MSE/MAE.
- KonIQ dataset evidence exists in `src/preprocess/make_koniq_csv.py:43-55`, which writes MOS data as `koniq10k`.
- FLIVE local naming evidence exists in `src/preprocess/make_flive_csv.py:12-24`, but the raw path is `data/raw/paq2piq` at `src/preprocess/make_flive_csv.py:5-8`.
- Both Flutter contracts mark outputs as MOS-like percent scalars and normalize by `/100`:
  - KonIQ: `aesthetic_model_contract.dart:366-377`
  - FLIVE: `aesthetic_model_contract.dart:379-390`
  - Normalization code: `aesthetic_model_contract.dart:353-362`

Final judgment:

- FLIVE: Dataset-based technical quality model, not direct paper architecture. Exact FLIVE-vs-PaQ-2-PiQ dataset identity is not verifiable from local evidence because local scripts name `flive_image` while reading `data/raw/paq2piq`.
- KonIQ: Dataset-based technical quality model, not direct paper architecture.

### 5.5 MUSIQ

Actual local usage evidence:

- WSL model code exists: `src/models/musiq.py`.
- WSL training script exists: `src/train/train_musiq.py`.
- WSL checkpoint exists: `/home/omen_pc1/photo_score_project/checkpoints/musiq_aadb_gpu/`.
- No MUSIQ `.tflite` found in WSL exports.
- No MUSIQ asset or code reference found in the primary Flutter repo.

Paper checklist if implemented:

- `src/models/musiq.py:31-52` implements a Transformer block.
- `src/models/musiq.py:55-117` builds token, position, scale, mask, transformer, and scalar score components.
- `src/datasets/native_size_dataset.py:184-241` builds multi-scale tokens from scale sizes.
- `src/train/train_musiq.py:33-49` defaults to scale sizes `224,384,512`.

Differences:

- The local implementation uses learned position embeddings; exact hash-based 2D spatial embedding is not verifiable from local evidence.
- No Flutter deployment exists.

Final judgment:

- WSL has a paper-inspired MUSIQ implementation, but MUSIQ is not used in the current Flutter app.

## 6. TFLite Input/Output and Preprocessing Verification

### Main Flutter A-cut TFLite files

| Model | Flutter file | Size | MTime | Input tensors | Output tensors | Quantization | Ops summary |
|---|---:|---:|---|---|---|---|---|
| NIMA | `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/assets/models/nima_mobile.tflite` | 24765068 | `2026-04-29 20:08:13 +0900` | `serving_default_keras_tensor_270:0`, `[1,224,224,3]`, float32 | `StatefulPartitionedCall_1:0`, `[1,10]`, float32 | input/output `(0.0, 0)` | 352 ops; includes `SOFTMAX`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`. |
| RGNet | `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/assets/models/rgnet_aadb_gpu.tflite` | 26085072 | `2026-04-29 20:08:14 +0900` | `serving_default_keras_tensor_270:0`, `[1,256,256,3]`, float32 | `StatefulPartitionedCall_1:0`, `[1,1]`, float32 | input/output `(0.0, 0)` | 410 ops; includes `BATCH_MATMUL`, `L2_NORMALIZATION`, `MATRIX_DIAG`, `SOFTMAX`. |
| A-LAMP | `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/assets/models/alamp_aadb_gpu.tflite` | 39158340 | `2026-04-29 20:08:13 +0900` | `serving_default_global_view:0`, `[1,384,384,3]`, float32; `serving_default_patches:0`, `[1,5,224,224,3]`, float32 | `StatefulPartitionedCall_1:0`, `[1,1]`, float32 | input/output `(0.0, 0)` | 717 ops; includes `SOFTMAX`, `SUM`, `TANH`, `CONCATENATION`. |
| FLIVE | `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/assets/models/flive_image_mobile.tflite` | 9545852 | `2026-04-29 20:08:13 +0900` | `serving_default_keras_tensor_154:0`, `[1,224,224,3]`, float32 | `StatefulPartitionedCall_1:0`, `[1,1]`, float32 | input/output `(0.0, 0)` | 66 ops; MobileNetV2-like conv/depthwise/full-connected ops. |
| KonIQ | `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/assets/models/koniq_mobile.tflite` | 9545852 | `2026-04-29 20:08:13 +0900` | `serving_default_keras_tensor_154:0`, `[1,224,224,3]`, float32 | `StatefulPartitionedCall_1:0`, `[1,1]`, float32 | input/output `(0.0, 0)` | 66 ops; MobileNetV2-like conv/depthwise/full-connected ops. |
| MUSIQ | None | N/A | N/A | N/A | N/A | N/A | No MUSIQ `.tflite` found. |

Signature evidence:

- `nima_mobile.tflite`: signature `serving_default`, input `keras_tensor_270`, output `output_0`.
- `rgnet_aadb_gpu.tflite`: signature `serving_default`, input `keras_tensor_270`, output `output_0`.
- `alamp_aadb_gpu.tflite`: signature `serving_default`, inputs `global_view`, `patches`, output `output_0`.
- `koniq_mobile.tflite`: signature `serving_default`, input `keras_tensor_154`, output `output_0`.
- `flive_image_mobile.tflite`: signature `serving_default`, input `keras_tensor_154`, output `output_0`.

### Preprocessing comparison

| Model | Training/export preprocessing evidence | Flutter preprocessing evidence | Match/mismatch judgment |
|---|---|---|---|
| NIMA | `src/datasets/ava_distribution_dataset.py:45-61` decodes RGB, converts to float32 `[0,1]`, train resize/crop/flip, eval direct resize. | `image_preprocessor.dart:84-99` direct resize, RGB order, channel `/255`; contract `224x224` at `aesthetic_model_contract.dart:405-416`. | Match for eval/export path: RGB, NHWC, float32, `[0,1]`, 224 square direct resize. |
| RGNet | `src/datasets/csv_dataset.py:7-19` and `train_rgnet.py:25-35` indicate RGB float `[0,1]`, image size 256, eval direct resize. | Signature input path derives 256 and calls `preprocessToRgbFloat32` at `tflite_aesthetic_service.dart:461-497`; direct resize at `image_preprocessor.dart:84-89`. | Match for fixed resize, RGB, `[0,1]`, 256. |
| A-LAMP | WSL uses aspect-preserving global `resize_with_pad` at `src/datasets/native_size_dataset.py:43-45`; saliency/adaptive patches at `:60-153`; 5 patches at `:156-175`. | Flutter uses direct global square resize at `tflite_aesthetic_service.dart:540-548`; fixed crop anchor patch batch at `image_preprocessor.dart:114-181`. | Mismatch. Flutter does not reproduce WSL saliency/adaptive patch selection or padded global preprocessing. |
| FLIVE | `src/datasets/csv_dataset.py:7-19` RGB float `[0,1]`, train crop/flip, eval direct resize; `tflite_presets.py:71-87` indicates 224 RGB `/255`. | Contract `224x224`, RGB `/255`, scalarPercent at `aesthetic_model_contract.dart:379-390`; direct resize at `image_preprocessor.dart:84-99`. | Match for eval/export path. |
| KonIQ | `src/datasets/csv_dataset.py:7-19`; `tflite_presets.py:53-70` indicates 224 RGB `/255`. | Contract `224x224`, RGB `/255`, scalarPercent at `aesthetic_model_contract.dart:366-377`; direct resize at `image_preprocessor.dart:84-99`. | Match for eval/export path. |
| MUSIQ | WSL tokenization in `src/datasets/native_size_dataset.py:184-241`; multi-scale in `src/train/train_musiq.py:33-49`. | No Flutter model or preprocessing path. | Not used in Flutter. |

No evidence of BGR channel order was found for these five deployed quality/aesthetic models. Flutter uses RGB channel order at `image_preprocessor.dart:94-99` and `:147-152`.

## 7. Score Scale Verification

| Model | Raw output | Flutter postprocessing | Normalized range evidence | Scale mismatch risk |
|---|---|---|---|---|
| NIMA | 10-bin distribution | `readRawScore` computes expectation at `aesthetic_model_contract.dart:339-350`; `normalizeOutput` converts `(mean - 1) / 9` at `:360-361`. | Contract output type distribution at `aesthetic_model_contract.dart:405-416`; TFLite output `[1,10]`. | Low for scale. Metadata missing in Flutter, but fallback contract matches inspected output. |
| RGNet | Scalar | Clamps scalar unit interval at `aesthetic_model_contract.dart:358-359`. | Contract scalarUnitInterval at `aesthetic_model_contract.dart:418-431`; TFLite output `[1,1]`. | Low for scale if model output remains sigmoid `[0,1]`; metadata missing in Flutter. |
| A-LAMP | Scalar | Clamps scalar unit interval at `aesthetic_model_contract.dart:358-359`. | Contract scalarUnitInterval at `aesthetic_model_contract.dart:433-446`; TFLite output `[1,1]`. | Low for scale, but high preprocessing mismatch risk. |
| FLIVE | MOS-like scalar | Divides by 100 and clamps at `aesthetic_model_contract.dart:356-357`. | Contract scalarPercent at `aesthetic_model_contract.dart:379-390`; metadata says MOS-like `/100`. | Low for scale. |
| KonIQ | MOS-like scalar | Divides by 100 and clamps at `aesthetic_model_contract.dart:356-357`. | Contract scalarPercent at `aesthetic_model_contract.dart:366-377`; metadata says MOS-like `/100`. | Low for scale. |
| MUSIQ | WSL scalar Keras output | No Flutter postprocessing. | No Flutter asset or code. | Not used. |

Ensemble verification:

- Technical weights are explicit: KonIQ `0.6`, FLIVE `0.4`; sum is `1.0`.
- Aesthetic weights are explicit: NIMA `0.10`, RGNet `0.50`, A-LAMP `0.40`; sum is `1.0`.
- `AestheticEnsembleWeights` normalizes any supplied weights at `aesthetic_ensemble_weights.dart:74-92`.
- `TfliteAestheticService._blend` re-normalizes by total available weight at `tflite_aesthetic_service.dart:656-671`.
- Final on-device score is `0.5 * technical + 0.5 * aesthetic` only when all three aesthetic scores exist; otherwise final score is technical-only at `photo_evaluation_service.dart:98-104`.
- Displayed score source in the primary A-cut screen is model-derived on-device scoring, not Firebase backend output, based on `a_cut_result_screen.dart:39-41` and `image_scoring_service.dart:73-82`.

## 8. Final Classification

| Model | Classification | Local usage note |
|---|---|---|
| NIMA | Close to direct paper implementation | Used by primary Flutter app in active aesthetic ensemble. |
| RGNet | Close to direct paper implementation | Used by primary Flutter app in active aesthetic ensemble. |
| A-LAMP / ALAMP | Paper-inspired implementation | Used by primary Flutter app in active aesthetic ensemble, but Flutter preprocessing differs from WSL training/export preprocessing. |
| FLIVE | Dataset-based model, not direct paper architecture | Used by primary Flutter app as technical quality model. |
| KonIQ | Dataset-based model, not direct paper architecture | Used by primary Flutter app as technical quality model. |
| MUSIQ | Paper-inspired implementation | WSL code/checkpoint exists, but no Flutter usage and no TFLite export found. |

## 9. Reproducibility Issues

- Git clone alone is not verifiable as sufficient.
  - WSL local `checkpoints/` size: `24G`.
  - WSL local `data/` size: `19G`.
  - WSL local `exports/` size: `180M`.
  - Flutter `assets/models/` size: `149M`.
- Git LFS requirement is not verifiable from local evidence.
  - No top-level `.gitattributes` found under `/home/omen_pc1/photo_score_project` at max depth 2.
  - `.gitattributes` files exist under `/home/omen_pc1/photo_score_project/checkpoints/NVILA-8B/` and `/home/omen_pc1/photo_score_project/checkpoints/VILA1.5-3b/`, which are outside the audited deployed TFLite model set.
- The primary Flutter checkout is dirty with `626` status lines. Reproducing exact current app behavior from the branch alone is not verifiable from local evidence.
- WSL training data and checkpoints are large local artifacts. Their availability after a clean clone is not verifiable from local evidence.
- Exact commands that copied WSL TFLite files into Flutter `assets/models/` are not verifiable from local evidence.
- Training validation histories for the direct NIMA, RGNet, A-LAMP, FLIVE, and KonIQ checkpoints are not verifiable from local evidence. TFLite parity verify JSON files exist.
- Active Flutter NIMA, RGNet, and A-LAMP metadata JSON files are absent from `assets/models/`; the app relies on hard-coded fallback contracts for those models.
- `/usr/bin/python` or `python` is not available; scripts require `python3` or the project virtual environment.

## 10. Items Requiring Further Confirmation

- Exact official dataset provenance for the local FLIVE model is not verifiable from local evidence because local code uses `data/raw/paq2piq` while naming the processed dataset `flive_image`.
- Exact copy/handoff command from WSL exports to Flutter assets is not verifiable from local evidence.
- Training logs and validation metric histories for the direct model checkpoints are not verifiable from local evidence.
- Whether the absent Flutter metadata JSON files for active NIMA, RGNet, and A-LAMP were intentionally omitted is not verifiable from local evidence.
- Whether production users receive the exact dirty `feat/acut` tree inspected here is not verifiable from local evidence.
- MUSIQ deployment status beyond local absence from Flutter assets/code is not verifiable from local evidence.
