# Independent Review of A-cut Model Paper-Basis Audit

## 1. Overall Verdict
The original audit report is **reliable**. It thoroughly distinguishes between local WSL training code, model artifacts, and actual Flutter runtime usage, grounding its claims in explicit line numbers and TFLite inspection rather than relying solely on filenames. Its conclusions appropriately identify discrepancies (e.g., backbone substitutions, A-LAMP preprocessing mismatches) and state what cannot be verified locally.

## 2. Claim-by-Claim Review

| Claim | Review result | Evidence found | Problem, if any | Recommended correction |
|---|---|---|---|---|
| 1. Flutter uses KonIQ + FLIVE for technical | Supported | `aesthetic_model_contract.dart:366-390`; TFLite models exist in `assets/models/`. | None | None |
| 2. Aesthetic ensemble uses NIMA + RGNet + A-LAMP | Supported | `aesthetic_model_contract.dart:513-518`; weights at `aesthetic_ensemble_weights.dart:1-10`. | None | None |
| 3. MUSIQ WSL only, no Flutter | Supported | WSL checkpoints at `checkpoints/musiq_aadb_gpu/`, `src/models/musiq.py`; no `.tflite` or Dart usage found. | None | None |
| 4. NIMA is close to direct paper implementation | Supported | `nima_distribution.py:43-50` (EMD loss), 10-bin output, TFLite output `[1,10]`. | Backbone is EfficientNetV2B0 instead of original. | None; report acknowledges backbone shift. |
| 5. RGNet is close to direct paper implementation | Supported | `rgnet.py:55-152` implements similarity matrix and graph convolution. | Backbone is EfficientNetV2B0 instead of DenseNet-121. | None; report explicitly calls it a "practical approximation." |
| 6. A-LAMP is paper-inspired | Supported | `alamp.py:101-130` uses multi-patch subnet; Flutter `image_preprocessor.dart:114-181` uses fixed anchor crops instead of adaptive saliency. | None | None |
| 7. FLIVE/KonIQ are dataset-based | Supported | `technical_regressor.py:3-17` uses generic MobileNetV2 with GAP for both. | None | None |

## 3. Model Classification Review

| Model | Original classification | Reviewer judgment | Reason |
|---|---|---|---|
| NIMA | Close to direct paper implementation | Agree | EMD/CDF loss and 10-bin expectation scoring are present. CNN backbone substitution is standard practice. |
| RGNet | Close to direct paper implementation | Agree | Implements core paper contributions: similarity adjacency matrix and region graph convolution, despite backbone upgrade. |
| A-LAMP | Paper-inspired implementation | Agree | WSL training includes saliency selection, but Flutter deployment uses fixed crops, failing to implement the paper's adaptive patch selection at runtime. |
| FLIVE | Dataset-based model, not direct paper architecture | Agree | No custom FLIVE architecture; relies on standard MobileNetV2 regressor trained on PaQ-2-PiQ data. |
| KonIQ | Dataset-based model, not direct paper architecture | Agree | No custom KonIQ architecture; relies on standard MobileNetV2 regressor trained on KonIQ-10k data. |
| MUSIQ | Paper-inspired implementation (not used) | Agree | Multi-scale Transformer logic exists in WSL, but no Flutter deployment or TFLite export is present. |

## 4. Evidence Quality Review
- **file paths:** Excellent. Clearly lists WSL paths (`/home/omen_pc1/...`) versus Flutter paths (`/mnt/c/Users/...`).
- **line numbers:** Excellent. Direct references to model implementations and Flutter integration logic.
- **command outputs:** Sufficient. `command_log.txt` verifies directory listings, file sizes, and Git states.
- **model metadata:** Checked. Report correctly notes missing metadata files in Flutter and verifies fallback logic.
- **TFLite inspection:** Excellent. Validates input/output shapes and specific ops (`BATCH_MATMUL`, `SOFTMAX`).
- **Flutter runtime usage evidence:** Strong. Examines exact `TfliteInterpreterManager` and `HybridPhotoEvaluationService` loading code.
- **WSL training/export evidence:** Strong. Tracks datasets, training scripts, and Keras checkpoints.

## 5. Paper-Basis Judgment Review

### NIMA
- Is “close to direct paper implementation” justified? Yes.
- Are 10-bin output, softmax, EMD/CDF loss, AVA/TID/LIVE evidence, and expectation scoring verified? Yes. Verified in `nima_distribution.py`, AVA dataset files, TFLite ops, and Flutter expectation post-processing.

### RGNet
- Is “close to direct paper implementation” justified? Yes.
- Are region composition graph, feature similarity adjacency, graph convolution, and aggregation verified? Yes. Verified in `rgnet.py` and TFLite BATCH_MATMUL / L2_NORMALIZATION ops.

### A-LAMP
- Is “paper-inspired” justified? Yes.
- Are adaptive patch selection, saliency, pattern diversity, multi-patch subnet, and layout-aware graph verified or missing? Missing in Flutter. Multi-patch branch exists, but adaptive saliency patch selection is replaced by fixed anchors in Dart `image_preprocessor.dart`.

### FLIVE / KonIQ
- Is “dataset-based technical quality model” justified? Yes.
- Are dataset, target MOS, backbone, loss, and metrics verified? Yes. Verified that both use MobileNetV2 backbone, trained via `train_regression.py` using `koniq10k` and `paq2piq` datasets.

### MUSIQ
- Is “not used by Flutter, WSL only” justified? Yes.
- Is there no TFLite asset and no Flutter loading path? Yes. Validated by file system search and Flutter app inspection.

## 6. Missing or Weak Evidence
- The exact commands used to copy exported models from WSL to the Flutter `assets/models/` directory are unknown (not verifiable).
- Training logs and validation curves for the models are not verifiable from local evidence.
- The reason why metadata JSON files for NIMA, RGNet, and A-LAMP are missing in the Flutter assets directory (while present in WSL) is unknown (not verifiable).
- The exact distinction between the raw `paq2piq` dataset path and the processed `flive_image` name lacks explicit external provenance tracking locally.

## 7. Recommended Fixes to the Original Report
The original report does an excellent job of bounding its claims to local evidence (repeatedly stating "not verifiable from local evidence"). 
- No specific wording corrections are required to reduce overclaiming, as the report is highly conservative.

## 8. Final Safe Summary
The A-cut aesthetic ensemble deployed in the Flutter application utilizes three models: NIMA, RGNet, and an A-LAMP variant. The implementations of NIMA and RGNet closely follow their respective paper architectures (incorporating EMD loss and graph convolutions), utilizing updated EfficientNetV2 backbones. The A-LAMP model employs a multi-patch architecture but diverges from the paper at runtime, relying on fixed crop anchors instead of adaptive saliency patch selection. For technical image quality scoring, the app deploys KonIQ and FLIVE models; these are not distinct architectural implementations, but rather standard regressors trained on the KonIQ-10k and PaQ-2-PiQ datasets. A MUSIQ model was developed experimentally but is not deployed in the current Flutter application. All runtime claims are reliably backed by on-device TFLite file inspection and active Dart contract code.