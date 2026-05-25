# A-LAMP Paper-Faithfulness Audit

## 1. Environment
- CWD: `/home/omen_pc1/photo_score_project`
- Date: Mon May 11 2026
- Git Status: Dirty tree (Modified files including `.gitignore`, `README.md`, `requirements-vila.txt`, `requirements.txt`, etc.)

## 2. Current Project A-LAMP Inventory
- **Current practical A-LAMP source files:** `src/models/alamp.py`, `src/train/train_alamp.py`, `src/datasets/native_size_dataset.py`
- **Current architecture summary:** A refined practical A-Lamp-style aesthetic model containing a separate global layout-aware branch and a multi-patch branch. Fuses branches using a learned patch attention weighting (`WeightedPatchPooling`).
- **Input shape:** `global_view` `[B, 384, 384, 3]` and `patches` `[B, 5, 224, 224, 3]`
- **Number of patch inputs:** 5
- **Global input shape:** 384x384
- **Backbone used:** EfficientNetV2B0 for the global branch, MobileNetV2 for the patches branch.
- **Patch extraction method:** Adaptive saliency-like patch selection (edge/variance) with heuristic search during WSL training; Flutter app uses fixed crop anchors.
- **Output type:** Scalar regression (linear)
- **Dataset used:** AADB for current model, predicting normalized regression scores.
- **Training objective:** MSE/MAE regression.
- **Export/deployment status:** Highly deployable, currently used in the primary Flutter app (`alamp_aadb_gpu.tflite`).
- **Best known metrics:** ~0.58 SRCC on AADB val set.
- **App-oriented vs paper-oriented:** Highly app-oriented implementation optimized for practical mobile performance.

## 3. A-LAMP Paper Protocol Evidence
- **Paper title:** A-Lamp: Adaptive Layout-Aware Multi-Patch deep convolutional neural network for photo aesthetic assessment
- **Task/dataset used:** AVA classification (typically) or distribution regression.
- **AVA classification protocol:** Binary (mean_score >= 5.0)
- **Label rule:** mean_score >= 5.0
- **Input preprocessing:** Saliency extraction based.
- **Patch selection method:** Saliency-map-based adaptive patch selection with diversity constraint.
- **Patch count:** 5 patches.
- **Patch size:** 224x224.
- **Global image branch or not:** Yes (Layout-aware subnet).
- **Backbone:** VGG16.
- **Whether VGG16 is used:** Yes.
- **Whether columns share weights:** Yes, shared backbone.
- **Multi-patch subnet design:** Aggregates multiple patches.
- **Layout-aware subnet design:** Includes Object/Global attribute graphs.
- **Object/global attribute graph design:** Detailed graph structures (missing from local codebase).
- **Loss:** CrossEntropy (for classification).
- **Optimizer/training schedule if available:** Not found locally.
- **Reported metrics:** Accuracy/F1.
- **Reported AVA result:** Not found locally.
- **Whether AADB is used by the paper or not:** Not found locally.

## 4. Paper vs Current Implementation Gap Matrix
See `alamp_gap_matrix.json`.

## 5. Official Reproduction Risk
- **High Risk:** The original paper uses complex Object/Global attribute graphs for its Layout-aware subnet which are not present in our codebase and whose exact construction rules are ambiguous or missing locally.
- **Missing Saliency:** We rely on basic edge/variance for WSL saliency, while the paper uses more robust saliency-map-based proposals.
- **Architecture Differences:** VGG16 is much heavier than EfficientNet/MobileNet. Deploying a true paper-faithful model to the mobile app is unfeasible. We must avoid claiming "official reproduction."

## 6. Feasible A-LAMP-paper-AVA Versions
See `alamp_feasibility_decision.json`.

## 7. Recommended Version To Implement First
**Version 0: A-LAMP-paper-AVA-v0 approximation**
An A-LAMP-paper-oriented approximation is the best starting point. This avoids the high risk and unknown details of the object/attribute graphs while shifting the architecture closer to the paper's shared VGG16 baseline and focusing on the AVA binary classification objective.

## 8. Gemini vs Codex Role Split
- **Gemini CLI:** Performed the paper-faithfulness audit, defined the gap matrix, and formulated the implementation plan.
- **Codex CLI:** Should be used next to implement `src/models/alamp_paper_ava.py`, the training script, and configuration.
- **User terminal (tmux/nohup):** Full dataset training once Codex has successfully implemented the scripts and completed smoke/mid-run validations.

## 9. Training/Evaluation Plan
Codex should generate isolated scripts specifically for `A-LAMP-paper-AVA-v0`. It will run a quick smoke test to verify graph construction, VGG16 loading, and loss computation without running a full training loop.

## 10. What Not To Claim
- Do not claim this is an "official A-LAMP reproduction".
- Do not use "same as the paper" or "paper-faithful". Always use "A-LAMP-paper-oriented approximation" or "A-LAMP-style model".
- Do not deploy this heavy VGG16-based model to the Flutter application.

## 11. Next Codex Prompt
```text
Implement the A-LAMP-paper-AVA-v0 approximation track. 
Create isolated A-LAMP-paper-AVA-v0 files only. 
Do not touch current practical A-LAMP models (`src/models/alamp.py`), Flutter code, or `forWeights/`.
Do not start full training; run smoke and mid-run only to verify construction.

Tasks:
1. Create `src/models/alamp_paper_ava.py`: Use a shared VGG16 backbone for both the global branch and patch branches. Use deterministic/adaptive patch extraction approximation and simple layout features. Include a binary AVA classifier head. Do not attempt to implement the exact object/global attribute graph.
2. Create `src/train/train_alamp_paper_ava.py` to train on the AVA dataset using binary labels (mean_score >= 5.0) and CrossEntropy loss.
3. Create `src/eval/evaluate_alamp_paper_ava.py` to calculate Accuracy, F1, etc.
4. Create configuration `configs/paper_benchmarks/alamp_paper_ava_classification.yaml`.
5. Output artifacts must go to `outputs/alamp_paper_ava_classification_YYYYMMDD/`.
```