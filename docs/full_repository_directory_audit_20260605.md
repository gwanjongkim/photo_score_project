# Full Repository Directory Audit (2026-06-05)

## 1. Executive Summary
This report provides a deep, directory-level audit of the `photo_score_project` repository (totaling 362GB). The repository is a massive capstone research environment containing source code, Python scripts, Flutter/TFLite export artifacts, external dependencies, and hundreds of gigabytes of models, weights, and datasets.

The source code is relatively clean, and no hardcoded secret values were found in tracked files. However, a significant **Git tracking risk** was discovered: over 5,700 files inside `functions/node_modules/` and over 3,000 files in `outputs/` are currently tracked by Git. These must be purged from Git tracking to prevent severe repository bloat before the final upload.

## 2. Final Decision
**`READY_TO_COMMIT_SOURCE_AFTER_REVIEW`**

The source code and documentation directories are safe to commit. However, the repository is currently blocked from a blind `git add .` due to thousands of untracked heavy artifacts and improperly tracked `node_modules`. Manual Git cache cleanup is required first.

## 3. Top-Level Directory Table

| Directory | Size | Likely Purpose | Classification |
| :--- | :--- | :--- | :--- |
| `src/` | 4.6M | Core Python architecture, training loops. | Source Code |
| `scripts/` | 3.2M | Evaluation, pipeline, parity execution scripts. | Source Code |
| `tools/` | 1.2M | Patch extraction, ONNX/TFLite export tools. | Source Code |
| `configs/` | 184K | JSON/YAML experiment configurations. | Source Code |
| `docs/` | 96K | Documentation, audits, readmes. | Documentation |
| `experiments/`| 1.9M | Reimplementation code (e.g., ICAA TF Native). | Source Code |
| `MDIQA/` | 1.3G | Cloned external submodule for baseline IQA. | External Dep / Models |
| `open_clip/` | 32M | Cloned external submodule for CLIP dependencies. | External Dep |
| `android_gemma_latency_harness/` | 96K | Android app for hardware latency testing. | Mobile Source Code |
| `functions/` | 88M | Firebase Cloud Functions backend code. | Source Code (Has `node_modules`) |
| `data/` | 33G | Raw images, processed benchmark CSVs. | Dataset |
| `outputs/` | 219G | Training logs, `.h5` weights, JSONL manifests, reports. | Output Artifacts |
| `checkpoints/`| 24G | VILA, NVILA HuggingFace `.safetensors`. | Model Artifacts |
| `models/` | 3.9G | Gemma 4 E4B, litert models, TFLite files. | Model Artifacts |
| `weights/` | 542M | Trained weight binaries. | Model Artifacts |
| `exports/` | 192M | TFLite deliverables explicitly exported for mobile. | Mobile Artifacts |
| `forWeights/` | 237M | Handoff metadata and previous weight iterations. | Handoff Artifacts |
| `forWeights2/`| 550M | Current model weight handoff directories. | Handoff Artifacts |
| `.venv*` | ~24G | Python virtual environments (GPU, TF, VILA, etc.).| Caches / Venvs |
| `test_*` | ~160M | Test images, sample outputs, validation scripts. | Test Artifacts |

## 4. Directory-by-Directory Findings

*   **`src/`, `scripts/`, `tools/`, `configs/`:** Contain the vast majority of the project's logic. Clean and ready to upload. Consists of Python files, shell scripts, and configs.
*   **`functions/`:** Contains TypeScript/JavaScript for Firebase. **Critical Issue:** `node_modules` is currently tracked by Git.
*   **`MDIQA/` & `open_clip/`:** External cloned repositories. `MDIQA` contains large `.pth` checkpoint files (over 1GB) that should not be tracked directly in this repository.
*   **`outputs/`:** The largest directory (219GB). Contains thousands of JSONL scoring manifests, HTML review pages, massive `.h5`/`.keras` model checkpoints, and HuggingFace caches. Only the `.md` reports inside this directory should be committed.
*   **`data/`:** Contains `raw/` and `processed/` subsets for AVA, FLIVE, etc. Correctly ignored by Git.
*   **`checkpoints/`, `models/`, `weights/`:** Contain pure binary weights (HuggingFace `model-*.safetensors`, `.tflite`, etc.). Should strictly not be committed to GitHub.
*   **`exports/`:** Contains final TFLite models for Flutter integration (e.g., `rgnet_aadb_gpu.tflite`). These are currently explicitly allowed in `.gitignore`. They are safe to upload *only* if each file is under 100MB.
*   **`forWeights/` & `forWeights2/`:** Include a mix of documentation (`HANDOFF.md`, `MODEL_CARD.md`) and `.tflite` model files meant for mobile developers.

## 5. Source Code Classification

*   **Core Logic (`src/`, `scripts/`, `tools/`):** Contains ~300 Python files doing dataset processing, distillation (`gemma_distill/`), model definitions (`alamp.py`, `rgnet.py`), and TFLite conversion. **Decision: Commit.**
*   **Configs (`configs/`):** Experiment YAMLs. **Decision: Commit.**
*   **Firebase (`functions/`):** Backend cloud functions. **Decision: Commit source, but purge `node_modules/`.**
*   **Submodules (`MDIQA/`, `open_clip/`):** Used for baseline evaluations. **Decision: Needs Cleanup.** Best practice is to ignore their internal checkpoints and use `.gitmodules` or strip their `.git` folders if tracking as raw source.
*   **Root Scripts (`*.py`):** Benchmarking and evaluation utilities. **Decision: Commit.**

## 6. Model / Data / Artifact Classification

*   **Massive Binaries (`+500MB`):** `checkpoints/NVILA-8B/` and `VILA1.5-3b/` `.safetensors`, `outputs/` `best.weights.h5` and `.keras` files, and `models/litert/gemma-4-E4B-it.litertlm` (3.5GB). **Decision: DO NOT COMMIT.** Blocked safely by `.gitignore`.
*   **Mobile Artifacts (`*.tflite`):** Located in `exports/tflite/` and `forWeights2/models/aesthetic/`. Ranging from 50MB to 168MB. **Decision: NEEDS USER DECISION.** GitHub blocks files >100MB. The 168MB `icaa_dat_tf_native_fp16.tflite` will be rejected by GitHub unless tracked via Git LFS.
*   **Outputs (`outputs/`):** Contains huge HuggingFace hub cache blobs and intermediate evaluation JSONL files. **Decision: Commit ONLY `.md` reports.**

## 7. Git Tracked/Untracked Risk Analysis

**Tracked Risks (CRITICAL):**
1.  `functions/node_modules/`: There are over 5,700 files tracked here. This is a severe anti-pattern and will bloat the repository.
2.  `outputs/`: There are 3,053 tracked files in `outputs/`. Many are small logs, but tracking intermediate experiment logs makes the git history massive and unreadable.
3.  `exports/tflite/` & `forWeights2/`: Some `.tflite` models are tracked. If any of these are over 100MB, `git push` will fail permanently requiring a history rewrite.

**Untracked Risks:**
*   Thousands of untracked files in `experiments/`, `outputs/`, and `MDIQA/`. A blind `git add .` will capture them and crash the upload.

## 8. Secret / Local Path Audit Summary

*   **Secrets:** **CLEAN.** Grep found zero hardcoded secrets. `GEMINI_API_KEY`, `HF_TOKEN`, and `GOOGLE_API_KEY` are successfully loaded via `os.environ.get()`. Dummy variables exist in `.env.example`. (Redacted during scan).
*   **Local Paths:** Hardcoded absolute paths (e.g., `/home/omen_pc1/...`) appear extensively inside `outputs/*.jsonl` manifests and log files. This is harmless output data. Source code scripts correctly utilize relative or OS-agnostic pathing.

## 9. Recommended `.gitignore` Changes

You must append the following to `.gitignore`:

```text
# Exclude node_modules everywhere
node_modules/

# Exclude MDIQA checkpoints
MDIQA/checkpoints/*.pth
MDIQA/**/*.pth

# Exclude large benchmark CSVs and output logs
outputs/**/*.csv
outputs/**/*.jsonl
outputs/**/*.log
outputs/**/*.html
outputs/**/*.json
```
*Note: Do not ignore `outputs/**/*.md` so capstone reports remain tracked.*

## 10. Recommended Commit Strategy

To safely upload this repository, you must execute a surgical clean-up of the Git cache before pushing. Follow the commands in the next section carefully.