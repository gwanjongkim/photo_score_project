# GitHub Readiness Audit (2026-06-05)

## 1. Overall Status
**Status:** `READY_WITH_USER_DECISIONS`

The repository is largely safe to upload to GitHub. The project structure is established, and the `.gitignore` is highly comprehensive, successfully blocking over 300GB of datasets, large language models, TFLite weights, and virtual environments. However, a few manual decisions must be made regarding moderately large outputs (CSVs) and specific TFLite model exports before committing.

## 2. Secrets Audit
**Result:** **CLEAN**

An extensive recursive search for potential secrets (`API_KEY`, `SECRET`, `TOKEN`, `PASSWORD`, etc.) across the codebase was conducted.
*   **Findings:** The codebase cleanly uses environment variables (e.g., `os.environ.get("GEMINI_API_KEY")`, `os.environ.get("HF_TOKEN")`).
*   No hardcoded secret values were found in tracked source files.
*   A `.env.example` file correctly provides dummy placeholders.

## 3. `.gitignore` Quality Audit
**Result:** **STRONG**

*   **Venvs:** Correctly ignored (`.venv`, `.venv_gpu`, `.venv_vila`, `.venv_tf`, `.venv_mdiqa`, `.venv_test`).
*   **Caches:** Python caches (`__pycache__`, `.pytest_cache`), IDE caches, and Hugging Face caches (`**/hf_cache/`, `**/hub/models--*/`) are properly ignored.
*   **Datasets:** `data/raw/`, `data/processed/`, etc., are correctly ignored.
*   **Models/Checkpoints:** Root `/models/`, `/checkpoints/`, and standard extension exclusions (`*.tflite`, `*.h5`, `*.safetensors`, etc.) effectively prevent uploading giant binaries.
*   **Recommendation:** Added `.agents/` and `.gemini/` to the ignore list as these are workspace-specific artifacts for the Gemini CLI and shouldn't be part of the capstone repository.

## 4. Large Files & Artifact Analysis
The project generates massive output artifacts, most of which are safely blocked. However, because `.gitignore` explicitly allows `!outputs/**/*.csv` and `!outputs/**/*.json` to preserve important capstone reports, some large benchmark files are currently tracked:

*   **Tracked Large CSVs (Needs User Decision):**
    *   `outputs/research_protocol_iqa_manifests_20260530/benchmark_all_research_protocol_iqa.csv` (106MB - Exceeds GitHub's 100MB limit)
    *   `outputs/research_protocol_iqa_manifests_20260530/benchmark_flive_paq2piq_toolbox_split.csv` (74MB - Triggers GitHub's 50MB warning)
*   **Action Required:** These files *must* either be excluded via `.gitignore`, zipped, or tracked via Git LFS before pushing.

## 5. Commit Candidates Classification

### A. Must Commit (Source & Docs)
*   `src/`, `configs/`, `scripts/`, `tools/`, `docs/`, `experiments/` (only Python and MD files).
*   `README.md`, `GEMINI.md`, `checklist.md`.
*   All evaluation scripts and reproducible training pipelines.

### B. Should Commit (Reasonable Size Evidences)
*   Final `.md` reports in `outputs/` (e.g., `outputs/acut_capstone_final_report_20260526/`).
*   Small JSON metadata and manifest summaries.
*   Lightweight CSVs (under 10MB) that prove model parity or benchmark results.

### C. Should NOT Commit (Blocked by .gitignore)
*   `data/` (All raw and processed images/CSVs).
*   `checkpoints/` (VILA, NVILA, Gemma).
*   Virtual environments (`.venv*`).
*   `outputs/**/merged_model/`, `.safetensors`, `.h5`, `.keras`.

### D. Needs User Decision (Manual Check Required)
*   Large benchmark CSVs in `outputs/` (>50MB).
*   Specific `.tflite` files in `exports/tflite/` and `assets/models/`. (Currently explicitly *allowed* in `.gitignore` for Flutter integration. Ensure they are under 100MB).
*   `MDIQA/` directory (Consider if this should be a git submodule instead of raw code if it's an external repo).