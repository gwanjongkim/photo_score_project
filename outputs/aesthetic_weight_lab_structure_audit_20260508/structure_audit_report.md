# Aesthetic Weight Lab Structure Audit

## 1. Environment
- **Path:** `/home/omen_pc1/photo_score_project`
- **Date:** `Fri May  8 21:58:09 KST 2026`
- **Git status summary:** Several modified and untracked files, particularly scripts in `src/infer` and `tools` (e.g., `tools/build_raw_aesthetic_scores_html.py`, `tools/run_aesthetic_ensemble_experiment.py`). The repository contains numerous temporary output directories and virtual environments.

## 2. Existing Output Directory
All files under: `outputs/test_vila_raw_aesthetic_scores_html_20260424`

- `contact_top10_aadb.jpg` (copied image/contact sheet)
- `contact_top10_alamp.jpg` (copied image/contact sheet)
- `contact_top10_nima.jpg` (copied image/contact sheet)
- `contact_top10_rgnet.jpg` (copied image/contact sheet)
- `contact_high_disagreement.jpg` (copied image/contact sheet)
- `ranked_aadb.csv` (raw score data)
- `ranked_alamp.csv` (raw score data)
- `ranked_nima.csv` (raw score data)
- `ranked_rgnet.csv` (raw score data)
- `raw_aesthetic_scores.csv` (raw score data)
- `review_index.html` (HTML report)
- `summary.json` (log/metadata)
- `thumbnail_warnings.csv` (log)
- `thumbs/*.jpg.jpg` (copied image thumbnails for the report)

## 3. Input Image Source
- **Where images came from:** `/home/omen_pc1/photo_score_project/test_vila` (Based on `summary.json`'s `input_dir`)
- **Number of images:** 50
- **Image file extensions:** `.jpg`, `.JPG`
- **Whether images are copied or linked in HTML:**
  - Thumbnails are generated and copied into `thumbs/` and are linked relatively (`src='thumbs/...'`).
  - Full-size original images are NOT copied. They are linked via relative path traversing outside the output folder (`href='../../test_vila/...'`), which breaks portability.

## 4. Script Discovery

| Script | Role | Evidence | CLI args | Hardcoded paths | Notes |
|---|---|---|---|---|---|
| `tools/run_aesthetic_ensemble_experiment.py` | Orchestrates inference and computes initial ensemble scores | Mentioned in `summary.json` as a scoring script | `--input_dir`, `--output_dir`, `--recursive`, `--extensions`, `--top_k`, `--no_contact_sheets` | `DEFAULT_MODEL_PATHS` contains local hardcoded checkpoint paths. | Uses `score_image_folder.py` internally |
| `src/infer/predict_quality_bundle.py` | Model loading and execution | Mentioned in `summary.json` | Inherits bundle args | None immediately visible | Core inference |
| `src/infer/score_image_folder.py` | Traverses folder and loops models | Mentioned in `summary.json` | None (used as module) | None | Core iterator |
| `tools/build_raw_aesthetic_scores_html.py` | Generates HTML report and thumbs | Mentioned as `review_builder_script` | `--source_csv`, `--source_summary`, `--output_dir`, `--top_k`, `--disagreement_k`, `--thumb_width`, `--thumb_height` | `DEFAULT_SOURCE_CSV`, `DEFAULT_SOURCE_SUMMARY`, `DEFAULT_OUTPUT_DIR` heavily depend on prior runs. | Implements live JS weighting |

## 5. Workflow Trace

**input images** (`test_vila/`)
→ **score calculation** (`tools/run_aesthetic_ensemble_experiment.py` via `predict_quality_bundle.py`)
→ **raw score output** (Writes to `outputs/test_vila_aesthetic_ensemble_experiment_20260423/scores_with_ensemble.csv` & `summary.json`)
→ **HTML generation** (`tools/build_raw_aesthetic_scores_html.py` reads the CSV/JSON and builds `review_index.html` and `thumbs/` inside `outputs/test_vila_raw_aesthetic_scores_html_20260424/`)

## 6. Model Usage

| Model | File path | Used by script | Configurable? | Notes |
|---|---|---|---|---|
| NIMA | `checkpoints/nima_ava_gpu/final_model.keras` | `run_aesthetic_ensemble_experiment.py` | Yes (via args) | |
| RGNet | `checkpoints/rgnet_aadb_gpu/final_model.keras` | `run_aesthetic_ensemble_experiment.py` | Yes (via args) | |
| A-LAMP | `checkpoints/alamp_aadb_gpu/final_model.keras` | `run_aesthetic_ensemble_experiment.py` | Yes (via args) | |
| AADB (Stage 5) | `checkpoints/composition_aadb_gpu/final_model.keras` | `run_aesthetic_ensemble_experiment.py` | Yes (via args) | Used as composition component |

## 7. Weight Handling
- **Where weights are defined:** Both in Python (`tools/run_aesthetic_ensemble_experiment.py`) for static calculation, and in HTML/JavaScript (`tools/build_raw_aesthetic_scores_html.py`) for live interaction.
- **Whether weights are hardcoded:** Yes, default weights are hardcoded in both scripts (`DEFAULT_WEIGHT_NIMA_UNIT = 0.333`, etc.).
- **Whether final score is recomputed in Python or HTML:** Both. Python computes `weighted_aesthetic_ensemble` in the CSV, but the HTML report uses Javascript to completely recalculate `combined-score` and `combined-rank` on the fly.
- **Whether interactive sliders exist:** Yes, `<input type="number">` forms exist in the generated HTML for live weight tuning.

## 8. HTML Report Structure
- **HTML file path:** `outputs/test_vila_raw_aesthetic_scores_html_20260424/review_index.html`
- **How thumbnails are displayed:** Base64 or local relative paths (`src='thumbs/001_...'`).
- **How scores are displayed:** HTML table rows color-coded by value (`<td class='score' style='background:hsl(...)'>`).
- **Whether JavaScript reranking exists:** Yes, the report recalculates when the weights sum to 1.0.
- **Whether report is portable to another machine:** No, full images link back to `../../test_vila/`.

## 9. Problems for GitHub/team use
- **outputs directory dependency:** Scripts look for hardcoded timestamped output folders.
- **Non-portable HTML image links:** Requires the original `test_vila/` folder structure to view full images.
- **Hardcoded model paths:** Expects `.keras` files inside `checkpoints/` which are likely ignored or too large for git.
- **Personal images:** `test_vila/` contains raw timestamped `.jpg` personal photos which shouldn't be shared.
- **No Git LFS:** The repository seems heavily bloated with large logs and `outputs/`, hindering simple clones.

## 10. Recommended Refactor Plan
- Migrate logic to a dedicated module:
  - `tools/run_aesthetic_ensemble_experiment.py` → `tools/aesthetic_weight_lab/run_aesthetic_weight_lab.py`
  - `tools/build_raw_aesthetic_scores_html.py` → `tools/aesthetic_weight_lab/html_report.py`
- Add an explicit configuration file: `configs/aesthetic_weight_lab.yaml`.
- Replace hardcoded directory lookups with parameterized `--experiment_dir` parsing.
- Modify HTML generation to either embed images via base64 or copy the original images into the HTML `assets/` folder to ensure portability.

## 11. Minimal GitHub Upload Set
- `tools/aesthetic_weight_lab/*.py` (The refactored scripts)
- `configs/aesthetic_weight_lab.yaml`
- `docs/aesthetic_weight_lab_usage.md` (Documentation explaining how to fetch models)
- A `test_samples/` directory with a few open-source/licensed images instead of personal datasets.
- *Strictly exclude:* large model weights, local `outputs/`, personal photos, and heavy logs.
