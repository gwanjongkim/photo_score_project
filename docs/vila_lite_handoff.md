# VILA-Lite Handoff

## What Was Added

- `src/data_prep/build_ava_captions_manifest.py`
  - Matches the AVA caption/comment CSVs to local AVA image files.
  - Writes `data/processed/ava_captions/{train,val,test}.csv`.
  - Keeps `image_path` repo-relative in the same style as the existing AVA manifests.
- `src/vila/prompt_sets.py`
  - Defines a practical prompt-pair preset for aesthetic reranking and explanation.
- `src/vila/model_loader.py`
  - Loads CLIP-like models from either Hugging Face or OpenCLIP.
  - Supports explicit model specs such as `hf:openai/clip-vit-base-patch32` and `open_clip:ViT-B-32:openai`.
- `src/vila/score_with_prompts.py`
  - Scores images against positive/negative prompt pairs.
  - Produces `vila_score`, per-prompt scores, prompt details, and JSON-friendly explanation signals.
- `src/vila/explain_selection.py`
  - Generates deterministic explanation text from prompt scores.
- `src/vila/explain_acut_selection.py`
  - Synthesizes final A-cut explanations from finalized selector scores, VILA prompt signals, and nearby candidate comparisons.
- `src/infer/score_image_folder_vila.py`
  - Scores a folder of images and writes `vila_scores.jsonl` and `vila_scores.csv`.
- `src/infer/smoke_test_vila_hf_clip.py`
  - Validates the Hugging Face CLIP path by loading `hf:openai/clip-vit-base-patch32`, encoding prompts and an image, and printing embedding shapes.
- `src/vila/datasets.py`
  - Adds a minimal AVA caption image/text dataset for future adaptation work.
- `src/train_vila_lite.py`
  - Adds a scaffold CLI for dataset validation and future contrastive or adapter tuning.
- `src/infer/select_best_shots.py`
  - Can now merge a VILA score CSV on `image_path`.
  - Can optionally use VILA as a top-pool normalized rerank signal.
  - Can attach VILA explanation text and explanation signals to final outputs.
- `src/infer/smoke_test_acut_reasoning.py`
  - Runs the reasoning-enabled selector, verifies the new explanation fields, and prints selected/rejected examples.

## Manifest Build

Run from the repo root:

```bash
PYTHONPATH=. python src/data_prep/build_ava_captions_manifest.py
```

This writes:

- `data/processed/ava_captions/train.csv`
- `data/processed/ava_captions/val.csv`
- `data/processed/ava_captions/test.csv`

Expected key columns:

- `image_path`
- `image_id`
- `comment`
- `mos`

## PyTorch-Side Setup

Use a separate PyTorch env for the VILA-lite path. The current repo already has `.venv_vila`; if you need to rebuild it:

```bash
python -m venv .venv_vila
source .venv_vila/bin/activate
pip install --upgrade pip
pip install torch torchvision transformers open_clip_torch pandas pillow
```

Important:

- The scorer does not ship pretrained CLIP weights inside this repo.
- On first use, Hugging Face or OpenCLIP weights must already be cached locally or be downloadable.
- If you want offline-only behavior, add `--local_files_only`.

## Folder Scoring

Example with a Hugging Face CLIP checkpoint:

```bash
PYTHONPATH=. ./.venv_vila/bin/python src/infer/score_image_folder_vila.py \
  --input_dir /path/to/candidates \
  --output_dir outputs/vila_scores_run \
  --model_name hf:openai/clip-vit-base-patch32 \
  --prompt_preset a_cut_basic \
  --recursive
```

Validated Hugging Face smoke path:

```bash
PYTHONPATH=. ./.venv_vila/bin/python src/infer/smoke_test_vila_hf_clip.py \
  --model_name hf:openai/clip-vit-base-patch32 \
  --image_path test_samples/KakaoTalk_20260330_180646779.jpg \
  --local_files_only
```

This path now validates that the Hugging Face CLIP backend returns final embedding tensors for both prompts and images before folder scoring starts.

Example with OpenCLIP:

```bash
PYTHONPATH=. ./.venv_vila/bin/python src/infer/score_image_folder_vila.py \
  --input_dir /path/to/candidates \
  --output_dir outputs/vila_scores_run \
  --model_name open_clip:ViT-B-32:openai \
  --prompt_preset a_cut_basic \
  --recursive
```

Outputs:

- `outputs/vila_scores_run/vila_scores.jsonl`
- `outputs/vila_scores_run/vila_scores.csv`

The CSV includes:

- `image_path`
- `vila_score`
- `prompt_good_image`
- `prompt_good_composition`
- `prompt_good_lighting`
- `prompt_clear_subject`
- `prompt_clean_background`
- `vila_explanation`
- `explanation_signals`

## Selector Integration

The selector still ranks candidates the same way:

- compute weighted aesthetic and technical components
- optionally rerank the top pool with pairwise and/or VILA
- optionally apply diversity-aware reranking
- write the finalized ranked rows

The new A-cut reasoning layer runs after that finalized ranking step. It does not rescore images. Instead it explains the existing choice using:

- finalized selector values such as `base_score`, `final_score_before_vila`, `final_score_after_rerank`, `final_score`, `aesthetic_component`, `technical_component`, `per_model_contributions`, and rerank deltas
- VILA prompt-side evidence such as `vila_score_raw`, `vila_score_normalized_in_pool`, `prompt_good_composition`, `prompt_good_lighting`, `prompt_clear_subject`, and `prompt_clean_background`
- nearby comparison context from the ranked list, including score deltas versus a higher-ranked, lower-ranked, or cut-line competitor

Required inputs:

- selector outputs containing the existing stage-5 score columns
- optional VILA CSV with `image_path`, `vila_score`, and prompt columns
- canonicalizable `image_path` values so selector rows and VILA rows merge cleanly

Use the existing TensorFlow-side selector env for the main bundle and selector:

```bash
PYTHONPATH=. ./.venv_gpu/bin/python src/infer/select_best_shots.py \
  --scores_csv outputs/stage5_scores/scores.csv \
  --vila_scores_csv outputs/vila_scores_run/vila_scores.csv \
  --enable_vila_rerank \
  --vila_rerank_weight 0.10 \
  --enable_vila_explanations \
  --enable_acut_reasoning \
  --reason_reference_mode nearest_competitor \
  --reason_detail_level full \
  --reason_include_model_contributions \
  --reason_include_vila_signals \
  --reason_include_comparison \
  --output_dir outputs/stage5_with_vila_reasoning \
  --top_k 5
```

### Why Raw `vila_score` Is Not Mixed Directly

The prompt-based `vila_score` lives on a different numeric scale from the selector ensemble score. Blending it directly into `final_score` as an absolute additive term creates a systematic downward drift as `--vila_rerank_weight` increases, even when VILA is only meant to act as local rerank support.

The selector now keeps the original ensemble score intact and uses VILA only as a pool-relative rerank signal:

1. Sort candidates by the selector backbone score.
2. Take the configured rerank pool.
3. Min-max normalize `vila_score_raw` inside that pool only.
4. Center the normalized values around the pool mean.
5. Convert the centered value into `vila_rerank_delta = vila_rerank_weight * centered_signal`.
6. Apply that delta to `final_score_before_vila` to get `final_score_after_rerank`.

This keeps VILA useful as explanation support, tie-break support, and local rerank support without globally depressing absolute scores.

Behavior:

- Existing ensemble logic remains the base ranking path.
- `base_score` stays untouched as the selector backbone score.
- `final_score_before_vila` is the score entering the VILA step.
- `vila_score_raw` is the unnormalized prompt scorer output from `vila_scores.csv`.
- `vila_score_normalized_in_pool` is only populated for rows inside the VILA rerank pool.
- `vila_rerank_delta` is zero-centered across the pool, so increasing VILA weight does not systematically push all scores downward.
- `final_score_after_rerank` is the pre-diversity score after pairwise and VILA rerank adjustments.
- `final_score` remains the final score after any later diversity step.
- When `--enable_acut_reasoning` is on, the selector writes deterministic decision explanations tied to the actual post-rerank order.

New output fields:

- `base_score`
- `final_score_before_vila`
- `vila_score_raw`
- `vila_score_normalized_in_pool`
- `vila_rerank_delta`
- `final_score_after_rerank`
- `final_score`
- `acut_short_reason`
- `acut_detailed_reason`
- `acut_comparison_reason`
- `acut_rejection_reason`
- `acut_explanation_structured`

Reasoning template behavior:

- Explanations remain deterministic and evidence-based.
- Template routing now distinguishes selected, rejected, tradeoff, strong technical win, strong composition win, balanced win, and weak-but-selected cases.
- Short reasons stay at one sentence.
- Detailed reasons stay within two to four sentences depending on `--reason_detail_level`.
- Sentence openings and structure vary deterministically by evidence pattern and image identity, rather than repeating one fixed skeleton.

Example reasoning output shape:

```json
{
  "status": "selected",
  "rank": 5,
  "selector": {
    "base_score": 0.529155,
    "final_score_before_vila": 0.529155,
    "final_score_after_rerank": 0.568042,
    "final_score": 0.568042,
    "aesthetic_component": 0.512813,
    "technical_component": 0.553669,
    "dominant_component": "technical"
  },
  "comparison": {
    "compared_to_image_path": "test_samples/KakaoTalk_20260330_180646779.jpg",
    "score_delta_vs_reference": 0.021508,
    "vila_delta_vs_reference": 0.085547,
    "key_advantages": ["composition", "background cleanliness"],
    "key_disadvantages": ["lighting"]
  },
  "text": {
    "template_family": "strong_technical",
    "short_reason": "Selected on technical consistency, with background cleanliness and composition keeping it ahead of nearby alternatives.",
    "detailed_reason": "Ranked #3 with a final score of 0.568, staying 0.049 above the cut line. The technical backbone stayed stronger, and VILA agreed on composition. The selector leaned on technical quality (0.554 vs 0.513 aesthetic), with the largest weighted contributions from FLIVE patch mean, FLIVE image, and AADB."
  }
}
```

Smoke test:

```bash
PYTHONPATH=. ./.venv_gpu/bin/python src/infer/smoke_test_acut_reasoning.py
```

If `outputs/stage5_scores/scores.csv` is missing, the smoke test falls back to the first available stage-5 score export under `outputs/` and prints which file it used.

## Training Scaffold

Dry-run the caption dataset scaffold:

```bash
PYTHONPATH=. ./.venv_vila/bin/python src/train_vila_lite.py \
  --train_csv data/processed/ava_captions/train.csv \
  --val_csv data/processed/ava_captions/val.csv \
  --batch_size 8
```

Optional model init check:

```bash
PYTHONPATH=. ./.venv_vila/bin/python src/train_vila_lite.py \
  --train_csv data/processed/ava_captions/train.csv \
  --val_csv data/processed/ava_captions/val.csv \
  --init_model \
  --model_name hf:openai/clip-vit-base-patch32 \
  --local_files_only
```

## Implemented Now

- AVA caption manifest creation with file validation and count reporting.
- Inference-first prompt-based VILA-lite scoring using pretrained CLIP-like models.
- Deterministic explanation generation from prompt scores.
- Folder-level VILA scoring CLI with CSV and JSONL outputs.
- Selector-side VILA reranking and explanation attachment.
- Selector-side deterministic A-cut reasoning tied to finalized selector ranking and nearby candidate comparisons.
- A minimal training scaffold for later fine-tuning work.

## Limitations

- The A-cut reasoning layer is deterministic and rule-based; it explains the selector outcome but does not learn better explanations from human preferences.
- Comparison wording is only as good as the available nearby ranked candidates; it does not reason over a larger search graph.
- If VILA scores are absent, explanations fall back to selector-only evidence.
- Prompt-based phrases are constrained to the current `a_cut_basic` prompt set and its column names.

## Still Future Work

- Actual contrastive fine-tuning on AVA image/comment pairs.
- Adapter or LoRA tuning for low-cost VILA-style adaptation.
- A learned VILA-R style reranker beyond prompt averaging.
- Prompt calibration or validation against human A-cut preferences.
- Automatic caching/downloading policy for CLIP weights inside project tooling.
