# Gemma Visual Grounding Diagnostic

## Run

Command:

```bash
LD_LIBRARY_PATH=/home/omen_pc1/photo_score_project/.venv_vila/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/lib/wsl/lib ./.venv_vila/bin/python -m tools.diagnose_gemma_visual_grounding --manifest outputs/gemini_api_teacher_aesthetic3_20260427_revised/test.jsonl --adapter_dir outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/final_adapter --base_model google/gemma-4-E4B-it --output_dir outputs/gemini_api_teacher_aesthetic3_20260427_revised/visual_grounding_diagnostic --hf_home outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/hf_cache --max_examples 6 --max_new_tokens 160 --seed 42 --load_in_4bit
```

## Image Handling

- Train, val, and test image paths all resolve.
- Training uses `datasets.Image()` on the manifest `image` column.
- Normal inference opens the manifest image path with PIL and passes it to the processor as `images=image_rgb`.
- The diagnostic processor inputs for image variants include `pixel_values` and `image_position_ids`.
- The text-only variant has no image tensors but still generates plausible JSON.

## Intervention Result

Mean similarity to original-image output:

- swapped image, same prompt: 0.2263
- blank black image, same prompt: 0.1713
- random noise image, same prompt: 0.1410
- text-only, no image marker: 0.2010

Interpretation:

- The model is not completely dropping images; outputs change when image tensors change.
- The output is still heavily prompt/state conditioned. Valid outputs keep the comment type implied by `selection_state`.
- Blank, noise, and text-only variants produce confident image descriptions, including repeated object hallucinations such as bird, portrait, landscape, sky, and water.

## Teacher Specificity

Heuristic audit over 282 clean teacher rows:

- rows with any concrete term: 272
- rows with concrete visible noun terms: 252
- rows with at least two specific image evidence terms: 188
- rows with generic aesthetic phrases: 215
- rows using generic composition/lighting/color/subject terms without a specific noun from the local list: 30

The teacher set contains useful concrete evidence, but generic aesthetic phrasing is common enough for a small student run to learn reusable templates.

## Recommendation

Do not scale directly to Stage B 1000 yet. First create a high-specificity 100-row teacher probe with a stricter prompt requiring concrete visible entities, spatial relations, and image-specific evidence, then rerun this same visual intervention diagnostic.
