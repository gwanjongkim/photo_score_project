# Gemma 4 E4B Gemini Teacher 282 Handoff

## Dataset

- Source directory: `outputs/gemini_api_teacher_aesthetic3_20260427_revised`
- Teacher source: Gemini API, Stage A 300 generation
- Clean teacher rows used: 282
- Rejected teacher rows excluded: 18
- `test_vila` policy: not included in train/val/test
- Split counts: train 223, val 25, test 34

## Teacher Schema

The assistant target is structured JSON:

```json
{
  "comment_type": "selected_explanation | near_miss_feedback | rejection_reason",
  "short_reason": "...",
  "detailed_reason": "...",
  "comparison_reason": "..." 
}
```

`comparison_reason` may also be `null`.

## Training

- Config: `configs/gemma4_e4b_gemini_teacher_282_train.json`
- Base model: `google/gemma-4-E4B-it`
- Start point: base Gemma, not pass2b, because the target changed from legacy three-line text to structured JSON.
- Method: QLoRA, 4-bit NF4, LoRA rank 8, alpha 16, dropout 0.05.
- Final LoRA targets: language-model attention/MLP projection regex only; vision/audio modules were excluded.
- Output directory: `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282`
- Final adapter: `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/final_adapter`
- Training completed: yes, 84 optimizer steps, 3 epochs.
- Final training loss: 2.5488
- Final validation loss: 0.3352
- Final validation mean token accuracy: 0.9248

Notes:
- An initial `all-linear` LoRA attempt failed because it wrapped the vision patch embedder and hit byte tensor dropout.
- A second suffix-target attempt still matched unsupported Gemma4 vision/audio clippable linear modules.
- The completed run used a full-name regex restricted to `model.language_model.layers`.

## Evaluation

Prediction files:

- Validation: `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/predictions/val_predictions.jsonl`
- Test: `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/predictions/test_predictions.jsonl`

Validation metrics:

- Matched predictions: 25
- Valid JSON rate: 1.0
- Required fields rate: 1.0
- Comment type valid rate: 1.0
- Exact comment type rate: 1.0
- Forbidden leakage count: 0
- Fake selected weakness count: 0
- Forced praise count: 0
- Near-miss structure fail count: 0
- Mean token F1 against teacher JSON: 0.4000

Test metrics:

- Matched predictions: 34
- Valid JSON rate: 1.0
- Required fields rate: 1.0
- Comment type valid rate: 1.0
- Exact comment type rate: 1.0
- Forbidden leakage count: 0
- Fake selected weakness count: 0
- Forced praise count: 0
- Near-miss structure fail count: 0
- Mean token F1 against teacher JSON: 0.4065

Qualitative result:

- The adapter learned the JSON schema and style constraints well.
- The outputs are not reliable enough for merge/export because qualitative samples show generic explanations and object hallucinations, such as describing a chain image as a lion portrait or a leaf macro as a bird in flight.

## Merge And Export

- Merge completed: no.
- Merge output requested path: `outputs/gemini_api_teacher_aesthetic3_20260427_revised/export_gemma4_282/merged_model`
- Reason merge was skipped: generation quality is structurally valid but visually under-grounded.
- Quantized export completed: no.
- Export summary: `outputs/gemini_api_teacher_aesthetic3_20260427_revised/export_gemma4_282/export_summary.json`
- Current export status: adapter bundle only.

## Mobile Readiness

Ready for mobile integration experiment: no.

The adapter is useful as a first proof that Gemma can learn the structured JSON contract from the Gemini teacher rows, but it is not yet a mobile candidate because visual grounding quality is insufficient.

## Remaining Blocker

The biggest blocker is improving visual grounding, not schema compliance. The next run should use more clean rows and/or a training setup that better preserves image-conditioned behavior before merge or mobile export.
