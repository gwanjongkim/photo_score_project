# Gemma 4 E4B A-cut Smoke Handoff

## Teacher definition

- Weight config: `configs/gemma4_e4b_aesthetic3_teacher.json`
- Inputs used only:
  - `nima_unit_score`
  - `rgnet_score`
  - `alamp_score`
- Formula:
  - `weighted_aesthetic_score = 0.214286 * nima_unit_score + 0.500000 * rgnet_score + 0.285714 * alamp_score`

## Teacher comment format

Every teacher target is exactly:

```text
Strength: ...
Weakness: ...
Verdict: selected because ... / Verdict: not selected because ...
```

## Teacher regeneration scope in this retry

- Local VILA regeneration was attempted on the smoke `train` / `val` / `test` subset only:
  - attempted rows: `15`
  - accepted rows: `0`
- Reason:
  - `VILA1.5-3b` on the available local path could run after the CPU dtype fix, but its outputs did not meet the trust gate for strict 3-line teacher use.
- Result:
  - the rebuilt smoke dataset uses `existing_teacher_output_fallback` for all kept rows
  - fallback comments are cleaner than the previous retry because the formatter now strips rank/score/template phrasing much more aggressively

## Smoke dataset

- Output root: `outputs/gemma4_e4b_aesthetic3_smoke`
- Kept rows after consistency filtering:
  - total: `49`
  - train: `8`
  - val: `3`
  - test: `2`
  - test_vila: `36`
- Teacher format validity:
  - `49 / 49` valid

## Environment and training result

- Correct runtime env for this retry: `.venv_vila`
- Verified stack after correction:
  - `torch 2.11.0+cu130`
  - `transformers 5.6.2`
  - `datasets 4.8.4`
  - `trl 1.2.0`
  - `peft 0.19.1`
  - `accelerate 1.13.0`
- Real smoke fine-tune completed with unrestricted execution:
  - base model: `google/gemma-4-E4B-it`
  - adapter output: `outputs/gemma4_e4b_aesthetic3_smoke/train_run/final_adapter`
  - epochs: `3`
  - train runtime: about `952.7s`

## Student evaluation

- Prediction file:
  - `outputs/gemma4_e4b_aesthetic3_smoke/evaluation/predictions_all.jsonl`
- Aggregate summary:
  - `outputs/gemma4_e4b_aesthetic3_smoke/evaluation/aggregate_summary.json`
- Key results:
  - predictions written: `49`
  - readable/generated predictions: `48`
  - corrupt-image failures: `1`
  - `test` format compliance: `1.0`
  - `test` mean token F1: `0.296255`
  - `test_vila` format compliance: `0.972222`
  - `test_vila` mean token F1: `0.272796`
- Main observed failure mode:
  - the student often collapses to numeric/rubric-style verdicts such as `weighted_aesthetic_score is low`
  - score-or-threshold mentions appeared in `41 / 48` generated predictions

## Export status

- Export summary:
  - `outputs/gemma4_e4b_aesthetic3_smoke/export/export_summary.json`
- Current state:
  - adapter export exists
  - quantized export does not
  - exported artifact:
    - `outputs/gemma4_e4b_aesthetic3_smoke/train_run/final_adapter`

## Biggest remaining weakness

- The student learned the strict 3-line shell, but it still overuses numeric prompt cues in the verdict and weakness lines instead of consistently grounding them in visible evidence.
