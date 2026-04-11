# Stage5 Operational Workflow

`stage5_reference` is the frozen default A-cut pipeline for this repo.
It keeps the accepted ranking stack intact:

- baseline AADB, KonIQ, FLIVE-image, FLIVE-patch
- NIMA
- MUSIQ
- A-Lamp
- RGNet
- pairwise rerank
- diversity-aware reranking

The frozen preset lives in `configs/stage5_reference.json`.
It points at the stable `checkpoints/*_gpu/final_model.keras` artifacts and uses `data/processed/aadb/val.csv` as the pairwise reference set.

## Run Stage5 On A New Folder

```bash
bash scripts/run_stage5_selector.sh --input_dir /path/to/new_photo_folder --run_label my_eval
```

This writes a clean run directory under `outputs/stage5_runs/` unless `--output_dir` is provided explicitly.

## Review One Run

```bash
PYTHONPATH=. ./.venv_gpu/bin/python tools/summarize_stage5_results.py outputs/stage5_runs/my_eval_YYYYMMDD_HHMMSS
```

Generated review artifacts:

- `review_topk.csv`
- `review_topk.md`
- `ranking_summary.json`
- `product_ranking.csv`
- `product_ranking.jsonl`

## Compare Multiple Runs

```bash
PYTHONPATH=. ./.venv_gpu/bin/python tools/compare_stage5_runs.py \
  outputs/stage5_runs/run_a \
  outputs/stage5_runs/run_b
```

Generated comparison artifacts:

- `comparison_summary.csv`
- `comparison_summary.json`
- `comparison_review.md`

## Ranking Review Focus

- Is rank 1 still the obvious best shot?
- Does the top-k order feel convincing without hand-waving about score scale?
- Did pairwise rerank help only in the top pool, instead of destabilizing the whole list?
- Did diversity penalties suppress near-duplicates without hiding clearly better frames?
- Is `product_ranking.csv` sufficient for downstream integration without re-parsing the raw selector JSON?
