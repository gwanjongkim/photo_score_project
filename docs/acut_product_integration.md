# A-cut Product Integration

## Default Command

Recommended default run:

```bash
PYTHONPATH=. ./.venv_gpu/bin/python src/infer/run_acut_pipeline.py \
  --scores_csv outputs/acut_stage5_full_with_pairwise/scores.csv \
  --vila_scores_csv outputs/vila_scores_run/vila_scores.csv \
  --output_dir outputs/acut_product_ready \
  --top_k 5
```

Recommended smoke test:

```bash
PYTHONPATH=. ./.venv_gpu/bin/python src/infer/smoke_test_acut_product_pipeline.py \
  --scores_csv outputs/acut_stage5_full_with_pairwise/scores.csv \
  --vila_scores_csv outputs/vila_scores_run/vila_scores.csv \
  --output_dir outputs/acut_product_ready_smoke \
  --top_k 5
```

Default behavior in `run_acut_pipeline.py`:

- Reuses the existing selector score inputs from `--input_dir`, `--scores_csv`, or `--scores_jsonl`.
- Merges `--vila_scores_csv` when provided.
- Enables VILA rerank and VILA explanations by default when `--vila_scores_csv` is present, unless explicitly disabled with `--disable_vila_rerank` or `--disable_vila_explanations`.
- Enables A-cut reasoning by default, unless explicitly disabled with `--disable_acut_reasoning`.
- Uses `--vila_rerank_weight 0.10` by default.
- Uses `--reason_reference_mode nearest_competitor` by default.
- Uses `--reason_detail_level full` by default.

## Stable Outputs

App-facing outputs:

- `output_dir/app_results.json`
- `output_dir/app_results.csv`
- `output_dir/top_k_summary.json`

Debug and review outputs:

- `output_dir/ranked_results.jsonl`
- `output_dir/ranked_results.csv`
- `output_dir/review_sheet.csv`
- `output_dir/top_k.txt`
- `output_dir/duplicate_groups.json` when diversity rerank is enabled
- `output_dir/scores.csv` and `output_dir/scores.jsonl` when the pipeline is asked to score from `--input_dir`

`app_results.json` is a compact JSON array. Each row contains:

- `rank`
- `image_path`
- `image_file_name`
- `selected`
- `status`
- `base_score`
- `final_score_after_rerank`
- `vila_score_raw`
- `vila_score_normalized_in_pool`
- `acut_short_reason`
- `acut_detailed_reason`
- `acut_comparison_reason`

The app results file intentionally stays as a compact array of rows. Schema and ranking metadata live in `top_k_summary.json` so the row payload stays stable and lightweight.

Use `--export_debug_reasoning` to append:

- `top_model_contributions`
- `explanation_structured`
- `vila_prompt_details`

## Score Field Meanings

- `base_score`: the selector backbone score before optional VILA rerank.
- `final_score_after_rerank`: the score after optional pairwise and VILA rerank, before any later diversity penalty.
- `vila_score_raw`: the unnormalized prompt-based VILA score merged from `vila_scores.csv`.
- `vila_score_normalized_in_pool`: the VILA score after pool-only normalization inside the rerank pool. This is only populated for rows inside the VILA rerank pool.

Canonical ranking semantics:

- `rank`, `selected`, and `status` are always the canonical final order for the run.
- When diversity reranking is disabled, `final_score_after_rerank` reflects the same final order as `rank`.
- When diversity reranking is enabled, `rank`, `selected`, and `status` reflect the post-diversity order, but `final_score_after_rerank` remains the pre-diversity score after pairwise and VILA rerank.
- For diversity-enabled runs, downstream consumers should treat `rank` and `status` as authoritative ordering fields, and treat `final_score_after_rerank` as score context rather than the final ordering key.
- `final_ordering_uses_diversity` is the parseable boolean for that distinction.
- `final_score_matches_final_ranking` is `true` only when `final_score_after_rerank` and the final rank order mean the same thing.

Recommended default weight:

- `0.10` is the project default and keeps VILA as a local tie-break and explanation signal.
- `0.15` is more aggressive and can create larger order changes inside the rerank pool.

## Review Sheet

`review_sheet.csv` is meant for quick manual inspection. It includes:

- all selected top-k rows
- a few near-cut rejected rows
- short reasons
- detailed reasons
- comparison reasons
- rejection reasons

## Summary Export

`top_k_summary.json` is the lightweight handoff summary for the app and quick inspection:

```json
{
  "schema_version": "acut_product_app.v1",
  "generated_at": "2026-04-03T12:34:56Z",
  "ranking_stage": "post_rerank",
  "score_semantics": "rank, selected, status, and final_score_after_rerank all reflect the same final post-rerank order because diversity reranking is disabled.",
  "diversity_enabled": false,
  "final_ordering_uses_diversity": false,
  "final_score_matches_final_ranking": true,
  "top_k": [
    {
      "rank": 1,
      "image_path": "test_samples/example_01.jpg",
      "image_file_name": "example_01.jpg",
      "final_score_after_rerank": 0.597029,
      "acut_short_reason": "Composition carried this frame into the A-cut, helped by technical quality."
    }
  ],
  "selected_count": 5,
  "rejected_count": 10,
  "pipeline_config": {
    "scores_csv": "outputs/acut_stage5_full_with_pairwise/scores.csv",
    "vila_scores_csv": "outputs/vila_scores_run/vila_scores.csv",
    "top_k": 5,
    "enable_vila_rerank": true,
    "vila_rerank_weight": 0.1,
    "enable_vila_explanations": true,
    "enable_acut_reasoning": true,
    "reason_reference_mode": "nearest_competitor",
    "reason_detail_level": "full"
  }
}
```

For diversity-enabled runs, the metadata changes to make the score semantics explicit:

```json
{
  "schema_version": "acut_product_app.v1",
  "ranking_stage": "post_diversity",
  "diversity_enabled": true,
  "final_ordering_uses_diversity": true,
  "final_score_matches_final_ranking": false,
  "score_semantics": "rank, selected, and status reflect the canonical final post-diversity order. final_score_after_rerank remains the pre-diversity score after pairwise and VILA rerank."
}
```
