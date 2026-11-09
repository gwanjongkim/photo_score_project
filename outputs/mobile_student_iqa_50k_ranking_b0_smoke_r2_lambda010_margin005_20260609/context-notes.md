# Phase 21 Context Notes

- Scope: one bounded B0 smoke only. No B1/B2, no full controlled training, no export, no Flutter, no production score, no commit.
- Ranking design: R2 within-dataset pairwise logistic ranking; margin 0.05; lambda_rank 0.10; neutral dataset sample weights.
- Implementation choice: create `scripts/technical_iqa/train_mobile_student_iqa_50k_ranking_smoke.py` to avoid changing default behavior of existing smoke/controlled runners.
- Reuse existing split, model, prediction, metric, safety, and official benchmark helpers where possible.
- Escalated `.venv_tf` run saw one TensorFlow GPU and completed in `242.86` seconds.
- Rank loss was finite: final `0.611503`; valid pair count was nonzero: overall mean `85.050667`, final epoch `84.528`.
- Final train total/huber/rank loss: `0.066094` / `0.004944` / `0.611503`.
- Official Track B Overall SRCC was `0.480260`, below Phase 19 W5-v2 `0.516337` and Phase 16 unweighted B0 `0.703290`.
- No collapse by current checks, but ranking smoke greatly widened prediction distribution: official pred_std `18.596494` and official severe FP `37`.
- Final decision: `RANKING_B0_SMOKE_NEEDS_LABEL_SCALING`.
