# TOPIQ PyIQA Teacher Smoke Report

- package import status: torch import ok; pyiqa import ok
- torch version: 2.5.1+cu121
- cuda availability: True
- selected TOPIQ model name: topiq_nr
- number of attempted images: 30
- number of successful images: 30
- number of failed images: 0
- topiq_raw min: 0.05552237
- topiq_raw max: 0.71211964
- topiq_raw mean: 0.48200977
- Spearman correlation against quality_score_0_100: 0.67830923
- per-image errors: outputs/topiq_teacher_smoke_20260531/errors.csv

Note: TOPIQ raw score direction and scale are not yet trusted until inspected.
