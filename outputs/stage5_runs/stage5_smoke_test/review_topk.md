# Stage5 Ranking Review

- Run: `stage5_smoke_test`
- Input dir: `/home/omen_pc1/photo_score_project/test_samples`
- Output dir: `outputs/stage5_runs/stage5_smoke_test`
- Images scored: `15`
- Top-5 signature: `KakaoTalk_20260330_180646779_10.jpg > KakaoTalk_20260330_180646779_07.jpg > KakaoTalk_20260330_180646779_11.jpg > KakaoTalk_20260330_180646779_08.jpg > KakaoTalk_20260330_180646779_13.jpg`
- Duplicate groups: `1`

## Top-5

| rank | image | final | base | pre-pairwise | pairwise delta | diversity | pairwise score | similar-to-higher |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | KakaoTalk_20260330_180646779_10.jpg | 0.5732 | 0.5732 | 0.5843 | -0.0111 | 0.0000 | 0.5101 |  |
| 2 | KakaoTalk_20260330_180646779_07.jpg | 0.5459 | 0.5459 | 0.5522 | -0.0063 | 0.0000 | 0.5101 | 0.7229 |
| 3 | KakaoTalk_20260330_180646779_11.jpg | 0.5410 | 0.5446 | 0.5507 | -0.0061 | 0.0036 | 0.5102 | 0.8218 |
| 4 | KakaoTalk_20260330_180646779_08.jpg | 0.5269 | 0.5269 | 0.5298 | -0.0029 | 0.0000 | 0.5102 | 0.7961 |
| 5 | KakaoTalk_20260330_180646779_13.jpg | 0.5262 | 0.5262 | 0.5290 | -0.0028 | 0.0000 | 0.5101 | 0.7052 |

## Duplicate Groups

| group | leader | size | max pair similarity | members |
| --- | --- | --- | --- | --- |
| 1 | KakaoTalk_20260330_180646779_04.jpg | 2 | 0.9008 | KakaoTalk_20260330_180646779_04.jpg, KakaoTalk_20260330_180646779_09.jpg |

## Ranking Checklist

- Best-shot plausibility: does rank 1 still feel like the obvious hero shot on manual review?
- Top-5 conviction: do the first few swaps still feel worse than the current order?
- Duplicate suppression: did any near-duplicates survive too high, or did diversity bury a clearly better frame?
- Product readiness: would `product_ranking.csv` be safe to hand to downstream code as the canonical order?
