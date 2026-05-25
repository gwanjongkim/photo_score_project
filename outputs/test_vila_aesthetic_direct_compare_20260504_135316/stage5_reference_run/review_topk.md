# Stage5 Ranking Review

- Run: `stage5_reference_run`
- Input dir: ``
- Output dir: `outputs/test_vila_aesthetic_direct_compare_20260504_135316/stage5_reference_run`
- Images scored: `50`
- Top-5 signature: `20231231_150055.jpg > 1675564423029-24.jpg > 20250204_174237(0).jpg > 1675342165226-3.jpg > 20250204_163550.jpg`
- Duplicate groups: `3`

## Top-5

| rank | image | final | base | pre-pairwise | pairwise delta | diversity | pairwise score | similar-to-higher |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 20231231_150055.jpg | 0.6303 | 0.6515 | 0.6515 | -0.0212 | 0.0000 | 0.5102 |  |
| 2 | 1675564423029-24.jpg | 0.6101 | 0.6277 | 0.6277 | -0.0176 | 0.0000 | 0.5101 | 0.5104 |
| 3 | 20250204_174237(0).jpg | 0.6036 | 0.6201 | 0.6201 | -0.0165 | 0.0000 | 0.5101 | 0.5246 |
| 4 | 1675342165226-3.jpg | 0.6017 | 0.6179 | 0.6179 | -0.0162 | 0.0000 | 0.5101 | 0.6179 |
| 5 | 20250204_163550.jpg | 0.5889 | 0.6028 | 0.6028 | -0.0139 | 0.0000 | 0.5101 | 0.5400 |

## Duplicate Groups

| group | leader | size | max pair similarity | members |
| --- | --- | --- | --- | --- |
| 1 | 1675342165226-3.jpg | 2 | 0.9026 | 1675342165226-3.jpg, 1675342165226-5.jpg |
| 2 | 1675564423029-9.jpg | 2 | 0.9194 | 1675564423029-9.jpg, 1675564423029-6.jpg |
| 3 | 1720499666964.jpg | 2 | 0.9722 | 1720499666964.jpg, 1720499666985.jpg |

## Ranking Checklist

- Best-shot plausibility: does rank 1 still feel like the obvious hero shot on manual review?
- Top-5 conviction: do the first few swaps still feel worse than the current order?
- Duplicate suppression: did any near-duplicates survive too high, or did diversity bury a clearly better frame?
- Product readiness: would `product_ranking.csv` be safe to hand to downstream code as the canonical order?
