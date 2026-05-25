# Stage5 Run Comparison

- Compared runs: `2`
- Top-k focus: `5`

| run | images | top1 | top-k signature | duplicate groups | top-k penalized | top-k pairwise adjusted |
| --- | --- | --- | --- | --- | --- | --- |
| acut_stage5_full_with_pairwise_fast | 15 | KakaoTalk_20260330_180646779_10.jpg | KakaoTalk_20260330_180646779_10.jpg > KakaoTalk_20260330_180646779_07.jpg > KakaoTalk_20260330_180646779_11.jpg > KakaoTalk_20260330_180646779_13.jpg > KakaoTalk_20260330_180646779_08.jpg | 1 | 1 | 4 |
| stage5_smoke_test | 15 | KakaoTalk_20260330_180646779_10.jpg | KakaoTalk_20260330_180646779_10.jpg > KakaoTalk_20260330_180646779_07.jpg > KakaoTalk_20260330_180646779_11.jpg > KakaoTalk_20260330_180646779_08.jpg > KakaoTalk_20260330_180646779_13.jpg | 1 | 1 | 5 |

## Per-Run Top-K

### acut_stage5_full_with_pairwise_fast

1. KakaoTalk_20260330_180646779_10.jpg | final=0.5732 | pairwise_delta= | diversity=0.0000
2. KakaoTalk_20260330_180646779_07.jpg | final=0.5457 | pairwise_delta= | diversity=0.0000
3. KakaoTalk_20260330_180646779_11.jpg | final=0.5410 | pairwise_delta= | diversity=0.0036
4. KakaoTalk_20260330_180646779_13.jpg | final=0.5290 | pairwise_delta= | diversity=0.0000
5. KakaoTalk_20260330_180646779_08.jpg | final=0.5270 | pairwise_delta= | diversity=0.0000

### stage5_smoke_test

1. KakaoTalk_20260330_180646779_10.jpg | final=0.5732 | pairwise_delta=-0.0111 | diversity=0.0000
2. KakaoTalk_20260330_180646779_07.jpg | final=0.5459 | pairwise_delta=-0.0063 | diversity=0.0000
3. KakaoTalk_20260330_180646779_11.jpg | final=0.5410 | pairwise_delta=-0.0061 | diversity=0.0036
4. KakaoTalk_20260330_180646779_08.jpg | final=0.5269 | pairwise_delta=-0.0029 | diversity=0.0000
5. KakaoTalk_20260330_180646779_13.jpg | final=0.5262 | pairwise_delta=-0.0028 | diversity=0.0000
