# TOPIQ Android Preprocess Debug Report

## 1. Summary

The requested Android TOPIQ preprocessing debug audit could not be completed because the input log `c6_topiq_preprocess_debug_20260529.log` was not available in the mounted local filesystem.

Checked locations:
- `/Users/gwanjung_mac/StudioProjects/pozy_app`: not mounted in this environment.
- `/home/omen_pc1/pozy_app`: Flutter checkout exists, but the log was not present.
- `/home/omen_pc1/photo_score_project`: no matching log or `[AcutTopiqPreprocessDebug]` / `[AcutC6Candidate]` lines were found.

Available WSL comparison file:
- `/home/omen_pc1/photo_score_project/outputs/c6_android_wsl_parity_mismatch_diagnosis_20260528/preprocess_debug/preprocess_stats.csv`

That WSL CSV contains model tensor stats for a small mismatch set, but it does not contain Android log entries, original image dimensions, resized content dimensions, pad dimensions, pad value, channel stats, or first tensor values. Therefore Android-vs-WSL resize/pad diagnosis is still blocked on the missing Android log.

## 2. Parsed Debug Entries

Parsed `[AcutTopiqPreprocessDebug]` entries: 0.

Reason: no source log was found locally.

Required fields not available:
- filename
- original width/height
- resized content width/height
- pad left/right/top/bottom
- pad value
- tensor min/max/mean/std
- channel means/stds
- first 12 tensor values

## 3. Tensor Range and Normalization Check

Android tensor range could not be verified because no `[AcutTopiqPreprocessDebug]` lines were available.

WSL TOPIQ stats visible in `preprocess_stats.csv` show raw `0..255`-scale tensors for the sampled mismatch images:

| filename | WSL TOPIQ min | WSL TOPIQ max | WSL TOPIQ mean | WSL TOPIQ std | shape |
| --- | ---: | ---: | ---: | ---: | --- |
| `20230201_181510.jpg` | 0.0 | 239.9513 | 84.6700 | 86.1044 | `[384, 384, 3]` |
| `1675564423029-24.jpg` | 0.0 | 255.0 | 60.9698 | 76.4964 | `[384, 384, 3]` |
| `20250204_174237(0).jpg` | 0.0 | 255.0 | 86.8087 | 66.0515 | `[384, 384, 3]` |
| `20250205_000515.jpg` | 0.0 | 255.0 | 22.0798 | 22.0868 | `[384, 384, 3]` |
| `20250204_135520.jpg` | 0.0 | 255.0 | 131.2021 | 85.2446 | `[384, 384, 3]` |

From WSL evidence only, the WSL side is not normalized to `0..1`. Android normalization status is unknown until the missing debug log is available.

## 4. Resize and Padding Check

Target size verification is blocked for Android because no debug entries were available.

Checks that still need the Android log:
- target is `384x384`
- resized content dimensions are consistent with aspect-ratio preserving resize
- left + content width + right equals `384`
- top + content height + bottom equals `384`
- pad value matches the expected TOPIQ pad value
- tensor first values agree with pad/content placement

The available WSL `preprocess_stats.csv` confirms TOPIQ tensor shape `[384, 384, 3]` for sampled mismatch images, but it does not expose resized content width/height or pad dimensions. Direct resize/pad dimension comparison is therefore not possible from this CSV alone.

## 5. C6 Candidate Lines

Parsed `[AcutC6Candidate]` entries: 0.

Reason: no source log was found locally.

No Android candidate scores could be matched to preprocessing entries.

## 6. Comparison Need with WSL Stats

The WSL stats file is locally available at:

`/home/omen_pc1/photo_score_project/outputs/c6_android_wsl_parity_mismatch_diagnosis_20260528/preprocess_debug/preprocess_stats.csv`

However, it only provides per-model aggregate tensor stats:
- filename
- model
- min
- max
- mean
- std
- shape

It does not provide the resize/pad geometry or first tensor values requested for the Android comparison. To complete the parity audit, copy or provide `c6_topiq_preprocess_debug_20260529.log` and ensure it contains `[AcutTopiqPreprocessDebug]` lines for the current top mismatches.

## 7. Likely Root Cause

Likely root cause remains `unknown` from the available local evidence.

Current evidence does not support these conclusions yet:
- pad dimension mismatch: not verifiable without Android dimensions.
- normalization mismatch: not verifiable on Android; WSL sampled TOPIQ tensors are `0..255`.
- RGB/channel mismatch: not verifiable without Android channel stats or first tensor values.
- tensor range mismatch: not verifiable without Android tensor min/max.
- resize rounding mismatch: plausible remaining candidate, but not locally proven.

The missing log is the blocking issue. The available WSL stats alone cannot distinguish resize rounding, pad geometry, channel order, or Android tensor range errors.

## 8. Recommendation

Next smallest diagnostic step: place `c6_topiq_preprocess_debug_20260529.log` in the Flutter checkout or provide its contents, then rerun this audit against the actual `[AcutTopiqPreprocessDebug]` and `[AcutC6Candidate]` lines.

The next log should include entries for at least these remaining high-mismatch images:
- `20230201_181300.jpg`
- `1675564423029-9.jpg`
- `20250129_102520.jpg`
- `1675564423029-24.jpg`
- `1715855266134.jpg`
- `20250129_103313.jpg`
- `1675564423029-6.jpg`
- `20240510_191716.jpg`
- `20250205_072644.jpg`
- `IMG_2785.JPG`
