# Context Notes

## 2026-05-30
- Previous tensor fingerprint comparison selected `pil_bilinear_resize_with_pad` as the closest tested backend.
- Previous comparison also found final content/pad dimensions matched `50/50`, but original decoded dimensions matched `0/50`.
- This follow-up isolates whether matching Android's logged decoded source dimensions reduces the remaining fingerprint gap.
- Final run processed 50 Android rows with zero failures and HEIF registered.
- Matching Android logged source dimensions produced source matches `50/50` and content/pad matches `50/50`, but worsened all key aggregate fingerprint metrics versus previous PIL bilinear.
- Decision: `B. Source-dimension matching does not improve parity; keep previous pil_bilinear backend and investigate another cause.`
