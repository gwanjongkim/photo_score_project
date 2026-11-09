# Context Notes

## 2026-05-29
- Prior backend ablation found PIL Lanczos + FP16 best by C6 metrics, but TensorFlow `resize_with_pad` best by raw TOPIQ MAE.
- This fingerprint comparison should decide preprocessing match using Android tensor fingerprints, not C6 scores alone.
- No training, model export, model mutation, Flutter repo edits, production source edits, or scoring changes are allowed.
- Initial fingerprint comparison points to PIL bilinear as the closest tested resize/pad backend, but also shows Android logged source dimensions are downsampled and do not match WSL's full EXIF-transposed decoded image dimensions.
- Final run generated 200 comparison rows across 50 Android rows and 4 backends with zero failures and HEIF registered.
- Closest tested backend: `pil_bilinear_resize_with_pad`; content/pad dimensions matched 50/50, original decoded source dimensions matched 0/50.
- Next smallest diagnostic is to reproduce Android's logged source dimensions before applying PIL bilinear and recompute fingerprints.
