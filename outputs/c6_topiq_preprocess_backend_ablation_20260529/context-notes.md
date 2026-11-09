# Context Notes

## 2026-05-29
- Current EXIF/HEIF-fixed parity is partial: C6 MAE about `1.3470`, Max AE `5.2387`, SRCC about `0.9690`.
- Remaining mismatch is suspected around TOPIQ resize backend/interpolation/rounding, FP16 vs FP32 TFLite behavior, and C6 cap branch sensitivity.
- The local Android preprocess debug report exists at `/home/omen_pc1/pozy_app/docs/acut_topiq_preprocess_debug_20260529.md`, but that prior audit could not parse Android tensor lines because the raw log was unavailable in this environment.
- This run uses the prompt-confirmed Android TOPIQ contract and compares backend outputs directly against Android CSV scores.
- Final ablation processed all 50 images with HEIF registered and zero failures.
- Best primary-sort result was `pil_lanczos_resize_with_pad` with TOPIQ FP16: C6 MAE `1.1397`, C6 Max AE `4.6256`, C6 SRCC `0.9744`, TOPIQ MAE `1.3893`, no score inversion.
- The current TensorFlow backend remained better on raw TOPIQ MAE (`1.1965`) but worse on C6 MAE/Max AE (`1.3470` / `5.2387`), likely due to C6 branch sensitivity and KonIQ interaction.
- Because C6-best and TOPIQ-best disagree, the report decision is `D. Need Flutter tensor export`; do not treat PIL Lanczos as the confirmed Android TOPIQ preprocessing match without tensor evidence.
