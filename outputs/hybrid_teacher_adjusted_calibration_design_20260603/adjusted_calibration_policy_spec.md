# Adjusted Calibration Policy Specification

## Scope

This specification defines diagnostic calibration candidates for the fixed Phase 4 2k hybrid teacher-label dry-run manifest.

The policies are not final training labels. They are candidates for a 2k revalidation pass only.

## Shared evaluation gates

A policy can be recommended only if:

- severe FP = `0`.
- FP = `0` preferred; FP policies are rejected unless explicitly diagnostic.
- SRCC drop from Policy A is `<= 0.02` per dataset.
- clipping is `<= 5%`.
- mean shift improves over Policy A.
- p95/p05 gap mismatch does not worsen over Policy A.
- range improvement is meaningful compared with Policy B.
- targets remain in `[0, 100]`.
- fallback proxy limitation is documented.

## Baselines

### Policy A: identity strict hybrid

Uses the existing selected Phase 4 target:

- KonIQ-10k: MDIQA `seed1`.
- SPAQ: MDIQA `xd`.
- FLIVE: Production.
- Fallback proxy: Production safety fallback.

### Policy B: 80/20 conservative blend

Uses existing diagnostic blend where available:

- KonIQ/SPAQ: `0.8 * MDIQA + 0.2 * Production`.
- FLIVE/Fallback: Production.
- If production is missing, the row is unavailable for Policy B.

## Adjusted candidates

### Policy F: capped source-step smoothing

Variants:

- `policy_f_shift_cap3`
- `policy_f_shift_cap5`
- `policy_f_shift_cap8`

Logic:

- Start from Policy A.
- Compute production-source central tendency from `production` and `production_safety_fallback`.
- Compute MDIQA-source central tendency from `mdiqa_seed1` and `mdiqa_xd`.
- Shift MDIQA-source targets toward production-source central tendency.
- Cap the absolute shift to `3`, `5`, or `8` points.
- Do not change rank order inside a dataset.
- Do not change variance.
- Clip to `[0, 100]`.

Purpose:

- Reduce source mean shift without lifting SPAQ tails through variance expansion.

Risk:

- SPAQ may still move upward. The revalidation script must reject any variant that creates FP.

### Policy G: MOS-residual conservative calibration

Variants:

- `policy_g_residual_25_cap5`
- `policy_g_residual_50_cap5`

Logic:

- Start from Policy A.
- For each dataset/source, compute `bias = median(target - MOS)`.
- Adjustment is `-fraction * bias`, with fraction `0.25` or `0.50`.
- Cap absolute adjustment to `5` points.
- Before applying to a source, test whether the source FP count would increase. If it would, set that source adjustment to `0`.
- Preserve rank order.
- Clip to `[0, 100]`.

Purpose:

- Reduce systematic bias while avoiding source-wide FP regression.

Risk:

- Uses MOS to design a calibration. It must remain diagnostic until frozen and validated on a larger dry-run.

### Policy H: FP-safe quantile clamp

Variants:

- `policy_h_a_fp_clamp`
- `policy_h_f5_fp_clamp`

Logic:

- Diagnostic only.
- Start from Policy A for `policy_h_a_fp_clamp`.
- Start from Policy F cap5 for `policy_h_f5_fp_clamp`.
- If MOS `<40` and target `>60`, clamp target to `60`.
- Clip to `[0, 100]`.

Purpose:

- Test whether calibration can be made safe if FP is the only blocker.

Reason not final:

- This policy uses MOS per row, so it cannot be directly used for unlabeled future teacher-label generation.

### Policy I: dataset-local z-score soft calibration

Variants:

- `policy_i_zsoft_25_cap5`
- `policy_i_zsoft_50_cap5`

Logic:

- Start from Policy A.
- For each dataset, compute raw target mean/std and MOS mean/std.
- Build a partial z-score target:
  - `z_target = raw_mean + (raw - raw_mean) * (mos_std / raw_std) + (mos_mean - raw_mean)`.
  - Soft target = `(1 - alpha) * raw + alpha * z_target`.
- Alpha is `0.25` or `0.50`.
- Cap per-row absolute adjustment to `5` points.
- Preserve rank order unless clipping/capping creates ties.
- Clip to `[0, 100]`.

Purpose:

- Test a softer version of Policy D that cannot aggressively expand tails.

Risk:

- Still MOS-derived and can raise SPAQ low-MOS rows. Revalidation must reject variants with FP.

## Full 2k revalidation command

```bash
python3 scripts/technical_iqa/run_hybrid_teacher_adjusted_calibration_2k.py
```

## Smoke command

```bash
python3 scripts/technical_iqa/run_hybrid_teacher_adjusted_calibration_2k.py --smoke
```
