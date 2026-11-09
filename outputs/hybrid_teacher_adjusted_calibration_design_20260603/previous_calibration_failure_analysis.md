# Previous Calibration Failure Analysis

## 1. Why Policy D reduced mean shift but worsened p95/p05 gap mismatch

Policy D aligned each source target distribution to its own MOS mean and standard deviation. This reduced source mean shift because MDIQA-source means moved closer to their MOS means, and production-source means also moved closer to their MOS means.

The failure was spread expansion. KonIQ and SPAQ MOS distributions are wider than their raw teacher-target distributions. Policy D therefore increased MDIQA-source spread:

- KonIQ target std expanded from about `4.89` to `11.31`.
- SPAQ target std expanded from about `18.32` to `21.13`.
- Production-source spread also expanded, but less in a way that did not close the source p95/p05 mismatch.

So Policy D improved the center but made the tails less compatible. That is why mean shift improved from `22.6235` to `17.5569`, while p95/p05 gap mismatch worsened from `34.9940` to `39.6805`.

## 2. Why Policy C/D/E created SPAQ FP cases

All three policies raised or expanded parts of the SPAQ target distribution:

- Policy C affine calibration lifted low-MOS SPAQ rows through a learned slope/intercept.
- Policy D mean/std matching expanded SPAQ toward a wider MOS distribution.
- Policy E used the same effective SPAQ mean/std transform because the SPAQ std ratio was inside the conservative cap.

Two SPAQ images crossed the FP rule after calibration:

- `data/raw/spaq/10619.jpg`: MOS `39.9`, calibrated target about `61-63`.
- `data/raw/spaq/11003.jpg`: MOS `26.6`, calibrated target about `61-63`.

They were not severe FP because targets stayed below `65`, but they violated the FP threshold of target `>60` when MOS `<40`.

## 3. Which target ranges caused FP

The FP cases occurred in the SPAQ low-MOS region:

- MOS below `40`.
- Raw Policy A target was safe.
- Calibrated target moved into roughly `61-63`.

The unsafe range is therefore the SPAQ low-MOS upper-tail after upward/variance-expanding calibration, specifically any calibration that lets MOS `<40` rows exceed target `60`.

## 4. Source-step issue diagnosis

The source-step issue is a mix:

- **MDIQA vs Production scale mismatch**: MDIQA-source targets have much lower mean than production-source targets. Baseline MDIQA-source vs production-source mean shift was `22.6235`.
- **True dataset MOS distribution difference**: SPAQ MOS is genuinely lower and wider than KonIQ/FLIVE in this sample, so forcing SPAQ upward would erase real data distribution differences.
- **Fallback proxy is FLIVE-like**: fallback proxy rows are FLIVE image rows with production scores. They match FLIVE production range closely and do not represent real mobile/unknown data.
- **SPAQ distribution is much lower**: SPAQ target and MOS distributions sit far below FLIVE/fallback. This creates a dataset boundary gap, but the good SPAQ SRCC and zero Policy A safety cases indicate that not all of this gap is artificial.

Conclusion: the problem is not solved by equalizing all dataset ranges. Adjusted calibration must reduce obvious source discontinuity while preserving true dataset differences and avoiding SPAQ low-MOS FP.

## 5. Constraints for adjusted calibration

Adjusted calibration should obey these constraints:

- Preserve rank order within each dataset whenever possible.
- Use small monotonic shifts before any variance expansion.
- Do not expand SPAQ upper tail aggressively.
- Keep severe FP at `0`.
- Prefer FP at `0`; any FP policy is diagnostic-only.
- Keep SRCC drop from Policy A at `<= 0.02` per dataset.
- Keep clipping at `<= 5%`, preferably `0`.
- Do not force SPAQ, KonIQ, FLIVE, and fallback to a shared mean.
- Treat fallback proxy as a mechanics test only, not evidence about real mobile/unknown data.
- Recommend 10k dry-run only if range improves meaningfully without worsening gap mismatch or safety.
