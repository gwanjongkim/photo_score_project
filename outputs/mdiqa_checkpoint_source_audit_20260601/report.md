# MDIQA Checkpoint Source Audit Report

## 1. Objective
Determine the availability and mapping of SPAQ and FLIVE dataset-specific checkpoints for MDIQA.

## 2. Findings

### 2.1. Local Discovery
The following key checkpoints were found in `MDIQA/checkpoints/`:
- `mdiqa_shared_5t_4a_mbv3_koniq_seed1.pth`: Standard KonIQ-10k fusion model.
- `mdiqa_shared_5t_4a_mbv3_koniq_xd.pth`: Cross-Dataset fusion model (466MB).
- `net_best.pth`: Unified best model for evaluation and restoration (105MB).

### 2.2. Code Evidence
- `train_MDIQA_xd_koniq.yml` validates on FLIVE using `mdiqa_shared_5t_4a_mbv3_koniq_xd.pth`.
- `MultiDimSharedIQAModel` in `multidim_shared_arch.py` hardcodes `net_best.pth`.
- No `.yml` files or code references exist for `MDIQA_SPAQ.pth` or `MDIQA_FLIVE.pth`.

### 2.3. External Audit (Drive/GitHub)
- Official Google Drive listing does not contain files named `SPAQ` or `FLIVE` for fusion models.
- GitHub README mentions SPAQ training for *technology heads*, but not for a specific fusion model.

## 3. Mapping Recommendations

| Variant | Recommended Checkpoint | Rationale |
|---|---|---|
| `mdiqa_koniq` | `mdiqa_shared_5t_4a_mbv3_koniq_seed1.pth` | Standard training target for KonIQ-10k. |
| `mdiqa_flive` | `mdiqa_shared_5t_4a_mbv3_koniq_xd.pth` | Cross-Dataset model explicitly tested on FLIVE in official configs. |
| `mdiqa_spaq` | `mdiqa_shared_5t_4a_mbv3_koniq_xd.pth` | Best available cross-dataset representative. |
| `mdiqa_unified`| `net_best.pth` | Official unified model for general tasks. |

## 4. Verdict
Specific `MDIQA_SPAQ.pth` and `MDIQA_FLIVE.pth` files do not exist.
The `koniq_xd.pth` model should be used for SPAQ and FLIVE evaluations.
`net_best.pth` should be benchmarked as a unified alternative.

## 5. Next Steps
1. Update `scripts/technical_iqa/run_mdiqa_variant_sanity_benchmark.py` to use `koniq_xd` and `net_best` for SPAQ/FLIVE.
2. Run sanity benchmark to confirm performance.
