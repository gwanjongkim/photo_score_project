# TOPIQ+FLIVE 60:40 Safety Gate Simulation Report

## 1. Executive Summary
We simulated various safety gates on the top log-only candidate (`topiq_existing_flive_grid_s06`, 60% TOPIQ + 40% FLIVE) to reduce its severe False Positives (FPs) from 3 down to 0, matching the current production model's safety. However, the simulation revealed that simple gates based on capping the candidate score using the production score failed to eliminate all severe FPs. 

**Final Decision: TOPIQ_FLIVE_GATE_NEEDS_TUNING**

## 2. Sources Inspected
- `outputs/official_protocol_technical_signal_benchmark_20260604/official_signal_predictions.csv`
- `outputs/official_protocol_technical_signal_benchmark_20260604/official_safety_cases.csv`

## 3. Baseline Candidate and Safety Problem
- **Candidate**: `topiq_flive_60_40`
- **SRCC**: 0.8069
- **Severe FPs**: 3 (vs 0 in production). 
- **The Problem**: Two of the three severe FPs received moderately high scores from the production model itself (52.33 and 62.56) despite having MOS < 40. Therefore, any gate that relies on a low production score to trigger a cap either misses these images entirely or caps them too high (>65), leaving them as severe FPs.

## 4. Gate Definitions
Seven safety gates were simulated:
- **Gate A**: If prod < 45 & base > 65, cap at prod + 10.
- **Gate B**: If prod < 50 & base > 65, cap at prod + 12.
- **Gate C**: If prod < 50 & base - prod > 20, cap at prod + 15.
- **Gate D**: If flive < 50 & topiq > 65, cap at max(flive + 12, 55).
- **Gate E**: If prod < 55 & base > 65, cap at prod + 15.
- **Gate F**: If prod < 50 & base > 65, soft blend (0.7 prod + 0.3 base).
- **Gate G**: Two-stage suspicious optimism gate (Gate A + B combined).

## 5. Overall Gate Metrics
No gate achieved 0 severe FPs.
- **Base SRCC**: 0.8069 | Severe FPs: 3
- **Gate A SRCC**: 0.8076 | Severe FPs: 2
- **Gate G SRCC**: 0.8071 | Severe FPs: 2
- **Gate D SRCC**: 0.8069 | Severe FPs: 3

## 6. Dataset-wise Gate Metrics
Dataset performance remained largely stable for conservative gates (like Gate A), but aggressive gates (like Gate E, which modified 267 rows) began dragging down KonIQ and SPAQ performance without resolving the safety issue.

## 7. Severe FP Fix Audit
- **Fixed by most gates**: 1 severe FP (`4980829250.jpg`), where prod = 44.76.
- **Remaining**: 2 severe FPs (`8090361774.jpg` prod=62.56, `10619.jpg` prod=52.33). Because production scored these >50, adding a buffer (+10 or +15) pushed the cap above 65, failing to resolve the severe FP classification.

## 8. Modified Row Analysis
- **Gate A**: Modified only 6 rows (highly surgical), fixed 1 severe FP.
- **Gate E**: Modified 267 rows, fixed 1 severe FP.
- **Gate D**: Modified 0 rows (condition never met).

## 9. Tradeoff Analysis
The fundamental tradeoff is that the production model is not a perfect oracle for "badness". Trying to force the new candidate to strictly obey the production model's upper bound fails when the production model itself is incorrectly optimistic.

## 10. Recommended Gate Candidate
None of the simulated gates are sufficient to reach 0 severe FPs. If forced to choose, **Gate A** is the most efficient (modifies only 6 rows, slightly improves overall SRCC to 0.8076) but it leaves 2 severe FPs. 

## 11. Final Decision
**TOPIQ_FLIVE_GATE_NEEDS_TUNING**

## 12. Next Actions
- Investigate alternative safety signals (e.g., MDIQA teacher scores, if feasible for offline auditing) to filter the training set.
- Consider accepting a non-zero severe FP rate for log-only profiling, given the massive overall SRCC improvement (0.71 -> 0.80).
